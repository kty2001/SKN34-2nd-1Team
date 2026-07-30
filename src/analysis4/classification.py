"""분석 4 트랙 A — 이탈 예측 (분류)

노트북 10절 트랙 A 설계를 구현한다.

- 시나리오 S1(전체) / S2(누수 의심 변수 제외)를 나란히 학습해 성능 차이를 보고한다.
- 파생변수 적용 전(raw) / 후(derived)를 각각 저장해 델타를 측정한다 (진행 순서 3~4단계).
- 불균형 처리는 기본 미적용 (9.9절 — 미적용이 대부분 지표에서 최고).

이 모듈은 Streamlit에 의존하지 않는다. 화면 표시는 pages/analysis4.py가 담당하고,
여기서는 DataFrame / dict 만 반환한다.

CLI 실행 (프로젝트 루트에서):
    python -m src.analysis4.classification
"""

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.common import SEED
from src.analysis4 import features as ft

# models/analysis4 — cwd가 아니라 이 파일 위치를 기준으로 잡는다.
MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "analysis4"

MODEL_NAMES = ["logreg", "rf", "xgb", "mlp"]

# 9.5절 — 시나리오마다 주력 모델이 다르다. S1은 XGBoost, S2는 LogisticRegression.
PRIMARY_MODEL = {"S1": "xgb", "S2": "logreg"}

SCORING = {
    "roc_auc": "roc_auc",
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
    "accuracy": "accuracy",
}


def build_models(balanced=False, pos_weight=None, params=None):
    """후보 모델 4종.

    스케일이 필요한 모델은 반드시 Pipeline에 StandardScaler를 넣는다.
    스케일러를 따로 저장하면 적용을 빠뜨려도 에러 없이 조용히 틀린 예측이 나온다.

    balanced=True 이면 불균형 보정을 켠다 (9.9절 — Recall 우선일 때만 사용).
    params={모델명: {하이퍼파라미터}} 로 튜닝 결과를 주입할 수 있다.
    """
    params = params or {}
    lr_kw = {"max_iter": 3000, "random_state": SEED}
    rf_kw = {"n_estimators": 300, "random_state": SEED, "n_jobs": -1}
    xgb_kw = {
        "n_estimators": 300,
        "learning_rate": 0.1,
        "max_depth": 4,
        "random_state": SEED,
        "eval_metric": "logloss",
        "n_jobs": -1,
    }
    if balanced:
        lr_kw["class_weight"] = "balanced"
        rf_kw["class_weight"] = "balanced"
        xgb_kw["scale_pos_weight"] = pos_weight

    lr_kw.update(params.get("logreg", {}))
    rf_kw.update(params.get("rf", {}))
    xgb_kw.update(params.get("xgb", {}))

    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(**lr_kw)),
        ]),
        "rf": RandomForestClassifier(**rf_kw),
        "xgb": XGBClassifier(**xgb_kw),
        # 9.3절 — 딥러닝 비교군. 성능 기대가 아니라 부적합 근거 제시용.
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                                    early_stopping=True, random_state=SEED)),
        ]),
    }


def compare_models(df=None, scenario="S1", derived=True, models=None, balanced=False):
    """교차검증으로 후보 모델 비교표 반환 (학습·저장 없음)"""
    ft.set_seed()
    X, y = ft.get_xy(df, scenario=scenario, derived=derived)
    pos_weight = (len(y) - y.sum()) / y.sum()
    candidates = build_models(balanced=balanced, pos_weight=pos_weight)
    names = models or MODEL_NAMES

    rows = []
    for name in names:
        cv_res = cross_validate(candidates[name], X, y, cv=ft.get_splitter(),
                                scoring=SCORING, n_jobs=1)
        row = {"모델": name}
        row.update({k: cv_res[f"test_{k}"].mean() for k in SCORING})
        row["auc_std"] = cv_res["test_roc_auc"].std()
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("roc_auc", ascending=False, ignore_index=True)
    out.insert(0, "구성", ft.scenario_tag(scenario, derived))
    return out


def bundle_path(scenario, derived, name, tag=None):
    """저장 경로 — 시나리오·파생 여부·모델명·단계를 파일명에 드러낸다.

    tag를 주면 기본 모델을 덮어쓰지 않고 별도 저장된다.
    '고도화 전/후'를 비교하려면 '전'이 남아 있어야 한다.
    """
    dv = "derived" if derived else "raw"
    suffix = f"_{tag}" if tag else ""
    return MODEL_DIR / f"clf_{scenario}_{dv}_{name}{suffix}.joblib"


def train_and_save(df=None, scenario="S1", derived=True, models=None,
                   balanced=False, tag=None, params=None, threshold=None):
    """홀드아웃으로 학습·평가 후 모델 + 메타데이터 저장 → 요약 DataFrame 반환

    학습은 train(75%)에서만 하고 test(25%)는 최종 보고용으로 남긴다.
    교차검증 점수도 train 안에서만 계산해 test 정보가 새지 않게 한다.

    params  : {모델명: {하이퍼파라미터}} — 튜닝 결과 주입용
    threshold: {모델명: float} — 분류 임계값. 생략 시 0.5
    """
    ft.set_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X_tr, X_te, y_tr, y_te = ft.get_holdout(df, scenario=scenario, derived=derived)
    pos_weight = (len(y_tr) - y_tr.sum()) / y_tr.sum()
    candidates = build_models(balanced=balanced, pos_weight=pos_weight, params=params)
    names = models or MODEL_NAMES
    thresholds = threshold or {}

    rows = []
    for name in names:
        model = candidates[name]
        cut = float(thresholds.get(name, 0.5))

        cv_res = cross_validate(model, X_tr, y_tr, cv=ft.get_splitter(),
                                scoring=SCORING, n_jobs=1)
        cv_scores = {k: float(cv_res[f"test_{k}"].mean()) for k in SCORING}

        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= cut).astype(int)
        test_scores = {
            "roc_auc": float(roc_auc_score(y_te, proba)),
            "f1": float(f1_score(y_te, pred)),
            "recall": float(recall_score(y_te, pred)),
            "precision": float(precision_score(y_te, pred)),
            "accuracy": float(accuracy_score(y_te, pred)),
        }

        bundle = {
            "model": model,
            "features": list(X_tr.columns),   # 예측 시 순서 검증용
            "scenario": scenario,
            "derived": derived,
            "balanced": balanced,
            "model_name": name,
            "stage": ft.STAGE_TUNED if tag else ft.STAGE_BASE,
            "threshold": cut,
            "params": (params or {}).get(name),
            "cv": cv_scores,
            "test": test_scores,
            "seed": SEED,
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        }
        path = bundle_path(scenario, derived, name, tag)
        joblib.dump(bundle, path)

        rows.append({
            "단계": ft.STAGE_TUNED if tag else ft.STAGE_BASE,
            "구성": ft.scenario_tag(scenario, derived),
            "모델": name,
            "임계값": cut,
            "cv_auc": cv_scores["roc_auc"],
            "test_auc": test_scores["roc_auc"],
            "test_f1": test_scores["f1"],
            "test_recall": test_scores["recall"],
            "test_precision": test_scores["precision"],
            "파일": path.name,
        })

    return pd.DataFrame(rows).sort_values("test_auc", ascending=False, ignore_index=True)


def load_bundle(scenario="S1", derived=True, name=None, tag=None):
    """저장한 모델 묶음 로드. name 생략 시 시나리오별 주력 모델(9.5절)."""
    name = name or PRIMARY_MODEL[scenario]
    path = bundle_path(scenario, derived, name, tag)
    if not path.exists():
        raise FileNotFoundError(
            f"모델이 없습니다: {path}\n"
            "먼저 학습하세요: python -m src.analysis4.classification"
        )
    return joblib.load(path)


def predict_proba(bundle, X):
    """저장된 피처 목록·순서를 검증한 뒤 이탈 확률 반환"""
    expected = bundle["features"]
    missing = [c for c in expected if c not in X.columns]
    if missing:
        raise KeyError(f"입력에 없는 피처: {missing}")
    # 순서가 어긋나면 에러 없이 조용히 틀리므로 저장된 순서로 다시 세운다.
    return bundle["model"].predict_proba(X[expected])[:, 1]


def find_best_threshold(y_true, proba, metric="f1"):
    """임계값 탐색 → (임계값, 점수). 기본 0.5가 최적이 아닐 수 있다."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    prec, rec = prec[:-1], rec[:-1]
    if metric == "f1":
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    elif metric == "recall":
        score = rec
    else:
        raise ValueError("metric은 'f1' 또는 'recall'")
    best = int(np.argmax(score))
    return float(thr[best]), float(score[best])


def evaluate(df=None, scenario="S1", derived=True, name=None, n_repeats=5):
    """저장 모델을 홀드아웃 test로 평가 → 지표·곡선·중요도 dict

    반환 키: metrics, curve_roc, confusion, importance, threshold, meta
    """
    ft.set_seed()
    bundle = load_bundle(scenario, derived, name)
    _, X_te, _, y_te = ft.get_holdout(df, scenario=scenario, derived=derived)

    proba = predict_proba(bundle, X_te)
    pred = (proba >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y_te, proba)
    best_thr, best_f1 = find_best_threshold(y_te, proba, "f1")

    perm = permutation_importance(bundle["model"], X_te[bundle["features"]], y_te,
                                  scoring="roc_auc", n_repeats=n_repeats,
                                  random_state=SEED, n_jobs=-1)
    importance = (pd.DataFrame({"피처": bundle["features"], "중요도": perm.importances_mean})
                  .sort_values("중요도", ascending=False, ignore_index=True))

    return {
        "metrics": pd.DataFrame([bundle["test"]]).T.rename(columns={0: "test"}),
        "curve_roc": pd.DataFrame({"fpr": fpr, "tpr": tpr}),
        "confusion": pd.DataFrame(
            confusion_matrix(y_te, pred),
            index=["실제 유지", "실제 이탈"], columns=["예측 유지", "예측 이탈"],
        ),
        "importance": importance,
        "threshold": {"기본": 0.5, "F1 최적": best_thr, "F1 최적 점수": best_f1},
        "meta": {k: bundle[k] for k in
                 ("scenario", "derived", "model_name", "seed", "n_train", "n_test", "trained_at")},
    }


# ────────────────────────────────────────────────────────────────
# 고도화 — Optuna 튜닝 + 임계값 최적화
# ────────────────────────────────────────────────────────────────
def threshold_from_cv(model, X_tr, y_tr, metric="f1"):
    """train 교차검증의 out-of-fold 예측으로 임계값 결정

    ⚠ test로 임계값을 고르면 '고도화 후' 성능이 부풀려져 전/후 비교가 무의미해진다.
      반드시 train 안에서만 정한다.
    """
    oof = cross_val_predict(model, X_tr, y_tr, cv=ft.get_splitter(),
                            method="predict_proba", n_jobs=1)[:, 1]
    return find_best_threshold(y_tr, oof, metric)[0]


def _search_space(name, trial):
    """Optuna 탐색 공간 — 모델별"""
    if name == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if name == "logreg":
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "solver": "liblinear",
        }
    if name == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }
    raise ValueError(f"튜닝 지원 대상이 아닙니다: {name}")


def tune(df=None, scenario="S1", derived=True, name=None, n_trials=30):
    """Optuna 하이퍼파라미터 탐색 → (최적 파라미터, 최적 CV AUC)

    train 안의 교차검증만 사용한다. test는 보지 않는다.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    ft.set_seed()
    name = name or PRIMARY_MODEL[scenario]
    X_tr, _, y_tr, _ = ft.get_holdout(df, scenario=scenario, derived=derived)
    pos_weight = (len(y_tr) - y_tr.sum()) / y_tr.sum()

    def objective(trial):
        space = _search_space(name, trial)
        model = build_models(pos_weight=pos_weight, params={name: space})[name]
        return cross_validate(model, X_tr, y_tr, cv=ft.get_splitter(),
                              scoring="roc_auc", n_jobs=1)["test_score"].mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = dict(study.best_params)
    if name == "logreg":
        best["solver"] = "liblinear"
    return best, float(study.best_value)


def train_advanced(df=None, scenario="S1", derived=True, name=None, n_trials=30):
    """고도화 — 튜닝 + 임계값 최적화 후 tag='tuned'로 저장 → 요약 DataFrame"""
    ft.set_seed()
    name = name or PRIMARY_MODEL[scenario]
    best_params, best_cv = tune(df, scenario, derived, name, n_trials)

    X_tr, _, y_tr, _ = ft.get_holdout(df, scenario=scenario, derived=derived)
    tuned_model = build_models(params={name: best_params})[name]
    cut = threshold_from_cv(tuned_model, X_tr, y_tr, metric="f1")

    out = train_and_save(df, scenario=scenario, derived=derived, models=[name],
                         tag="tuned", params={name: best_params},
                         threshold={name: cut})
    out["튜닝_cv_auc"] = best_cv
    return out


def compare_stages(df=None, configs=None):
    """고도화 전/후 비교표 → (상세, 피벗)

    저장된 모델이 없으면 (빈 DataFrame, None). 반환 형태를 항상 2개 튜플로 고정한다 —
    호출부가 타입을 보고 분기하지 않아도 되게.
    """
    configs = configs or [("S1", True), ("S2", False)]
    rows = []
    for scenario, derived in configs:
        name = PRIMARY_MODEL[scenario]
        for tag, stage in ((None, ft.STAGE_BASE), ("tuned", ft.STAGE_TUNED)):
            path = bundle_path(scenario, derived, name, tag)
            if not path.exists():
                continue
            bundle = joblib.load(path)
            rows.append({
                "구성": ft.scenario_tag(scenario, derived),
                "모델": name,
                "단계": stage,
                "임계값": bundle.get("threshold", 0.5),
                **{k: round(v, 4) for k, v in bundle["test"].items()},
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out, None

    pivot = out.pivot_table(index=["구성", "모델"], columns="단계",
                            values=["roc_auc", "f1", "recall"])
    for metric in ("roc_auc", "f1", "recall"):
        if (metric, ft.STAGE_TUNED) in pivot.columns:
            pivot[(metric, "변화")] = (pivot[(metric, ft.STAGE_TUNED)]
                                      - pivot[(metric, ft.STAGE_BASE)])
    return out, pivot.round(4).sort_index(axis=1)


# ────────────────────────────────────────────────────────────────
# 통합 실행
# ────────────────────────────────────────────────────────────────
def run_all(df=None, n_trials=30):
    """트랙 A 전체 — 기본 4구성 학습 후 주력 모델 고도화

    반환: dict — 'summary'(long 형식, 트랙 통합용) + 상세표들
    """
    ft.set_seed()
    frames = [train_and_save(df, scenario=s, derived=d)
              for s in ("S1", "S2") for d in (False, True)]
    detail = pd.concat(frames, ignore_index=True)

    advanced = pd.concat(
        [train_advanced(df, scenario=s, derived=d, n_trials=n_trials)
         for s, d in (("S1", True), ("S2", False))],
        ignore_index=True) if n_trials else pd.DataFrame()

    full = pd.concat([detail, advanced], ignore_index=True) if len(advanced) else detail

    # long 형식 요약 (features.SUMMARY_COLUMNS 규약)
    rows = []
    for _, r in full.iterrows():
        for metric in ("test_auc", "test_f1", "test_recall", "test_precision"):
            rows.append({"트랙": "A 분류", "단계": r["단계"], "구성": r["구성"],
                         "모델": r["모델"], "지표": metric, "값": r[metric]})
    summary = ft.make_summary(rows)

    pivot = detail.pivot_table(index="모델", columns="구성", values="test_auc")
    delta = pd.DataFrame({
        f"{s} 델타": pivot[f"{s}/derived"] - pivot[f"{s}/raw"]
        for s in ("S1", "S2")
        if f"{s}/raw" in pivot.columns and f"{s}/derived" in pivot.columns
    })

    result = {"summary": summary, "detail": full, "auc_pivot": pivot, "derived_delta": delta}
    if len(advanced):
        result["stage_detail"], result["stage_compare"] = compare_stages(df)
    return result


if __name__ == "__main__":
    ft.set_seed()
    r = run_all()

    print("=" * 78)
    print("트랙 A — 구성별 학습 결과 (holdout test 기준)")
    print("=" * 78)
    print(r["detail"].round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("기본 모델 test AUC")
    print("=" * 78)
    print(r["auc_pivot"].round(4).to_string())
    print("\n파생변수 적용 델타 (derived - raw)")
    print(r["derived_delta"].round(4).to_string())

    if "stage_compare" in r:
        print("\n" + "=" * 78)
        print("고도화 전/후 비교 (주력 모델)")
        print("=" * 78)
        print(r["stage_compare"].to_string())

    print("\n" + "=" * 78)
    print(f"통합 요약 (long) — {len(r['summary'])}행")
    print("=" * 78)
    print(r["summary"].head(8).to_string(index=False))
    print(f"\n저장 파일 수: {len(list(MODEL_DIR.glob('clf_*.joblib')))}")
