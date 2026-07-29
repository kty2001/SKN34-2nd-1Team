"""분석 4 트랙 B — 신규 가입자 이탈 타이밍 예측

노트북 10절 트랙 B 설계를 구현한다. 타깃은 `Lifetime`(첫 방문 이후 경과 개월 수).

⚠ 데이터 제약 (9절 / 10절 트랙 B)
    이탈 시점이 관측된 고객은 이탈자 1,061명뿐이고 91.8%가 0~2개월에 몰려 있다
    (평균 0.99, 표준편차 1.11). 유지자 2,939명은 "아직 이탈하지 않음" = 우측 절단이다.
    따라서 단순 회귀(B-3)는 평균 예측(RMSE 약 1.11)을 이기기 어렵다.

세 가지 접근을 비교한다.
    B-1  순서형 분류 — 0개월 / 1개월 / 2개월+   (이탈자 1,061명)
    B-2  생존분석   — 유지자를 censored로 포함  (전체 4,000명)   ← 실측 결과 유일하게 성공
    B-3  회귀       — 비교군                    (이탈자 1,061명)

★ 실행 결과 (signal_check 로 재현 가능)
    B-2 만 작동한다 (C-index 0.88). B-1 과 B-3 는 베이스라인을 넘지 못한다.
    원인은 모델이 아니라 데이터다 — **이탈자 내부에는 이탈 시점을 가르는 신호가 없다.**
    이탈자 1,061명 안에서 피처와 개월수의 상관은 최대 |r|=0.079에 그치고,
    이진으로 재정의해도 AUC 0.44~0.47(무작위 이하)이다.
    반면 같은 피처로 "이탈 여부"를 예측하면 AUC 0.966이 나온다.
    즉 이 피처들은 '누가 이탈하는가'는 알지만 '언제 이탈하는가'는 모른다.
    B-1/B-3의 실패는 보고서에서 **삭제할 것이 아니라 근거와 함께 제시할 결과**다.

Streamlit에 의존하지 않는다. DataFrame / dict 만 반환한다.

CLI 실행 (프로젝트 루트에서):
    python -m src.analysis4.regression
"""

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight
from statsmodels.miscmodels.ordinal_model import OrderedModel
from xgboost import XGBClassifier, XGBRegressor

from src.common import SEED
from src.analysis4 import features as ft

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "analysis4"

# B-1 타깃 구간 — 이탈자의 91.8%가 0~2개월에 몰려 있어 3범주로 나누면 38 / 39 / 23%로 균형이 좋다.
TIMING_BINS = [-0.1, 0.5, 1.5, np.inf]
TIMING_LABELS = ["0개월", "1개월", "2개월+"]

# 접근별 대표 모델 — load_bundle()에서 name 생략 시 사용
DEFAULT_MODEL = {"b1": "xgb", "b2": "cox", "b3": "xgb"}


# ────────────────────────────────────────────────────────────────
# 저장 · 로드
# ────────────────────────────────────────────────────────────────
def bundle_path(approach, scenario="S1", derived=True, name=None):
    """저장 경로 — approach 는 'b1' / 'b2' / 'b3'

    저장하는 쪽과 읽는 쪽이 같은 함수를 쓰게 해 파일명이 어긋나지 않도록 한다.
    """
    if approach not in DEFAULT_MODEL:
        raise ValueError(f"approach는 {list(DEFAULT_MODEL)} 중 하나여야 합니다: {approach!r}")
    name = name or DEFAULT_MODEL[approach]
    dv = "derived" if derived else "raw"
    return MODEL_DIR / f"reg_{approach}_{scenario}_{dv}_{name}.joblib"


def load_bundle(approach="b2", scenario="S1", derived=True, name=None):
    """저장한 트랙 B 모델 묶음 로드"""
    path = bundle_path(approach, scenario, derived, name)
    if not path.exists():
        raise FileNotFoundError(
            f"모델이 없습니다: {path}\n"
            "먼저 학습하세요: python -m src.analysis4.regression"
        )
    return joblib.load(path)


def _align(bundle, X):
    """저장된 피처 목록·순서로 입력을 다시 세운다.

    순서가 어긋나도 예외 없이 조용히 틀린 값이 나오므로 예측 전에 반드시 거친다.
    """
    expected = bundle["features"]
    missing = [c for c in expected if c not in X.columns]
    if missing:
        raise KeyError(f"입력에 없는 피처: {missing}")
    return X[expected]


def predict_class_proba(bundle, X):
    """B-1 — 구간별 확률 (열 순서는 bundle['labels'])"""
    return np.asarray(bundle["model"].predict_proba(_align(bundle, X)))


def predict_months(bundle, X):
    """B-3 — 예상 이탈 개월수"""
    return np.asarray(bundle["model"].predict(_align(bundle, X)), dtype=float)


def cox_risk(bundle, X):
    """B-2 — Cox 선형예측자 = 상대 위험도

    ⚠ 절대 시점이 아니다. 값 자체에는 단위가 없고 **다른 고객과의 순위 비교**로만 읽는다.
      (베이스라인 위험함수를 저장하지 않으므로 개별 생존곡선은 복원하지 않는다.)
    """
    Z = bundle["scaler"].transform(_align(bundle, X))
    return np.asarray(Z @ bundle["params"], dtype=float)


# ────────────────────────────────────────────────────────────────
# 데이터 구성
# ────────────────────────────────────────────────────────────────
def make_timing_data(df=None, scenario="S1", derived=True):
    """이탈자만 뽑아 (X, 개월수, 3범주 라벨) 반환 — B-1 / B-3 공용"""
    df = ft.load_data() if df is None else df
    work = ft.add_derived(df) if derived else df

    churned = work[work[ft.TARGET] == 1]
    X = churned[ft.get_timing_features(scenario, derived)].copy()
    y_months = churned[ft.TIMING_TARGET].astype(float).copy()
    y_class = pd.cut(y_months, TIMING_BINS, labels=[0, 1, 2]).astype(int)
    return X, y_months, y_class


def make_survival_data(df=None, scenario="S1", derived=True):
    """전체 고객 (X, 기간, 이벤트) 반환 — B-2 생존분석용

    유지자는 event=0(우측 절단)으로 들어가 4,000명을 모두 쓴다.
    """
    df = ft.load_data() if df is None else df
    work = ft.add_derived(df) if derived else df
    X = work[ft.get_timing_features(scenario, derived)].copy()
    time = work[ft.TIMING_TARGET].astype(float).copy()
    event = work[ft.TARGET].astype(int).copy()
    return X, time, event


def _split(X, y_strat, test_size=0.25):
    """트랙 B 전용 분할 (타깃이 달라 트랙 A의 get_holdout을 쓸 수 없다)"""
    return train_test_split(X, y_strat, test_size=test_size,
                            stratify=y_strat, random_state=SEED)


# ────────────────────────────────────────────────────────────────
# B-1  순서형 분류 (주력)
# ────────────────────────────────────────────────────────────────
class OrderedLogit:
    """statsmodels OrderedModel 래퍼 — 등급 순서를 실제로 활용하는 유일한 모델

    sklearn 다항 분류는 0/1/2를 순서 없는 범주로 취급하지만,
    이 모델은 "0개월 < 1개월 < 2개월+"라는 순서 정보를 사용한다.
    """

    def __init__(self, seed=SEED):
        self.seed = seed
        self.scaler = StandardScaler()

    def fit(self, X, y):
        self.columns_ = list(X.columns)
        Z = self.scaler.fit_transform(X)
        self.res_ = OrderedModel(np.asarray(y), Z, distr="logit").fit(
            method="bfgs", disp=0, maxiter=300)
        return self

    def predict_proba(self, X):
        Z = self.scaler.transform(X[self.columns_])
        return np.asarray(self.res_.predict(Z))

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def build_timing_classifiers():
    """B-1 후보 — 베이스라인 2종 포함

    ⚠ macro-F1을 볼 때 `baseline(최빈)`만 두면 모델이 실제보다 좋아 보인다.
      최빈 클래스만 답하면 나머지 두 클래스의 F1이 0이라 macro 평균이 0.19까지 내려가기 때문이다.
      계층 비율대로 무작위 추측하는 `baseline(무작위)`가 정직한 기준선이다.
    """
    return {
        "baseline(최빈)": lambda: DummyClassifier(strategy="most_frequent", random_state=SEED),
        "baseline(무작위)": lambda: DummyClassifier(strategy="stratified", random_state=SEED),
        "ordered_logit": lambda: OrderedLogit(seed=SEED),
        "logreg": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, random_state=SEED)),
        ]),
        "rf": lambda: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "xgb": lambda: XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=4,
                                     objective="multi:softprob", num_class=3,
                                     random_state=SEED, eval_metric="mlogloss", n_jobs=-1),
    }


def _cv_classify(factory, X, y, cv):
    """모델 팩토리 기반 수동 교차검증 (OrderedLogit도 동일하게 처리)"""
    f1s, accs = [], []
    for tr, te in cv.split(X, y):
        model = factory().fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        f1s.append(f1_score(y.iloc[te], pred, average="macro"))
        accs.append(accuracy_score(y.iloc[te], pred))
    return float(np.mean(f1s)), float(np.mean(accs))


def train_ordinal(df=None, scenario="S1", derived=True, save=True):
    """B-1 학습·평가 → 요약 DataFrame"""
    ft.set_seed()
    X, _, y = make_timing_data(df, scenario, derived)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    cv = ft.get_splitter()

    rows = []
    for name, factory in build_timing_classifiers().items():
        cv_f1, cv_acc = _cv_classify(factory, X_tr, y_tr, cv)

        model = factory().fit(X_tr, y_tr)
        pred = model.predict(X_te)
        row = {
            "구성": f"{scenario}/{'derived' if derived else 'raw'}",
            "모델": name,
            "cv_macro_f1": cv_f1,
            "cv_accuracy": cv_acc,
            "test_macro_f1": f1_score(y_te, pred, average="macro"),
            "test_accuracy": accuracy_score(y_te, pred),
        }
        rows.append(row)

        if save and not name.startswith("baseline"):
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump({
                "model": model, "features": list(X_tr.columns), "approach": "B-1",
                "scenario": scenario, "derived": derived, "model_name": name,
                "labels": TIMING_LABELS, "cv": {"macro_f1": cv_f1, "accuracy": cv_acc},
                "test": {k: row[k] for k in ("test_macro_f1", "test_accuracy")},
                "seed": SEED, "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
                "trained_at": datetime.now().isoformat(timespec="seconds"),
            }, bundle_path("b1", scenario, derived, name))

    return pd.DataFrame(rows).sort_values("test_macro_f1", ascending=False, ignore_index=True)


# ────────────────────────────────────────────────────────────────
# B-2  생존분석
# ────────────────────────────────────────────────────────────────
def concordance_index(time, risk, event):
    """Harrell's C-index (우측 절단 대응)

    비교 가능한 쌍: 이벤트가 발생한 i 와, i보다 오래 관측된 j.
    위험도가 높은 쪽이 먼저 이탈하면 일치(concordant).
    0.5 = 무작위, 1.0 = 완벽.
    """
    time = np.asarray(time, dtype=float)
    risk = np.asarray(risk, dtype=float)
    event = np.asarray(event).astype(bool)

    concordant = 0.0
    comparable = 0.0
    for i in np.flatnonzero(event):
        later = time > time[i]
        n_later = int(later.sum())
        if n_later == 0:
            continue
        other = risk[later]
        concordant += float((risk[i] > other).sum()) + 0.5 * float((risk[i] == other).sum())
        comparable += n_later
    return concordant / comparable if comparable else float("nan")


def survival_km(df=None, by=None):
    """Kaplan-Meier 생존곡선 → DataFrame(시점, 생존확률, [그룹])

    by 에 컬럼명을 주면 그룹별 곡선을 세로로 쌓아 반환한다.
    """
    df = ft.load_data() if df is None else df

    def _curve(sub, label=None):
        sf = SurvfuncRight(sub[ft.TIMING_TARGET], sub[ft.TARGET])
        out = pd.DataFrame({"개월": sf.surv_times, "생존확률": sf.surv_prob})
        if label is not None:
            out[by] = label
        return out

    if by is None:
        return _curve(df)
    return pd.concat([_curve(sub, val) for val, sub in df.groupby(by)], ignore_index=True)


def survival_cox(df=None, scenario="S1", derived=True, save=True):
    """B-2 Cox 비례위험모형 → dict(계수표, C-index, 요약)"""
    ft.set_seed()
    X, time, event = make_survival_data(df, scenario, derived)

    # 계수 비교가 가능하도록 표준화 후 적합
    scaler = StandardScaler()
    Z = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    idx_tr, idx_te = train_test_split(X.index, test_size=0.25,
                                      stratify=event, random_state=SEED)
    res = PHReg(time.loc[idx_tr], Z.loc[idx_tr], status=event.loc[idx_tr]).fit()

    risk_te = Z.loc[idx_te].values @ res.params            # 선형예측자 = 위험도
    c_index = concordance_index(time.loc[idx_te], risk_te, event.loc[idx_te])

    coef = (pd.DataFrame({"피처": X.columns, "계수": res.params, "p_value": res.pvalues})
            .assign(위험비=lambda t: np.exp(t["계수"]))
            .sort_values("계수", key=np.abs, ascending=False, ignore_index=True))

    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "params": res.params, "features": list(X.columns), "scaler": scaler,
            "approach": "B-2", "scenario": scenario, "derived": derived,
            "model_name": "cox", "c_index": float(c_index), "seed": SEED,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        }, bundle_path("b2", scenario, derived))

    return {"coef": coef, "c_index": float(c_index),
            "n_total": int(len(X)), "n_event": int(event.sum()),
            "censored_rate": float(1 - event.mean())}


# ────────────────────────────────────────────────────────────────
# B-3  회귀 (비교군)
# ────────────────────────────────────────────────────────────────
def build_timing_regressors():
    return {
        "baseline(평균)": DummyRegressor(strategy="mean"),
        "linreg": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "rf": RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
        "xgb": XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=4,
                            random_state=SEED, n_jobs=-1),
    }


def train_regression(df=None, scenario="S1", derived=True, save=True):
    """B-3 학습·평가 → 요약 DataFrame (베이스라인 RMSE 대비 개선폭으로 판정)"""
    ft.set_seed()
    X, y_months, y_class = make_timing_data(df, scenario, derived)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_months, test_size=0.25,
                                              stratify=y_class, random_state=SEED)
    rows = []
    for name, model in build_timing_regressors().items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
        rows.append({
            "구성": f"{scenario}/{'derived' if derived else 'raw'}",
            "모델": name,
            "test_rmse": rmse,
            "test_mae": float(mean_absolute_error(y_te, pred)),
        })
        if save and name != "baseline(평균)":
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump({
                "model": model, "features": list(X_tr.columns), "approach": "B-3",
                "scenario": scenario, "derived": derived, "model_name": name,
                "test": {"rmse": rmse}, "seed": SEED,
                "trained_at": datetime.now().isoformat(timespec="seconds"),
            }, bundle_path("b3", scenario, derived, name))

    out = pd.DataFrame(rows).sort_values("test_rmse", ignore_index=True)
    base = out.loc[out["모델"] == "baseline(평균)", "test_rmse"].iloc[0]
    out["베이스라인_대비"] = (out["test_rmse"] - base).round(4)
    return out


# ────────────────────────────────────────────────────────────────
# 통합 실행
# ────────────────────────────────────────────────────────────────
def target_distribution(df=None):
    """B-1 타깃 분포 — 3범주가 균형 잡혔는지 확인용"""
    _, y_months, y_class = make_timing_data(df)
    vc = y_class.value_counts().sort_index()
    return pd.DataFrame({
        "구간": TIMING_LABELS,
        "인원": vc.values,
        "비율(%)": (vc.values / len(y_class) * 100).round(1),
    }).assign(개월수_평균=[round(y_months[y_class == i].mean(), 2) for i in range(3)])


def signal_check(df=None, scenario="S1", derived=True):
    """이탈자 내부에 '시점' 신호가 있는지 진단 → dict

    B-1/B-3 실패의 원인이 모델이 아니라 데이터임을 보이는 근거.
    """
    ft.set_seed()
    X, y_months, _ = make_timing_data(df, scenario, derived)

    corr = (pd.concat([X, y_months.rename("개월수")], axis=1)
            .corr(numeric_only=True)["개월수"].drop("개월수"))
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)

    # 이진으로 재정의해도 신호가 없는지 확인
    binary = {}
    for label, yb in (("0개월 vs 1개월+", (y_months > 0).astype(int)),
                      ("0~1개월 vs 2개월+", (y_months >= 2).astype(int))):
        X_tr, X_te, y_tr, y_te = train_test_split(X, yb, test_size=0.25,
                                                  stratify=yb, random_state=SEED)
        model = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=4,
                              random_state=SEED, eval_metric="logloss", n_jobs=-1)
        model.fit(X_tr, y_tr)
        binary[label] = float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))

    return {
        "상관계수": corr.round(3),
        "최대_절대상관": float(corr.abs().max()),
        "이진_재정의_AUC": binary,
    }


def run_all(df=None, scenario="S1", derived=True):
    """트랙 B 전체 — B-1 / B-2 / B-3

    반환: dict — 'summary'(long 형식, 트랙 통합용) + 상세표들
    """
    ft.set_seed()
    config = ft.scenario_tag(scenario, derived)

    b1 = train_ordinal(df, scenario, derived)
    b2 = survival_cox(df, scenario, derived)
    b3 = train_regression(df, scenario, derived)

    # long 형식 요약 (features.SUMMARY_COLUMNS 규약)
    rows = []
    for _, r in b1.iterrows():
        for metric in ("test_macro_f1", "test_accuracy"):
            rows.append({"트랙": "B 타이밍", "단계": ft.STAGE_BASE, "구성": f"{config} (B-1)",
                         "모델": r["모델"], "지표": metric, "값": r[metric]})
    rows.append({"트랙": "B 타이밍", "단계": ft.STAGE_BASE, "구성": f"{config} (B-2)",
                 "모델": "cox", "지표": "c_index", "값": b2["c_index"]})
    for _, r in b3.iterrows():
        for metric in ("test_rmse", "test_mae"):
            rows.append({"트랙": "B 타이밍", "단계": ft.STAGE_BASE, "구성": f"{config} (B-3)",
                         "모델": r["모델"], "지표": metric, "값": r[metric]})

    return {
        "summary": ft.make_summary(rows),
        "target": target_distribution(df),
        "signal": signal_check(df, scenario, derived),
        "b1": b1,
        "b2_coef": b2["coef"],
        "b2_meta": {k: v for k, v in b2.items() if k != "coef"},
        "b2_km": survival_km(df),
        "b3": b3,
    }


if __name__ == "__main__":
    ft.set_seed()
    r = run_all()

    print("=" * 78)
    print("트랙 B — 타깃 분포 (이탈자 1,061명)")
    print("=" * 78)
    print(r["target"].to_string(index=False))

    print("\n" + "=" * 78)
    print("B-1  순서형 분류")
    print("=" * 78)
    print(r["b1"].round(4).to_string(index=False))
    print("  ※ 정직한 기준선은 baseline(무작위)다. 이를 넘지 못하면 실패로 판정한다.")

    print("\n" + "=" * 78)
    print("신호 진단 — 이탈자 내부에 '시점' 정보가 있는가")
    print("=" * 78)
    sig = r["signal"]
    print(sig["상관계수"].to_string())
    print(f"\n  최대 절대상관: {sig['최대_절대상관']:.3f}")
    for k, v in sig["이진_재정의_AUC"].items():
        print(f"  이진 재정의 [{k}] AUC = {v:.4f}  (0.5=무작위)")

    print("\n" + "=" * 78)
    print("B-2  생존분석 (Cox) — 전체 4,000명, 절단 {:.1%}".format(r["b2_meta"]["censored_rate"]))
    print("=" * 78)
    print(f"C-index: {r['b2_meta']['c_index']:.4f}   (0.5=무작위, 1.0=완벽)")
    print(r["b2_coef"].round(4).to_string(index=False))

    print("\nKaplan-Meier 생존곡선")
    print(r["b2_km"].round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("B-3  회귀 (비교군)")
    print("=" * 78)
    print(r["b3"].round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print(f"통합 요약 (long) — {len(r['summary'])}행")
    print("=" * 78)
    print(r["summary"].head(6).to_string(index=False))
    print(f"\n트랙 B 파일 수: {len(list(MODEL_DIR.glob('reg_*.joblib')))}")
