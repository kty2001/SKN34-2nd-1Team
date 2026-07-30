"""분석 4 — 피처 선택 사전 검증 (노트북 9절의 실행 코드)

노트북 9절은 판단의 근거가 되는 실험들을 담고 있으나 **결과 수치만 마크다운으로
남아 있고 실행 코드가 없었다.** 보고서 3.1절이 그 수치를 인용하므로, 재현할 수 없는
숫자가 보고서에 남지 않도록 이 모듈로 전부 코드화한다.

여기 있는 실험은 학습·저장을 하지 않는다. 모델을 고르기 **전에** 무엇을 넣고 뺄지
정하기 위한 측정이며, 결정 결과는 features.py의 상수(DROP_ALWAYS·SCENARIOS·
CLUSTER_FEATURES)로 고정되어 있다. 즉 이 모듈은 그 상수들의 근거를 되짚는다.

공통 실험 조건 — 별도 언급이 없으면 전부 이 설정이다.
    분할     StratifiedKFold(5, shuffle=True, random_state=42)  ← ft.get_splitter()
    지표     ROC-AUC / F1
    데이터   전체 4,000명 (홀드아웃을 떼지 않는다. 여기서는 모델을 저장하지 않으므로
             train/test 누수 문제가 없고, 전체를 쓰면 분산이 줄어 비교가 안정적이다)

CLI 실행 (프로젝트 루트에서):
    python -m src.analysis4.validation            # 전체
    python -m src.analysis4.validation --quick    # MLP·학습곡선 제외 (빠름)
"""

import sys

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_validate, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.api import Logit, add_constant
from xgboost import XGBClassifier

from src.common import SEED, load_data
from src.analysis4 import features as ft
from src.analysis4 import classification as clf

# 9.5절 시나리오 사다리 — S1/S2는 features.SCENARIOS와 동일하고,
# S3/S4는 "어디까지 빼면 무너지는가"를 보기 위한 진단용이라 학습에는 쓰지 않는다.
FREQ_COLS = ["Avg_class_frequency_total", ft.LEAK_SUSPECT]

SCENARIO_LADDER = {
    "S1 전체": list(ft.BASE_FEATURES),
    "S2 직전달 방문 제외": [c for c in ft.BASE_FEATURES if c != ft.LEAK_SUSPECT],
    "S3 방문빈도 전부 제외": [c for c in ft.BASE_FEATURES if c not in FREQ_COLS],
    "S4 S3 + Lifetime 제외": [c for c in ft.BASE_FEATURES
                              if c not in FREQ_COLS + [ft.TIMING_TARGET]],
}

SCORING = {"roc_auc": "roc_auc", "f1": "f1"}


def _cv(model, X, y, scoring=None):
    """공통 교차검증 → {지표: 평균}"""
    scoring = scoring or SCORING
    res = cross_validate(model, X, y, cv=ft.get_splitter(), scoring=scoring, n_jobs=1)
    return {k: float(res[f"test_{k}"].mean()) for k in scoring}


def _xgb():
    return XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=4,
                         random_state=SEED, eval_metric="logloss", n_jobs=-1)


# ────────────────────────────────────────────────────────────────
# 9.1  gender / Phone — 제거 근거
# ────────────────────────────────────────────────────────────────
def chi_square(df=None, cols=None):
    """이진 변수와 Churn의 독립성 검정 → DataFrame

    p가 크면 '이탈과 독립'이다. 귀무가설(독립)을 기각할 수 없다는 뜻이므로
    "관계가 없다"의 증명은 아니지만, 다른 변수와 나란히 두면 신호 크기를 비교할 수 있다.
    """
    df = load_data() if df is None else df
    cols = cols or ["gender", "Phone", "Near_Location", "Partner",
                    "Promo_friends", "Group_visits"]
    rows = []
    for col in cols:
        table = pd.crosstab(df[col], df[ft.TARGET])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        rows.append({"변수": col, "카이제곱": round(chi2, 4), "자유도": int(dof),
                     "p_value": p, "독립 여부": "독립" if p > 0.05 else "관련"})
    return pd.DataFrame(rows).sort_values("p_value", ascending=False, ignore_index=True)


def logit_pvalues(df=None):
    """전체 설명변수(13개) 로지스틱 계수의 p값 → DataFrame

    카이제곱은 변수 하나만 보고, 이 검정은 나머지 변수를 통제한 뒤 본다.
    둘 다 무신호로 나와야 제거 근거가 된다.
    """
    df = load_data() if df is None else df
    X = df.drop(columns=[ft.TARGET])
    Z = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    res = Logit(df[ft.TARGET].values, add_constant(Z)).fit(disp=0)
    out = (pd.DataFrame({"변수": Z.columns, "계수": res.params[1:].values,
                         "p_value": res.pvalues[1:].values})
           .sort_values("p_value", ascending=False, ignore_index=True))
    out["순위(p 큰 순)"] = out.index + 1
    return out


def ablation(df=None, groups=None):
    """변수군을 뺐을 때의 CV AUC 변화 → DataFrame

    '중요도가 낮다'가 아니라 '빼도 성능이 안 떨어진다'가 제거의 근거다.
    양수면 제거가 이득이다.
    """
    df = load_data() if df is None else df
    groups = groups or {"gender + Phone": ft.DROP_ALWAYS, "Age": ["Age"]}
    full = ft.BASE_FEATURES + ft.DROP_ALWAYS          # 원본 13개
    y = df[ft.TARGET]

    models = {"logreg": Pipeline([("scaler", StandardScaler()),
                                  ("model", LogisticRegression(max_iter=3000,
                                                               random_state=SEED))]),
              "xgb": _xgb()}
    rows = []
    for label, drop in groups.items():
        keep = [c for c in full if c not in drop]
        for name, model in models.items():
            with_auc = _cv(model, df[full], y)["roc_auc"]
            without_auc = _cv(model, df[keep], y)["roc_auc"]
            rows.append({"제거 대상": label, "모델": name,
                         "13개 전체": round(with_auc, 4),
                         "제거 후": round(without_auc, 4),
                         "변화": round(without_auc - with_auc, 4)})
    return pd.DataFrame(rows)


def cluster_noise_effect(df=None, k=3):
    """군집 축에 gender/Phone을 넣었을 때와 뺐을 때의 실루엣 → DataFrame

    분류에서는 무해한 변수가 거리 기반 군집에서는 잡음이 된다.
    KMeans는 모든 축을 동등하게 보기 때문이다.
    """
    df = load_data() if df is None else df
    rows = []
    for label, cols in (("9축 (gender·Phone 제외)", list(ft.CLUSTER_FEATURES)),
                        ("11축 (gender·Phone 포함)", ft.CLUSTER_FEATURES + ft.DROP_ALWAYS)):
        Z = StandardScaler().fit_transform(df[cols])
        labels = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(Z)
        rows.append({"군집 축": label, "축 수": len(cols),
                     "실루엣": round(float(silhouette_score(Z, labels)), 4)})
    out = pd.DataFrame(rows)
    base = out.loc[out["축 수"] == len(ft.CLUSTER_FEATURES), "실루엣"].iloc[0]
    other = out.loc[out["축 수"] != len(ft.CLUSTER_FEATURES), "실루엣"].iloc[0]
    out.attrs["개선율(%)"] = round((base - other) / other * 100, 1)
    return out


# ────────────────────────────────────────────────────────────────
# 9.2  Age — 유지 근거
# ────────────────────────────────────────────────────────────────
def age_profile(df=None):
    """연령대별 이탈률 + Lifetime과의 상관 → (DataFrame, 상관계수)

    Age가 Lifetime의 대리변수라면 굳이 둘 다 넣을 필요가 없다. 상관이 낮으면 독립 신호다.
    """
    df = load_data() if df is None else df
    bins = pd.cut(df["Age"], [17, 24, 27, 30, 33, 41],
                  labels=["18~24세", "25~27세", "28~30세", "31~33세", "34세+"])
    out = df.groupby(bins, observed=True)[ft.TARGET].agg(["size", "mean"])
    out.columns = ["인원", "이탈률(%)"]
    out["이탈률(%)"] = (out["이탈률(%)"] * 100).round(1)
    return out, round(float(df["Age"].corr(df[ft.TIMING_TARGET])), 3)


def age_importance_rank(df=None):
    """최종 구성(S1/derived, xgb)에서 Age의 순열 중요도 순위 → (순위, 전체 수, 표)"""
    ev = clf.evaluate(df, scenario="S1", derived=True, name="xgb", n_repeats=5)
    imp = ev["importance"]
    rank = int(imp.index[imp["피처"] == "Age"][0]) + 1
    return rank, len(imp), imp


# ────────────────────────────────────────────────────────────────
# 9.3  딥러닝 — 비교군으로만
# ────────────────────────────────────────────────────────────────
def model_family_compare(df=None, scenario="S1", derived=True):
    """트리 · 선형 · MLP 2종 비교 → DataFrame

    MLP를 넣는 목적은 성능 기대가 아니라 '정형 4,000행에는 부적합'의 근거 확보다.
    은닉층을 키운 쪽도 함께 재 크기 문제가 아님을 보인다.
    """
    X, y = ft.get_xy(df, scenario=scenario, derived=derived)
    models = {
        "XGBoost": _xgb(),
        "LogisticRegression": Pipeline([("scaler", StandardScaler()),
                                        ("model", LogisticRegression(max_iter=3000,
                                                                     random_state=SEED))]),
        "MLP(64,32)": Pipeline([("scaler", StandardScaler()),
                                ("model", MLPClassifier(hidden_layer_sizes=(64, 32),
                                                        max_iter=1000, early_stopping=True,
                                                        random_state=SEED))]),
        "MLP(128,64,32)": Pipeline([("scaler", StandardScaler()),
                                    ("model", MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                                                            max_iter=1000, early_stopping=True,
                                                            random_state=SEED))]),
    }
    rows = [{"모델": name, **{k: round(v, 4) for k, v in _cv(m, X, y).items()}}
            for name, m in models.items()]
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False, ignore_index=True)


def learning_curve_check(df=None, scenario="S1", derived=True):
    """학습곡선 → (DataFrame, 마지막 구간 기울기)

    데이터를 더 모으면 나아지는가. 마지막 구간의 기울기가 0에 가까우면 포화다.
    """
    X, y = ft.get_xy(df, scenario=scenario, derived=derived)
    sizes, _, test = learning_curve(_xgb(), X, y, cv=ft.get_splitter(),
                                    scoring="roc_auc", n_jobs=1,
                                    train_sizes=np.linspace(0.2, 1.0, 5))
    mean = test.mean(axis=1)
    out = pd.DataFrame({"학습 표본": sizes, "CV AUC": mean.round(4)})
    return out, round(float(mean[-1] - mean[-2]), 4)


# ────────────────────────────────────────────────────────────────
# 9.5  누수 의심 변수 — 시나리오 사다리
# ────────────────────────────────────────────────────────────────
def scenario_ladder(df=None):
    """S1~S4를 같은 조건에서 비교 → DataFrame

    S1과 S2의 차이가 작으면 '한 변수에 기생하지 않는다'는 근거가 된다.
    S3·S4는 어느 변수가 실제 핵심인지 드러내기 위한 진단용이다.
    """
    df = load_data() if df is None else df
    y = df[ft.TARGET]
    rows = []
    for label, cols in SCENARIO_LADDER.items():
        scores = _cv(_xgb(), df[cols], y)
        rows.append({"시나리오": label, "피처 수": len(cols),
                     **{k: round(v, 4) for k, v in scores.items()}})
    out = pd.DataFrame(rows)
    out["S1 대비 AUC"] = (out["roc_auc"] - out.loc[0, "roc_auc"]).round(4)
    return out


# ────────────────────────────────────────────────────────────────
# 9.6 / 9.9  이상치 · 불균형
# ────────────────────────────────────────────────────────────────
def outlier_profile(df=None, col=None):
    """IQR 이상치 집단의 이탈률 → DataFrame

    이상치를 지우기 전에 그 집단이 어떤 고객인지 본다.
    """
    df = load_data() if df is None else df
    col = col or ft.TIMING_TARGET
    q1, q3 = df[col].quantile([0.25, 0.75])
    hi = q3 + 1.5 * (q3 - q1)
    mask = df[col] > hi
    return pd.DataFrame([
        {"구분": f"{col} 이상치 (> {hi:.1f})", "인원": int(mask.sum()),
         "이탈률(%)": round(float(df.loc[mask, ft.TARGET].mean() * 100), 1)},
        {"구분": "그 외", "인원": int((~mask).sum()),
         "이탈률(%)": round(float(df.loc[~mask, ft.TARGET].mean() * 100), 1)},
    ])


def imbalance_compare(df=None, scenario="S1", derived=True):
    """불균형 처리 3방식 비교 → DataFrame (미적용 / class_weight / SMOTE)"""
    X, y = ft.get_xy(df, scenario=scenario, derived=derived)
    pos_weight = (len(y) - y.sum()) / y.sum()
    scoring = {"roc_auc": "roc_auc", "f1": "f1",
               "recall": "recall", "precision": "precision"}

    variants = {
        "미적용": _xgb(),
        "scale_pos_weight": clf.build_models(balanced=True, pos_weight=pos_weight)["xgb"],
        "SMOTE": ImbPipeline([("smote", SMOTE(random_state=SEED)), ("model", _xgb())]),
    }
    rows = [{"방식": name, **{k: round(v, 4) for k, v in _cv(m, X, y, scoring).items()}}
            for name, m in variants.items()]
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────
# 트랙 A 확률 등급 vs 군집 등급
# ────────────────────────────────────────────────────────────────
def risk_terciles(df=None, scenario="S1", derived=True, name="xgb", tag=None):
    """저장 모델의 이탈 확률을 3분위로 나눈 구간별 실제 이탈률 → DataFrame

    군집 등급(트랙 C)과 나란히 두면 '등급은 확률로, 설명은 군집으로'의 근거가 된다.
    """
    df = load_data() if df is None else df
    bundle = clf.load_bundle(scenario, derived, name, tag)
    X, y = ft.get_xy(df, scenario=scenario, derived=derived)
    proba = clf.predict_proba(bundle, X)

    tercile = pd.qcut(proba, 3, labels=["저위험", "중위험", "고위험"])
    out = pd.DataFrame({"이탈": y.values, "구간": tercile}).groupby(
        "구간", observed=True)["이탈"].agg(["size", "mean"])
    out.columns = ["인원", "이탈률(%)"]
    out["이탈률(%)"] = (out["이탈률(%)"] * 100).round(1)
    return out


def _show(title, table, note=None):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(table if isinstance(table, str) else table.to_string(
        index=not isinstance(table.index, pd.RangeIndex)))
    if note:
        print(f"  ※ {note}")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    ft.set_seed()
    df = load_data()

    _show("9.1  카이제곱 독립성 검정", chi_square(df))
    _show("9.1  로지스틱 계수 p값 (13개 전체, 표준화 후)", logit_pvalues(df).round(4))
    _show("9.1 / 9.2  변수군 제거 시 CV AUC 변화", ablation(df),
          "양수면 제거가 이득")

    noise = cluster_noise_effect(df)
    _show("9.1  군집 축에서의 gender·Phone 효과", noise,
          f"9축이 11축보다 실루엣 {noise.attrs['개선율(%)']}% 높음")

    prof, corr = age_profile(df)
    _show("9.2  연령대별 이탈률", prof, f"Age ↔ Lifetime 상관 {corr}")
    rank, total, imp = age_importance_rank(df)
    _show("9.2  Age 순열 중요도 순위", imp.round(4),
          f"Age는 {total}개 중 {rank}위")

    _show("9.5  시나리오 사다리 (xgb · 5-fold CV · 원본 피처)", scenario_ladder(df))
    _show("9.6  Lifetime 이상치 프로파일", outlier_profile(df))
    _show("9.9  불균형 처리 비교", imbalance_compare(df))
    _show("트랙 A 확률 3분위별 실제 이탈률 (S1/derived · xgb 기본)", risk_terciles(df))

    if quick:
        print("\n(--quick: 딥러닝 비교·학습곡선 생략)")
    else:
        _show("9.3  모델 계열 비교 (S1/derived · 5-fold CV)", model_family_compare(df))
        lc, slope = learning_curve_check(df)
        _show("9.10  학습곡선", lc, f"마지막 구간 기울기 {slope:+.4f}")
