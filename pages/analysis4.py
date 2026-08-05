"""분석 4 — 헬스장 고객 이탈 (Streamlit 페이지)

탭 구성은 src/analysis4 모듈 구조를 그대로 따른다.
    EDA           eda.py
    분류          classification.py  (트랙 A) — 고도화 전/후
    이탈 타이밍   regression.py      (트랙 B) — 회귀 두 가지 비교
                                       모듈의 순서형 분류(B-1)는 학습 코드에만 두고 화면에서는 뺐다.
                                       그래서 화면 번호가 모듈 태그와 다르다 — REG_TAG 참고.
                                         화면 B-1 = 모듈 B-3 (최소제곱 회귀)
                                         화면 B-2 = 모듈 B-2 (Cox 회귀)
    군집화        clustering.py      (트랙 C) — 2D·3D 분포 + 리텐션 솔루션

이 페이지는 학습하지 않는다. models/analysis4 의 저장 번들과 통합 요약(summary.joblib),
전처리 검증 결과(validation.joblib)를 읽어 표시만 한다.
    학습        python -m src.analysis4.train
    전처리 검증  python -m src.analysis4.validation

EDA 탭 끝의 '전처리 결정 근거'와 각 탭 끝의 '모델 사양'은 계산하지 않고 위 두 파일과
번들 메타데이터만 읽는다. 특히 사양표는 rf 번들이 개당 10MB 안팎이라 모든 파일을 열지
않고 트랙별 주력 모델만 연다 — 나머지는 파일 목록(이름·크기)으로만 보여준다.

샘플 입력 예측은 별도 예측 페이지로 옮기기로 해서 이 페이지에서는 주석 처리했다.
분류(탭 2)·타이밍(탭 3)의 예측 블록과 그 전용 헬퍼(입력 폼·모집단 캐시)가 대상이며,
지우지 않고 남긴 이유는 옮겨 갈 페이지에서 그대로 참고하기 위함이다.
되살릴 때는 `# [예측 이전]` 표시가 붙은 블록을 함께 해제하면 된다.

차트는 Altair(축 범위 제어) / Plotly(3D)로 그린다. 화면 폭에 맞춰 늘어나고 높이는
CHART_H · TALL_H 두 값으로 통일한다. eda.py의 matplotlib Figure는 쓰지 않는다 —
PNG라 반응형이 아니고 Figure마다 종횡비가 달랐다. (보고서용 이미지는 eda.py가 계속 담당한다.)
표는 eda.all_tables()를 그대로 받아 쓴다.
"""

from datetime import datetime

import altair as alt
import pandas as pd
import plotly.express as px
import streamlit as st

from src.common import load_data
from src.analysis4 import classification as clf
from src.analysis4 import clustering as cl
from src.analysis4 import eda
from src.analysis4 import features as ft
from src.analysis4 import regression as rg
from src.analysis4 import train as tr
from src.analysis4 import validation as va

TRACK_A, TRACK_B, TRACK_C = "A 분류", "B 타이밍", "C 군집"

# 화면 표기 -> (scenario, derived)
CONFIGS = {
    "S1/derived": ("S1", True),
    "S1/raw": ("S1", False),
    "S2/derived": ("S2", True),
    "S2/raw": ("S2", False),
}

NOT_TRAINED = "학습 결과 없음. 프로젝트 루트에서 `python -m src.analysis4.train` 을 먼저 실행할 것."
NOT_VALIDATED = ("전처리 검증 결과 없음. 프로젝트 루트에서 "
                 "`python -m src.analysis4.validation` 을 먼저 실행할 것.")

# 저장 폴더 — 사양표의 파일 목록에 쓴다. 경로는 train 모듈이 이미 갖고 있어 다시 적지 않는다.
MODEL_DIR = tr.MODEL_DIR

# 사양표에 노출할 하이퍼파라미터 — get_params()는 수십 개를 쏟아내므로
# build_models()·군집 파이프라인이 실제로 지정하거나 Optuna가 건드리는 키만 추린다.
SPEC_PARAM_KEYS = [
    "n_estimators", "learning_rate", "max_depth", "min_child_weight",
    "subsample", "colsample_bytree", "gamma", "reg_alpha", "reg_lambda",
    "C", "penalty", "solver", "max_iter", "class_weight", "scale_pos_weight",
    "hidden_layer_sizes", "early_stopping", "n_clusters", "n_init", "random_state",
]

# 트랙 B 화면 번호 -> summary의 구성 태그.
# 순서형 분류(모듈의 B-1)를 화면에서 빼면서 번호를 앞으로 당겼다. 모듈·보고서의 태그는 그대로이므로
# 필터는 원래 태그로 걸고 제목만 새 번호로 쓴다. 여기서 한 번만 맞춰 두면 둘이 어긋나지 않는다.
REG_TAG = {"B-1": "(B-3)", "B-2": "(B-2)"}

# summary에 저장된 모델명 -> 화면 표기. 저장된 값은 그대로 두고 표시할 때만 바꾼다.
BASELINE_MEAN = "baseline(평균)"
BASELINE_MEAN_LABEL = "Baseline(Mean Selection)"
REG_MODEL_LABEL = {BASELINE_MEAN: BASELINE_MEAN_LABEL}

# 화면 B-1의 대표 모델. 개요표의 결과 값과 샘플 예측이 같은 모델을 가리키도록 여기서 고정한다.
# rg.load_bundle의 기본값은 xgb라, 생략하면 '최소제곱 회귀'라고 적힌 자리에 다른 모델 값이 들어간다.
REG_OLS_MODEL = "linreg"

# [예측 이전] 예측 폼에서 받을 원본 변수 — 파생변수는 받지 않고 add_derived()로 계산한다.
# 화면에서 직접 입력받으면 학습 때와 정의가 어긋날 수 있다.
# RAW_INPUT_COLS = list(ft.BASE_FEATURES)

LABELS = {
    "Contract_period": "계약 기간(개월)",
    "Near_Location": "헬스장 근처 거주",
    "Partner": "제휴사 직원",
    "Promo_friends": "친구 추천 가입",
    "Group_visits": "그룹수업 참여",
    "Age": "나이",
    "Avg_additional_charges_total": "부가 서비스 총액",
    "Month_to_end_contract": "계약 잔여 개월",
    "Lifetime": "가입 후 경과 개월",
    "Avg_class_frequency_total": "주당 평균 방문(전체)",
    "Avg_class_frequency_current_month": "주당 평균 방문(직전 달)",
}

# 등급별 색 — 2D·3D 산점도가 같은 색을 쓰도록 고정
TIER_COLORS = {"유지군": "#4C78A8", "이탈 위험군": "#F58518", "이탈군": "#E45756"}

# 차트 높이는 두 값만 쓴다 — 폭은 컨테이너에 맞추고 높이만 통일해야 스크롤이 들쭉날쭉하지 않다.
CHART_H = 340          # 일반 차트
TALL_H = 480           # 항목이 많은 차트 (히트맵 · 15행 막대)

# 보고서 5.2 트랙 D — 군집 프로파일에서 도출한 등급별 대응
TIER_PLAYBOOK = {
    "유지군": {
        "해석": "장기계약 + 제휴사 + 친구추천이 겹친 우량 집단",
        "액션": "**친구 추천 및 제휴 채널**를 통해 신규 회원 증가를 꾀함",
    },
    "이탈 위험군": {
        "해석": "헬스장 근처에 살아 접근성은 높지만 **계약 기간**이 짧음",
        "액션": "**장기계약 전환 인센티브 + 그룹 수업 무료 체험**",
    },
    "이탈군": {
        "해석": "위험군과 비교해 물리적 접근성인 **거주지 근접성**이 낮음",
        "액션": "방문 빈도에 덜 의존하는 상품(주말 집중반 등)과 가입 첫 달 집중 관리로 접근",
    },
}

# 파생변수 정의 — 9.4절. 값은 features.py의 상수를 그대로 쓴다(정의가 갈리지 않게).
DERIVED_DOC = pd.DataFrame([
    {"파생변수": ft.CONTRACT_USED,
     "정의": "1 − 잔여개월 / 계약개월",
     "대체 대상": ft.DERIVED_REPLACES[ft.CONTRACT_USED],
     "효과": "예측 기여 거의 없음(+0.0001) · VIF 19.18 → 1.41 로 계수 해석이 가능해짐"},
    {"파생변수": ft.VISIT_DELTA,
     "정의": "직전달 − 전체 방문빈도",
     "대체 대상": ft.DERIVED_REPLACES[ft.VISIT_DELTA],
     "효과": "RF +0.0137 · 선형모델 +0 (두 원본의 선형 결합이라 새 정보가 없음)"},
])

# 보고서 5.2 — 솔루션은 개입 가능한 변수에서만 도출한다.
INTERVENTION_TABLE = pd.DataFrame([
    {"구분": "고정 (개입 불가)", "변수": "Age, Near_Location", "활용": "대상 선정 기준"},
    {"구분": "레버 (개입 가능)", "변수": "Contract_period, Group_visits, Promo_friends, Partner", "활용": "실제 수단"},
    {"구분": "모니터링", "변수": "Lifetime, 방문 빈도", "활용": "개입 타이밍·효과 측정"},
])


# ── 캐시 래퍼 ─────────────────────────────────────────────────────
# 저장된 모델을 읽어 지표만 계산하므로 재학습은 없다.
# 위젯 조작 때마다 스크립트가 다시 도는 만큼, 무거운 계산은 전부 캐시에 태운다.

@st.cache_data(show_spinner=False)
def cached_summary():
    s = tr.load_summary()
    return s["summary"], s["created_at"], s["seed"]


@st.cache_data(show_spinner=False)
def cached_validation():
    """전처리 결정 근거 — validation.py가 저장해 둔 측정 결과를 읽기만 한다.

    여기 담긴 교차검증(제거 실험·불균형 비교·학습곡선)은 수십 초가 걸려 화면에서
    즉석 계산할 수 없다. 대신 화면에 수치를 적어 두면 재학습 때 코드와 어긋나므로
    summary.joblib과 같은 방식으로 파일에서 읽는다.
    """
    return va.load_validation()


@st.cache_data(show_spinner=False)
def cached_eda_tables(df):
    return eda.all_tables(df)


@st.cache_data(show_spinner=False)
def cached_vif_pairs(df):
    """다중공선성 쌍 — 상관계수 + 각 변수의 VIF

    all_tables()에 묶지 않고 따로 부른다. cache_data는 감싼 함수 본문으로만 키를 만들어,
    안쪽 eda 함수가 바뀌어도 실행 중인 앱이 옛 표를 계속 내주기 때문이다.
    """
    return eda.multicollinear_pairs(df)


@st.cache_data(show_spinner=False)
def cached_corr_long(df):
    """상관행렬을 long 형식으로 — 위쪽 삼각형은 같은 정보의 반복이라 가린다."""
    corr = eda.correlation(df).round(2)
    cols = list(corr.columns)
    order = {c: i for i, c in enumerate(cols)}
    long = (corr.rename_axis("변수1").reset_index()
            .melt(id_vars="변수1", var_name="변수2", value_name="상관계수"))
    long = long[long["변수1"].map(order) >= long["변수2"].map(order)].copy()
    return long, cols


@st.cache_data(show_spinner=False)
def cached_histogram(df, col, bins=30):
    """이탈 여부별 히스토그램을 미리 집계 — 원본 4,000행을 브라우저로 보내지 않는다."""
    binned = pd.cut(df[col], bins=bins)
    out = (df.groupby([binned, ft.TARGET], observed=True).size()
           .reset_index(name="인원"))
    out.columns = ["구간", "이탈", "인원"]
    # 구간의 시작과 끝을 모두 넘긴다 — bin="binned"는 x2가 없으면 막대 폭이 정해지지 않아
    # 축만 그려지고 막대가 사라진다.
    out["시작"] = out["구간"].apply(lambda iv: float(iv.left))
    out["끝"] = out["구간"].apply(lambda iv: float(iv.right))
    out["구분"] = out["이탈"].map({0: "유지", 1: "이탈"})
    return out[["시작", "끝", "구분", "인원"]]


@st.cache_data(show_spinner="모델 평가 중…")
def cached_clf_eval(df, scenario, derived, name):
    return clf.evaluate(df, scenario=scenario, derived=derived, name=name)


@st.cache_data(show_spinner=False)
def cached_stage_compare(df):
    """(상세, 피벗) 반환 — 고도화 모델이 없으면 (빈 DataFrame, None)"""
    return clf.compare_stages(df)


@st.cache_data(show_spinner="생존분석 적합 중…")
def cached_cox(df):
    # save=False — 페이지 조회가 저장된 모델 파일을 덮어쓰지 않게 한다.
    return rg.survival_cox(df, save=False)


@st.cache_data(show_spinner=False)
def cached_km(df):
    return rg.survival_km(df)


@st.cache_data(show_spinner="시점 신호 진단 중…")
def cached_signal(df):
    return rg.signal_check(df)


# [예측 이전] 개인 위험도를 백분위로 옮길 때만 쓰던 캐시 — 예측 블록과 함께 비활성.
# @st.cache_data(show_spinner=False)
# def cached_cox_population(df):
#     """전체 고객의 Cox 위험도 분포 — 개인 위험도를 백분위로 옮길 때 쓴다."""
#     bundle = rg.load_bundle("b2")
#     return pd.Series(rg.cox_risk(bundle, ft.add_derived(df)))


@st.cache_data(show_spinner="군집 프로파일 계산 중…")
def cached_cluster(df):
    bundle = cl.load_bundle()
    labels = bundle["pipeline"].predict(df[bundle["features"]])
    loadings, var = cl.pca_loadings(df, n_components=3)
    return {
        "silhouette": bundle["silhouette"],
        "mapping": bundle["mapping"],
        "profile": cl.profile_clusters(df, labels),
        "pca2": cl.pca_2d(df, labels),
        "pca3": cl.pca_3d(df, labels),
        "loadings": loadings,
        "var": var,
        "k": bundle["k"],
    }


@st.cache_data(show_spinner="k 탐색 중…")
def cached_search_k(df):
    out = cl.search_k(df)
    # 리스트를 담은 컬럼은 표로 그릴 수 없어 문자열로 편다.
    out["군집별_이탈률"] = out["군집별_이탈률"].apply(lambda v: ", ".join(f"{x:.1f}" for x in v))
    return out


@st.cache_data(show_spinner=False)
def cached_retention_actions(df):
    """보고서 5.2의 액션을 현재 데이터에서 다시 계산 — 인원·이탈률을 하드코딩하지 않는다."""
    def block(mask):
        sub = df[mask]
        return int(len(sub)), round(float(sub[ft.TARGET].mean()) * 100, 1)

    rows = []
    n, rate = block((df["Age"] <= 27) & (df["Lifetime"] <= 1))
    rows.append({"우선순위": 1, "액션": "27세 이하 신규 회원 온보딩",
                 "대상": "27세 이하 × 가입 0~1개월", "인원": n, "현재 이탈률(%)": rate,
                 "수단": "그룹수업 배정 · 첫 달 집중 관리"})

    n, rate = block((df["Contract_period"] == 1) & (df["Group_visits"] == 0))
    rows.append({"우선순위": 2, "액션": "단기계약 · 그룹수업 미참여 전환",
                 "대상": "1개월 계약 × 그룹수업 미참여", "인원": n, "현재 이탈률(%)": rate,
                 "수단": "장기계약 전환 인센티브 · 그룹수업 무료 체험"})

    n, rate = block((df["Promo_friends"] == 0) & (df["Partner"] == 0))
    rows.append({"우선순위": 3, "액션": "친구추천 · 제휴 채널 확대",
                 "대상": "추천·제휴 경로 없이 가입", "인원": n, "현재 이탈률(%)": rate,
                 "수단": "추천 리워드 · 제휴사 확대"})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def cached_lever_effects(df):
    """레버별 이탈률 격차 — 솔루션의 근거표"""
    rows = []
    for col in ("Group_visits", "Promo_friends", "Partner"):
        rate = df.groupby(col)[ft.TARGET].mean() * 100
        rows.append({"레버": LABELS[col],
                     "미해당 이탈률(%)": round(float(rate.loc[0]), 1),
                     "해당 이탈률(%)": round(float(rate.loc[1]), 1),
                     "격차(%p)": round(float(rate.loc[0] - rate.loc[1]), 1)})

    rate = df.groupby("Contract_period")[ft.TARGET].mean() * 100
    lo, hi = rate.index.min(), rate.index.max()
    rows.append({"레버": f"계약 {hi}개월 (vs {lo}개월)",
                 "미해당 이탈률(%)": round(float(rate.loc[lo]), 1),
                 "해당 이탈률(%)": round(float(rate.loc[hi]), 1),
                 "격차(%p)": round(float(rate.loc[lo] - rate.loc[hi]), 1)})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def cached_scenarios():
    """네 구성의 실제 피처 목록 — 화면 설명과 학습 코드가 어긋나지 않도록 직접 뽑아 보여준다."""
    return ft.describe_scenarios()


# ── 모델 사양 ─────────────────────────────────────────────────────
# 저장 번들의 메타데이터만 읽는다. 여기서 다시 적합하지 않으므로 화면에 뜨는 사양은
# 실제로 models/analysis4 에 들어 있는 모델의 사양이다.
def algo_label(estimator):
    """추정기 이름. Pipeline이면 단계를 이어 붙여 스케일러 포함 여부까지 드러낸다."""
    if hasattr(estimator, "steps"):
        return " + ".join(type(step).__name__ for _, step in estimator.steps)
    return type(estimator).__name__


def model_params(estimator):
    """하이퍼파라미터에서 SPEC_PARAM_KEYS만 추린다 (Pipeline이면 마지막 단계 기준).

    get_params()는 기본값까지 수십 개를 내놓아 그대로 띄우면 무엇을 정한 값인지 안 보인다.
    sklearn이 폐기 예정 인자에 넣어 두는 'deprecated' 자리표시자도 뺀다 — 설정값이 아니다.
    """
    inner = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
    got = inner.get_params()
    return {k: got[k] for k in SPEC_PARAM_KEYS
            if got.get(k) is not None and got.get(k) != "deprecated"}


def format_param(value):
    """하이퍼파라미터 한 칸 — 없으면 '—', 정수로 떨어지는 실수는 소수점을 떼고 쓴다."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:g}" if float(value).is_integer() else f"{value:.5g}"
    return str(value)


def params_table(params):
    """{모델 라벨: {파라미터: 값}} -> 표

    모델마다 조정하는 파라미터가 달라 빈칸이 생기고 값 타입(int·float·str·tuple)도 섞인다.
    열 dtype에 맡기면 빈칸이 NaN으로, 300이 300.0으로 나오므로 칸마다 직접 문자열로 만든다.
    """
    keys = list(dict.fromkeys(key for spec in params.values() for key in spec))
    return pd.DataFrame([
        {"하이퍼파라미터": key,
         **{label: format_param(spec.get(key)) for label, spec in params.items()}}
        for key in keys
    ])


def sample_text(bundle):
    """번들에 기록된 표본 수 — 트랙마다 저장 필드가 달라 없으면 '—'로 둔다."""
    n_train, n_test = bundle.get("n_train"), bundle.get("n_test")
    if n_train is None and n_test is None:
        return "—"
    if n_test is None:
        return f"{n_train:,}"
    return f"{n_train:,} / {n_test:,}"


@st.cache_data(show_spinner=False)
def cached_model_files(prefix):
    """저장 파일 목록 — joblib을 열지 않고 파일 이름·크기만 읽는다."""
    rows = [{"파일": p.name,
             "크기(MB)": round(p.stat().st_size / 1024 ** 2, 2),
             "저장 시각": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")}
            for p in sorted(MODEL_DIR.glob(f"{prefix}*.joblib"))]
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def cached_clf_specs():
    """분류 주력 모델의 사양 + 하이퍼파라미터

    네 구성 × 4모델 = 18개 번들을 전부 열면 rf만으로 50MB가 넘는다. 보고 대상인
    시나리오별 주력 모델(S1/derived xgb · S2/raw logreg)의 기본·고도화만 연다.
    모델 객체를 그대로 들고 있으므로 cache_data(피클 복제)가 아니라 cache_resource.
    """
    rows, params = [], {}
    for scenario, derived in (("S1", True), ("S2", False)):
        name = clf.PRIMARY_MODEL[scenario]
        config = ft.scenario_tag(scenario, derived)
        for tag, stage in ((None, ft.STAGE_BASE), ("tuned", ft.STAGE_TUNED)):
            try:
                bundle = clf.load_bundle(scenario, derived, name, tag)
            except FileNotFoundError:
                continue
            rows.append({
                "구성": config,
                "단계": bundle.get("stage", stage),
                "알고리즘": algo_label(bundle["model"]),
                "피처 수": len(bundle["features"]),
                "임계값": round(float(bundle.get("threshold", 0.5)), 4),
                "test AUC": round(float(bundle["test"]["roc_auc"]), 4),
                "test F1": round(float(bundle["test"]["f1"]), 4),
                "표본 (train/test)": sample_text(bundle),
                "seed": bundle["seed"],
                "학습 일시": bundle["trained_at"].replace("T", " "),
                "파일": clf.bundle_path(scenario, derived, name, tag).name,
            })
            params[f"{config} · {name} · {bundle.get('stage', stage)}"] = model_params(bundle["model"])
    return pd.DataFrame(rows), params


@st.cache_resource(show_spinner=False)
def cached_reg_specs():
    """트랙 B 사양 — 화면 B-1(모듈 b3 최소제곱)과 B-2(Cox)

    Cox는 sklearn 추정기가 아니라 statsmodels PHReg라 번들 구조가 다르다.
    모델 객체 대신 계수(params)와 표준화 scaler가 저장되므로 따로 세운다.
    """
    rows, params = [], {}

    try:
        b3 = rg.load_bundle("b3", name=REG_OLS_MODEL)
    except FileNotFoundError:
        b3 = None
    if b3 is not None:
        rows.append({
            "화면 구분": "B-1 최소제곱 회귀 (비교군)",
            "알고리즘": algo_label(b3["model"]),
            "피처 수": len(b3["features"]),
            "지표": "test RMSE",
            "값": round(float(b3["test"]["rmse"]), 4),
            "표본 (train/test)": sample_text(b3),
            "seed": b3["seed"],
            "학습 일시": b3["trained_at"].replace("T", " "),
            "파일": rg.bundle_path("b3", name=REG_OLS_MODEL).name,
        })
        params["B-1 최소제곱 회귀"] = model_params(b3["model"])

    try:
        b2 = rg.load_bundle("b2")
    except FileNotFoundError:
        b2 = None
    if b2 is not None:
        rows.append({
            "화면 구분": "B-2 Cox 비례위험 회귀 (주력)",
            # 저장된 스케일러 종류를 직접 읽는다 — 표준화 기준은 train만 보고 정한 값이다.
            "알고리즘": f"{type(b2['scaler']).__name__} + PHReg (statsmodels)",
            "피처 수": len(b2["features"]),
            "지표": "C-index",
            "값": round(float(b2["c_index"]), 4),
            "표본 (train/test)": sample_text(b2),
            "seed": b2["seed"],
            "학습 일시": b2["trained_at"].replace("T", " "),
            "파일": rg.bundle_path("b2").name,
        })
    return pd.DataFrame(rows), params


@st.cache_resource(show_spinner=False)
def cached_cluster_spec():
    """트랙 C 사양 — 파이프라인(스케일러 + KMeans) 구성과 등급 매핑 기준"""
    bundle = cl.load_bundle()
    pipe = bundle["pipeline"]
    row = {
        "알고리즘": algo_label(pipe),
        "군집 수 k": bundle["k"],
        "축(피처) 수": len(bundle["features"]),
        "실루엣": round(float(bundle["silhouette"]), 4),
        "표본": f"{bundle['n']:,}",
        "seed": bundle["seed"],
        "학습 일시": bundle["trained_at"].replace("T", " "),
        "파일": cl.bundle_path(bundle["k"]).name,
    }
    return pd.DataFrame([row]), {"KMeans": model_params(pipe)}


def render_model_spec(spec, params, file_prefix, note=None, params_note=None):
    """탭 하단 공통 '모델 사양' 섹션 — 표 + 하이퍼파라미터 + 저장 파일 목록"""
    st.subheader("모델 사양")
    if spec.empty:
        st.info(NOT_TRAINED)
        return
    st.dataframe(spec, hide_index=True, width="stretch")
    if note:
        st.caption(note)
    # 조정할 파라미터가 없는 모델(최소제곱 회귀·Cox)만 있으면 표가 비므로 접이식 자체를 열지 않는다.
    table = params_table(params) if params else pd.DataFrame()
    if not table.empty:
        with st.expander("하이퍼파라미터"):
            if params_note:
                st.caption(params_note)
            st.dataframe(table, hide_index=True, width="stretch")
    with st.expander("저장된 모델 파일"):
        st.dataframe(cached_model_files(file_prefix), hide_index=True, width="stretch")


def metric_table(summary, track, config_contains=None):
    """long 요약 -> 모델 × 지표 표"""
    sub = summary[summary["트랙"] == track]
    if config_contains:
        sub = sub[sub["구성"].str.contains(config_contains, regex=False)]
    if sub.empty:
        return None
    return sub.pivot_table(index="모델", columns="지표", values="값").round(4)


def show_metric_table(table, index_label, index_width=140, value_width=95):
    """지표 표를 열 폭까지 지정해 띄운다.

    width="stretch"에 맡기면 남는 폭이 모든 열에 똑같이 나눠져, 인덱스는 이름을 못 담을 만큼
    좁아지고 숫자 열만 넓어진다. 인덱스는 가장 긴 이름에, 숫자 열은 자릿수에 맞춘다.
    표 자체는 content 폭으로 — 열 몇 개짜리 표를 화면 끝까지 늘리면 셀 사이가 휑해진다.
    """
    st.dataframe(
        table, width="content",
        column_config={
            "_index": st.column_config.TextColumn(index_label, width=index_width),
            **{c: st.column_config.NumberColumn(c, width=value_width, format="%.4f")
               for c in table.columns},
        })


# ── 차트 ──────────────────────────────────────────────────────────
def label_color(condition):
    """히트맵 셀 위 글자색 — 진한 셀에는 흰 글씨.

    ⚠ `alt.Color("컬럼:N", scale=None)`으로 색을 직접 넣지 말 것.
      Vega-Lite v6는 `"scale": null`에서 컴파일이 깨져 **차트가 통째로 렌더되지 않는다**
      (에러도 안 뜨고 빈 자리만 남는다). 조건식으로 값을 직접 지정한다.
    """
    return alt.condition(condition, alt.value("white"), alt.value("#222222"))


def roc_chart(curve):
    """ROC 곡선 — 축을 0~1로 고정한다(데이터 범위에 따라 축이 흔들리면 곡선 비교가 안 된다).

    두 축의 범위가 같으므로 폭도 픽셀로 고정해 정사각형에 가깝게 그린다.
    컨테이너에 맡기면 화면이 넓을수록 가로로만 늘어나 곡선이 납작해진다.
    """
    scale = alt.Scale(domain=[0, 1], nice=False, clamp=True)
    ref = alt.Chart(pd.DataFrame({"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]})).mark_line(
        color="#999", strokeDash=[4, 4]).encode(x="fpr:Q", y="tpr:Q")
    line = alt.Chart(curve).mark_line(color=eda.COLOR_STAY, clip=True).encode(
        x=alt.X("fpr:Q", title="False Positive Rate", scale=scale),
        y=alt.Y("tpr:Q", title="True Positive Rate", scale=scale),
        tooltip=[alt.Tooltip("fpr:Q", format=".3f"), alt.Tooltip("tpr:Q", format=".3f")],
    )
    return (ref + line).properties(width=420, height=360)


def confusion_chart(cm):
    """혼동행렬 히트맵 — 셀 색은 건수, 숫자는 그 위에 얹는다.

    ⚠ 크기를 컨테이너에 맡기지 않고 픽셀로 고정한다.
      좁은 컬럼 안에서 폭이 0으로 잡히면 rect가 접혀 **숫자만 남는다.**
      2×2 행렬은 늘릴 이유도 없다.
    """
    long = (cm.rename_axis("실제").reset_index()
            .melt(id_vars="실제", var_name="예측", value_name="건수"))
    cutoff = float(long["건수"].max()) * 0.5

    base = alt.Chart(long).encode(
        # rect는 band 스케일이어야 셀을 채운다. 레이어에서 스케일이 point로 잡히면 폭이 0이 된다.
        x=alt.X("예측:N", title=None, sort=list(cm.columns), scale=alt.Scale(type="band"),
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y("실제:N", title=None, sort=list(cm.index), scale=alt.Scale(type="band")),
    )
    heat = base.mark_rect().encode(
        color=alt.Color("건수:Q", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=["실제:N", "예측:N", "건수:Q"],
    )
    text = base.mark_text(fontSize=20, fontWeight="bold").encode(
        text="건수:Q", color=label_color(alt.datum["건수"] > cutoff))
    return (heat + text).properties(width=320, height=300)


def importance_chart(imp):
    """순열 중요도 — 중요도 내림차순(위→아래), x축은 0부터."""
    top = imp.sort_values("중요도", ascending=False)
    upper = float(top["중요도"].max())
    upper = upper * 1.05 if upper > 0 else 1.0
    return alt.Chart(top).mark_bar(color=eda.COLOR_STAY, clip=True).encode(
        x=alt.X("중요도:Q", title="AUC 감소량",
                scale=alt.Scale(domain=[0, upper], nice=False, clamp=True)),
        y=alt.Y("피처:N", sort="-x", title=None),
        tooltip=["피처:N", alt.Tooltip("중요도:Q", format=".4f")],
    ).properties(height=340)


def km_chart(km):
    """Kaplan-Meier 곡선 — x는 0개월부터, y는 0~1. 계단함수이므로 step-after로 잇는다."""
    upper = float(km["개월"].max())
    return alt.Chart(km).mark_line(interpolate="step-after", color=eda.COLOR_CHURN).encode(
        x=alt.X("개월:Q", title="가입 후 개월",
                scale=alt.Scale(domain=[0, upper], nice=False, clamp=True)),
        y=alt.Y("생존확률:Q", title="잔존 확률",
                scale=alt.Scale(domain=[0, 1], nice=False)),
        tooltip=[alt.Tooltip("개월:Q"), alt.Tooltip("생존확률:Q", format=".3f")],
    ).properties(height=300)


def target_chart(table, height=CHART_H):
    """타깃 분포 막대 — 값 라벨은 미리 만들어 텍스트 레이어로 얹는다."""
    data = table.copy()
    data["라벨"] = data.apply(lambda r: f"{r['인원']:,}명 ({r['비율(%)']:.1f}%)", axis=1)
    order = list(data["구분"])

    base = alt.Chart(data).encode(
        x=alt.X("구분:N", title=None, sort=order),
        y=alt.Y("인원:Q", title="고객 수",
                scale=alt.Scale(domain=[0, float(data["인원"].max()) * 1.25], nice=False)),
    )
    bars = base.mark_bar(size=80).encode(
        color=alt.Color("구분:N", sort=order, legend=None,
                        scale=alt.Scale(domain=order, range=[eda.COLOR_STAY, eda.COLOR_CHURN])),
        tooltip=["구분:N", "인원:Q", "비율(%):Q"],
    )
    labels = base.mark_text(dy=-10, fontSize=13).encode(text="라벨:N")
    return (bars + labels).properties(height=height)


def churn_rate_chart(table, overall, height=TALL_H):
    """변수별·값별 이탈률 — 격차가 큰 항목이 위로 오도록 정렬한다."""
    data = table.copy()
    data["항목"] = data["변수"] + " = " + data["값"].astype(str)

    bars = alt.Chart(data).mark_bar(color=eda.COLOR_STAY).encode(
        x=alt.X("이탈률(%):Q", scale=alt.Scale(domain=[0, 55], nice=False)),
        y=alt.Y("항목:N", sort="-x", title=None),
        tooltip=["변수:N", "값:N", "인원:Q", "이탈률(%):Q", "격차(%p):Q"],
    )
    # 기준선의 필드명이 다르면 Vega가 두 축 제목을 이어 붙인다 — 제목을 지워 막대 쪽만 남긴다.
    mean_rule = alt.Chart(pd.DataFrame({"전체 평균": [overall]})).mark_rule(
        color=eda.COLOR_CHURN, strokeDash=[4, 4], size=2).encode(
        x=alt.X("전체 평균:Q", title=None),
        tooltip=[alt.Tooltip("전체 평균:Q", format=".1f")])
    return (bars + mean_rule).properties(height=height)


def histogram_chart(hist, col):
    """이탈 여부별 히스토그램 — 겹쳐 그린다(stack=None을 빼면 Vega가 쌓아버린다)."""
    return alt.Chart(hist).mark_bar(opacity=0.6).encode(
        x=alt.X("시작:Q", title=col, bin="binned"),
        x2="끝:Q",
        y=alt.Y("인원:Q", title="고객 수", stack=None),
        color=alt.Color("구분:N", title=None,
                        scale=alt.Scale(domain=["유지", "이탈"],
                                        range=[eda.COLOR_STAY, eda.COLOR_CHURN])),
        tooltip=[alt.Tooltip("시작:Q", format=".2f"), alt.Tooltip("끝:Q", format=".2f"),
                 "구분:N", "인원:Q"],
    ).properties(height=CHART_H)


def corr_chart(long, cols, cell=46):
    """상관계수 히트맵 (하삼각)

    폭은 스펙에 넣지 않는다 — Streamlit이 `autosize: fit`을 붙여 어차피 컨테이너에 맞춘다.
    대신 호출부에서 폭이 확정된 컨테이너에 담고, 여기서는 높이만 칸 수에 비례해 잡는다.
    (그래야 셀이 정사각형에 가깝게 유지된다.)
    """
    base = alt.Chart(long).encode(
        # 세로로 세운다. -45도로 눕히면 `Avg_class_frequency_current_month` 같은 긴 이름의
        # 가로 폭이 칸 폭을 넘어, Vega가 겹치는 라벨을 하나 걸러 지워버린다.
        # labelOverlap=False로 지우는 것도 막고, labelLimit=0으로 말줄임도 없앤다.
        x=alt.X("변수2:N", title=None, sort=cols, scale=alt.Scale(type="band"),
                axis=alt.Axis(labelAngle=-90, labelOverlap=False, labelLimit=0,
                              labelAlign="right", labelBaseline="middle")),
        y=alt.Y("변수1:N", title=None, sort=cols, scale=alt.Scale(type="band"),
                axis=alt.Axis(labelOverlap=False, labelLimit=0)),
    )
    heat = base.mark_rect().encode(
        color=alt.Color("상관계수:Q", title="r",
                        scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True)),
        tooltip=["변수1:N", "변수2:N", alt.Tooltip("상관계수:Q", format=".2f")],
    )
    text = base.mark_text(fontSize=9).encode(
        text=alt.Text("상관계수:Q", format=".2f"),
        color=label_color((alt.datum["상관계수"] > 0.6) | (alt.datum["상관계수"] < -0.6)))
    return (heat + text).properties(height=cell * len(cols))


def lifetime_chart(table):
    """유지 기간 구간별 이탈률"""
    data = table.reset_index()
    data.columns = ["구간", "인원", "이탈률(%)"]
    order = list(data["구간"].astype(str))
    data["구간"] = data["구간"].astype(str)

    base = alt.Chart(data).encode(
        x=alt.X("구간:N", title=None, sort=order),
        y=alt.Y("이탈률(%):Q", scale=alt.Scale(domain=[0, 100], nice=False)),
    )
    bars = base.mark_bar(color=eda.COLOR_CHURN, size=60).encode(
        tooltip=["구간:N", "인원:Q", "이탈률(%):Q"])
    labels = base.mark_text(dy=-10, fontSize=12).encode(
        text=alt.Text("이탈률(%):Q", format=".1f"))
    return (bars + labels).properties(height=CHART_H)


def cross_chart(cross, x_title, y_title, counts=None):
    """교차표 → 이탈률 히트맵

    counts를 주면 그 칸의 인원(=이탈률의 분모)과 이탈자 수를 툴팁에 함께 띄운다.
    칸마다 분모가 다르다는 사실이 마우스만 올려도 보여야 오해가 없다.
    """
    long = (cross.rename_axis(y_title).reset_index()
            .melt(id_vars=y_title, var_name=x_title, value_name="이탈률(%)"))
    if counts is not None:
        n = (counts.rename_axis(y_title).reset_index()
             .melt(id_vars=y_title, var_name=x_title, value_name="인원"))
        long = long.merge(n, on=[y_title, x_title])
        long["이탈자"] = (long["인원"] * long["이탈률(%)"] / 100).round().astype(int)
    cutoff = float(long["이탈률(%)"].max()) * 0.55
    # 문자열로 바뀌면 Vega가 알파벳순으로 세운다(~27세가 맨 뒤로 감). 원래 순서를 넘긴다.
    x_order = [str(v) for v in cross.columns]
    y_order = [str(v) for v in cross.index]
    long[x_title] = long[x_title].astype(str)
    long[y_title] = long[y_title].astype(str)

    base = alt.Chart(long).encode(
        x=alt.X(f"{x_title}:N", title=x_title, sort=x_order, scale=alt.Scale(type="band"),
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{y_title}:N", title=y_title, sort=y_order, scale=alt.Scale(type="band")),
    )
    tooltip = [f"{y_title}:N", f"{x_title}:N", "이탈률(%):Q"]
    if counts is not None:
        tooltip += [alt.Tooltip("이탈자:Q", title="이탈자 수"),
                    alt.Tooltip("인원:Q", title="이 칸 인원(분모)")]
    heat = base.mark_rect().encode(
        color=alt.Color("이탈률(%):Q", scale=alt.Scale(scheme="reds"), legend=None),
        tooltip=tooltip,
    )
    text = base.mark_text(fontSize=13).encode(
        text=alt.Text("이탈률(%):Q", format=".1f"),
        color=label_color(alt.datum["이탈률(%)"] > cutoff))
    # 셀이 몇 개 안 되는 행렬이라 폭도 고정한다 (confusion_chart와 같은 이유).
    return (heat + text).properties(width=300, height=CHART_H)


def search_k_chart(ks, height=318):
    """k별 실루엣 — y축을 데이터 범위로 좁혀 변화를 보이게 한다.

    실루엣이 0.160~0.189로 폭이 0.03도 안 돼, 0부터 그리면 선이 축 위쪽에 눌려 붙어
    어느 k가 최선인지 구분되지 않는다. 위아래로 범위의 15%만 여백을 준다.
    최댓값 k에는 세로 점선을 얹어 눈이 먼저 가게 한다.
    """
    lo, hi = float(ks["실루엣"].min()), float(ks["실루엣"].max())
    pad = (hi - lo) * 0.15 or 0.01
    best_k = float(ks.loc[ks["실루엣"].idxmax(), "k"])

    # 두 레이어가 x축을 공유하므로 같은 인코딩을 쓴다. 점선 쪽에 그냥 "k:Q"를 주면
    # 수치형 기본값(zero=True)이 딸려와 도메인이 합쳐지고, 축이 2가 아니라 0부터 시작한다.
    x = alt.X("k:Q", title="k",
              scale=alt.Scale(domain=[float(ks["k"].min()), float(ks["k"].max())],
                              nice=False, zero=False),
              axis=alt.Axis(tickMinStep=1))

    line = alt.Chart(ks).mark_line(point=True, color=eda.COLOR_STAY).encode(
        x=x,
        y=alt.Y("실루엣:Q", scale=alt.Scale(domain=[lo - pad, hi + pad], nice=False),
                axis=alt.Axis(format=".3f")),
        tooltip=["k:Q", alt.Tooltip("실루엣:Q", format=".4f")],
    )
    best = alt.Chart(pd.DataFrame({"k": [best_k]})).mark_rule(
        color=eda.COLOR_CHURN, strokeDash=[4, 4]).encode(x=x)
    return (best + line).properties(height=height)


def learning_curve_chart(curve, height=CHART_H):
    """학습곡선 — 요점은 '마지막 구간이 평평한가'라 y축을 데이터 범위로 좁힌다.

    0부터 그리면 0.984~0.988의 변화가 선 하나로 뭉개져 포화 여부가 보이지 않는다.
    """
    lo, hi = float(curve["CV AUC"].min()), float(curve["CV AUC"].max())
    pad = (hi - lo) * 0.2 or 0.001
    return alt.Chart(curve).mark_line(point=True, color=eda.COLOR_STAY).encode(
        x=alt.X("학습 표본:Q", title="학습 표본 수", scale=alt.Scale(nice=False, padding=20)),
        y=alt.Y("CV AUC:Q", title="CV ROC-AUC",
                scale=alt.Scale(domain=[lo - pad, hi + pad], nice=False),
                axis=alt.Axis(format=".4f")),
        tooltip=[alt.Tooltip("학습 표본:Q", title="학습 표본", format=",.0f"),
                 alt.Tooltip("CV AUC:Q", format=".4f")],
    ).properties(height=height)


# ── [예측 이전] 예측 입력 폼 ───────────────────────────────────────
# 예측 페이지로 옮기기로 해 비활성. 이 블록은 두 예측 블록에서만 쓰였다.
# @st.cache_data(show_spinner=False)
# def input_specs(df):
#     """열별 입력 위젯 사양 — 범위·기본값을 데이터에서 뽑는다."""
#     spec = {}
#     for col in RAW_INPUT_COLS:
#         s = df[col]
#         values = sorted(s.dropna().unique().tolist())
#         spec[col] = {
#             "min": float(s.min()),
#             "max": float(s.max()),
#             "median": float(s.median()),
#             "choices": values if len(values) <= 6 else None,
#             "int": bool(pd.api.types.is_integer_dtype(s)),
#         }
#     return spec
#
#
# def _coerce(spec, value):
#     """세션에 넣을 값을 위젯이 받아들이는 타입으로 맞춘다."""
#     if spec["choices"] is not None:
#         return min(spec["choices"], key=lambda o: abs(o - float(value)))
#     return int(value) if spec["int"] else round(float(value), 2)
#
#
# def _widget(container, col, spec, key):
#     label = LABELS.get(col, col)
#     if spec["choices"] is not None:
#         if spec["choices"] == [0, 1]:
#             return container.selectbox(label, [0, 1], key=key,
#                                        format_func=lambda v: "예" if v == 1 else "아니오")
#         return container.selectbox(label, spec["choices"], key=key,
#                                    format_func=lambda v: f"{v:g}")
#     if spec["int"]:
#         return container.number_input(label, min_value=int(spec["min"]), max_value=int(spec["max"]),
#                                       step=1, key=key)
#     return container.number_input(label, min_value=float(spec["min"]), max_value=float(spec["max"]),
#                                   step=0.1, format="%.2f", key=key)
#
#
# def raw_input_form(df, prefix, exclude=(), n_cols=3):
#     """원본 변수 입력 폼 → 제출 후 1행 DataFrame, 아직이면 None
#
#     파생변수는 받지 않는다. 원본을 받아 features.add_derived()로 계산해야
#     학습할 때와 같은 정의가 보장된다.
#     """
#     specs = input_specs(df)
#     cols = [c for c in RAW_INPUT_COLS if c not in exclude]
#     for col in cols:
#         st.session_state.setdefault(f"{prefix}_{col}", _coerce(specs[col], specs[col]["median"]))
#
#     if st.button("데이터에서 무작위 고객 불러오기", key=f"{prefix}_sample"):
#         row = df.sample(1).iloc[0]
#         for col in cols:
#             st.session_state[f"{prefix}_{col}"] = _coerce(specs[col], row[col])
#         st.session_state[f"{prefix}_done"] = True
#         st.rerun()
#
#     with st.form(f"{prefix}_form"):
#         grid = st.columns(n_cols)
#         for i, col in enumerate(cols):
#             _widget(grid[i % n_cols], col, specs[col], f"{prefix}_{col}")
#         submitted = st.form_submit_button("예측 실행", type="primary")
#
#     if submitted:
#         st.session_state[f"{prefix}_done"] = True
#     if not st.session_state.get(f"{prefix}_done"):
#         return None
#     # 폼 위젯은 제출 시점에만 세션에 반영되므로, 세션 값 = 마지막으로 제출한 값이다.
#     return pd.DataFrame([{col: st.session_state[f"{prefix}_{col}"] for col in cols}])


# ── 탭 1 · EDA ────────────────────────────────────────────────────
def early_churn_share(df, lifetime_table):
    """가입 0~1개월 구간이 전체 이탈자에서 차지하는 비중 → (인원, 전체, 비율%)"""
    early = lifetime_table.head(2)
    n_early = int(round((early["인원"] * early["이탈률(%)"] / 100).sum()))
    n_total = int(df[ft.TARGET].sum())
    return n_early, n_total, n_early / n_total * 100


def render_eda(df):
    tables = cached_eda_tables(df)
    overall = float(df[ft.TARGET].mean() * 100)

    st.subheader("데이터 개요")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(tables["기본 정보"], hide_index=True, width="stretch")
    with col2:
        st.dataframe(tables["타깃 분포"], hide_index=True, width="stretch")

    st.subheader("이탈 분포")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("유지 / 이탈 비율")
        # 옆 그래프와 높이를 맞춘다 — 나란히 놓았을 때 아래쪽이 어긋나 보이지 않게.
        st.altair_chart(target_chart(tables["타깃 분포"], height=TALL_H), width="stretch")
    with col2:
        st.caption(f"변수별 이탈률 (빨간 점선 = 전체 평균 {overall:.1f}%)")
        st.altair_chart(churn_rate_chart(tables["변수별 이탈률"], overall), width="stretch")
    with st.expander("변수별 이탈률 표로 보기"):
        st.dataframe(tables["변수별 이탈률"], hide_index=True, width="stretch")

    st.subheader("수치형 변수 분포")
    # 6개를 한 화면에 격자로 넣으면 각 그래프가 작아진다. 하나씩 크게 본다.
    col = st.selectbox("변수 선택", eda.NUMERIC, key="eda_numeric")
    st.altair_chart(histogram_chart(cached_histogram(df, col), col), width="stretch")
    st.caption("파랑 = 유지, 빨강 = 이탈 (겹쳐 그림)")

    st.subheader("상관관계 및 다중공선성")
    long, cols = cached_corr_long(df)
    # 폭이 확정된 컨테이너 안에서 stretch — width="content"로 두면 컨테이너 폭이 내용에,
    # 내용 폭이 컨테이너에 의존하는 순환이 생겨 차트가 찌그러진다.
    with st.container(width=900):
        st.altair_chart(corr_chart(long, cols), width="stretch")
    
    st.markdown("**다중공선성 쌍** — 설명변수끼리 `|r| ≥ 0.5` 로 겹치는 조합")
    st.dataframe(cached_vif_pairs(df), hide_index=True, width="stretch")
    st.markdown("상관계수는 두 변수만, VIF는 나머지 변수 전부를 함께 본 값. "
                "VIF 10 이상이면 그 변수의 분산이 다른 변수들로 10배 넘게 부풀려졌다는 뜻.")
    st.markdown("히트맵에서 대각선을 뺀 진한 칸이 이 표에 그대로 나옴. 겹치는 두 변수를 함께 넣으면 "
                "회귀계수가 서로 나눠 가져 해석이 어긋남. 그래서 파생변수 `계약소진율`로 "
                "`Month_to_end_contract`를 대체했고(VIF 19.18 → 1.41), 트랙 C 군집 축에서는 "
                "각 쌍의 한쪽을 제외함.")

    st.subheader("유지 기간 · 교차 분석")
    st.altair_chart(lifetime_chart(tables["유지기간별 이탈률"]), width="stretch")
    n_early, n_total, share = early_churn_share(df, tables["유지기간별 이탈률"])
    st.markdown(f"전체 이탈자 {n_total:,}명 중 {n_early:,}명"
                f"({share:.1f}%)이 가입 1개월 이내에 나가며 3개월을 넘긴 회원의 이탈은 사실상 사라짐. "
                f"가입 초기를 잘 넘기는 것이 중요함을 알 수 있음")

    # 0/1·개월 수를 그대로 축에 걸면 안 읽힌다. 차트와 설명이 같은 표를 보도록 한 번만 바꾼다.
    rename = dict(columns={0: "미참여", 1: "참여"}, index={1: "1개월", 6: "6개월", 12: "12개월"})
    contract_cross = tables["계약×그룹수업"].rename(**rename)
    contract_n = eda.contract_group_cross(df, metric="count").rename(**rename)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("계약 기간 × 그룹수업 참여별 이탈률(%)")
        with st.container(width=380):
            st.altair_chart(cross_chart(contract_cross, "그룹수업 참여", "계약 기간",
                                        counts=contract_n),
                            width="stretch")
    with col2:
        st.markdown("연령대 × 유지기간별 이탈률(%)")
        with st.container(width=380):
            st.altair_chart(cross_chart(tables["연령×유지기간"], "연령대", "유지 기간",
                                        counts=eda.age_lifetime_cross(df, metric="count")),
                            width="stretch")

    st.divider()
    render_preprocessing()


# ── EDA 탭 · 전처리 결정 근거 ─────────────────────────────────────
# 파생변수 정의는 상수(DERIVED_DOC)라 항상 보이고, 측정 근거는 validation.joblib이
# 있을 때만 보인다. 분류 탭에도 같은 DERIVED_DOC을 띄우는데, 거기서는 S1/S2 구성 설명의
# 일부라 맥락이 필요해 옮기지 않고 양쪽에서 같은 상수를 쓴다.
def decision_summary(v):
    """전처리 결정 요약표 — 수치는 저장된 검증 결과에서 뽑는다 (화면에 적어 두지 않는다)."""
    chi = v["chi_square"].set_index("변수")["p_value"]
    delta = v["ablation"].pivot_table(index="제거 대상", columns="모델", values="변화")
    outlier, others = v["outlier"].iloc[0], v["outlier"].iloc[1]
    imbalance = v["imbalance"].set_index("방식")
    auc_gap = float(imbalance["roc_auc"].max() - imbalance["roc_auc"].min())

    rows = [
        {"대상": "gender · Phone", "결정": "제거",
         "근거": (f"카이제곱 p={chi['gender']:.4f} / {chi['Phone']:.4f} (이탈과 독립) · "
                 f"제거 시 CV AUC logreg {delta.loc['gender + Phone', 'logreg']:+.4f} · "
                 f"xgb {delta.loc['gender + Phone', 'xgb']:+.4f} · "
                 f"군집 실루엣 {v['cluster_noise_gain']:+.1f}%")},
        {"대상": "Age", "결정": "유지",
         "근거": (f"제거 시 CV AUC logreg {delta.loc['Age', 'logreg']:+.4f} · "
                 f"xgb {delta.loc['Age', 'xgb']:+.4f} (둘 다 하락) · "
                 f"Lifetime과의 상관 {v['age_lifetime_corr']}로 독립 신호")},
    ]
    rows += [{"대상": row["대체 대상"], "결정": f"{row['파생변수']}로 대체", "근거": row["효과"]}
             for _, row in DERIVED_DOC.iterrows()]
    rows += [
        {"대상": f"{ft.TIMING_TARGET} 이상치", "결정": "제거하지 않음",
         "근거": (f"{int(outlier['인원']):,}명의 이탈률이 {outlier['이탈률(%)']}% "
                 f"(그 외 {int(others['인원']):,}명은 {others['이탈률(%)']}%) — 최우량 장기회원")},
        {"대상": "클래스 불균형", "결정": "처리 미적용",
         "근거": (f"세 방식의 AUC 차이 {auc_gap:.4f} · Precision은 미적용이 최고"
                 f"({imbalance.loc['미적용', 'precision']:.4f})")},
        {"대상": "데이터 추가 수집", "결정": "불필요",
         "근거": f"학습곡선 마지막 구간 기울기 {v['learning_curve_slope']:+.4f} (포화)"},
    ]
    return pd.DataFrame(rows)


def render_preprocessing():
    st.subheader("전처리 결정 근거")

    st.markdown("**파생변수 정의** — 기존 변수 대체")
    st.dataframe(DERIVED_DOC, hide_index=True, width="stretch")
    st.caption(f"`{ft.VISIT_DELTA}`는 `{ft.LEAK_SUSPECT}`를 품고 있어 S2에서는 쓰지 않음. "
               "비율(÷)이 아니라 차이(−)로 정의한 이유는 `Avg_class_frequency_total == 0`인 고객이 "
               "있어 `Zero Division`이 발생하기 때문")

    try:
        v = cached_validation()
    except FileNotFoundError:
        st.info(NOT_VALIDATED)
        return

    st.markdown("**결정 요약**")
    st.dataframe(decision_summary(v), hide_index=True, width="stretch")

    with st.expander("제거 — `gender` · `Phone` (세 방향 모두 무신호)"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("카이제곱 독립성 검정 — 변수 하나만 보고 판단")
            st.dataframe(v["chi_square"].round(4), hide_index=True, width="stretch")
        with col2:
            st.markdown("로지스틱 계수 p값 (원본 13개 · 표준화 후) — 나머지 변수를 통제하고 판단")
            st.dataframe(v["logit_pvalues"].round(4).head(6), hide_index=True, width="stretch")
        st.caption("두 검정은 보는 방향이 다르므로 **둘 다** 무신호로 나와야 제거 근거가 됨")

        st.markdown("제거 시 CV AUC 변화 — 양수면 제거가 이득")
        st.dataframe(v["ablation"][v["ablation"]["제거 대상"] == "gender + Phone"],
                     hide_index=True, width="stretch")

        st.markdown("군집 축에서의 효과")
        st.dataframe(v["cluster_noise"], hide_index=True, width="stretch")
        st.caption(f"9축이 11축보다 실루엣 {v['cluster_noise_gain']}% 높음 — KMeans는 모든 축을 "
                   "동등하게 보므로 분류에서 무해한 변수가 거리 계산에서는 잡음이 됨")

    with st.expander("유지 — `Age` (빼면 두 모델 모두 성능 하락)"):
        rank, total = v["age_importance_rank"]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("제거 시 CV AUC 변화 — 음수면 하락")
            st.dataframe(v["ablation"][v["ablation"]["제거 대상"] == "Age"],
                         hide_index=True, width="stretch")
            st.caption(f"순열 중요도 {total}개 중 {rank}위 · "
                       f"`{ft.TIMING_TARGET}`과의 상관 {v['age_lifetime_corr']}")
        with col2:
            st.markdown("연령대별 이탈률")
            st.dataframe(v["age_profile"].reset_index().rename(columns={"Age": "연령대"}),
                         hide_index=True, width="stretch")
        st.caption(f"`{ft.TIMING_TARGET}`의 대리변수일 가능성을 검토했으나, 같은 유지기간 구간 "
                   "안에서도 연령 차이가 유지되어(위 교차분석) 독립 신호임이 확인됨")

    with st.expander("하지 않기로 한 것 — 이상치 제거 · 불균형 처리 · 추가 수집"):
        st.markdown(f"**이상치 — 제거하지 않음** (`{ft.TIMING_TARGET}` IQR 기준)")
        st.dataframe(v["outlier"], hide_index=True, width="stretch")
        st.caption("이상치 집단이 이탈률 0%인 최우량 장기회원이라, 지우면 신호가 가장 강한 구간을 버리게 됨")

        st.markdown("**불균형 처리 — 기본 미적용** (이탈 26.5% · 2.8 : 1)")
        st.dataframe(v["imbalance"], hide_index=True, width="stretch")
        st.caption("지표별로 승자가 갈림 — Precision은 미적용, Recall은 SMOTE. 미적용을 기본으로 두는 "
                   "이유는 전처리 단계가 줄고 잘못된 개입 비용을 줄이는 Precision이 가장 높기 때문. "
                   "Recall이 필요하면 `build_models(balanced=True)`로 전환")

        st.markdown("**데이터 추가 수집 — 불필요**")
        col1, col2 = st.columns([3, 2])
        with col1:
            st.altair_chart(learning_curve_chart(v["learning_curve"]), width="stretch")
        with col2:
            st.dataframe(v["learning_curve"], hide_index=True, width="stretch",
                         height=ROW_PX * (len(v["learning_curve"]) + 1) + 3)
        st.caption(f"마지막 구간 기울기 {v['learning_curve_slope']:+.4f} — 표본을 늘려도 "
                   "더 오르지 않는 포화 상태")

    st.caption(f"저장된 검증 결과를 읽어 표시 — 검증 {v['created_at'].replace('T', ' ')} "
               f"· seed {v['seed']}")


# ── 탭 2 · 분류 (트랙 A) ──────────────────────────────────────────
# [예측 이전] 샘플 입력 예측 — 예측 페이지로 옮기기로 해 비활성.
# def render_predict_classification(df, scenario, derived, name):
#     st.subheader("샘플 입력 예측")
#     st.markdown(f"위에서 고른 구성(**{ft.scenario_tag(scenario, derived)} · {name}**)의 저장 모델로 "
#                 "고객 한 명의 이탈 확률을 계산. 파생변수는 입력값에서 자동 계산.")
#
#     raw = raw_input_form(df, "a4_clf")
#     if raw is None:
#         st.markdown("값을 채우고 **예측 실행** 누를 것.")
#         return
#
#     # 고도화 모델이 있으면 그쪽을 쓴다 (임계값까지 최적화된 버전).
#     try:
#         bundle, stage = clf.load_bundle(scenario, derived, name, tag="tuned"), ft.STAGE_TUNED
#     except FileNotFoundError:
#         try:
#             bundle, stage = clf.load_bundle(scenario, derived, name), ft.STAGE_BASE
#         except FileNotFoundError as e:
#             st.warning(str(e))
#             return
#
#     work = ft.add_derived(raw) if derived else raw
#     proba = float(clf.predict_proba(bundle, work)[0])
#     threshold = float(bundle.get("threshold", 0.5))
#
#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("이탈 확률", f"{proba * 100:.1f}%")
#     col2.metric("판정", "이탈 위험" if proba >= threshold else "유지",
#                 help=f"임계값 {threshold:.3f} 기준")
#     col3.metric("적용 임계값", f"{threshold:.3f}", help=f"{stage} 모델")
#     try:
#         tier = cl.predict_tier(cl.load_bundle(), raw).iloc[0]
#         col4.metric("군집 등급", tier, help="트랙 C 프로파일 기준")
#     except (FileNotFoundError, KeyError):
#         col4.metric("군집 등급", "—", help="군집 모델 없음")
#
#     st.progress(min(max(proba, 0.0), 1.0))
#     st.caption(f"사용 모델: {bundle['model_name']} · 단계 {stage} · 학습 {bundle['trained_at']}")
#     with st.expander("모델에 실제로 들어간 값 보기"):
#         st.dataframe(work[bundle["features"]], hide_index=True, width="stretch")


def render_classification(df, summary):
    st.markdown(
        f"- **시나리오** — 누수 의심 변수(`{ft.LEAK_SUSPECT}`)를 **넣은 S1** / **뺀 S2**\n"
        "- **파생변수** — 원본 그대로인 **raw** / 파생변수로 바꾼 **derived**\n\n"
    )
    with st.expander("네 구성의 피처와 파생변수 정의"):
        st.markdown("**대체된 파생변수**")
        st.dataframe(DERIVED_DOC, hide_index=True, width="stretch")
        st.warning(f"`{ft.VISIT_DELTA}`는 `{ft.LEAK_SUSPECT}`를 품고 있어 S2에서는 쓰지 않음.\n\n"
                   f"S2는 `{ft.CONTRACT_USED}`만 대체되고 피처가 10개로 유지됨 — S2에서 derived와 raw의 성능 차이가 거의 없는 이유.")

    st.subheader("구성별 성능")
    if summary is None:
        st.info(NOT_TRAINED)
    else:
        base = summary[(summary["트랙"] == TRACK_A)
                       & (summary["단계"] == ft.STAGE_BASE)
                       & (summary["지표"] == "test_auc")]
        if base.empty:
            st.info(NOT_TRAINED)
        else:
            # 행 = 구성, 열 = 모델. 구성이 이 표의 주어라 세로로 세운다.
            # 열은 알파벳순이 아니라 MODEL_NAMES 순서로 — 다른 표·차트와 모델 순서를 맞춘다.
            pivot = (base.pivot_table(index="구성", columns="모델", values="값")
                     .reindex(columns=clf.MODEL_NAMES).round(4))
            show_metric_table(pivot, "구성", index_width=150)
            with st.expander("전체 지표 보기 (F1 / Recall / Precision)"):
                other = summary[(summary["트랙"] == TRACK_A) & (summary["지표"] != "test_auc")]
                st.dataframe(
                    other.pivot_table(index=["단계", "구성", "모델"], columns="지표", values="값").round(4),
                    width="stretch")

    st.subheader("모델 상세 평가")
    col1, col2 = st.columns(2)
    config = col1.selectbox("구성", list(CONFIGS))
    scenario, derived = CONFIGS[config]
    models = clf.MODEL_NAMES
    name = col2.selectbox("모델", models, index=models.index(clf.PRIMARY_MODEL[scenario]))

    try:
        ev = cached_clf_eval(df, scenario, derived, name)
    except FileNotFoundError as e:
        st.warning(str(e))
    else:
        scores = ev["metrics"]["test"]
        for col, key in zip(st.columns(5), ["roc_auc", "f1", "recall", "precision", "accuracy"]):
            col.metric(key, f"{scores[key]:.4f}")

        # 컬럼으로 나누면 폭이 고정된 두 차트 사이에 남는 공간이 그대로 빈칸이 된다.
        # 가로 컨테이너에 담아 내용 크기만큼만 차지하게 하고 가운데로 모은다.
        with st.container(horizontal=True, horizontal_alignment="center", gap="large"):
            with st.container(width=460):
                st.caption("ROC 곡선 (점선 = 무작위)")
                st.altair_chart(roc_chart(ev["curve_roc"]), width="stretch")
            with st.container(width=360):
                st.caption("혼동행렬 (임계값 0.5)")
                st.altair_chart(confusion_chart(ev["confusion"]), width="stretch")

        # evaluate()가 계산해 둔 F1 최적 임계값. 위 지표·혼동행렬은 0.5 기준이므로
        # 임계값을 옮기면 어디까지 올라가는지 같이 밝혀야 0.5가 최적이 아니라는 게 보인다.
        thr = ev["threshold"]
        st.caption(f"위 지표와 혼동행렬은 임계값 {thr['기본']:.2f} 기준. 이 모델의 F1 최적 임계값은 "
                   f"{thr['F1 최적']:.3f}이며 그때 F1 {thr['F1 최적 점수']:.4f} — 고도화 단계에서는 "
                   "이 값을 test가 아닌 train 교차검증(OOF)으로 다시 구해 적용")

        st.caption("순열 중요도 — 피처를 섞었을 때의 test AUC 감소량 (내림차순)")
        st.altair_chart(importance_chart(ev["importance"]), width="stretch")
        
        meta = ev["meta"]
        st.caption(f"train {meta['n_train']:,} / test {meta['n_test']:,} · seed {meta['seed']} "
                   f"· 학습 {meta['trained_at']}")

    st.subheader("고도화 전/후 (주력 모델)")
    detail, pivot = cached_stage_compare(df)
    if detail.empty:
        st.info("고도화 모델 없음. `python -m src.analysis4.train` (--no-tune 없이) 실행 필요.")
    else:
        st.dataframe(detail, hide_index=True, width="stretch")
        if pivot is not None:
            flat = pivot.copy()
            flat.columns = [" · ".join(c) for c in flat.columns]
            st.caption("지표별 변화 (고도화 − 기본)")
            st.dataframe(flat.reset_index(), hide_index=True, width="stretch")
        st.markdown("Optuna 하이퍼파라미터 탐색 + 임계값 최적화")

    # [예측 이전] st.divider() + render_predict_classification(df, scenario, derived, name)

    st.divider()
    spec, params = cached_clf_specs()
    render_model_spec(
        spec, params, "clf_",
        note="시나리오별 주력 모델(S1은 xgb · S2는 logreg)의 기본·고도화 사양. "
             "임계값은 고도화 단계에서 train 교차검증(OOF)으로 다시 구한 값",
        params_note="고도화 행은 Optuna(30 trial)가 찾은 값, 기본 행은 `build_models()`가 "
                    "지정한 값. 설정하지 않은 기본값은 표에서 뺌")


# ── 탭 3 · 이탈 타이밍 (트랙 B) ───────────────────────────────────
# [예측 이전] 샘플 입력 예측 — 예측 페이지로 옮기기로 해 비활성.
# def render_predict_regression(df):
#     st.subheader("샘플 입력 예측")
#     st.markdown("두 접근을 함께 표시. `가입 후 경과 개월`은 트랙 B의 타깃이므로 입력받지 않음.")
#
#     raw = raw_input_form(df, "a4_reg", exclude=("Lifetime",))
#     if raw is None:
#         st.markdown("값을 채우고 **예측 실행** 누를 것.")
#         return
#
#     work = ft.add_derived(raw)
#     try:
#         cox_bundle = rg.load_bundle("b2")
#         # 화면의 B-1 = 모듈의 b3. 파일명은 모듈 태그를 따르므로 그대로 부른다.
#         # name은 반드시 넘긴다 — 생략하면 regression.DEFAULT_MODEL의 xgb가 로드되어
#         # 화면 라벨(최소제곱 회귀)·개요표(linreg)와 다른 모델의 예측값이 나온다.
#         ols_bundle = rg.load_bundle("b3", name=REG_OLS_MODEL)
#     except FileNotFoundError as e:
#         st.warning(str(e))
#         return
#
#     months = float(rg.predict_months(ols_bundle, work)[0])
#     risk = float(rg.cox_risk(cox_bundle, work)[0])
#     population = cached_cox_population(df)
#     percentile = float((population < risk).mean() * 100)
#
#     col1, col2 = st.columns(2)
#     col1.metric("Cox 위험도 순위 (B-2)", f"상위 {100 - percentile:.0f}%",
#                 help=f"선형예측자 {risk:+.3f} · 전체 {len(population):,}명 중 백분위")
#     col2.metric("예상 개월수 (B-1)", f"{months:.2f}개월",
#                 help=f"최소제곱 회귀 예측 — {BASELINE_MEAN_LABEL} 대비 개선이 없는 모델")
#
#     st.warning(f"**B-1 예측은 참고용.** 이 모델은 홀드아웃에서 {BASELINE_MEAN_LABEL}을 넘지 "
#                "못함. 신뢰할 수 있는 값은 B-2의 **상대 순위**뿐이며, 그마저 절대 시점이 아님.")


def regression_overview(cox, summary):
    """두 회귀를 한 표로 세운다 — 좌변에 무엇을 두느냐가 성패를 갈랐다는 게 이 탭의 요지다."""
    b1 = metric_table(summary, TRACK_B, REG_TAG["B-1"]) if summary is not None else None
    # 비-베이스라인 중 최솟값이 아니라 REG_OLS_MODEL의 값을 쓴다.
    # 이 열의 제목이 '최소제곱 회귀'이므로 다른 모델(rf·xgb)의 숫자가 들어가면 라벨과 어긋난다.
    if (b1 is not None and "test_rmse" in b1.columns
            and {BASELINE_MEAN, REG_OLS_MODEL} <= set(b1.index)):
        base = float(b1.loc[BASELINE_MEAN, "test_rmse"])
        ols = float(b1.loc[REG_OLS_MODEL, "test_rmse"])
        result = f"RMSE {ols:.4f} — {BASELINE_MEAN_LABEL} {base:.4f}도 못 이김"
    else:
        result = "—"

    return pd.DataFrame([
        {"항목": "예측 대상 (좌변)", "B-1 최소제곱 회귀": f"개월 수 `{ft.TIMING_TARGET}` 자체",
         "B-2 Cox 비례위험 회귀": "이탈 위험함수 h(t|x)"},
        {"항목": "학습 표본", "B-1 최소제곱 회귀": f"이탈자 {cox['n_event']:,}명만",
         "B-2 Cox 비례위험 회귀": f"전체 {cox['n_total']:,}명 (유지자는 우측 절단)"},
        {"항목": "평가 지표", "B-1 최소제곱 회귀": "RMSE (낮을수록 좋음)",
         "B-2 Cox 비례위험 회귀": "C-index (0.5 = 무작위)"},
        {"항목": "결과", "B-1 최소제곱 회귀": result,
         "B-2 Cox 비례위험 회귀": f"C-index {cox['c_index']:.4f}"},
    ])


def render_regression(df, summary):
    st.markdown(
        f"트랙 B — 이탈 **시점**(`{ft.TIMING_TARGET}`) 예측. 최소제곱 회귀 모델과 Cox 비례위험 회귀 모델 비교")

    cox = cached_cox(df)
    st.dataframe(regression_overview(cox, summary), hide_index=True, width="stretch")

    # 표에는 네 모델이 함께 오르므로 제목은 접근 이름으로 두고, 대표 모델은 아래 설명에서 밝힌다.
    st.subheader("B-1 회귀 — 개월 수 직접 예측 (비교군)")
    if summary is None:
        st.info(NOT_TRAINED)
    else:
        b1 = metric_table(summary, TRACK_B, REG_TAG["B-1"])
        if b1 is None:
            st.info(NOT_TRAINED)
        else:
            # 인덱스 폭은 가장 긴 이름(Baseline(Mean Selection))에 맞춘다.
            show_metric_table(b1.rename(index=REG_MODEL_LABEL), "모델",
                              index_width=210, value_width=110)
            st.markdown(
                f"`{BASELINE_MEAN_LABEL}`은 피처를 보지 않고 학습 평균만 답하는 기본 모델. "
                f"대표 모델은 최소제곱 회귀(`{REG_OLS_MODEL}`)로, 위 개요표의 결과 값과 "
                "아래 샘플 예측이 모두 이 모델")

    st.subheader("B-2 생존분석 — Cox 비례위험 회귀 (주력)")
    col1, col2, col3 = st.columns(3)
    col1.metric("C-index", f"{cox['c_index']:.4f}",
                help=f"0.5 = 무작위, 1.0 = 완벽 · holdout test {cox['n_test']:,}명에서 측정")
    col2.metric("분석 대상", f"{cox['n_total']:,}명",
                help=f"유지자를 우측 절단으로 포함해 전원 사용 "
                     f"(적합 {cox['n_train']:,}명 / 평가 {cox['n_test']:,}명)")
    col3.metric("절단(유지자) 비율", f"{cox['censored_rate'] * 100:.1f}%")
    st.markdown("B-1이 버린 유지자를 '아직 이탈하지 않음'으로 되살려 모든 데이터 행을 사용")

    st.caption("Kaplan-Meier 생존곡선 — 가입 후 개월별 잔존 확률")
    st.altair_chart(km_chart(cached_km(df)), width="stretch")

    st.caption("Cox 계수 (표준화 후) — 양수면 이탈 위험 증가, 위험비는 exp(계수). "
               "표준화 기준(평균·표준편차)은 train만 보고 정하므로 test 정보가 C-index에 섞이지 않음")
    st.dataframe(cox["coef"].round(4), hide_index=True, width="stretch")

    st.subheader("B-1 실패 원인 분석")
    st.markdown("이탈자만 남긴 뒤 각 피처와 이탈까지 걸린 개월 수 간의 피어슨 상관계수 확인")

    sig = cached_signal(df)
    col1, col2, col3 = st.columns(3)
    col1.metric("최대 | 피어슨 상관계수 |", f"{sig['최대_절대상관']:.3f}",
                help=f"이탈자 내부에서 피처와 `{ft.TIMING_TARGET}`(개월 수)의 상관계수 절댓값 중 최댓값")
    for col, (label, auc) in zip((col2, col3), sig["이진_재정의_AUC"].items()):
        col.metric(f"AUC — {label}", f"{auc:.3f}", help="0.5 미만이면 무작위 이하")

    corr = (sig["상관계수"].rename("개월 수와의 피어슨 상관계수")
            .rename_axis("피처").reset_index())
    st.dataframe(corr, hide_index=True, width="stretch")

    # [예측 이전] st.divider() + render_predict_regression(df)

    st.divider()
    spec, params = cached_reg_specs()
    render_model_spec(
        spec, params, "reg_")


# ── 탭 4 · 군집화 (트랙 C) ────────────────────────────────────────
PROFILE_INDEX_LABEL = "항목"
ROW_PX = 35    # st.dataframe 기본 행 높이 — 표 높이를 행 수로 환산할 때 쓴다


def tier_profile(profile, mapping):
    """군집 번호 컬럼을 등급명으로 바꾸고 이탈률 오름차순으로 세운다.

    인덱스에도 이름을 붙인다 — 이름이 없으면 화면에서 피처 이름 위 헤더가 빈칸으로 나온다.
    """
    rename = {f"군집{row['군집']}": row["등급"] for _, row in mapping.iterrows()}
    ordered = mapping.sort_values("이탈률")["등급"].tolist()
    return profile.rename(columns=rename)[ordered].rename_axis(PROFILE_INDEX_LABEL)


def tier_playbook_md(mapping):
    """등급 매핑 + 해석 · 대응을 한 표로 → 마크다운 문자열

    셀렉트박스로 한 등급씩 넘기면 정작 등급 간 비교가 안 되고, 인원·이탈률·프로파일은
    위 표와 그대로 겹친다. 세 줄이니 한 화면에 세운다.
    st.dataframe이 아니라 마크다운인 이유는 두 가지다 — 셀 텍스트가 줄바꿈되지 않아
    문장이 잘리고, `**강조**` 표기가 그대로 노출된다.
    """
    lines = ["| 등급 | 인원 | 이탈률 | 해석 | 대응 |",
             "|---|---:|---:|---|---|"]
    for _, row in mapping.sort_values("이탈률").iterrows():
        play = TIER_PLAYBOOK.get(row["등급"], {})
        lines.append(
            f"| **{row['등급']}** | {int(row['인원']):,}명 | {row['이탈률']:.1f}% | "
            f"{play.get('해석', '—')} | {play.get('액션', '—')} |")
    return "\n".join(lines)


def show_profile_table(table, value_width=130):
    """프로파일 표 — 인덱스는 가장 긴 피처 이름에, 값 열은 자릿수에 맞춘다.

    높이를 행 수에 맞춰 직접 잡는다. 기본값은 10행까지만 보이고 나머지는 세로 스크롤로
    숨는데, 이 표는 11행이라 마지막 피처가 잘린다. 스크롤 없이 전부 보여야 비교가 된다.
    """
    st.dataframe(
        table, width="content", height=ROW_PX * (len(table) + 1) + 3,
        column_config={
            "_index": st.column_config.TextColumn(PROFILE_INDEX_LABEL, width=250),
            **{c: st.column_config.NumberColumn(c, width=value_width, format="%.2f")
               for c in table.columns},
        })


def axis_story(loadings, var):
    """각 주성분을 지배하는 변수 2개로 축의 의미를 설명한다."""
    lines = []
    for i, pc in enumerate(loadings.columns):
        s = loadings[pc]
        top = s.abs().sort_values(ascending=False).head(2).index.tolist()
        parts = " · ".join(f"`{f}` ({s[f]:+.2f})" for f in top)
        lines.append(f"- **{pc}** — 설명분산 {var[i]:.1%} · 지배 변수 {parts}")
    return "\n".join(lines)


def render_clustering(df):
    st.markdown("트랙 C — 이탈 여부(정답)를 빼고 9개 축으로 고객을 나눈 뒤, "
                "사후 이탈률로 등급명을 붙임.")

    try:
        views = cached_cluster(df)
    except FileNotFoundError as e:
        st.warning(str(e))
        return

    mapping = views["mapping"]
    spread = float(mapping["이탈률"].max() - mapping["이탈률"].min())
    col1, col2, col3 = st.columns(3)
    col1.metric("군집 수", views["k"])
    col2.metric("실루엣", f"{views['silhouette']:.4f}", help="1에 가까울수록 군집이 뚜렷함")
    col3.metric("이탈률 격차", f"{spread:.1f}%p", help="최고 이탈률 군집 − 최저 이탈률 군집")
    
    st.subheader("등급 매핑 · 대응")
    st.markdown(tier_playbook_md(mapping))

    st.subheader("군집 프로파일")
    show_profile_table(tier_profile(views["profile"], mapping))

    st.subheader("PCA 분포 — 2D · 3D")
    var = views["var"]
    col1, col2 = st.columns(2)
    common = dict(color="등급", opacity=0.55,
                  category_orders={"등급": list(TIER_COLORS)},
                  color_discrete_map=TIER_COLORS, height=430)
    with col1:
        st.caption(f"2차원 — 설명분산 합계 {sum(var[:2]):.1%}")
        st.plotly_chart(px.scatter(views["pca2"], x="PC1", y="PC2", **common),
                        width="stretch", key="pca2")
    with col2:
        st.caption(f"3차원 — 설명분산 합계 {sum(var[:3]):.1%} (끌어서 회전)")
        fig3d = px.scatter_3d(views["pca3"], x="PC1", y="PC2", z="PC3", **common)
        # 3D 기본 마커(5px)는 4,000점에서 서로 겹쳐 군집이 뭉개진다.
        fig3d.update_traces(marker=dict(size=2, line=dict(width=0)))
        st.plotly_chart(fig3d, width="stretch", key="pca3")

    st.markdown("**축이 무엇을 뜻하는가**")
    st.markdown(axis_story(views["loadings"], var))
    with st.expander("주성분 로딩 전체 보기"):
        st.dataframe(views["loadings"], width="stretch")
    st.markdown(f"3차원으로 늘려도 전체 분산의 {sum(var[:3]):.1%}만 설명함")

    st.subheader("고객 유지 방안")
    
    st.markdown("**실행 액션**")
    st.dataframe(cached_retention_actions(df), hide_index=True, width="stretch")

    st.markdown("**근거 — 레버별 이탈률 격차**")
    st.dataframe(cached_lever_effects(df), hide_index=True, width="stretch")

    with st.expander("솔루션을 어떤 변수에서 뽑았나"):
        st.dataframe(INTERVENTION_TABLE, hide_index=True, width="stretch")
        
    st.subheader("k 탐색")
    ks = cached_search_k(df)
    # 표는 세 열이면 충분하다 — inertia는 이 페이지에서 쓰지 않는 엘보 지표고,
    # 군집별_이탈률은 k만큼 길어지는 목록이라 옆에 두면 그래프 자리를 다 먹는다.
    col1, col2 = st.columns([3, 2])
    with col1:
        st.altair_chart(search_k_chart(ks), width="stretch")
    with col2:
        st.dataframe(ks[["k", "실루엣", "최소최대_격차"]].round(4), hide_index=True,
                     width="stretch", height=ROW_PX * (len(ks) + 1) + 3)

    st.divider()
    try:
        spec, params = cached_cluster_spec()
    except FileNotFoundError as e:
        st.subheader("모델 사양")
        st.warning(str(e))
    else:
        render_model_spec(
            spec, params, "cluster_")


# ── 페이지 ────────────────────────────────────────────────────────
# 탭 버튼은 기본값이 글자 폭에 딱 붙어 작다. flex로 폭을 균등 분배해 가로로 채운다.
# BaseWeb 구조라 선택자를 data-baseweb 속성으로 잡는다 — 클래스명은 빌드마다 바뀐다.
st.markdown(
    """
    <style>
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        width: 100%;
        gap: 6px;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        flex: 1 1 0;              /* 남는 폭을 탭 수만큼 똑같이 나눠 가진다 */
        justify-content: center;
        padding: 14px 8px;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] p {
        font-size: 18px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("분석 4 — 헬스장 고객 이탈")

df = load_data()

if df is None:
    st.warning("데이터를 불러올 수 없음.")
else:
    try:
        summary, created_at, seed = cached_summary()
    except FileNotFoundError:
        summary, created_at, seed = None, None, None
        st.warning(NOT_TRAINED)

    if summary is not None:
        head = tr.headline(summary)
        for col, (_, row) in zip(st.columns(max(len(head), 1)), head.iterrows()):
            col.metric(f"{row['트랙']} · {row['지표']}", f"{row['최고값']:.4f}",
                       help=f"{row['구성']} · {row['모델']} · {row['단계']}")
        st.caption(f"저장된 학습 결과를 읽어 표시 — 학습 {created_at} · seed {seed}")

    tab_eda, tab_clf, tab_reg, tab_clu = st.tabs(
        ["EDA", "이탈 여부 예측", "이탈 타이밍 예측", "회원 군집화"])

    with tab_eda:
        render_eda(df)
    with tab_clf:
        render_classification(df, summary)
    with tab_reg:
        render_regression(df, summary)
    with tab_clu:
        render_clustering(df)
