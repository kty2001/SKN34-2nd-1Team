"""analysis4 샘플 입력 예측 모듈

원본 분석 페이지에서 샘플 입력/예측에 필요한 코드만 분리한 모듈.
학습은 하지 않고 저장된 모델 번들을 불러와 예측만 수행한다.
"""

import pandas as pd
import streamlit as st

from src.analysis4 import classification as clf
from src.analysis4 import clustering as cl
from src.analysis4 import features as ft
from src.analysis4 import regression as rg


# 원본 데이터에서 사용자 입력용으로 받을 변수
RAW_INPUT_COLS = list(ft.BASE_FEATURES)

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

REG_OLS_MODEL = "linreg"
BASELINE_MEAN_LABEL = "Baseline(Mean Selection)"


@st.cache_data(show_spinner=False)
def input_specs(df):
    """데이터에서 입력 위젯의 범위·기본값·선택지를 생성한다."""
    spec = {}
    for col in RAW_INPUT_COLS:
        s = df[col]
        values = sorted(s.dropna().unique().tolist())
        spec[col] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
            "choices": values if len(values) <= 6 else None,
            "int": bool(pd.api.types.is_integer_dtype(s)),
        }
    return spec


def _coerce(spec, value):
    """위젯에 넣을 수 있는 타입으로 값을 변환한다."""
    if spec["choices"] is not None:
        return min(spec["choices"], key=lambda o: abs(o - float(value)))
    return int(value) if spec["int"] else round(float(value), 2)


def _widget(container, col, spec, key):
    """변수 타입에 맞는 Streamlit 입력 위젯을 생성한다."""
    label = LABELS.get(col, col)

    if spec["choices"] is not None:
        if spec["choices"] == [0, 1]:
            return container.selectbox(
                label, [0, 1], key=key,
                format_func=lambda v: "예" if v == 1 else "아니오"
            )
        return container.selectbox(
            label, spec["choices"], key=key,
            format_func=lambda v: f"{v:g}"
        )

    if spec["int"]:
        return container.number_input(
            label,
            min_value=int(spec["min"]),
            max_value=int(spec["max"]),
            step=1,
            key=key,
        )

    return container.number_input(
        label,
        min_value=float(spec["min"]),
        max_value=float(spec["max"]),
        step=0.1,
        format="%.2f",
        key=key,
    )


def raw_input_form(df, prefix, exclude=(), n_cols=3):
    """원본 변수 입력 → 제출 후 1행 DataFrame 반환."""
    specs = input_specs(df)
    cols = [c for c in RAW_INPUT_COLS if c not in exclude]

    for col in cols:
        st.session_state.setdefault(
            f"{prefix}_{col}",
            _coerce(specs[col], specs[col]["median"])
        )

    if st.button("데이터에서 무작위 고객 불러오기", key=f"{prefix}_sample"):
        row = df.sample(1).iloc[0]

        for col in cols:
            st.session_state[f"{prefix}_{col}"] = _coerce(specs[col], row[col])

        st.session_state[f"{prefix}_done"] = True
        st.rerun()

    with st.form(f"{prefix}_form"):
        grid = st.columns(n_cols)

        for i, col in enumerate(cols):
            _widget(
                grid[i % n_cols],
                col,
                specs[col],
                f"{prefix}_{col}",
            )

        submitted = st.form_submit_button("예측 실행", type="primary")

    if submitted:
        st.session_state[f"{prefix}_done"] = True

    if not st.session_state.get(f"{prefix}_done"):
        return None

    return pd.DataFrame([
        {col: st.session_state[f"{prefix}_{col}"] for col in cols}
    ])


@st.cache_data(show_spinner=False)
def cached_cox_population(df):
    """전체 고객의 Cox 위험도 분포."""
    bundle = rg.load_bundle("b2")
    return pd.Series(rg.cox_risk(bundle, ft.add_derived(df)))


def predict_classification(df, scenario, derived, name, prefix="a4_clf"):
    """이탈 여부 예측 UI를 표시한다."""
    st.subheader("이탈 여부 샘플 예측")
    st.markdown(
        f"선택 구성(**{ft.scenario_tag(scenario, derived)} · {name}**)의 "
        "저장 모델로 고객 한 명의 이탈 확률을 계산합니다."
    )

    raw = raw_input_form(df, prefix)

    if raw is None:
        st.info("값을 입력한 후 **예측 실행**을 눌러주세요.")
        return

    try:
        try:
            bundle = clf.load_bundle(
                scenario, derived, name, tag="tuned"
            )
            stage = ft.STAGE_TUNED
        except FileNotFoundError:
            bundle = clf.load_bundle(
                scenario, derived, name
            )
            stage = ft.STAGE_BASE

    except FileNotFoundError as e:
        st.warning(str(e))
        return

    work = ft.add_derived(raw) if derived else raw
    proba = float(clf.predict_proba(bundle, work)[0])
    threshold = float(bundle.get("threshold", 0.5))

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("이탈 확률", f"{proba * 100:.1f}%")
    col2.metric(
        "판정",
        "이탈 위험" if proba >= threshold else "유지",
        help=f"임계값 {threshold:.3f} 기준",
    )
    col3.metric(
        "적용 임계값",
        f"{threshold:.3f}",
        help=f"{stage} 모델",
    )

    try:
        tier = cl.predict_tier(
            cl.load_bundle(),
            raw
        ).iloc[0]
        col4.metric(
            "군집 등급",
            tier,
            help="트랙 C 프로파일 기준",
        )
    except (FileNotFoundError, KeyError):
        col4.metric(
            "군집 등급",
            "—",
            help="군집 모델 없음",
        )

    st.progress(min(max(proba, 0.0), 1.0))

    st.caption(
        f"사용 모델: {bundle['model_name']} · "
        f"단계 {stage} · 학습 {bundle['trained_at']}"
    )

    with st.expander("모델에 실제로 들어간 값 보기"):
        st.dataframe(
            work[bundle["features"]],
            hide_index=True,
            width="stretch",
        )


def predict_regression(df, prefix="a4_reg"):
    """이탈 타이밍 샘플 예측 UI를 표시한다."""
    st.subheader("이탈 타이밍 샘플 예측")
    st.markdown(
        "두 접근을 함께 표시합니다. "
        "`가입 후 경과 개월(Lifetime)`은 예측 대상이므로 입력받지 않습니다."
    )

    raw = raw_input_form(
        df,
        prefix,
        exclude=("Lifetime",),
    )

    if raw is None:
        st.info("값을 입력한 후 **예측 실행**을 눌러주세요.")
        return

    work = ft.add_derived(raw)

    try:
        cox_bundle = rg.load_bundle("b2")
        ols_bundle = rg.load_bundle(
            "b3",
            name=REG_OLS_MODEL,
        )
    except FileNotFoundError as e:
        st.warning(str(e))
        return

    months = float(
        rg.predict_months(ols_bundle, work)[0]
    )

    risk = float(
        rg.cox_risk(cox_bundle, work)[0]
    )

    population = cached_cox_population(df)
    percentile = float(
        (population < risk).mean() * 100
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Cox 위험도 순위",
        f"상위 {100 - percentile:.0f}%",
        help=f"선형예측자 {risk:+.3f} · 전체 {len(population):,}명 중 백분위",
    )

    col2.metric(
        "예상 개월수",
        f"{months:.2f}개월",
        help=f"최소제곱 회귀 예측 — {BASELINE_MEAN_LABEL} 대비 개선이 없는 모델",
    )

    st.warning(
        f"**B-1 예측은 참고용입니다.** "
        f"이 모델은 홀드아웃에서 {BASELINE_MEAN_LABEL}을 넘지 못했습니다. "
        "상대적으로 신뢰할 수 있는 값은 B-2의 상대 순위입니다."
    )


def render_prediction_page(df):
    """분석 4 예측 전용 페이지."""
    st.title("분석 4 — 샘플 고객 예측")

    tab_clf, tab_reg = st.tabs([
        "이탈 여부 예측",
        "이탈 타이밍 예측",
    ])

    with tab_clf:
        col1, col2 = st.columns(2)

        from src.analysis4 import features as ft

        configs = {
            "S1/derived": ("S1", True),
            "S1/raw": ("S1", False),
            "S2/derived": ("S2", True),
            "S2/raw": ("S2", False),
        }

        config = col1.selectbox(
            "구성",
            list(configs),
            key="predict_config",
        )

        scenario, derived = configs[config]

        models = clf.MODEL_NAMES
        name = col2.selectbox(
            "모델",
            models,
            index=models.index(
                clf.PRIMARY_MODEL[scenario]
            ),
            key="predict_model",
        )

        predict_classification(
            df,
            scenario,
            derived,
            name,
        )

    with tab_reg:
        predict_regression(df)


__all__ = [
    "raw_input_form",
    "predict_classification",
    "predict_regression",
    "render_prediction_page",
]
