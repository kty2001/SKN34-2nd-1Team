import joblib
from pathlib import Path

import pandas as pd
import streamlit as st


MODEL_DIR = Path("models/analysis2")

FEATURES = [
    "gender",
    "Near_Location",
    "Partner",
    "Promo_friends",
    "Phone",
    "Contract_period",
    "Group_visits",
    "Age",
    "Avg_additional_charges_total",
    "Month_to_end_contract",
    "Lifetime",
    "Avg_class_frequency_total",
    "Avg_class_frequency_current_month",
]

LABELS = {
    "gender": "성별",
    "Near_Location": "헬스장 근처 거주 여부",
    "Partner": "제휴 파트너 가입 여부",
    "Promo_friends": "친구 추천 가입 여부",
    "Phone": "전화번호 제공 여부",
    "Contract_period": "계약 기간 (개월)",
    "Group_visits": "그룹 운동 참여 여부",
    "Age": "나이",
    "Avg_additional_charges_total": "부가서비스 총 지출액",
    "Month_to_end_contract": "계약 종료까지 남은 개월",
    "Lifetime": "가입 기간 (개월)",
    "Avg_class_frequency_total": "전체 평균 수업 참여 빈도",
    "Avg_class_frequency_current_month": "최근 월 평균 수업 참여 빈도",
}


@st.cache_resource
def load_model(model_name="random_forest_advanced"):
    """저장된 분석2 모델을 불러온다."""
    model_path = MODEL_DIR / f"{model_name}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    with open(model_path, "rb") as f:
        return joblib.load(model_path)


def predict_sample(input_data, model_name="random_forest_advanced"):
    """입력 데이터 1건을 분석2 모델로 예측한다."""
    model = load_model(model_name)

    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.Series):
        input_data = input_data.to_frame().T
    elif not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame(input_data, columns=FEATURES)

    input_data = input_data[FEATURES]

    prediction = int(model.predict(input_data)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_data)[0][1])

    return prediction, probability


def render_prediction_form(df, model_name="random_forest_advanced"):
    """Streamlit에서 분석2 샘플 입력 및 예측 화면을 표시한다."""
    st.write("고객의 이용 특성을 입력하면 결정트리 또는 랜덤포레스트 모델로 이탈 여부를 예측합니다.")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("성별", [0, 1])
        near_location = st.selectbox("헬스장 근처 거주 여부", [0, 1])
        partner = st.selectbox("제휴 파트너 가입 여부", [0, 1])
        promo_friends = st.selectbox("친구 추천 가입 여부", [0, 1])
        phone = st.selectbox("전화번호 제공 여부", [0, 1])
        contract_period = st.number_input(
            "계약 기간 (개월)",
            min_value=1.0,
            value=3.0,
            step=1.0,
        )
        group_visits = st.selectbox("그룹 운동 참여 여부", [0, 1])

    with col2:
        age = st.number_input(
            "나이",
            min_value=float(df["Age"].min()),
            max_value=float(df["Age"].max()),
            value=float(df["Age"].median()),
            step=1.0,
        )
        additional_charges = st.number_input(
            "부가서비스 총 지출액",
            min_value=0.0,
            value=float(df["Avg_additional_charges_total"].median()),
            step=10.0,
        )
        month_to_end = st.number_input(
            "계약 종료까지 남은 개월",
            min_value=0.0,
            value=float(df["Month_to_end_contract"].median()),
            step=1.0,
        )
        lifetime = st.number_input(
            "가입 기간 (개월)",
            min_value=0.0,
            value=float(df["Lifetime"].median()),
            step=1.0,
        )
        avg_freq_total = st.number_input(
            "전체 평균 수업 참여 빈도",
            min_value=0.0,
            value=float(df["Avg_class_frequency_total"].median()),
            step=0.1,
        )
        avg_freq_current = st.number_input(
            "최근 월 평균 수업 참여 빈도",
            min_value=0.0,
            value=float(df["Avg_class_frequency_current_month"].median()),
            step=0.1,
        )

    input_data = {
        "gender": gender,
        "Near_Location": near_location,
        "Partner": partner,
        "Promo_friends": promo_friends,
        "Phone": phone,
        "Contract_period": contract_period,
        "Group_visits": group_visits,
        "Age": age,
        "Avg_additional_charges_total": additional_charges,
        "Month_to_end_contract": month_to_end,
        "Lifetime": lifetime,
        "Avg_class_frequency_total": avg_freq_total,
        "Avg_class_frequency_current_month": avg_freq_current,
    }

    if st.button("🔍 이탈 여부 예측", key="analysis2_predict", use_container_width=True):
        prediction, probability = predict_sample(input_data, model_name)

        st.markdown("---")

        if prediction == 1:
            st.error("⚠️ 예측 결과: **이탈 고객**")
        else:
            st.success("✅ 예측 결과: **유지 고객**")

        if probability is not None:
            col1, col2 = st.columns(2)
            col1.metric("이탈 확률", f"{probability * 100:.1f}%")
            col2.metric("유지 확률", f"{(1 - probability) * 100:.1f}%")

        with st.expander("입력 데이터 확인"):
            st.dataframe(
                pd.DataFrame([input_data]),
                hide_index=True,
                use_container_width=True,
            )


def render_prediction_page(df):
    """분석2 예측 페이지 전체 UI."""
    st.subheader("📊 분석2 고객 이탈 예측")

    model_name = st.selectbox(
        "사용할 모델",
        [
            "random_forest_advanced",
            "decision_tree_advanced",
            "random_forest_base",
            "decision_tree_base",
        ],
        format_func=lambda x: {
            "random_forest_advanced": "Random Forest (고도화)",
            "decision_tree_advanced": "Decision Tree (고도화)",
            "random_forest_base": "Random Forest (기본)",
            "decision_tree_base": "Decision Tree (기본)",
        }[x],
    )

    render_prediction_form(df, model_name)