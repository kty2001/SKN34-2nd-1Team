import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "models/analysis1/churn_cluster_model.joblib"

PREDICT_FEATURES = [
    "Lifetime",
    "Age",
    "Avg_class_frequency_current_month",
    "Contract_period"
]


def load_predict_model():
    return joblib.load(MODEL_PATH)


def predict_sample(lifetime, age, avg_frequency, contract_period):
    model = load_predict_model()

    sample = pd.DataFrame([{
        "Lifetime": lifetime,
        "Age": age,
        "Avg_class_frequency_current_month": avg_frequency,
        "Contract_period": contract_period
    }])

    return model.predict(sample)[0]


def render_prediction_page():
    st.subheader("📊 분석1 고객 군집 예측")
    st.write("고객 정보를 입력하면 K-Means 모델을 통해 고객군을 예측합니다.")

    col1, col2 = st.columns(2)

    with col1:
        lifetime = st.number_input(
            "이용 기간 (개월)",
            min_value=0.0,
            value=3.0,
            step=1.0
        )

        age = st.number_input(
            "나이",
            min_value=10.0,
            max_value=100.0,
            value=30.0,
            step=1.0
        )

    with col2:
        avg_freq = st.number_input(
            "최근 월 평균 방문 빈도",
            min_value=0.0,
            value=2.0,
            step=0.1
        )

        contract_period = st.number_input(
            "계약 기간 (개월)",
            min_value=0.0,
            value=3.0,
            step=1.0
        )

    st.markdown("---")

    if st.button("🔍 고객군 예측", use_container_width=True):
        cluster = predict_sample(
            lifetime,
            age,
            avg_freq,
            contract_period
        )

        st.success(f"예측 결과: **Cluster {cluster}**")

        cluster_name = {
            0: "VIP 고객",
            1: "장기계약 고객",
            2: "이탈 위험 고객",
        }

        if cluster in cluster_name:
            st.info(f"해당 고객군: **{cluster_name[cluster]}**")