import streamlit as st
from src.analysis1.predict import predict_sample

st.set_page_config(
    page_title="샘플 예측",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 고객 이탈 군집 샘플 예측")
st.write("고객의 이용 특성을 입력하면 해당 고객이 어느 클러스터에 속하는지 예측합니다.")

tab1, tab2, tab3, tab4 = st.tabs(["분석1", "분석2", "분석3", "분석4"])

with tab1:
    st.subheader("📊 분석1 고객 군집 예측")
    st.write("고객 정보를 입력하면 K-Means 모델을 통해 고객군을 예측합니다.")

    col1, col2 = st.columns(2)

    with col1:
        lifetime = st.number_input("이용 기간 (개월)", min_value=0.0, value=3.0, step=1.0)
        age = st.number_input("나이", min_value=10.0, max_value=100.0, value=30.0, step=1.0)

    with col2:
        avg_freq = st.number_input("최근 월 평균 방문 빈도", min_value=0.0, value=2.0, step=0.1)
        contract_period = st.number_input("계약 기간 (개월)", min_value=0.0, value=3.0, step=1.0)

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
            0: "장기 충성 고객",
            1: "고계약 집중 고객",
            2: "신규·라이트 유저",
            3: "이탈 위험 고객"
        }

        if cluster in cluster_name:
            st.info(f"해당 고객군: **{cluster_name[cluster]}**")

with tab2:
    st.subheader("📊 분석2")
    st.info("분석2 예측 모델을 준비 중입니다.")

with tab3:
    st.subheader("📊 분석3")
    st.info("분석3 예측 모델을 준비 중입니다.")

with tab4:
    st.subheader("📊 분석4")
    st.info("분석4 예측 모델을 준비 중입니다.")