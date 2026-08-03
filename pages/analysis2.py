import streamlit as st
from src.common import load_data  # 데이터 불러오기 모듈
from src.analysis2 import eda_test, evaluate_test
from src.analysis2.evaluate import plot_confusion_matrix, plot_feature_importance, MODEL_NAMES

st.set_page_config(page_title="분석2: 결정트리 & 랜덤포레스트", page_icon="🌳")
st.title("분석2: 결정트리 & 랜덤포레스트")
st.write("헬스장 회원 이탈 예측 - 트리 기반 모델(결정트리, 랜덤포레스트)로 분석합니다.")

# 데이터 불러오기
df = load_data()

if df is not None:
    tab1, tab2, tab3 = st.tabs(["EDA", "학습/추론 평가", "고도화 전/후 평가"])

    with tab1:
        st.subheader("EDA")
        st.dataframe(eda_test())

    with tab2:
        st.subheader("학습/추론 평가 (고도화 전 기본 모델)")
        st.dataframe(evaluate_test(stage="base"))

        st.markdown("**Confusion Matrix**")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_confusion_matrix("decision_tree", stage="base"))
        with col2:
            st.pyplot(plot_confusion_matrix("random_forest", stage="base"))

    with tab3:
        st.subheader("고도화 전/후 평가")

        st.markdown("**고도화 전 (base)**")
        st.dataframe(evaluate_test(stage="base"))

        st.markdown("**고도화 후 (advanced)**")
        st.dataframe(evaluate_test(stage="advanced"))

        st.markdown("**Feature Importance (고도화 후 모델 기준)**")
        for name in MODEL_NAMES:
            st.pyplot(plot_feature_importance(name, stage="advanced"))
else:
    st.warning("데이터를 불러올 수 없습니다.")