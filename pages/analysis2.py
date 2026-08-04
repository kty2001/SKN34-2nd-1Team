import streamlit as st
from src.common import load_data
from src.analysis2 import (
    eda_test, evaluate_test,
    get_key_features_by_churn, plot_correlation_heatmap,
    plot_confusion_matrix, plot_feature_importance, MODEL_NAMES,
)

st.set_page_config(page_title="회원 이탈 예측", page_icon="🌳")
st.title("🌳 회원 이탈 예측")
st.write("가입기간·수업 참여빈도 기반 트리 모델(결정트리, 랜덤포레스트)로 이탈 여부를 예측합니다.")

# 데이터 불러오기
df = load_data()

if df is not None:
    tab1, tab2, tab3 = st.tabs(["EDA", "학습/추론 평가", "고도화 전/후 평가"])

    with tab1:
        st.subheader("EDA")
        st.dataframe(eda_test())

        st.markdown("**이탈(1) vs 잔류(0) 그룹 평균 비교**")
        st.dataframe(get_key_features_by_churn(df))

        st.markdown("**피처 간 상관관계**")
        st.pyplot(plot_correlation_heatmap(df))

        st.info("📌 이탈 회원은 잔류 회원 대비 계약기간(Contract_period), 가입기간(Lifetime)이 뚜렷하게 짧고 부가서비스 지출도 낮게 나타남 — '짧은 계약 + 낮은 참여'가 이탈의 전형적 패턴")

    with tab2:
        st.subheader("학습/추론 평가 (고도화 전 기본 모델)")
        st.dataframe(evaluate_test(stage="base"))

        st.markdown("**Confusion Matrix**")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_confusion_matrix("decision_tree", stage="base"))
        with col2:
            st.pyplot(plot_confusion_matrix("random_forest", stage="base"))

        st.info("📌 랜덤포레스트가 결정트리보다 이탈자를 잔류로 잘못 예측하는 비율(FN)이 낮아, 이탈자를 놓치지 않는 측면에서 더 우수함")

    with tab3:
        st.subheader("고도화 전/후 평가")

        st.markdown("**고도화 전 (base)**")
        st.dataframe(evaluate_test(stage="base"))

        st.markdown("**고도화 후 (advanced)**")
        st.dataframe(evaluate_test(stage="advanced"))

        st.markdown("**Feature Importance (고도화 후 모델 기준)**")
        for name in MODEL_NAMES:
            st.pyplot(plot_feature_importance(name, stage="advanced"))

        st.info("📌 가입기간(Lifetime)이 이탈 예측에 가장 큰 영향을 미치며, 최근 수업 참여빈도와 연령이 뒤를 이음 — 최종 채택 모델은 Recall/F1/ROC-AUC 기준 우수한 Random Forest(advanced)")
else:
    st.warning("데이터를 불러올 수 없습니다.")