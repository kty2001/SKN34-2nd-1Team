import streamlit as st

from src.common import load_data
from src.analysis4 import eda

# 주제·설명은 추후 확정 / 학습·고도화 탭은 evaluate.py, advanced.py 작성 후 연결
st.title("분석 4 — 헬스장 고객 이탈")
st.write("탐색적 분석 결과")

df = load_data()

if df is None:
    st.warning("데이터를 불러올 수 없습니다.")
else:
    tab1, tab2, tab3 = st.tabs(["EDA", "학습/추론 평가", "고도화 전/후 평가"])

    with tab1:
        tables = eda.all_tables(df)
        figures = eda.all_figures(df)

        st.subheader("데이터 개요")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(tables["기본 정보"], hide_index=True, use_container_width=True)
        with col2:
            st.dataframe(tables["타깃 분포"], hide_index=True, use_container_width=True)
        st.pyplot(figures["타깃 분포"])

        st.subheader("변수별 이탈률")
        st.pyplot(figures["변수별 이탈률"])
        st.dataframe(tables["변수별 이탈률"], hide_index=True, use_container_width=True)

        st.subheader("수치형 변수 분포")
        st.pyplot(figures["수치형 분포"])
        st.dataframe(tables["이탈 여부별 평균"], use_container_width=True)

        st.subheader("상관관계 및 다중공선성")
        st.pyplot(figures["상관 히트맵"])
        st.dataframe(tables["다중공선성 쌍"], hide_index=True, use_container_width=True)

        st.subheader("유지 기간 · 교차 분석")
        st.pyplot(figures["유지기간별 이탈률"])
        col1, col2 = st.columns(2)
        with col1:
            st.caption("계약 기간 × 그룹수업 참여별 이탈률(%)")
            st.dataframe(tables["계약×그룹수업"], use_container_width=True)
        with col2:
            st.caption("연령대 × 유지기간별 이탈률(%)")
            st.dataframe(tables["연령×유지기간"], use_container_width=True)

    with tab2:
        st.subheader("학습/추론 평가")
        st.info("준비 중 — src/analysis4/evaluate.py 작성 후 연결")

    with tab3:
        st.subheader("고도화 전/후 평가")
        st.info("준비 중 — src/analysis4/advanced.py 작성 후 연결")
