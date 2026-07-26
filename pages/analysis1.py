import streamlit as st
from src.common import SEED, load_data # 데이터 불러오기 모듈
from src.analysis1 import eda_test, evaluate_test # 예시 추후 수정

# 내용 추후 수정
st.set_page_config(page_title="주제", page_icon="📊")
st.title("주제")
st.write("간략설명")

# 데이터 불러오기
df = load_data()

if df is not None:
    tab1, tab2, tab3 = st.tabs(["EDA", "학습/추론 평가", "고도화 전/후 평가"])
	
	# 탭 별 내용은 추후 수정
    with tab1:
        st.subheader("EDA")
        st.dataframe(eda_test())

    with tab2:
        st.subheader("학습/추론 평가")
        st.write(evaluate_test())

    with tab3:
        st.subheader("고도화 전/후 평가")
        st.write(evaluate_test())
else:
    st.warning("데이터를 불러올 수 없습니다.")