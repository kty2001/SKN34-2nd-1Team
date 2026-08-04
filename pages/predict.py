import streamlit as st
from src.common import load_data
from src.analysis1.predict import render_prediction_page as render_analysis1_prediction
from src.analysis2.predict import render_prediction_page as render_analysis2_prediction
from src.analysis3.predict import render_prediction_page as render_analysis3_prediction
from src.analysis4.predict import render_prediction_page as render_analysis4_prediction

st.set_page_config(
    page_title="샘플 예측",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 고객 이탈 군집 샘플 예측")
st.write("고객의 이용 특성을 입력하면 해당 고객의 분석 결과를 예측합니다.")

tab1, tab2, tab3, tab4 = st.tabs(["분석1", "분석2", "분석3", "분석4"])

with tab1:
    df = load_data()

    if df is None:
        st.warning("데이터를 불러올 수 없습니다.")
    else:
        render_analysis1_prediction()

with tab2:
    df = load_data()

    if df is None:
        st.warning("데이터를 불러올 수 없습니다.")
    else:
        render_analysis2_prediction(df)

with tab3:
    df = load_data()

    if df is None:
        st.warning("데이터를 불러올 수 없습니다.")
    else:
        render_analysis3_prediction(df)
        
with tab4:
    df = load_data()

    if df is None:
        st.warning("데이터를 불러올 수 없습니다.")
    else:
        render_analysis4_prediction(df)