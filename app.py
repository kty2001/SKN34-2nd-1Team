import streamlit as st

# 사이드바 css 적용
st.markdown(
    """
    <style>
    /* 사이드바 전체 */
    [data-testid="stSidebarNavItems"] a span {
        font-size: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(
    page_title="고객 이탈 분석",
    page_icon="📊",
    layout="wide"
)

pages = {
    "메뉴": [
        st.Page(
            "pages/dashboard.py",
            title="대시보드",
            icon="🏠"
        ),

		# 이름은 추후에 수정
        st.Page(
			"pages/analysis1.py",
			title="분석1",
			icon="📊"
		),

		st.Page(
			"pages/analysis2.py",
			title="분석2",
			icon="📈"
		),

		st.Page(
			"pages/analysis3.py",
			title="분석3",
			icon="🔍"
		),

		st.Page(
			"pages/analysis4.py",
			title="분석4",
			icon="📋"
		),

		st.Page(
			"pages/sample_pred.py",
			title="예측",
			icon="🔮"
		)
    ]
}

pg = st.navigation(pages)

pg.run()