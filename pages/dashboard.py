import streamlit as st
import altair as alt
import plotly.express as px
from src.common import SEED, load_data

# 데이터 불러오기
df = load_data()

st.title("고객 이탈 분석 대시보드")

st.divider()

# 기본 데이터
total_customers = len(df)
churn_customers = df["Churn"].sum()
active_customers = total_customers - churn_customers
churn_rate = (
    churn_customers / total_customers
) * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="전체 고객",
        value=f"{total_customers:,}명"
    )

with col2:
    st.metric(
        label="유지 고객",
        value=f"{active_customers:,}명"
    )

with col3:
    st.metric(
        label="이탈 고객",
        value=f"{churn_customers:,}명"
    )

with col4:
    st.metric(
        label="전체 이탈률",
        value=f"{churn_rate:.1f}%"
    )

st.divider()

st.header("주요 분석")

col1, col2, col3 = st.columns(3)

with col1:

    st.subheader("계약 기간별 고객 비율")

    contract_period_count = (
        df["Contract_period"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    contract_period_count.columns = [
        "계약 기간",
        "고객 수"
    ]
    
    contract_period_count["계약 기간"] = (
            contract_period_count["계약 기간"].astype(str) + "개월"
        )
    

    contract_period_chart = px.pie(
        contract_period_count,
        names="계약 기간",
        values="고객 수",
        hole=0.5
    )
    
    contract_period_chart.update_traces(
        textposition="inside",
        texttemplate="%{percent}",
        textfont=dict(
            color="white",
            size=18
        ),
        hovertemplate=(
            "계약 기간: %{label}<br>"
            "고객 수: %{value:,}명<br>"
            "<extra></extra>"
        )
    )
    
    contract_period_chart.update_layout(
        height=430
    )

    st.plotly_chart(
        contract_period_chart,
        use_container_width=True
    )

with col2:

    st.subheader("회원 이용 기간별 고객 수")

    lifetime_count = (
        df["Lifetime"]
        .value_counts(
            bins=5,
            sort=False
        )
        .reset_index()
    )

    lifetime_count.columns = [
        "회원 이용 기간",
        "고객 수"
    ]

    lifetime_count["정렬 기준"] = (
        lifetime_count["회원 이용 기간"]
        .apply(lambda x: x.left)
    )

    lifetime_count["회원 이용 기간"] = (
        lifetime_count["회원 이용 기간"]
        .apply(
            lambda x: f"{max(0, x.left):.0f}~{x.right:.0f}개월"
        )
    )

    lifetime_chart = (
        alt.Chart(lifetime_count)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "회원 이용 기간:O",
                title="회원 이용 기간",
                sort=alt.SortField(
                    field="정렬 기준",
                    order="ascending"
                ),
                axis=alt.Axis(labelAngle=0)
            ),
            y=alt.Y(
                "고객 수:Q",
                title="고객 수"
            )
        )
    )

    st.altair_chart(
        lifetime_chart,
        use_container_width=True
    )
    
with col3:

    st.subheader("계약 종료까지 남은 기간별 고객 수")

    month_end_count = (
        df["Month_to_end_contract"]
        .value_counts()
        .reset_index()
    )

    month_end_count.columns = [
        "계약 종료까지 남은 기간",
        "고객 수"
    ]
    
    month_end_count["계약 종료까지 남은 기간"] = (
        month_end_count["계약 종료까지 남은 기간"]
        .astype(int)
        .astype(str)
        + "개월"
    )

    chart = (
        alt.Chart(month_end_count)
        .mark_bar()
        .encode(
            x=alt.X(
                "계약 종료까지 남은 기간:O",
                title="계약 종료까지 남은 기간",
                sort=alt.SortField(
                    field="고객 수",
                    order="descending"
                ),
                axis=alt.Axis(labelAngle=0)
            ),
            y=alt.Y(
                "고객 수:Q",
                title="고객 수"
            )
        )
    )

    st.altair_chart(
        chart,
        use_container_width=True
    )
    
    st.divider()

st.header("핵심 인사이트")

# 주요 지표 계산
top_contract = (
    df["Contract_period"]
    .value_counts()
    .idxmax()
)

avg_lifetime = df["Lifetime"].mean()
avg_age = df["Age"].mean()
avg_frequency = df["Avg_class_frequency_current_month"].mean()

contract_churn = (
    df.groupby("Contract_period")["Churn"]
    .mean()
    .sort_values(ascending=False)
)

highest_churn_contract = contract_churn.index[0]
highest_churn_rate = contract_churn.iloc[0] * 100

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("⚠️ 이탈 위험 계약")
    st.metric(
        "이탈률이 가장 높은 계약",
        f"{highest_churn_contract}개월"
    )
    st.write(f"해당 계약의 이탈률은 **{highest_churn_rate:.1f}%**입니다.")

with col2:
    st.subheader("📋 주요 계약기간")
    st.metric(
        "가장 많은 고객이 선택",
        f"{top_contract}개월"
    )
    st.write("해당 계약기간을 선택한 고객이 가장 많습니다.")

with col3:
    st.subheader("👥 평균 고객 특성")
    st.write(f"평균 이용기간 **{avg_lifetime:.1f}개월**")
    st.write(f"평균 나이 **{avg_age:.1f}세**")
    st.write(f"평균 방문빈도 **{avg_frequency:.1f}회/월**")