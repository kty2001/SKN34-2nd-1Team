<<<<<<< HEAD
# eda
from src.common import SEED, load_data # 데이터 불러오기 모듈
=======
"""
src/analysis2/eda.py
분석2 (결정트리 / 랜덤포레스트) - 탐색적 데이터 분석(EDA)
>>>>>>> main

gym_churn_us.csv 컬럼:
gender, Near_Location, Partner, Promo_friends, Phone, Contract_period,
Group_visits, Age, Avg_additional_charges_total, Month_to_end_contract,
Lifetime, Avg_class_frequency_total, Avg_class_frequency_current_month, Churn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"  # Windows 한글 폰트
plt.rcParams["axes.unicode_minus"] = False      # 마이너스 기호 깨짐 방지


def get_summary(df: pd.DataFrame) -> pd.DataFrame:
    # 이 함수를 쓴 이유: 전체 컬럼의 기초 통계(평균, 표준편차, 결측치 등)를 한눈에 보기 위해
    summary = df.describe().T
    summary["missing"] = df.isnull().sum()
    summary["dtype"] = df.dtypes
    return summary


def get_churn_rate(df: pd.DataFrame) -> float:
    # 이 함수를 쓴 이유: 전체 이탈률이 몇 %인지 확인 (클래스 불균형 체크 -> 나중에 SMOTE 필요한지 판단 근거)
    return df["Churn"].mean()


def plot_churn_distribution(df: pd.DataFrame):
    # 이 함수를 쓴 이유: 이탈(1) vs 잔류(0) 회원 수 비율을 막대그래프로 확인
    fig, ax = plt.subplots(figsize=(5, 4))
    df["Churn"].value_counts().sort_index().plot(
        kind="bar", ax=ax, color=["#4C72B0", "#DD8452"]
    )
    ax.set_xticklabels(["잔류(0)", "이탈(1)"], rotation=0)
    ax.set_title("이탈 여부 분포")
    ax.set_ylabel("회원 수")
    return fig


def plot_feature_by_churn(df: pd.DataFrame, feature: str):
    # 이 함수를 쓴 이유: 특정 피처가 이탈 그룹과 잔류 그룹에서 어떻게 다르게 분포하는지 보기 위해
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="Churn", y=feature, ax=ax)
    ax.set_xticklabels(["잔류(0)", "이탈(1)"])
    ax.set_title(f"{feature} - 이탈 여부별 분포")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    # 이 함수를 쓴 이유: 피처간 상관관계 확인 (다중공선성 체크 + Churn과 상관 높은 피처 파악)
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("피처 간 상관관계")
    return fig


def get_key_features_by_churn(df: pd.DataFrame) -> pd.DataFrame:
    # 이 함수를 쓴 이유: 이탈(1) 그룹과 잔류(0) 그룹의 평균값 차이가 큰 피처를 표로 정리해서 인사이트 뽑기
    grouped = df.groupby("Churn").mean(numeric_only=True).T
    grouped.columns = ["잔류(0) 평균", "이탈(1) 평균"]
    grouped["차이"] = grouped["이탈(1) 평균"] - grouped["잔류(0) 평균"]
    return grouped.sort_values("차이", key=abs, ascending=False)


def test2(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    __init__.py 에서 `from .eda import test2 as eda_test` 로 불러오는 함수.
    pages/analysis2.py 의 tab1(EDA)에서 st.dataframe(eda_test())로 호출됨.
    df를 안 넘기면 공통 데이터 로더(src/common/__init__.py)에서 직접 불러옴.
    """
    if df is None:
        from src.common import load_data
        df = load_data()
    return get_summary(df)