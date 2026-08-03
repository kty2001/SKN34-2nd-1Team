"""
src/analysis2/eda.py
분석2 (결정트리 / 랜덤포레스트) - 탐색적 데이터 분석(EDA)

gym_churn_us.csv 컬럼:
gender, Near_Location, Partner, Promo_friends, Phone, Contract_period,
Group_visits, Age, Avg_additional_charges_total, Month_to_end_contract,
Lifetime, Avg_class_frequency_total, Avg_class_frequency_current_month, Churn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def get_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.describe().T
    summary["missing"] = df.isnull().sum()
    summary["dtype"] = df.dtypes
    return summary


def get_churn_rate(df: pd.DataFrame) -> float:
    return df["Churn"].mean()


def plot_churn_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 4))
    df["Churn"].value_counts().sort_index().plot(
        kind="bar", ax=ax, color=["#4C72B0", "#DD8452"]
    )
    ax.set_xticklabels(["잔류(0)", "이탈(1)"], rotation=0)
    ax.set_title("이탈 여부 분포")
    ax.set_ylabel("회원 수")
    return fig


def plot_feature_by_churn(df: pd.DataFrame, feature: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="Churn", y=feature, ax=ax)
    ax.set_xticklabels(["잔류(0)", "이탈(1)"])
    ax.set_title(f"{feature} - 이탈 여부별 분포")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("피처 간 상관관계")
    return fig


def get_key_features_by_churn(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("Churn").mean(numeric_only=True).T
    grouped.columns = ["잔류(0) 평균", "이탈(1) 평균"]
    grouped["차이"] = grouped["이탈(1) 평균"] - grouped["잔류(0) 평균"]
    return grouped.sort_values("차이", key=abs, ascending=False)


def test2(df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        from src.common import load_data
        df = load_data()
    return get_summary(df)