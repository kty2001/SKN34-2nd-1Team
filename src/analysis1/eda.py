# eda
from src.common import SEED, load_data # 데이터 불러오기 모듈

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 분석에 사용하는 Feature
CLUSTER_FEATURES = [
    "Lifetime",
    "Age",
    "Avg_class_frequency_current_month",
    "Contract_period"
]

# 분석 데이터 생성
def get_cluster_data(df):
    features = CLUSTER_FEATURES + ["Churn"]
    return df[features].copy()

# Feature 기본 통계
def get_descriptive_stats(df):
    return df[CLUSTER_FEATURES].describe()

# 결측치 확인
def get_missing_values(df):
    return df[CLUSTER_FEATURES + ["Churn"]].isna().sum()

# 중복 데이터 개수
def get_duplicate_count(df):
    return df.duplicated().sum()

# Feature 분포 그래프
def plot_feature_distribution(df):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.flatten()

    for idx, col in enumerate(CLUSTER_FEATURES):
        sns.histplot(
            df,
            x=col,
            bins=30,
            kde=True,
            ax=axes[idx]
        )

        axes[idx].set_title(f"{col} 분포")
        axes[idx].grid(alpha=0.3)

    plt.tight_layout()

    return fig

# Feature 상관관계
def plot_correlation(df):

    corr = df[CLUSTER_FEATURES].corr()

    short_names = {
        "Lifetime": "Life",
        "Age": "Age",
        "Avg_class_frequency_current_month": "AvgFreq",
        "Contract_period": "Contract"
    }

    corr = corr.rename(index=short_names, columns=short_names)

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)

    ax.set_title("변수간 상관관계")
    plt.tight_layout()

    return fig

def get_feature_name_mapping():
    return pd.DataFrame({
        "약어": ["Life", "Age", "AvgFreq", "Contract"],
        "원본 컬럼명": [
            "Lifetime",
            "Age",
            "Avg_class_frequency_current_month",
            "Contract_period"
        ],
        "설명": [
            "헬스장 이용 기간(개월)",
            "고객의 나이",
            "최근 월의 평균 방문 빈도",
            "고객의 계약 기간"
        ]
    })