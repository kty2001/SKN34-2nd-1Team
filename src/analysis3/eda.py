"""
탐색적 데이터 분석 (노트북 2절): 이탈 분포, 상관관계, 수치형/이진형 변수 분석.
"""
import matplotlib.pyplot as plt
import seaborn as sns

from .config import NUM_FEATURES, BINARY_FEATURES, PALETTE


def plot_churn_distribution(df):
    churn_rate = df["Churn"].value_counts(normalize=True)
    print(churn_rate)

    plt.figure(figsize=(4, 4))
    sns.countplot(x="Churn", data=df, palette=PALETTE)
    plt.title(f"이탈 여부 분포 (이탈률 {churn_rate[1]:.1%})")
    plt.xlabel("Churn (0=잔존, 1=이탈)")
    plt.show()

    return churn_rate


def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
    plt.title("피처 간 상관관계")
    plt.tight_layout()
    plt.show()

    print("Churn과의 상관계수 (절대값 기준 정렬):")
    print(corr["Churn"].drop("Churn").abs().sort_values(ascending=False))

    return corr


def plot_numeric_distributions(df, num_features=NUM_FEATURES):
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for ax, col in zip(axes.flat, num_features):
        sns.kdeplot(data=df, x=col, hue="Churn", fill=True, common_norm=False,
                    alpha=0.4, ax=ax, palette=PALETTE)
        ax.set_title(col)
    for ax in axes.flat[len(num_features):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_binary_churn_rates(df, binary_features=BINARY_FEATURES):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.flat, binary_features):
        rate = df.groupby(col)["Churn"].mean()
        sns.barplot(x=rate.index, y=rate.values, ax=ax, color="#4C72B0")
        ax.set_title(col)
        ax.set_ylabel("이탈률")
        ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()
