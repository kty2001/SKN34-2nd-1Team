# 평가
from src.common import SEED, load_data # 데이터 불러오기 모듈

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Train / Test 검증
def validate_model(pipeline, X_train, X_test):
    scaler = pipeline.named_steps["scaler"]

    # 스케일링
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 군집 예측
    train_cluster = pipeline.predict(X_train)
    test_cluster = pipeline.predict(X_test)

    # Silhouette Score
    train_score = silhouette_score(X_train_scaled, train_cluster)
    test_score = silhouette_score(X_test_scaled, test_cluster)

    # 군집별 데이터 개수
    train_count = (
        pd.Series(train_cluster)
        .value_counts()
        .sort_index()
    )

    test_count = (
        pd.Series(test_cluster)
        .value_counts()
        .sort_index()
    )

    return {
        "train_cluster": train_cluster,
        "test_cluster": test_cluster,
        "train_score": train_score,
        "test_score": test_score,
        "train_count": train_count,
        "test_count": test_count
    }

# Train / Test Silhouette 비교
def plot_validation(result):
    score_df = pd.DataFrame({
        "Train": [result["train_score"]],
        "Test": [result["test_score"]]
    })

    fig, ax = plt.subplots(figsize=(6, 3))

    score_df.plot(kind="bar", ax=ax)

    ax.set_title("Train / Test 실루엣 점수 비교")
    ax.set_ylabel("실루엣 점수")
    ax.set_xticklabels(["실루엣 점수"], rotation=0)
    plt.tight_layout()

    return fig


# 군집별 Feature 평균
def get_cluster_mean(train_result, cluster_features):
    return (train_result.groupby("Cluster")[cluster_features].mean())

# 군집별 Feature 평균 시각화
def plot_cluster_feature(cluster_mean):
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        cluster_mean.T,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        ax=ax
    )

    ax.set_title("군집별 특성 평균")
    plt.tight_layout()

    return fig

# Cluster별 Churn
def evaluate_cluster_churn(train_result, test_result):
    train = (train_result.groupby("Cluster")["Churn"].mean()
    )

    test = (test_result.groupby("Cluster")["Churn"].mean())

    return pd.DataFrame({"Train": train, "Test": test})

# Cluster별 Churn 시각화
def plot_cluster_churn(cluster_churn):
    fig, ax = plt.subplots(figsize=(8, 5))

    cluster_churn.plot(kind="bar", ax=ax)

    ax.set_title("클러스터별 이탈률")
    ax.set_xlabel("클러스터")
    ax.set_ylabel("이탈률" )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()

    return fig


# PCA 군집 시각화
def plot_cluster_pca(pipeline, X, cluster):
    scaler = pipeline.named_steps["scaler"]
    X_scaled = scaler.transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "Cluster": cluster})

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette="Set2",
        ax=ax
    )

    ax.set_title("PCA 군집 분포도")
    plt.tight_layout()

    return fig