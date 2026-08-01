# 고도화
from src.common import SEED, load_data  # 데이터 불러오기 모듈

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 고도화 Pipeline
def preprocess_pipeline_upgrade(n_cluster, seed=42):
    pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("kmeans", KMeans(n_clusters=n_cluster, random_state=seed, n_init=10))
    ])
    
    return pipeline

# 고도화 모델 학습
def train_upgrade_model(X_train, n_cluster=3, seed=42):
    pipeline = preprocess_pipeline_upgrade(n_cluster, seed)
    pipeline.fit(X_train)
    
    return pipeline

# 고도화 모델 평가
def evaluate_upgrade(pipeline, X_train, X_test):
    scaler = pipeline.named_steps["scaler"]

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_cluster = pipeline.predict(X_train)
    test_cluster = pipeline.predict(X_test)

    train_score = silhouette_score(X_train_scaled, train_cluster)
    test_score = silhouette_score(X_test_scaled, test_cluster)

    return {
        "train_score": train_score,
        "test_score": test_score,
        "train_cluster": train_cluster,
        "test_cluster": test_cluster
    }


# 고도화 전 / 후 Silhouette 비교
def compare_before_after(before_result, after_result):
    return pd.DataFrame({
        "Before": [before_result["train_score"], before_result["test_score"]],
        "After": [after_result["train_score"], after_result["test_score"]]
    }, index=["Train", "Test"])


# 고도화 전 / 후 시각화
def plot_before_after(comparison):
    fig, ax = plt.subplots(figsize=(6, 4))

    comparison.plot(kind="bar", ax=ax)

    ax.set_title("고도화 전 / 후 실루엣 점수 비교")
    ax.set_xlabel("데이터셋")
    ax.set_ylabel("실루엣 점수")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    plt.tight_layout()
    
    return fig