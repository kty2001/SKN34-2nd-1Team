"""
K-Means 군집화 기반 회원 세그먼트 분석 (노트북 8절).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .config import RANDOM_STATE, SCORING
from .modeling import cross_validate_models


def evaluate_k_range(X_train_scaled, k_range=range(2, 11), random_state=RANDOM_STATE):
    inertias = []
    sil_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_train_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_train_scaled, labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(list(k_range), inertias, marker="o")
    axes[0].set_title("Elbow Method (Inertia)")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(list(k_range), sil_scores, marker="o", color="#DD8452")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette Score")

    plt.tight_layout()
    plt.show()

    return pd.Series(sil_scores, index=list(k_range), name="silhouette").round(4)


def fit_kmeans(X_train_scaled, X_test_scaled, k=4, random_state=RANDOM_STATE):
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    cluster_train = kmeans.fit_predict(X_train_scaled)
    cluster_test = kmeans.predict(X_test_scaled)
    return kmeans, cluster_train, cluster_test


def build_cluster_profile(X_train, y_train, cluster_train):
    profile = X_train.reset_index(drop=True).copy()
    profile["Cluster"] = cluster_train
    profile["Churn"] = y_train.reset_index(drop=True)

    cluster_profile = profile.groupby("Cluster").agg(
        회원수=("Churn", "size"),
        이탈률=("Churn", "mean"),
        평균나이=("Age", "mean"),
        평균Lifetime=("Lifetime", "mean"),
        평균계약기간=("Contract_period", "mean"),
        평균방문빈도_이번달=("Avg_class_frequency_current_month", "mean"),
        평균부가지출=("Avg_additional_charges_total", "mean"),
    ).round(2)

    return profile, cluster_profile


def plot_cluster_churn_rate(cluster_profile):
    plt.figure(figsize=(5, 4))
    sns.barplot(x=cluster_profile.index, y=cluster_profile["이탈률"], color="#4C72B0")
    plt.title("클러스터별 이탈률")
    plt.xlabel("Cluster")
    plt.ylabel("이탈률")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


def build_cluster_features(X_train_scaled, X_test_scaled, cluster_train, cluster_test, feature_names):
    cluster_dummies_train = pd.get_dummies(pd.Series(cluster_train, name="Cluster"), prefix="Cluster")
    cluster_dummies_test = pd.get_dummies(pd.Series(cluster_test, name="Cluster"), prefix="Cluster")
    cluster_dummies_test = cluster_dummies_test.reindex(columns=cluster_dummies_train.columns, fill_value=0)

    feature_names_cl = list(feature_names) + list(cluster_dummies_train.columns)
    X_train_cl = np.hstack([X_train_scaled, cluster_dummies_train.values.astype(float)])
    X_test_cl = np.hstack([X_test_scaled, cluster_dummies_test.values.astype(float)])

    print("클러스터 피처 결합 후 shape:", X_train_cl.shape, X_test_cl.shape)

    return X_train_cl, X_test_cl, feature_names_cl


def compare_cluster_feature_effect(models, X_train_cl, y_train, cv, cv_results_df, scoring=SCORING):
    cv_results_cl_df = cross_validate_models(models, X_train_cl, y_train, cv, scoring=scoring)

    comparison = pd.concat(
        {"Baseline": cv_results_df, "Cluster 피처 추가": cv_results_cl_df}, axis=1
    )

    return comparison.round(4)
