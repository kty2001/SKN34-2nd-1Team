"""
이탈 방지 시뮬레이션(Counterfactual 분석) (노트북 11절).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_FEATURES_TO_ADJUST = ["Contract_period", "Lifetime", "Avg_class_frequency_total", "Month_to_end_contract"]


def compute_cluster_targets(profile, features=DEFAULT_FEATURES_TO_ADJUST):
    cluster_targets = profile.groupby("Cluster")[features + ["Churn"]].mean().rename(columns={"Churn": "이탈률"})

    low_churn_cluster = cluster_targets["이탈률"].idxmin()
    target_values = cluster_targets.loc[low_churn_cluster]

    print(f"기준(저이탈) 세그먼트: Cluster {low_churn_cluster} (이탈률 {target_values['이탈률']:.1%})")

    return cluster_targets.round(3), low_churn_cluster, target_values


def compute_cluster_targets_full(profile, features=None):
    features = features or [
        "Contract_period", "Lifetime", "Avg_class_frequency_total",
        "Avg_class_frequency_current_month", "Month_to_end_contract",
    ]
    cluster_targets_full = profile.groupby("Cluster")[features + ["Churn"]].mean()
    return cluster_targets_full.round(3)


def compute_baseline_churned_probability(X_test, y_test, final_model, scaler):
    X_test_churned_raw = X_test[y_test == 1].reset_index(drop=True)
    proba_before = final_model.predict_proba(scaler.transform(X_test_churned_raw))[:, 1]

    print(f"테스트 세트 내 실제 이탈 회원 수: {len(X_test_churned_raw)}")
    print(f"조정 전 평균 예측 이탈확률: {proba_before.mean():.4f}")

    return X_test_churned_raw, proba_before


def simulate_feature_adjustments(X_test_churned_raw, proba_before, final_model, scaler,
                                  target_values, features_to_adjust=DEFAULT_FEATURES_TO_ADJUST):
    sim_records = []
    for feat in features_to_adjust:
        X_cf = X_test_churned_raw.copy()
        target = target_values[feat]
        X_cf[feat] = np.maximum(X_cf[feat], target)
        proba_after = final_model.predict_proba(scaler.transform(X_cf))[:, 1]

        sim_records.append({
            "피처": feat,
            "기준값(Cluster 평균)": target,
            "조정 전 평균 이탈확률": proba_before.mean(),
            "조정 후 평균 이탈확률": proba_after.mean(),
            "평균 감소폭(%p)": (proba_before.mean() - proba_after.mean()) * 100,
            "확률이 감소한 회원 비율": (proba_after < proba_before).mean(),
        })

    sim_df = pd.DataFrame(sim_records).set_index("피처")
    return sim_df.sort_values("평균 감소폭(%p)", ascending=False).round(4)


def plot_simulation_results(sim_df, low_churn_cluster):
    plt.figure(figsize=(8, 5))
    order = sim_df.sort_values("평균 감소폭(%p)", ascending=False).index
    x = np.arange(len(order))
    width = 0.35
    plt.bar(x - width / 2, sim_df.loc[order, "조정 전 평균 이탈확률"], width, label="조정 전", color="#DD8452")
    plt.bar(x + width / 2, sim_df.loc[order, "조정 후 평균 이탈확률"], width,
            label=f"조정 후(Cluster {low_churn_cluster} 수준)", color="#4C72B0")
    plt.xticks(x, order, rotation=15)
    plt.ylabel("평균 예측 이탈확률")
    plt.title("피처별 상향 조정 시뮬레이션: 조정 전후 평균 이탈확률")
    plt.legend()
    plt.tight_layout()
    plt.show()


def simulate_all_features_combined(X_test_churned_raw, proba_before, final_model, scaler,
                                    target_values, features_to_adjust=DEFAULT_FEATURES_TO_ADJUST):
    X_cf_all = X_test_churned_raw.copy()
    for feat in features_to_adjust:
        X_cf_all[feat] = np.maximum(X_cf_all[feat], target_values[feat])
    proba_after_all = final_model.predict_proba(scaler.transform(X_cf_all))[:, 1]

    print(f"4개 피처 동시 조정 후 평균 예측 이탈확률: {proba_after_all.mean():.4f} "
          f"(조정 전 {proba_before.mean():.4f} 대비 {(proba_before.mean() - proba_after_all.mean()) * 100:.2f}%p 감소)")

    return proba_after_all


def simulate_frequency_adjustment_both(X_test_churned_raw, proba_before, final_model, scaler,
                                        target_values, cluster_targets_full, low_churn_cluster):
    current_month_target = cluster_targets_full.loc[low_churn_cluster, "Avg_class_frequency_current_month"]

    X_cf_freq_both = X_test_churned_raw.copy()
    X_cf_freq_both["Avg_class_frequency_total"] = np.maximum(
        X_cf_freq_both["Avg_class_frequency_total"], target_values["Avg_class_frequency_total"]
    )
    X_cf_freq_both["Avg_class_frequency_current_month"] = np.maximum(
        X_cf_freq_both["Avg_class_frequency_current_month"], current_month_target
    )
    proba_after_freq_both = final_model.predict_proba(scaler.transform(X_cf_freq_both))[:, 1]

    print(f"방문 빈도 2개 피처(전체 평균 + 이번 달) 동시 조정 후 평균 예측 이탈확률: {proba_after_freq_both.mean():.4f} "
          f"(조정 전 {proba_before.mean():.4f} 대비 {(proba_before.mean() - proba_after_freq_both.mean()) * 100:.2f}%p 변화)")

    return proba_after_freq_both
