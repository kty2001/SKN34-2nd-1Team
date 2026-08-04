<<<<<<< HEAD
<<<<<<< HEAD
"""
gym_churn_prediction.ipynb의 분석 흐름을 기능별 모듈로 나눈 패키지.

- config: 전역 설정(랜덤 시드, 스코어링, 플롯 스타일)
- data_loading: 데이터 로드 및 기본 확인
- eda: 탐색적 데이터 분석
- preprocessing: 데이터 분리 및 스케일링
- modeling: 기본 분류 모델 학습/비교/평가
- clustering: K-Means 군집화 기반 회원 세그먼트 분석
- advanced_modeling: XGBoost 추가, SMOTE, 하이퍼파라미터/임계값 튜닝
- simulation: 이탈 방지 시뮬레이션(Counterfactual 분석)
"""
from .config import set_style, RANDOM_STATE, SCORING, NUM_FEATURES, BINARY_FEATURES, DATA_PATH
from .data_loading import load_data, basic_info
from .eda import (
    plot_churn_distribution,
    plot_correlation_heatmap,
    plot_numeric_distributions,
    plot_binary_churn_rates,
)
from .preprocessing import split_and_scale
from .modeling import (
    get_cv,
    get_baseline_models,
    cross_validate_models,
    plot_cv_results,
    select_best_model,
    fit_and_evaluate,
    plot_confusion_and_roc,
    plot_feature_importance,
)
from .clustering import (
    evaluate_k_range,
    fit_kmeans,
    build_cluster_profile,
    plot_cluster_churn_rate,
    build_cluster_features,
    compare_cluster_feature_effect,
)
from .advanced_modeling import (
    add_xgboost,
    cross_validate_with_smote,
    tune_xgboost,
    evaluate_final_model,
    find_best_threshold,
    evaluate_threshold,
    plot_final_feature_importance,
)
from .simulation import (
    compute_cluster_targets,
    compute_cluster_targets_full,
    compute_baseline_churned_probability,
    simulate_feature_adjustments,
    plot_simulation_results,
    simulate_all_features_combined,
    simulate_frequency_adjustment_both,
)

__all__ = [
    "set_style", "RANDOM_STATE", "SCORING", "NUM_FEATURES", "BINARY_FEATURES", "DATA_PATH",
    "load_data", "basic_info",
    "plot_churn_distribution", "plot_correlation_heatmap", "plot_numeric_distributions", "plot_binary_churn_rates",
    "split_and_scale",
    "get_cv", "get_baseline_models", "cross_validate_models", "plot_cv_results",
    "select_best_model", "fit_and_evaluate", "plot_confusion_and_roc", "plot_feature_importance",
    "evaluate_k_range", "fit_kmeans", "build_cluster_profile", "plot_cluster_churn_rate",
    "build_cluster_features", "compare_cluster_feature_effect",
    "add_xgboost", "cross_validate_with_smote", "tune_xgboost", "evaluate_final_model",
    "find_best_threshold", "evaluate_threshold", "plot_final_feature_importance",
    "compute_cluster_targets", "compute_cluster_targets_full", "compute_baseline_churned_probability",
    "simulate_feature_adjustments", "plot_simulation_results",
    "simulate_all_features_combined", "simulate_frequency_adjustment_both",
    "run_all",
]


def run_all(data_path=DATA_PATH):
    """노트북 전체 분석 파이프라인을 처음부터 끝까지 순서대로 실행한다."""
    set_style()

    df = load_data(data_path)
    basic_info(df)

    plot_churn_distribution(df)
    plot_correlation_heatmap(df)
    plot_numeric_distributions(df)
    plot_binary_churn_rates(df)

    split = split_and_scale(df)
    X, y = split["X"], split["y"]
    X_train, X_test = split["X_train"], split["X_test"]
    y_train, y_test = split["y_train"], split["y_test"]
    X_train_scaled, X_test_scaled = split["X_train_scaled"], split["X_test_scaled"]
    scaler = split["scaler"]

    cv = get_cv()
    models = get_baseline_models()
    cv_results_df = cross_validate_models(models, X_train_scaled, y_train, cv)
    plot_cv_results(cv_results_df)

    best_model_name, best_model = select_best_model(models, cv_results_df)
    y_pred, y_proba, test_scores = fit_and_evaluate(best_model, X_train_scaled, y_train, X_test_scaled, y_test)
    plot_confusion_and_roc(y_test, y_pred, y_proba, best_model_name)
    plot_feature_importance(best_model, X.columns, best_model_name)

    evaluate_k_range(X_train_scaled)
    kmeans, cluster_train, cluster_test = fit_kmeans(X_train_scaled, X_test_scaled)
    profile, cluster_profile = build_cluster_profile(X_train, y_train, cluster_train)
    plot_cluster_churn_rate(cluster_profile)

    X_train_cl, X_test_cl, feature_names_cl = build_cluster_features(
        X_train_scaled, X_test_scaled, cluster_train, cluster_test, X.columns
    )
    compare_cluster_feature_effect(models, X_train_cl, y_train, cv, cv_results_df)

    models_v2 = add_xgboost(models)
    cv_results_v2_df = cross_validate_models(models_v2, X_train_scaled, y_train, cv)
    cross_validate_with_smote(models_v2, X_train_scaled, y_train, cv, cv_results_v2_df)

    xgb_search = tune_xgboost(X_train_scaled, y_train, cv)
    final_model, y_pred_final, y_proba_final, final_scores, final_comparison = evaluate_final_model(
        xgb_search, X_test_scaled, y_test, test_scores
    )
    plot_confusion_and_roc(y_test, y_pred_final, y_proba_final, "Tuned XGBoost",
                            cm_title="Confusion Matrix (튜닝된 XGBoost)")

    best_threshold = find_best_threshold(final_model, X_train_scaled, y_train, cv)
    evaluate_threshold(y_test, y_proba_final, best_threshold, final_scores)
    plot_final_feature_importance(final_model, X.columns)

    cluster_targets, low_churn_cluster, target_values = compute_cluster_targets(profile)
    cluster_targets_full = compute_cluster_targets_full(profile)

    X_test_churned_raw, proba_before = compute_baseline_churned_probability(X_test, y_test, final_model, scaler)
    sim_df = simulate_feature_adjustments(X_test_churned_raw, proba_before, final_model, scaler, target_values)
    plot_simulation_results(sim_df, low_churn_cluster)
    simulate_all_features_combined(X_test_churned_raw, proba_before, final_model, scaler, target_values)
    simulate_frequency_adjustment_both(
        X_test_churned_raw, proba_before, final_model, scaler,
        target_values, cluster_targets_full, low_churn_cluster,
    )

    return {
        "df": df,
        "final_model": final_model,
        "scaler": scaler,
        "final_scores": final_scores,
        "cluster_profile": cluster_profile,
        "sim_df": sim_df,
    }
=======
=======
>>>>>>> upstream/develop
# 예시 추후 수정
from .eda import test2 as eda_test
from .evaluate import test3 as evaluate_test

# 사용법
"""
from src.analysis1 import eda_test, evaluate_test
<<<<<<< HEAD
"""
>>>>>>> 216054d562f7b743ba18e71e1f7c19e3194538b7
=======
"""
>>>>>>> upstream/develop
