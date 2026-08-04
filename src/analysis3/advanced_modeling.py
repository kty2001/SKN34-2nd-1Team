"""
고도화(Advanced Modeling): XGBoost 추가, SMOTE, 하이퍼파라미터/임계값 튜닝 (노트북 9절).
"""
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve,
)

from .config import RANDOM_STATE, SCORING


def add_xgboost(models, random_state=RANDOM_STATE):
    models_v2 = dict(models)
    models_v2["XGBoost"] = XGBClassifier(
        n_estimators=300, random_state=random_state, eval_metric="logloss"
    )
    return models_v2


def cross_validate_with_smote(models_v2, X_train_scaled, y_train, cv, cv_results_v2_df,
                               random_state=RANDOM_STATE, scoring=SCORING):
    smote_results = {}
    for name, model in models_v2.items():
        pipe = ImbPipeline([
            ("smote", SMOTE(random_state=random_state)),
            ("clf", model),
        ])
        scores = {
            metric: cross_val_score(pipe, X_train_scaled, y_train, cv=cv, scoring=metric).mean()
            for metric in scoring
        }
        smote_results[name] = scores

    smote_results_df = pd.DataFrame(smote_results).T

    smote_comparison = pd.concat(
        {"기존": cv_results_v2_df, "SMOTE 적용": smote_results_df}, axis=1
    )

    return smote_comparison.round(4)


def tune_xgboost(X_train_scaled, y_train, cv, random_state=RANDOM_STATE):
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.3],
    }

    xgb_search = RandomizedSearchCV(
        XGBClassifier(random_state=random_state, eval_metric="logloss"),
        param_distributions=param_dist,
        n_iter=60,
        scoring="f1",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
    )
    xgb_search.fit(X_train_scaled, y_train)

    print("최적 파라미터:", xgb_search.best_params_)
    print("최적 CV F1:", round(xgb_search.best_score_, 4))

    return xgb_search


def evaluate_final_model(xgb_search, X_test_scaled, y_test, baseline_scores):
    final_model = xgb_search.best_estimator_

    y_pred_final = final_model.predict(X_test_scaled)
    y_proba_final = final_model.predict_proba(X_test_scaled)[:, 1]

    final_scores = {
        "Accuracy": accuracy_score(y_test, y_pred_final),
        "Precision": precision_score(y_test, y_pred_final),
        "Recall": recall_score(y_test, y_pred_final),
        "F1": f1_score(y_test, y_pred_final),
        "ROC-AUC": roc_auc_score(y_test, y_proba_final),
    }

    final_comparison = pd.DataFrame({
        "Baseline (섹션 5, Gradient Boosting)": baseline_scores,
        "튜닝된 XGBoost (섹션 9)": final_scores,
    })

    return final_model, y_pred_final, y_proba_final, final_scores, final_comparison.round(4)


def find_best_threshold(final_model, X_train_scaled, y_train, cv):
    oof_proba = cross_val_predict(final_model, X_train_scaled, y_train, cv=cv, method="predict_proba")[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_train, oof_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = f1s[:-1].argmax()
    best_threshold = thresholds[best_idx]

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, precisions[:-1], label="Precision")
    plt.plot(thresholds, recalls[:-1], label="Recall")
    plt.plot(thresholds, f1s[:-1], label="F1")
    plt.axvline(best_threshold, color="gray", linestyle="--", label=f"Best threshold={best_threshold:.2f}")
    plt.axvline(0.5, color="black", linestyle=":", label="Default=0.50")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.title("임계값에 따른 Precision/Recall/F1 (학습 데이터 OOF 기준)")
    plt.tight_layout()
    plt.show()

    print(f"F1 최적 임계값: {best_threshold:.3f} (OOF F1={f1s[best_idx]:.4f}, "
          f"Precision={precisions[best_idx]:.4f}, Recall={recalls[best_idx]:.4f})")

    return best_threshold


def evaluate_threshold(y_test, y_proba_final, best_threshold, final_scores):
    y_pred_thresh = (y_proba_final >= best_threshold).astype(int)

    threshold_scores = {
        "Accuracy": accuracy_score(y_test, y_pred_thresh),
        "Precision": precision_score(y_test, y_pred_thresh),
        "Recall": recall_score(y_test, y_pred_thresh),
        "F1": f1_score(y_test, y_pred_thresh),
        "ROC-AUC": final_scores["ROC-AUC"],
    }

    return pd.DataFrame({
        "임계값 0.5": final_scores,
        f"임계값 {best_threshold:.2f}": threshold_scores,
    }).round(4)


def plot_final_feature_importance(final_model, feature_names):
    final_importance = pd.Series(
        final_model.feature_importances_, index=feature_names
    ).sort_values(ascending=True)

    plt.figure(figsize=(8, 6))
    final_importance.plot(kind="barh", color="#4C72B0")
    plt.title("튜닝된 XGBoost - 피처 중요도")
    plt.tight_layout()
    plt.show()

    return final_importance
