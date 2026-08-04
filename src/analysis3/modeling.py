"""
기본 분류 모델 학습/비교/평가 및 피처 중요도 분석 (노트북 4~6절).
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay, classification_report,
)

from .config import RANDOM_STATE, SCORING


def get_cv(n_splits=5, random_state=RANDOM_STATE):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def get_baseline_models(random_state=RANDOM_STATE):
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    }


def cross_validate_models(models, X_train_scaled, y_train, cv, scoring=SCORING):
    cv_results = {}
    for name, model in models.items():
        scores = {
            metric: cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring=metric).mean()
            for metric in scoring
        }
        cv_results[name] = scores

    return pd.DataFrame(cv_results).T


def plot_cv_results(cv_results_df, title="모델별 5-fold 교차검증 평균 성능"):
    cv_results_df.plot(kind="bar", figsize=(11, 5), rot=0)
    plt.title(title)
    plt.ylabel("score")
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def select_best_model(models, cv_results_df, metric="f1"):
    best_model_name = cv_results_df[metric].idxmax()
    print("선택된 모델:", best_model_name)
    return best_model_name, models[best_model_name]


def fit_and_evaluate(model, X_train_scaled, y_train, X_test_scaled, y_test):
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    test_scores = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }

    return y_pred, y_proba, test_scores


def plot_confusion_and_roc(y_test, y_pred, y_proba, model_name, cm_title="Confusion Matrix"):
    print(classification_report(y_test, y_pred, target_names=["잔존(0)", "이탈(1)"]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["잔존", "이탈"], cmap="Blues", ax=axes[0], colorbar=False
    )
    axes[0].set_title(cm_title)

    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1], name=model_name)
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    axes[1].set_title("ROC Curve")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=feature_names)
        title = f"{model_name} - 피처 중요도"
    else:
        importance = pd.Series(model.coef_[0], index=feature_names).abs()
        title = f"{model_name} - 계수 절댓값(중요도)"

    importance = importance.sort_values(ascending=True)

    plt.figure(figsize=(8, 6))
    importance.plot(kind="barh", color="#4C72B0")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return importance
