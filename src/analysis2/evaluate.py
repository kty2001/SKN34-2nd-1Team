"""
src/analysis2/evaluate.py
분석2 (결정트리 / 랜덤포레스트) - 모델 평가

train.py에서 저장한 모델(models/analysis2/*_base.pkl)과 테스트셋(test_data.pkl)을
불러와서 Accuracy/Precision/Recall/F1/ROC-AUC 계산 + Confusion Matrix 시각화.

stage 파라미터로 base(고도화 전) / advanced(고도화 후) 결과를 구분해서 불러올 수 있음.
- 기본 모델: models/analysis2/decision_tree_base.pkl, random_forest_base.pkl
- 고도화 모델: models/analysis2/decision_tree_advanced.pkl, random_forest_advanced.pkl (advanced.py에서 저장)
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

MODEL_DIR = "models/analysis2"
MODEL_NAMES = ["decision_tree", "random_forest"]


def load_test_data():
    return joblib.load(os.path.join(MODEL_DIR, "test_data.pkl"))


def load_model(name: str, stage: str = "base"):
    path = os.path.join(MODEL_DIR, f"{name}_{stage}.pkl")
    return joblib.load(path)


def get_metrics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }


def get_metrics_table(stage: str = "base") -> pd.DataFrame:
    X_test, y_test = load_test_data()
    rows = {}
    for name in MODEL_NAMES:
        model = load_model(name, stage=stage)
        rows[name] = get_metrics(model, X_test, y_test)
    return pd.DataFrame(rows).T.round(4)


def plot_confusion_matrix(name: str, stage: str = "base"):
    X_test, y_test = load_test_data()
    model = load_model(name, stage=stage)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["잔류(0)", "이탈(1)"])
    ax.set_yticklabels(["잔류(0)", "이탈(1)"])
    ax.set_xlabel("예측")
    ax.set_ylabel("실제")
    ax.set_title(f"{name} ({stage}) Confusion Matrix")
    return fig


def plot_feature_importance(name: str, stage: str = "base", top_n: int = 10):
    X_test, _ = load_test_data()
    model = load_model(name, stage=stage)

    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(importance_df["feature"], importance_df["importance"], color="skyblue")
    ax.invert_yaxis()
    ax.set_xlabel("중요도")
    ax.set_title(f"{name} ({stage}) Feature Importance")
    return fig


def test3(stage: str = "base"):
    return get_metrics_table(stage=stage)


if __name__ == "__main__":
    print(test3(stage="base"))