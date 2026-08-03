<<<<<<< HEAD
# 평가
from src.common import SEED, load_data # 데이터 불러오기 모듈
=======
"""
src/analysis2/evaluate.py
분석2 (결정트리 / 랜덤포레스트) - 모델 평가
>>>>>>> main

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
plt.rcParams["font.family"] = "Malgun Gothic"  # Windows 한글 폰트
plt.rcParams["axes.unicode_minus"] = False      # 마이너스 기호 깨짐 방지

MODEL_DIR = "models/analysis2"
MODEL_NAMES = ["decision_tree", "random_forest"]


def load_test_data():
    # 이 함수를 쓴 이유: train.py에서 저장해둔 테스트셋을 그대로 불러와서
    # 평가할 때마다 다시 분할할 필요 없이 동일한 기준으로 비교하기 위해
    return joblib.load(os.path.join(MODEL_DIR, "test_data.pkl"))


def load_model(name: str, stage: str = "base"):
    # 이 함수를 쓴 이유: stage(base/advanced)에 맞는 저장된 모델 파일을 불러오기 위해
    path = os.path.join(MODEL_DIR, f"{name}_{stage}.pkl")
    return joblib.load(path)


def get_metrics(model, X_test, y_test) -> dict:
    # 이 함수를 쓴 이유: 이탈 예측은 Recall(이탈자를 놓치지 않는 정도)이 특히 중요해서
    # 5개 지표를 한번에 계산해서 비교
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
    # 이 함수를 쓴 이유: 결정트리 vs 랜덤포레스트 성능을 표 하나로 한눈에 비교하기 위해
    X_test, y_test = load_test_data()
    rows = {}
    for name in MODEL_NAMES:
        model = load_model(name, stage=stage)
        rows[name] = get_metrics(model, X_test, y_test)
    return pd.DataFrame(rows).T.round(4)


def plot_confusion_matrix(name: str, stage: str = "base"):
    # 이 함수를 쓴 이유: 어떤 유형의 오분류(이탈자를 놓침 vs 잔류자를 이탈로 잘못 예측)가
    # 더 많은지 시각적으로 확인하기 위해
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
    # 이 함수를 쓴 이유: 이탈에 가장 큰 영향을 주는 피처가 뭔지 비즈니스 인사이트로 뽑기 위해
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
    """
    __init__.py 에서 `from .evaluate import test3 as evaluate_test` 로 불러오는 함수.
    pages/analysis2.py 의 tab2(학습/추론 평가), tab3(고도화 전/후 평가)에서 호출됨.
    tab2 -> evaluate_test() (stage="base"), tab3 -> evaluate_test(stage="advanced")
    """
    return get_metrics_table(stage=stage)


if __name__ == "__main__":
    print(test3(stage="base"))