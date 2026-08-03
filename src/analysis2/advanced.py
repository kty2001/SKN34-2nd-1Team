<<<<<<< HEAD
# 고도화
from src.common import SEED, load_data # 데이터 불러오기 모듈
=======
"""
src/analysis2/advanced.py
분석2 (결정트리 / 랜덤포레스트) - 모델 고도화

train.py에서 저장한 기본 모델(models/analysis2/*_base.pkl)과 동일한 학습/테스트셋을 이용해
GridSearchCV로 하이퍼파라미터를 튜닝하고, 최종 모델을 *_advanced.pkl 로 저장한다.
evaluate.py의 test3(stage="advanced")에서 이 결과를 불러와 base와 비교.
"""

import os
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.common import load_data
from src.analysis2.train import prepare_data

MODEL_DIR = "models/analysis2"

# 이 값을 쓴 이유: 결정트리는 과적합 방지(깊이 제한), 랜덤포레스트는 트리 개수/깊이 위주로 탐색
DT_PARAM_GRID = {
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_leaf": [1, 2, 4, 8],
}

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "max_features": ["sqrt", "log2"],
}


def tune_decision_tree(X_train, y_train):
    # 이 함수를 쓴 이유: 결정트리 기본 모델은 과적합되기 쉬워서 depth/leaf 조합을 그리드서치로 탐색
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        DT_PARAM_GRID,
        scoring="recall",  # 이탈자를 놓치지 않는 게 중요해서 recall 기준으로 최적화
        cv=5,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def tune_random_forest(X_train, y_train):
    # 이 함수를 쓴 이유: 랜덤포레스트는 트리 개수/깊이/피처 샘플링 조합이 성능에 큰 영향을 줘서 탐색
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        RF_PARAM_GRID,
        scoring="recall",
        cv=5,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def save_advanced_models(models: dict):
    # 이 함수를 쓴 이유: 튜닝된 최종 모델을 base와 구분되는 이름(_advanced.pkl)으로 저장
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}_advanced.pkl"))

>>>>>>> main

def test1():
    """
    직접 실행(python -m src.analysis2.advanced)하면
    튜닝 -> models/analysis2/*_advanced.pkl 저장까지 한번에 진행.
    반환값: 각 모델의 best_params (README/보고서에 그대로 기록하면 됨)
    """
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)

    dt_model, dt_params = tune_decision_tree(X_train, y_train)
    rf_model, rf_params = tune_random_forest(X_train, y_train)

    save_advanced_models({"decision_tree": dt_model, "random_forest": rf_model})

    return {
        "decision_tree_best_params": dt_params,
        "random_forest_best_params": rf_params,
    }


# 해당 페이지를 직접 실행 후 모델 저장
if __name__ == "__main__":
    print(test1())