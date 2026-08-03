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
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        DT_PARAM_GRID,
        scoring="recall",
        cv=5,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def tune_random_forest(X_train, y_train):
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
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}_advanced.pkl"))


def test1():
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)

    dt_model, dt_params = tune_decision_tree(X_train, y_train)
    rf_model, rf_params = tune_random_forest(X_train, y_train)

    save_advanced_models({"decision_tree": dt_model, "random_forest": rf_model})

    return {
        "decision_tree_best_params": dt_params,
        "random_forest_best_params": rf_params,
    }


if __name__ == "__main__":
    print(test1())