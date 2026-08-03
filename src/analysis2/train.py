<<<<<<< HEAD
# 학습
from src.common import SEED, load_data # 데이터 불러오기 모듈
=======
"""
src/analysis2/train.py
분석2 (결정트리 / 랜덤포레스트) - 모델 학습 및 저장

데이터 정제 및 분리 후, 결정트리/랜덤포레스트 기본(튜닝 전) 모델을 학습하고
models/analysis2/ 에 저장한다.
evaluate.py(평가), advanced.py(고도화)에서 여기서 저장한 모델을 불러와서 사용.
"""

import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.common import load_data  # 데이터 불러오기 모듈

MODEL_DIR = "models/analysis2"


def prepare_data(df):
    # 이 함수를 쓴 이유: 전체 컬럼이 이미 숫자형(gender, Near_Location 등도 0/1)이라
    # 별도 인코딩 없이 X, y만 분리하면 됨
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    # 이 함수를 쓴 이유: 결정트리, 랜덤포레스트 기본(튜닝 전) 모델 두 개를 각각 학습
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    return {"decision_tree": dt, "random_forest": rf}


def save_models(models, X_test, y_test):
    # 이 함수를 쓴 이유: 학습된 모델 + 테스트셋을 저장해서
    # evaluate.py / advanced.py 에서 다시 학습할 필요 없이 그대로 불러다 쓰기 위해
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}_base.pkl"))
    joblib.dump((X_test, y_test), os.path.join(MODEL_DIR, "test_data.pkl"))

>>>>>>> main

def test4():
    """
    직접 실행(python -m src.analysis2.train)하면 학습 -> 저장까지 한번에 진행.
    반환값은 간단한 확인용 (기본 정확도) — 실제 평가는 evaluate.py(test3)에서 상세히 진행.
    """
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    models = train_models(X_train, y_train)
    save_models(models, X_test, y_test)
    return {name: round(model.score(X_test, y_test), 4) for name, model in models.items()}


# 해당 페이지를 직접 실행 후 모델 저장
if __name__ == "__main__":
    print(test4())