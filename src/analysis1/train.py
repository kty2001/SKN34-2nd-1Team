# 학습
from src.common import SEED, load_data # 데이터 불러오기 모듈
import joblib
import pandas as pd

MODEL_PATH = "models/analysis1/churn_cluster_model.joblib"

# 저장된 모델 불러오기
def load_model():
    pipeline = joblib.load(MODEL_PATH)

    return pipeline

# 군집 예측
def predict_cluster(pipeline, X):
    cluster = pipeline.predict(X)

    return cluster

# 예측 결과 생성
def make_cluster_result(X, y, cluster):
    result = X.copy()

    result["Cluster"] = cluster
    result["Churn"] = y.values

    return result