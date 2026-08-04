<<<<<<< HEAD
<<<<<<< HEAD
"""
Streamlit 앱에서 사용할 모델/스케일러/메타데이터를 models/ 폴더에 저장한다.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from src.common import load_data

RANDOM_STATE = 42
MODEL_DIR = Path("models/analysis3")
MODEL_DIR.mkdir(exist_ok=True)

df = load_data()
X = df.drop(columns="Churn")
y = df["Churn"]
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

param_dist = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.3],
}

print("XGBoost 하이퍼파라미터 탐색 중...")
xgb_search = RandomizedSearchCV(
    XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
    param_distributions=param_dist,
    n_iter=60,
    scoring="f1",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
xgb_search.fit(X_train_scaled, y_train)
print("최적 파라미터:", xgb_search.best_params_)
print("최적 CV F1:", round(xgb_search.best_score_, 4))

final_model = xgb_search.best_estimator_

y_pred = final_model.predict(X_test_scaled)
y_proba = final_model.predict_proba(X_test_scaled)[:, 1]

test_scores = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
}
print("테스트 성능:", {k: round(v, 4) for k, v in test_scores.items()})

# 학습 데이터 OOF 예측으로 F1 최적 임계값 산출
oof_proba = cross_val_predict(final_model, X_train_scaled, y_train, cv=cv, method="predict_proba")[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_train, oof_proba)
f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
best_idx = f1s[:-1].argmax()
best_threshold = float(thresholds[best_idx])
print(f"F1 최적 임계값: {best_threshold:.3f}")

importance = pd.Series(final_model.feature_importances_, index=feature_names).sort_values(ascending=False)

# 저이탈(low-churn) 회원 프로필 추출: 이탈률이 가장 낮은 클러스터의 실제 이용 패턴 평균값을
# "개선 목표치"로 사용해, 개별 회원에게 어떤 항목을 얼마나 올리면 좋을지 제안하는 데 활용한다.
from sklearn.cluster import KMeans

K = 4
kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
cluster_train = kmeans.fit_predict(X_train_scaled)

profile = X_train.reset_index(drop=True).copy()
profile["Cluster"] = cluster_train
profile["Churn"] = y_train.reset_index(drop=True)

actionable_features = [
    "Contract_period", "Month_to_end_contract", "Lifetime",
    "Avg_class_frequency_total", "Avg_class_frequency_current_month", "Group_visits",
]
cluster_churn = profile.groupby("Cluster")["Churn"].mean()
low_churn_cluster = int(cluster_churn.idxmin())
cluster_targets = profile.groupby("Cluster")[actionable_features].mean().loc[low_churn_cluster]
print(f"\n저이탈 클러스터: {low_churn_cluster} (이탈률 {cluster_churn.loc[low_churn_cluster]:.1%})")
print("개선 목표치:", cluster_targets.round(2).to_dict())

joblib.dump(final_model, MODEL_DIR / "churn_model.pkl")
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

metadata = {
    "feature_names": feature_names,
    "best_threshold": best_threshold,
    "test_scores": test_scores,
    "feature_importance": importance.round(4).to_dict(),
    "feature_stats": {
        col: {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "mean": float(X[col].mean()),
        }
        for col in feature_names
    },
    "actionable_features": actionable_features,
    "low_churn_cluster": low_churn_cluster,
    "low_churn_rate": float(cluster_churn.loc[low_churn_cluster]),
    "cluster_targets": cluster_targets.round(4).to_dict(),
}
with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"\n저장 완료 -> {MODEL_DIR}/churn_model.pkl, scaler.pkl, metadata.json")
=======
# 학습
from src.common import load_data # 데이터 불러오기 모듈
=======
# 학습
from src.common import SEED, load_data # 데이터 불러오기 모듈
>>>>>>> upstream/develop

# 데이터 정제 및 분리 후
# 해당 모듈에서 모델 저장 후 평가 및 고도화 시 모델 블러와서 사용 
# 저장 경로 models\analysis3
# 예시
def test4():
    df = load_data()
    return df.shape

# 해당 페이지를 직접 실행 후 모델 저장
if __name__ == "__main__":
<<<<<<< HEAD
    test4()
>>>>>>> 216054d562f7b743ba18e71e1f7c19e3194538b7
=======
    test4()
>>>>>>> upstream/develop
