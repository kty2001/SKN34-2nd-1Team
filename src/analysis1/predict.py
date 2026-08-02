import joblib
import pandas as pd

MODEL_PATH = "models/analysis1/churn_cluster_model.joblib"

PREDICT_FEATURES = [
    "Lifetime",
    "Age",
    "Avg_class_frequency_current_month",
    "Contract_period"
]

def load_predict_model():
    return joblib.load(MODEL_PATH)

def predict_sample(lifetime, age, avg_frequency, contract_period):
    model = load_predict_model()
    sample = pd.DataFrame([{
        "Lifetime": lifetime,
        "Age": age,
        "Avg_class_frequency_current_month": avg_frequency,
        "Contract_period": contract_period
    }])
    cluster = model.predict(sample)[0]
    return cluster