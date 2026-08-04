"""
gym_churn_prediction.ipynb 분석 전반에서 공유되는 전역 설정(랜덤 시드, 스코어링 지표, 플롯 스타일).
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
DATA_PATH = "data/gym_churn_us.csv"

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

NUM_FEATURES = [
    "Age", "Lifetime", "Contract_period", "Month_to_end_contract",
    "Avg_class_frequency_total", "Avg_class_frequency_current_month",
    "Avg_additional_charges_total",
]

BINARY_FEATURES = ["gender", "Near_Location", "Partner", "Promo_friends", "Phone", "Group_visits"]

PALETTE = ["#4C72B0", "#DD8452"]


def set_style():
    sns.set_theme(style="whitegrid")
    mpl.rcParams["font.family"] = "Malgun Gothic"
    mpl.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 100
