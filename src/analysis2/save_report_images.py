"""
src/analysis2/save_report_images.py
리포트(analysis2_report.md)에 삽입할 그래프들을 이미지 파일로 저장

실행: python -m src.analysis2.save_report_images
"""

import os
from src.common import load_data
from src.analysis2.eda import (
    plot_churn_distribution,
    plot_correlation_heatmap,
)
from src.analysis2.evaluate import (
    plot_confusion_matrix,
    plot_feature_importance,
)

IMG_DIR = "reports/images/analysis2"


def save_all_images():
    os.makedirs(IMG_DIR, exist_ok=True)
    df = load_data()

    plot_churn_distribution(df).savefig(f"{IMG_DIR}/01_churn_distribution.png", dpi=150, bbox_inches="tight")
    plot_correlation_heatmap(df).savefig(f"{IMG_DIR}/02_correlation_heatmap.png", dpi=150, bbox_inches="tight")

    plot_confusion_matrix("decision_tree", stage="base").savefig(f"{IMG_DIR}/03_dt_confusion_base.png", dpi=150, bbox_inches="tight")
    plot_confusion_matrix("random_forest", stage="base").savefig(f"{IMG_DIR}/04_rf_confusion_base.png", dpi=150, bbox_inches="tight")

    plot_feature_importance("decision_tree", stage="advanced").savefig(f"{IMG_DIR}/05_dt_importance_advanced.png", dpi=150, bbox_inches="tight")
    plot_feature_importance("random_forest", stage="advanced").savefig(f"{IMG_DIR}/06_rf_importance_advanced.png", dpi=150, bbox_inches="tight")

    print(f"이미지 저장 완료: {IMG_DIR}")


if __name__ == "__main__":
    save_all_images()