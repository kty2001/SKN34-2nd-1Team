"""
헬스장 회원 이탈(Churn) 분석 결과 대시보드.
- models/analysis3 폴더의 저장된 모델(scaler.pkl, churn_model.pkl, metadata.json)과
  원본 데이터를 이용해 EDA, 모델 성능, 군집 분석, 이탈 방지 시뮬레이션 결과를 시각화한다.
- 모델/스케일러가 없으면 `python -m src.analysis3.train`을 먼저 실행해 생성해야 한다.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

MODEL_DIR = Path("./models/analysis3")
DATA_PATH = Path("./data/gym_churn_us.csv")

# 상태(risk) 색상 - 절대 색상만으로 의미를 전달하지 않고 아이콘/텍스트를 함께 사용한다.
STATUS_COLORS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "critical": "#d03b3b",
}
SEQ_BLUE = "#2a78d6"
SEQ_ORANGE = "#eb6834"
STAY_COLOR = SEQ_BLUE       # 잔존(0) / 조정 후
CHURN_COLOR = SEQ_ORANGE    # 이탈(1) / 조정 전
CLUSTER_COLORS = [SEQ_BLUE, STATUS_COLORS["good"], STATUS_COLORS["critical"], STATUS_COLORS["warning"]]

FEATURE_LABELS = {
    "gender": "성별",
    "Near_Location": "헬스장 근처 거주",
    "Partner": "제휴사 직원",
    "Promo_friends": "지인 추천 가입",
    "Phone": "전화번호 등록",
    "Contract_period": "계약 기간(개월)",
    "Group_visits": "그룹 수업 참여",
    "Age": "나이",
    "Avg_additional_charges_total": "부가서비스 평균 지출액",
    "Month_to_end_contract": "계약 만료까지 남은 개월",
    "Lifetime": "가입 후 경과 개월",
    "Avg_class_frequency_total": "전체 기간 평균 주간 방문 빈도",
    "Avg_class_frequency_current_month": "이번 달 평균 주간 방문 빈도",
}

ACTION_SUGGESTIONS = {
    "Contract_period": "장기 계약(6개월·12개월) 전환 시 할인/사은품 프로모션 제안",
    "Month_to_end_contract": "계약 만료 임박 회원 대상 조기 갱신 쿠폰 발송, 만료 전 리마인드 알림 발송",
    "Lifetime": "가입 초기(1~3개월) 웰컴콜, 목표 설정 상담 등 초기 리텐션 프로그램 운영",
    "Avg_class_frequency_total": "개인 맞춤 운동 알림 푸시, 출석 리워드 포인트 제공",
    "Avg_class_frequency_current_month": "이번 달 방문이 저조한 회원 대상 PT 무료 체험·이벤트 클래스 안내",
    "Group_visits": "그룹 수업 무료 체험 쿠폰 제공, 그룹 수업 전용 할인 프로그램 안내",
}

NUM_FEATURES = [
    "Age", "Contract_period", "Month_to_end_contract", "Lifetime",
    "Avg_class_frequency_total", "Avg_class_frequency_current_month",
    "Avg_additional_charges_total",
]
BINARY_FEATURES = ["gender", "Near_Location", "Partner", "Promo_friends", "Phone", "Group_visits"]

st.set_page_config(page_title="이탈 예측 모델과 리텐션 시뮬레이션 결과", page_icon="🏋️", layout="wide")


def _style_ax(ax, grid_axis="y"):
    fig = ax.get_figure()
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#e1e0d9", linewidth=0.8)
        ax.set_axisbelow(True)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "churn_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    with open(MODEL_DIR / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, scaler, metadata


@st.cache_data
def load_raw_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def split_data(df: pd.DataFrame, feature_names: list[str]):
    X = df[feature_names]
    y = df["Churn"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


@st.cache_data
def evaluate_model(_model, _scaler, X_test, y_test):
    X_test_scaled = _scaler.transform(X_test)
    y_pred = _model.predict(X_test_scaled)
    y_proba = _model.predict_proba(X_test_scaled)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    return cm, fpr, tpr


@st.cache_data
def compute_cluster_profile(_scaler, X_train, y_train):
    X_train_scaled = _scaler.transform(X_train)
    labels = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X_train_scaled)

    profile = X_train.reset_index(drop=True).copy()
    profile["Cluster"] = labels
    profile["Churn"] = y_train.reset_index(drop=True).values

    summary = profile.groupby("Cluster").apply(
        lambda g: pd.Series({
            "회원수": len(g),
            "이탈률": g["Churn"].mean(),
            "평균 계약기간(개월)": g["Contract_period"].mean(),
            "이번달 평균 방문빈도": g["Avg_class_frequency_current_month"].mean(),
        }),
        include_groups=False,
    ).reset_index()
    return summary


@st.cache_data
def simulate_population(_model, _scaler, X_test, y_test, feature_names, actionable_features, cluster_targets):
    churned = X_test[y_test == 1].copy()
    proba_before = _model.predict_proba(_scaler.transform(churned[feature_names]))[:, 1]

    rows = []
    for feat in actionable_features:
        target = cluster_targets.get(feat)
        if target is None:
            continue
        adjusted = churned.copy()
        if feat == "Group_visits":
            adjusted[feat] = 1
        else:
            adjusted[feat] = np.maximum(adjusted[feat].to_numpy(), target)
        proba_after = _model.predict_proba(_scaler.transform(adjusted[feature_names]))[:, 1]
        rows.append({
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "target": target,
            "before": float(proba_before.mean()),
            "after": float(proba_after.mean()),
            "reduction": float(proba_before.mean() - proba_after.mean()),
            "pct_improved": float((proba_after < proba_before).mean() * 100),
        })

    result = pd.DataFrame(rows).sort_values("reduction", ascending=False).reset_index(drop=True)
    return result, float(proba_before.mean()), len(churned)


if not (MODEL_DIR / "churn_model.pkl").exists():
    st.error("저장된 모델이 없습니다. 먼저 터미널에서 `python -m src.analysis3.train`을 실행해 모델을 생성하세요.")
    st.stop()

model, scaler, metadata = load_artifacts()
feature_names = metadata["feature_names"]
test_scores = metadata["test_scores"]
importance = pd.Series(metadata["feature_importance"]).sort_values(ascending=True)
actionable_features = metadata.get("actionable_features", [])
cluster_targets = metadata.get("cluster_targets", {})
low_churn_rate = metadata.get("low_churn_rate")

df = load_raw_data()
X_train, X_test, y_train, y_test = split_data(df, feature_names)

st.title("🏋️ 이탈 예측 모델과 리텐션 시뮬레이션 결과")
st.caption("회원 4,000명의 이용 데이터를 기반으로 이탈 요인을 탐색하고, 튜닝된 XGBoost 모델과 K-Means 군집화 결과, "
           "이탈 방지 시뮬레이션까지 정리했습니다.")

with st.sidebar:
    st.header("모델 정보")
    st.metric("F1-score", f"{test_scores['f1']:.3f}")
    st.metric("ROC-AUC", f"{test_scores['roc_auc']:.3f}")
    st.metric("Accuracy", f"{test_scores['accuracy']:.3f}")
    st.caption(f"이탈 판정 임계값: {metadata['best_threshold']:.2f} (F1 최적값 기준)")
    st.divider()
    st.caption("모델: 튜닝된 XGBoost · 학습 데이터: data/gym_churn_us.csv")

tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🤖 모델 평가", "🧩 군집 분석", "💡 이탈 방지 시뮬레이션"])

# ──────────────────────────────────────────────────────────────
# 탭 1. EDA
# ──────────────────────────────────────────────────────────────
with tab1:
    st.header("탐색적 데이터 분석")

    churn_rate = df["Churn"].mean()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("전체 회원 수", f"{len(df):,}명")
    col2.metric("전체 이탈률", f"{churn_rate:.1%}")
    col3.metric("결측치", int(df.isna().sum().sum()))
    col4.metric("중복 행", int(df.duplicated().sum()))

    st.markdown("---")
    st.subheader("이탈 여부 분포")
    counts = df["Churn"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.5, 4))
    bars = ax.bar(["잔존 (0)", "이탈 (1)"], counts.values, color=[STAY_COLOR, CHURN_COLOR], width=0.55)
    ax.bar_label(bars, labels=[f"{v:,}명 ({v / len(df):.1%})" for v in counts.values], padding=4)
    ax.set_ylabel("회원 수")
    _style_ax(ax)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("##### 📌 분석 요약")
    st.write(f"잔존 회원이 {1 - churn_rate:.1%}, 이탈 회원이 {churn_rate:.1%}로, 극단적인 불균형 데이터는 아닙니다.")

    st.markdown("---")
    st.subheader("변수 간 상관관계")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
                annot_kws={"size": 6.5}, cbar_kws={"shrink": 0.8}, ax=ax)
    fig.patch.set_facecolor("#fcfcfb")
    plt.tight_layout()
    st.pyplot(fig)
    top_corr = corr["Churn"].drop("Churn").abs().sort_values(ascending=False).head(5)
    st.markdown("##### 📌 분석 요약")
    top_list = ", ".join(f"{FEATURE_LABELS.get(k, k)}({v:.3f})" for k, v in top_corr.items())
    st.write(f"`Churn`과의 상관계수(절댓값)가 높은 변수는 {top_list} 순입니다. "
             "`Avg_class_frequency_total`과 `Avg_class_frequency_current_month`는 서로 강하게 상관되어 있어 "
             "(다중공선성) 선형 모델 계수를 해석할 때 주의가 필요합니다.")

    st.markdown("---")
    st.subheader("수치형 변수의 이탈 여부별 분포")
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, col in zip(axes.flat, NUM_FEATURES):
        sns.kdeplot(data=df, x=col, hue="Churn", fill=True, common_norm=False, alpha=0.4, ax=ax,
                    palette={0: STAY_COLOR, 1: CHURN_COLOR})
        ax.set_title(FEATURE_LABELS.get(col, col), fontsize=10)
        ax.set_xlabel("")
        _style_ax(ax, grid_axis=None)
    for ax in axes.flat[len(NUM_FEATURES):]:
        ax.axis("off")
    fig.patch.set_facecolor("#fcfcfb")
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("##### 📌 분석 요약")
    st.write("이탈 회원은 잔존 회원 대비 `Lifetime`(가입 경과 개월)이 짧고, 방문 빈도(특히 이번 달)가 낮으며, "
             "계약 기간과 잔여 계약 기간이 짧은 쪽에 밀집되어 있는 경향이 뚜렷합니다.")

    st.markdown("---")
    st.subheader("이진 변수별 이탈률")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.flat, BINARY_FEATURES):
        rate = df.groupby(col)["Churn"].mean()
        ax.bar(rate.index.astype(str), rate.values, color=SEQ_BLUE, width=0.5)
        ax.set_title(FEATURE_LABELS.get(col, col), fontsize=10)
        ax.set_ylabel("이탈률")
        ax.set_ylim(0, 1)
        _style_ax(ax)
    fig.patch.set_facecolor("#fcfcfb")
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("##### 📌 분석 요약")
    st.write("헬스장 근처에 거주하거나(`Near_Location`), 제휴사 소속이거나(`Partner`), 지인 추천으로 가입했거나"
             "(`Promo_friends`), 그룹 수업에 참여하는(`Group_visits`) 회원일수록 이탈률이 낮습니다. "
             "반면 `gender`와 `Phone`은 이탈률에 뚜렷한 차이를 만들지 않습니다.")

# ──────────────────────────────────────────────────────────────
# 탭 2. 모델 평가
# ──────────────────────────────────────────────────────────────
with tab2:
    st.header("모델 학습 / 평가")
    st.caption("튜닝된 XGBoost — 하이퍼파라미터 탐색(RandomizedSearchCV, F1 기준) 이후의 최종 모델입니다.")

    cols = st.columns(5)
    for col, key, label in zip(
        cols, ["accuracy", "precision", "recall", "f1", "roc_auc"],
        ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
    ):
        col.metric(label, f"{test_scores[key]:.4f}")

    st.markdown("---")
    st.subheader("Confusion Matrix & ROC Curve")
    cm, fpr, tpr = evaluate_model(model, scaler, X_test, y_test)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["잔존(0)", "이탈(1)"], yticklabels=["잔존(0)", "이탈(1)"])
        ax.set_xlabel("예측")
        ax.set_ylabel("실제")
        fig.patch.set_facecolor("#fcfcfb")
        plt.tight_layout()
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(fpr, tpr, color=SEQ_BLUE, linewidth=2, label=f"ROC (AUC={test_scores['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], color="#c9c8c2", linestyle="--", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", frameon=False)
        _style_ax(ax)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("##### 📌 분석 요약")
    st.write(f"테스트 세트 기준 Accuracy {test_scores['accuracy']:.1%}, F1 {test_scores['f1']:.4f}, "
             f"ROC-AUC {test_scores['roc_auc']:.4f}로 우수한 성능을 보입니다. "
             "잔존(0) 클래스의 정밀도·재현율은 매우 높은 반면, 이탈(1) 클래스는 재현율이 상대적으로 낮아 "
             "실제 이탈 회원 일부를 놓치는 경향이 있습니다.")

    st.markdown("---")
    st.subheader("피처 중요도")
    top_importance = importance.tail(8)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([FEATURE_LABELS.get(i, i) for i in top_importance.index], top_importance.values,
            color=SEQ_BLUE, height=0.6)
    ax.set_xlabel("중요도")
    _style_ax(ax, grid_axis="x")
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("##### 📌 분석 요약")
    top3 = importance.sort_values(ascending=False).head(3)
    top3_list = ", ".join(f"{FEATURE_LABELS.get(k, k)}({v:.2f})" for k, v in top3.items())
    st.write(f"계약·이용 기간과 관련된 변수({top3_list})의 중요도 합이 전체의 대부분을 차지해, "
             "계약 조건과 초기 리텐션 관리가 이탈 예측의 핵심 축임을 보여줍니다.")

# ──────────────────────────────────────────────────────────────
# 탭 3. 군집 분석
# ──────────────────────────────────────────────────────────────
with tab3:
    st.header("K-Means 군집화 기반 회원 세그먼트")
    st.caption("학습 데이터에만 K-Means(k=4)를 적합하여, 이용 패턴이 유사한 회원 그룹을 도출했습니다.")

    cluster_summary = compute_cluster_profile(scaler, X_train, y_train)
    display_summary = cluster_summary.copy()
    display_summary["이탈률"] = display_summary["이탈률"].map(lambda v: f"{v:.1%}")
    display_summary["평균 계약기간(개월)"] = display_summary["평균 계약기간(개월)"].round(1)
    display_summary["이번달 평균 방문빈도"] = display_summary["이번달 평균 방문빈도"].round(2)
    st.dataframe(display_summary, hide_index=True, width="stretch")

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        [f"Cluster {c}" for c in cluster_summary["Cluster"]],
        cluster_summary["이탈률"],
        color=CLUSTER_COLORS[:len(cluster_summary)],
    )
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in cluster_summary["이탈률"]], padding=4)
    ax.set_ylabel("이탈률")
    ax.set_ylim(0, max(cluster_summary["이탈률"]) * 1.25)
    _style_ax(ax)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("##### 📌 분석 요약")
    worst = cluster_summary.loc[cluster_summary["이탈률"].idxmax()]
    best = cluster_summary.loc[cluster_summary["이탈률"].idxmin()]
    st.write(
        f"Cluster {int(worst['Cluster'])}은 평균 계약기간 {worst['평균 계약기간(개월)']:.1f}개월, "
        f"방문빈도 {worst['이번달 평균 방문빈도']:.2f}회로 이탈률이 {worst['이탈률']:.1%}에 달해 가장 우선적인 "
        f"리텐션 타겟입니다. 반면 Cluster {int(best['Cluster'])}은 평균 계약기간 {best['평균 계약기간(개월)']:.1f}개월로 "
        f"이탈률이 {best['이탈률']:.1%}에 불과해, 계약 기간과 방문 빈도가 이탈 방지에 중요한 레버임을 재확인시켜 줍니다."
    )

# ──────────────────────────────────────────────────────────────
# 탭 4. 이탈 방지 시뮬레이션 (Counterfactual)
# ──────────────────────────────────────────────────────────────
with tab4:
    st.header("💡 이탈 방지 시뮬레이션 (Counterfactual 분석)")
    sim_df, before_mean, n_churned = simulate_population(
        model, scaler, X_test, y_test, feature_names, actionable_features, cluster_targets
    )
    st.caption(
        f"테스트 세트에서 실제로 이탈한 회원 {n_churned}명을 대상으로, 저이탈 세그먼트(이탈률 {low_churn_rate:.1%})의 "
        "평균 이용 패턴을 목표치로 삼아 각 항목을 그 수준까지 상향 조정했을 때 예측 이탈확률이 어떻게 바뀌는지 계산했습니다."
    )

    col1, col2 = st.columns(2)
    col1.metric("대상 회원 수", f"{n_churned}명")
    col2.metric("조정 전 평균 예측 이탈확률", f"{before_mean:.1%}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(sim_df))
    width = 0.35
    bars_before = ax.bar(x - width / 2, sim_df["before"], width, label="조정 전", color=CHURN_COLOR)
    bars_after = ax.bar(x + width / 2, sim_df["after"], width, label="조정 후", color=STAY_COLOR)
    ax.bar_label(bars_before, labels=[f"{v:.0%}" for v in sim_df["before"]], padding=2, fontsize=8)
    ax.bar_label(bars_after, labels=[f"{v:.0%}" for v in sim_df["after"]], padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(sim_df["label"], rotation=12)
    ax.set_ylabel("평균 예측 이탈확률")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", frameon=False)
    _style_ax(ax)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("##### 📌 분석 요약")
    best_row = sim_df.iloc[0]
    st.write(
        f"`{best_row['label']}`을(를) 저이탈 세그먼트 수준({best_row['target']:.2f})까지 올렸을 때 평균 이탈확률이 "
        f"{best_row['reduction']:.1%}p 감소해(대상자의 {best_row['pct_improved']:.0f}%에서 개선) 가장 효과가 컸습니다. "
        "`Avg_class_frequency_total`은 `Avg_class_frequency_current_month`와 다중공선성이 있어 단독으로 올리면 "
        "오히려 역효과가 날 수 있으므로, 두 방문 빈도 지표를 함께 개선하는 방향을 권장합니다."
    )

    st.markdown("###### 실행 예시")
    for _, r in sim_df.iterrows():
        suggestion = ACTION_SUGGESTIONS.get(r["feature"], "")
        st.markdown(
            f"- **{r['label']}**: 이탈확률 {r['reduction']:.1%}p 감소 예상 (대상자의 {r['pct_improved']:.0f}%에서 개선)\n"
            f"  - 실행 예시: {suggestion}"
        )
