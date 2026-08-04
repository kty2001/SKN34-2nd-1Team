"""
헬스장 회원 이탈(Churn) 예측 Streamlit 앱.
- models/ 폴더의 저장된 모델(scaler.pkl, churn_model.pkl, metadata.json)을 불러와
  사용자가 입력한 값으로 이탈 확률을 예측한다.
- 모델/스케일러가 없으면 `python train_model.py`로 먼저 생성해야 한다.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import streamlit as st

mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

MODEL_DIR = Path("./models/analysis3")

# 상태(risk) 색상 - 절대 색상만으로 의미를 전달하지 않고 아이콘/텍스트를 함께 사용한다.
STATUS_COLORS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "critical": "#d03b3b",
}
SEQ_BLUE = "#2a78d6"
SEQ_ORANGE = "#eb6834"

FEATURE_LABELS = {
    "Contract_period": "계약 기간(개월)",
    "Month_to_end_contract": "계약 만료까지 남은 개월",
    "Lifetime": "가입 후 경과 개월",
    "Avg_class_frequency_total": "전체 기간 평균 주간 방문 빈도",
    "Avg_class_frequency_current_month": "이번 달 평균 주간 방문 빈도",
    "Group_visits": "그룹 수업 참여 여부",
}

ACTION_SUGGESTIONS = {
    "Contract_period": "장기 계약(6개월·12개월) 전환 시 할인/사은품 프로모션 제안",
    "Month_to_end_contract": "계약 만료 임박 회원 대상 조기 갱신 쿠폰 발송, 만료 전 리마인드 알림 발송",
    "Lifetime": "가입 초기(1~3개월) 웰컴콜, 목표 설정 상담 등 초기 리텐션 프로그램 운영",
    "Avg_class_frequency_total": "개인 맞춤 운동 알림 푸시, 출석 리워드 포인트 제공",
    "Avg_class_frequency_current_month": "이번 달 방문이 저조한 회원 대상 PT 무료 체험·이벤트 클래스 안내",
    "Group_visits": "그룹 수업 무료 체험 쿠폰 제공, 그룹 수업 전용 할인 프로그램 안내",
}

st.set_page_config(page_title="헬스장 회원 이탈 예측", page_icon="🏋️", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "churn_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    with open(MODEL_DIR / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, scaler, metadata


if not (MODEL_DIR / "churn_model.pkl").exists():
    st.error("저장된 모델이 없습니다. 먼저 터미널에서 `python src/analysis3/train.py`를 실행해 모델을 생성하세요.")
    st.stop()

model, scaler, metadata = load_artifacts()
feature_names = metadata["feature_names"]
threshold = metadata["best_threshold"]
stats = metadata["feature_stats"]

st.title("🏋️ 헬스장 회원 이탈(Churn) 예측")
st.caption("각 항목에 값을 입력하면 튜닝된 XGBoost 모델이 해당 회원의 이탈 확률을 예측합니다.")

with st.sidebar:
    st.header("모델 정보")
    ts = metadata["test_scores"]
    st.metric("F1-score", f"{ts['f1']:.3f}")
    st.metric("ROC-AUC", f"{ts['roc_auc']:.3f}")
    st.metric("Accuracy", f"{ts['accuracy']:.3f}")
    st.caption(f"이탈 판정 임계값: {threshold:.2f} (0.5가 아닌 F1 최적값 기준)")
    st.divider()
    st.caption("모델: 튜닝된 XGBoost · 학습 데이터: data/gym_churn_us.csv")

st.subheader("회원 정보 입력")

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**기본 정보**")
        gender = st.selectbox(
            "성별 코드 (gender)", options=[0, 1],
            help="데이터셋 원본 인코딩 값 그대로 (0 또는 1)",
        )
        age = st.number_input("나이 (Age)", min_value=10, max_value=100, value=int(round(stats["Age"]["mean"])), step=1)
        near_location = st.selectbox(
            "헬스장 근처 거주/근무 여부 (Near_Location)", options=["예", "아니오"], index=0,
        )
        phone = st.selectbox("전화번호 등록 여부 (Phone)", options=["예", "아니오"], index=0)

        st.markdown("**계약/제휴 정보**")
        partner = st.selectbox("제휴사 직원 여부 (Partner)", options=["예", "아니오"], index=1)
        promo_friends = st.selectbox("지인 추천 가입 여부 (Promo_friends)", options=["예", "아니오"], index=1)
        contract_period = st.number_input(
            "계약 기간, 개월 (Contract_period)", min_value=1, max_value=36,
            value=int(round(stats["Contract_period"]["mean"])), step=1,
        )
        month_to_end = st.number_input(
            "계약 만료까지 남은 개월 (Month_to_end_contract)", min_value=0.0, max_value=36.0,
            value=round(stats["Month_to_end_contract"]["mean"], 1), step=0.5,
        )

    with col2:
        st.markdown("**이용 패턴**")
        lifetime = st.number_input(
            "가입 후 경과 개월 (Lifetime)", min_value=0, max_value=120,
            value=int(round(stats["Lifetime"]["mean"])), step=1,
        )
        group_visits = st.selectbox("그룹 수업 참여 여부 (Group_visits)", options=["예", "아니오"], index=1)
        avg_freq_total = st.number_input(
            "전체 기간 평균 주간 방문 빈도 (Avg_class_frequency_total)",
            min_value=0.0, max_value=15.0, value=round(stats["Avg_class_frequency_total"]["mean"], 2), step=0.1,
        )
        avg_freq_month = st.number_input(
            "이번 달 평균 주간 방문 빈도 (Avg_class_frequency_current_month)",
            min_value=0.0, max_value=15.0, value=round(stats["Avg_class_frequency_current_month"]["mean"], 2), step=0.1,
        )
        avg_charges = st.number_input(
            "부가서비스 평균 지출액, $ (Avg_additional_charges_total)",
            min_value=0.0, max_value=2000.0, value=round(stats["Avg_additional_charges_total"]["mean"], 2), step=1.0,
        )

    submitted = st.form_submit_button("이탈 확률 예측", use_container_width=True, type="primary")

if submitted:
    yes_no = lambda v: 1 if v == "예" else 0

    row = {
        "gender": gender,
        "Near_Location": yes_no(near_location),
        "Partner": yes_no(partner),
        "Promo_friends": yes_no(promo_friends),
        "Phone": yes_no(phone),
        "Contract_period": contract_period,
        "Group_visits": yes_no(group_visits),
        "Age": age,
        "Avg_additional_charges_total": avg_charges,
        "Month_to_end_contract": month_to_end,
        "Lifetime": lifetime,
        "Avg_class_frequency_total": avg_freq_total,
        "Avg_class_frequency_current_month": avg_freq_month,
    }
    X_input = pd.DataFrame([row])[feature_names]
    X_scaled = scaler.transform(X_input)
    proba = float(model.predict_proba(X_scaled)[0, 1])
    is_churn = proba >= threshold

    if proba < 0.3:
        status, icon, label = "good", "🟢", "낮음"
    elif proba < 0.6:
        status, icon, label = "warning", "🟡", "보통"
    else:
        status, icon, label = "critical", "🔴", "높음"
    color = STATUS_COLORS[status]

    st.divider()
    st.subheader("예측 결과")

    r1, r2, r3 = st.columns(3)
    r1.metric("이탈 확률", f"{proba:.1%}")
    r2.markdown(
        f"<div style='font-size:0.875rem;color:#52514e;margin-bottom:2px;'>이탈 위험도</div>"
        f"<div style='font-size:1.5rem;font-weight:600;color:{color};'>{icon} {label}</div>",
        unsafe_allow_html=True,
    )
    r3.markdown(
        f"<div style='font-size:0.875rem;color:#52514e;margin-bottom:2px;'>모델 판정 (임계값 {threshold:.2f})</div>"
        f"<div style='font-size:1.5rem;font-weight:600;color:{'#d03b3b' if is_churn else '#0ca30c'};'>"
        f"{'⚠️ 이탈 예상' if is_churn else '✅ 잔류 예상'}</div>",
        unsafe_allow_html=True,
    )

    st.progress(min(max(proba, 0.0), 1.0))

    # ------------------------------------------------------------------
    # 이탈률을 낮추기 위한 개선 제안 (반사실적 시뮬레이션)
    # 이탈률이 가장 낮은 회원 클러스터(저이탈군)의 평균 이용 패턴을 목표치로 삼아,
    # 현재 입력값에서 각 항목만 그 목표치로 올렸을 때 예측 확률이 어떻게 바뀌는지 계산한다.
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("💡 이탈률을 낮추려면 무엇을 바꿔야 할까?")

    actionable_features = metadata.get("actionable_features", [])
    cluster_targets = metadata.get("cluster_targets", {})
    low_churn_rate = metadata.get("low_churn_rate")

    def simulate(overrides: dict) -> float:
        r = dict(row)
        r.update(overrides)
        Xr = pd.DataFrame([r])[feature_names]
        Xs = scaler.transform(Xr)
        return float(model.predict_proba(Xs)[0, 1])

    recommendations = []
    for feat in actionable_features:
        target_raw = cluster_targets.get(feat)
        if target_raw is None:
            continue
        current_val = row[feat]
        if feat == "Group_visits":
            if current_val >= 1 or target_raw <= 0.5:
                continue
            target_val = 1
        else:
            target_val = max(current_val, target_raw)
            if target_val - current_val < 1e-6:
                continue
        new_proba = simulate({feat: target_val})
        reduction = proba - new_proba
        if reduction > 0.001:
            recommendations.append({
                "feature": feat,
                "label": FEATURE_LABELS.get(feat, feat),
                "current": current_val,
                "target": target_val,
                "new_proba": new_proba,
                "reduction": reduction,
            })

    recommendations.sort(key=lambda d: d["reduction"], reverse=True)

    if not recommendations:
        st.success(
            "현재 입력값은 이미 이탈률이 낮은 회원군(저이탈 클러스터, 평균 이탈률 "
            f"{low_churn_rate:.1%})과 이용 패턴이 비슷합니다. 추가로 개선을 제안할 항목이 없습니다."
        )
    else:
        combined_overrides = {r["feature"]: r["target"] for r in recommendations}
        combined_proba = simulate(combined_overrides)

        st.caption(
            f"이탈률이 가장 낮은 회원 그룹(저이탈 클러스터, 평균 이탈률 {low_churn_rate:.1%})의 "
            "이용 패턴을 목표치로 삼아, 각 항목을 그 수준까지 올렸을 때의 예측 확률 변화입니다."
        )

        # 1) 항목별 조정 전/후 확률 비교 막대그래프
        labels = [r["label"] for r in recommendations] + ["전체 적용"]
        before_vals = [proba] * len(recommendations) + [proba]
        after_vals = [r["new_proba"] for r in recommendations] + [combined_proba]

        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.patch.set_facecolor("#fcfcfb")
        ax.set_facecolor("#fcfcfb")
        bars_before = ax.bar(x - width / 2, before_vals, width, label="현재", color=SEQ_ORANGE)
        bars_after = ax.bar(x + width / 2, after_vals, width, label="조정 후", color=SEQ_BLUE)
        ax.bar_label(bars_before, fmt="%.0f%%", labels=[f"{v:.0%}" for v in before_vals], padding=2, fontsize=8)
        ax.bar_label(bars_after, fmt="%.0f%%", labels=[f"{v:.0%}" for v in after_vals], padding=2, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12)
        ax.set_ylabel("예측 이탈 확률")
        ax.set_ylim(0, 1)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", frameon=False)
        plt.tight_layout()
        st.pyplot(fig)

        # 2) 상위 항목에 대해 수치를 조금씩 올릴 때 확률이 어떻게 바뀌는지 보여주는 변화 곡선
        numeric_recs = [r for r in recommendations if r["feature"] != "Group_visits"][:2]
        if numeric_recs:
            st.markdown("###### 수치를 단계적으로 올릴 때 예측 이탈 확률 변화")
            cols = st.columns(len(numeric_recs))
            for c, r in zip(cols, numeric_recs):
                feat = r["feature"]
                sweep_vals = np.linspace(r["current"], r["target"], 10)
                sweep_proba = [simulate({feat: v}) for v in sweep_vals]
                fig, ax = plt.subplots(figsize=(4.2, 3.4))
                fig.patch.set_facecolor("#fcfcfb")
                ax.set_facecolor("#fcfcfb")
                ax.plot(sweep_vals, sweep_proba, color=SEQ_BLUE, linewidth=2)
                ax.scatter([r["current"]], [proba], color="#898781", zorder=5, s=40, label="현재")
                ax.scatter([r["target"]], [r["new_proba"]], color=STATUS_COLORS["good"], zorder=5, s=50, label="목표")
                ax.annotate(f"{proba:.0%}", (r["current"], proba), textcoords="offset points",
                            xytext=(-5, 8), fontsize=8, color="#52514e")
                ax.annotate(f"{r['new_proba']:.0%}", (r["target"], r["new_proba"]), textcoords="offset points",
                            xytext=(-5, -14), fontsize=8, color=STATUS_COLORS["good"])
                ax.set_title(r["label"], fontsize=10)
                ax.set_xlabel(r["label"], fontsize=8)
                ax.set_ylabel("예측 이탈 확률", fontsize=8)
                ax.set_ylim(0, 1)
                for spine in ["top", "right"]:
                    ax.spines[spine].set_visible(False)
                ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
                ax.set_axisbelow(True)
                ax.legend(loc="upper right", frameon=False, fontsize=7)
                plt.tight_layout()
                c.pyplot(fig)

        # 3) 실행 예시 텍스트
        st.markdown("###### 실행 예시")
        for r in recommendations:
            cur_str = f"{r['current']:.2f}" if isinstance(r["current"], float) else str(r["current"])
            tgt_str = f"{r['target']:.2f}" if isinstance(r["target"], float) else str(r["target"])
            suggestion = ACTION_SUGGESTIONS.get(r["feature"], "")
            st.markdown(
                f"- **{r['label']}**: {cur_str} → **{tgt_str}** "
                f"(이탈 확률 {r['reduction']:.1%}p 감소 예상)\n"
                f"  - 실행 예시: {suggestion}"
            )
        st.markdown(
            f"- **전체 항목 동시 적용 시**: 예측 이탈 확률 {proba:.1%} → **{combined_proba:.1%}** "
            f"({proba - combined_proba:.1%}p 감소 예상)"
        )

    st.markdown("##### 이탈 예측에 영향을 준 주요 요인 (모델 전체 특성 중요도)")
    importance = pd.Series(metadata["feature_importance"]).sort_values(ascending=True).tail(8)
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.barh(importance.index, importance.values, color=SEQ_BLUE, height=0.6)
    ax.set_xlabel("중요도")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)

    with st.expander("입력값 확인"):
        st.dataframe(X_input, use_container_width=True)