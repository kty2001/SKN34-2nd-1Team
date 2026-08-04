import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import streamlit as st


MODEL_DIR = Path("models/analysis3")

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


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "churn_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

    with open(MODEL_DIR / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    return model, scaler, metadata


def predict_sample(row, model, scaler, feature_names, threshold):
    X_input = pd.DataFrame([row])[feature_names]
    X_scaled = scaler.transform(X_input)
    proba = float(model.predict_proba(X_scaled)[0, 1])
    is_churn = proba >= threshold

    return X_input, proba, is_churn


def render_prediction_page(df):
    mpl.rcParams["font.family"] = "Malgun Gothic"
    mpl.rcParams["axes.unicode_minus"] = False

    model, scaler, metadata = load_artifacts()

    feature_names = metadata["feature_names"]
    threshold = metadata["best_threshold"]
    stats = metadata["feature_stats"]

    st.subheader("🏋️ 헬스장 회원 이탈(Churn) 예측")
    st.caption(
        "각 항목에 값을 입력하면 튜닝된 XGBoost 모델이 "
        "해당 회원의 이탈 확률을 예측합니다."
    )

    with st.expander("📊 모델 성능 정보"):
        ts = metadata["test_scores"]

        col1, col2, col3 = st.columns(3)

        col1.metric("F1-score", f"{ts['f1']:.3f}")
        col2.metric("ROC-AUC", f"{ts['roc_auc']:.3f}")
        col3.metric("Accuracy", f"{ts['accuracy']:.3f}")

        st.caption(
            f"이탈 판정 임계값: {threshold:.2f} "
            "(F1 최적값 기준)"
        )

    st.markdown("---")
    st.subheader("회원 정보 입력")

    with st.form("analysis3_churn_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**기본 정보**")

            gender = st.selectbox(
                "성별 코드 (gender)",
                options=[0, 1],
                help="데이터셋 원본 인코딩 값 그대로 (0 또는 1)",
            )

            age = st.number_input(
                "나이 (Age)",
                min_value=10,
                max_value=100,
                value=int(round(stats["Age"]["mean"])),
                step=1,
            )

            near_location = st.selectbox(
                "헬스장 근처 거주/근무 여부",
                options=["예", "아니오"],
                index=0,
            )

            phone = st.selectbox(
                "전화번호 등록 여부",
                options=["예", "아니오"],
                index=0,
            )

            st.markdown("**계약/제휴 정보**")

            partner = st.selectbox(
                "제휴사 직원 여부",
                options=["예", "아니오"],
                index=1,
            )

            promo_friends = st.selectbox(
                "지인 추천 가입 여부",
                options=["예", "아니오"],
                index=1,
            )

            contract_period = st.number_input(
                "계약 기간, 개월",
                min_value=1,
                max_value=36,
                value=int(round(stats["Contract_period"]["mean"])),
                step=1,
            )

            month_to_end = st.number_input(
                "계약 만료까지 남은 개월",
                min_value=0.0,
                max_value=36.0,
                value=round(
                    stats["Month_to_end_contract"]["mean"],
                    1,
                ),
                step=0.5,
            )

        with col2:
            st.markdown("**이용 패턴**")

            lifetime = st.number_input(
                "가입 후 경과 개월",
                min_value=0,
                max_value=120,
                value=int(round(stats["Lifetime"]["mean"])),
                step=1,
            )

            group_visits = st.selectbox(
                "그룹 수업 참여 여부",
                options=["예", "아니오"],
                index=1,
            )

            avg_freq_total = st.number_input(
                "전체 기간 평균 주간 방문 빈도",
                min_value=0.0,
                max_value=15.0,
                value=round(
                    stats["Avg_class_frequency_total"]["mean"],
                    2,
                ),
                step=0.1,
            )

            avg_freq_month = st.number_input(
                "이번 달 평균 주간 방문 빈도",
                min_value=0.0,
                max_value=15.0,
                value=round(
                    stats["Avg_class_frequency_current_month"]["mean"],
                    2,
                ),
                step=0.1,
            )

            avg_charges = st.number_input(
                "부가서비스 평균 지출액, $",
                min_value=0.0,
                max_value=2000.0,
                value=round(
                    stats["Avg_additional_charges_total"]["mean"],
                    2,
                ),
                step=1.0,
            )

        submitted = st.form_submit_button(
            "🔍 이탈 확률 예측",
            use_container_width=True,
            type="primary",
        )

    if not submitted:
        return

    yes_no = lambda value: 1 if value == "예" else 0

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

    X_input, proba, is_churn = predict_sample(
        row,
        model,
        scaler,
        feature_names,
        threshold,
    )

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

    r1.metric(
        "이탈 확률",
        f"{proba:.1%}",
    )

    r2.markdown(
        f"<div style='font-size:0.875rem;'>이탈 위험도</div>"
        f"<div style='font-size:1.5rem;font-weight:600;color:{color};'>"
        f"{icon} {label}</div>",
        unsafe_allow_html=True,
    )

    result_color = "#d03b3b" if is_churn else "#0ca30c"
    result_text = "⚠️ 이탈 예상" if is_churn else "✅ 잔류 예상"

    r3.markdown(
        f"<div style='font-size:0.875rem;'>"
        f"모델 판정 (임계값 {threshold:.2f})</div>"
        f"<div style='font-size:1.5rem;font-weight:600;color:{result_color};'>"
        f"{result_text}</div>",
        unsafe_allow_html=True,
    )

    st.progress(min(max(proba, 0.0), 1.0))

    # 이탈률 개선 시뮬레이션
    st.divider()
    st.subheader("💡 이탈률을 낮추려면 무엇을 바꿔야 할까?")

    actionable_features = metadata.get("actionable_features", [])
    cluster_targets = metadata.get("cluster_targets", {})
    low_churn_rate = metadata.get("low_churn_rate", 0)

    def simulate(overrides):
        simulated_row = dict(row)
        simulated_row.update(overrides)

        Xr = pd.DataFrame([simulated_row])[feature_names]
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

    recommendations.sort(
        key=lambda item: item["reduction"],
        reverse=True,
    )

    if not recommendations:
        st.success(
            "현재 입력값은 이미 이탈률이 낮은 회원군과 "
            f"이용 패턴이 비슷합니다. "
            f"(저이탈 클러스터 평균 이탈률 {low_churn_rate:.1%})"
        )

    else:
        combined_overrides = {
            item["feature"]: item["target"]
            for item in recommendations
        }

        combined_proba = simulate(combined_overrides)

        st.caption(
            f"저이탈 클러스터(평균 이탈률 {low_churn_rate:.1%})의 "
            "이용 패턴을 목표치로 설정한 시뮬레이션입니다."
        )

        # 항목별 조정 전/후 비교
        labels = [
            item["label"]
            for item in recommendations
        ] + ["전체 적용"]

        before_vals = [proba] * len(recommendations) + [proba]

        after_vals = [
            item["new_proba"]
            for item in recommendations
        ] + [combined_proba]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 4.5))

        ax.bar(
            x - width / 2,
            before_vals,
            width,
            label="현재",
            color=SEQ_ORANGE,
        )

        ax.bar(
            x + width / 2,
            after_vals,
            width,
            label="조정 후",
            color=SEQ_BLUE,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12)
        ax.set_ylabel("예측 이탈 확률")
        ax.set_ylim(0, 1)
        ax.legend()

        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        # 주요 항목 변화 곡선
        numeric_recs = [
            item
            for item in recommendations
            if item["feature"] != "Group_visits"
        ][:2]

        if numeric_recs:
            st.markdown("###### 수치를 단계적으로 올릴 때 예측 이탈 확률 변화")

            cols = st.columns(len(numeric_recs))

            for col, item in zip(cols, numeric_recs):
                feat = item["feature"]

                sweep_vals = np.linspace(
                    item["current"],
                    item["target"],
                    10,
                )

                sweep_proba = [
                    simulate({feat: value})
                    for value in sweep_vals
                ]

                fig, ax = plt.subplots(figsize=(4.2, 3.4))

                ax.plot(
                    sweep_vals,
                    sweep_proba,
                    color=SEQ_BLUE,
                    linewidth=2,
                )

                ax.scatter(
                    [item["current"]],
                    [proba],
                    color="#898781",
                    zorder=5,
                    s=40,
                )

                ax.scatter(
                    [item["target"]],
                    [item["new_proba"]],
                    color=STATUS_COLORS["good"],
                    zorder=5,
                    s=50,
                )

                ax.set_title(
                    item["label"],
                    fontsize=10,
                )

                ax.set_xlabel(
                    item["label"],
                    fontsize=8,
                )

                ax.set_ylabel(
                    "예측 이탈 확률",
                    fontsize=8,
                )

                ax.set_ylim(0, 1)
                ax.grid(axis="y")

                plt.tight_layout()

                col.pyplot(fig)
                plt.close(fig)

        # 실행 예시
        st.markdown("###### 실행 예시")

        for item in recommendations:
            cur_str = (
                f"{item['current']:.2f}"
                if isinstance(item["current"], float)
                else str(item["current"])
            )

            tgt_str = (
                f"{item['target']:.2f}"
                if isinstance(item["target"], float)
                else str(item["target"])
            )

            suggestion = ACTION_SUGGESTIONS.get(
                item["feature"],
                "",
            )

            st.markdown(
                f"- **{item['label']}**: "
                f"{cur_str} → **{tgt_str}** "
                f"(이탈 확률 {item['reduction']:.1%}p 감소 예상)\n"
                f"  - 실행 예시: {suggestion}"
            )

        st.markdown(
            f"- **전체 항목 동시 적용 시**: "
            f"예측 이탈 확률 {proba:.1%} → "
            f"**{combined_proba:.1%}** "
            f"({proba - combined_proba:.1%}p 감소 예상)"
        )

    # Feature Importance
    st.markdown(
        "##### 이탈 예측에 영향을 준 주요 요인"
    )

    importance = (
        pd.Series(metadata["feature_importance"])
        .sort_values(ascending=True)
        .tail(8)
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.barh(
        importance.index,
        importance.values,
        color=SEQ_BLUE,
        height=0.6,
    )

    ax.set_xlabel("중요도")
    ax.grid(axis="x")

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    with st.expander("입력값 확인"):
        st.dataframe(
            X_input,
            use_container_width=True,
        )