import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt

from src.common import load_data
from src.analysis1 import (
    get_cluster_data,
    get_descriptive_stats,
    get_missing_values,
    get_duplicate_count,
    plot_feature_distribution,
    plot_correlation,
    load_model,
    make_cluster_result,
    validate_model,
    plot_validation,
    get_cluster_mean,
    plot_cluster_feature,
    evaluate_cluster_churn,
    plot_cluster_churn,
    plot_cluster_pca,
    train_upgrade_model,
    evaluate_upgrade,
    compare_before_after,
    plot_before_after,
    get_feature_name_mapping
)

# 차트 기본 설정
plt.rcParams['font.family'] = 'Malgun Gothic' # 한글 폰트
plt.rcParams['axes.unicode_minus'] = False

# 페이지 설정
st.set_page_config(
    page_title="고객 이탈률 분석",
    page_icon="📊",
    layout="wide"
)

# 제목
st.title("📊 고객 이용 특성과 계약 기간에 따른 이탈률 분석")
st.write(
    "고객의 이용기간, 나이, 최근 방문 빈도, 계약기간을 기반으로 "
    "고객군을 분류하고 군집별 이탈률을 분석합니다."
)

# 데이터 불러오기
df = load_data()

if df is not None:
    # 분석 데이터 준비
    cluster_df = get_cluster_data(df)
    cluster_features = ["Lifetime", "Age", "Avg_class_frequency_current_month", "Contract_period"]
    target = "Churn"

    # 탭 구성
    tab1, tab2, tab3 = st.tabs([
        "📊 EDA",
        "🤖 학습 / 추론 평가",
        "🚀 고도화 전 / 후 평가"
    ])

    # EDA
    with tab1:
        st.header("탐색적 데이터 분석")

        # 기본 정보 메트릭
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("전체 고객 수", len(cluster_df))
        with col2:
            st.metric("분석 Feature", len(cluster_features))
        with col3:
            churn_rate = cluster_df["Churn"].mean() * 100
            st.metric("전체 이탈률", f"{churn_rate:.2f}%")
            
        # 통계
        st.markdown("---")
        st.subheader("기초 통계")
        st.dataframe(get_descriptive_stats(cluster_df), use_container_width=True)
        
        # 데이터 품질
        st.markdown("---")
        st.subheader("데이터 품질 확인")
        col1, col2 = st.columns(2)
        with col1:
            st.write("결측치")
            st.dataframe(get_missing_values(cluster_df))
        with col2:
            st.metric("중복 데이터", get_duplicate_count(cluster_df))

        # 분포
        st.markdown("---")
        st.subheader("Feature 분포")
        st.pyplot(plot_feature_distribution(cluster_df))
        st.markdown("### 📌 분석 요약")
        st.write("대부분의 고객이 짧은 이용기간과 짧은 계약기간에 몰려 있어, 분포가 한쪽으로 편향된 형태를 보입니다.")

        # 상관관계
        st.markdown("---")
        st.subheader("Feature 상관관계")
        st.pyplot(plot_correlation(cluster_df))
        st.markdown("### 📌 분석 요약")
        st.write("변수 간 상관관계가 크지 않게 나타나, 각 특성이 서로 다른 정보를 담고 있어 군집 분석에 상대적으로 유용하게 쓰일 가능성이 큽니다.")

    # 학습 / 추론 및 평가
    with tab2:
        st.header("학습 / 추론 및 평가")

        # 모델 불러오기 및 데이터 분리
        pipeline = load_model()
        X = cluster_df[cluster_features]
        y = cluster_df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 검증 진행
        validation_result = validate_model(pipeline, X_train, X_test)

        # Score 메트릭
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Train Silhouette", f"{validation_result['train_score']:.4f}")
        with col2:
            st.metric("Test Silhouette", f"{validation_result['test_score']:.4f}")

        # Train/Test Score 시각화
        st.markdown("---")
        st.subheader("Train / Test 군집 품질")
        st.pyplot(plot_validation(validation_result))
        st.markdown("### 📌 분석 요약")
        st.write("학습 데이터와 테스트 데이터의 군집 품질 점수가 거의 비슷하게 나타나, 군집화 결과가 비교적 안정적으로 유지되고 있습니다.")

        # 결과 데이터 생성
        train_result = make_cluster_result(X_train, y_train, validation_result["train_cluster"])
        test_result = make_cluster_result(X_test, y_test, validation_result["test_cluster"])

        # Cluster별 특성
        st.markdown("---")
        st.subheader("Cluster별 고객 특성")
        cluster_mean = get_cluster_mean(train_result, cluster_features)
        st.dataframe(cluster_mean.style.format("{:.2f}"), use_container_width=True)
        short_names = {"Lifetime": "Life", "Age": "Age", "Avg_class_frequency_current_month": "AvgFreq", "Contract_period": "Contract"}
        cluster_mean = cluster_mean.rename(columns=short_names)
        st.pyplot(plot_cluster_feature(cluster_mean))
        st.markdown("### 📌 분석 요약")
        st.write("Cluster 0은 '일반 유저', 1은 '고계약 고객', 2는 '장기 충성 고객', 3은 '신규·라이트 유저'로 요약할 수 있습니다.")
        
        # Cluster별 이탈률
        st.markdown("---")
        st.subheader("Cluster별 이탈률")
        cluster_churn = evaluate_cluster_churn(train_result, test_result)
        st.dataframe(cluster_churn.style.format("{:.2%}"), use_container_width=True)
        st.pyplot(plot_cluster_churn(cluster_churn))
        st.markdown("### 📌 분석 요약")
        st.write("일부 클러스터의 이탈률이 매우 높게 나타나, 이탈 위험이 큰 고객군을 식별하는 데 의미가 있습니다.")

        # PCA 군집 분포
        st.markdown("---")
        st.subheader("PCA 군집 분포")
        st.pyplot(plot_cluster_pca(pipeline, X_train, validation_result["train_cluster"]))
        st.markdown("### 📌 분석 요약")
        st.write("PCA 공간에서도 군집이 일부 구분되지만, 경계가 완전히 선명하지 않아 모델 개선 여지가 남아 있습니다.")

    # 고도화 전 / 후 평가
    with tab3:
        st.header("고도화 전 / 후 평가")
        st.write("StandardScaler 기반 K-Means와 RobustScaler 기반 K-Means의 군집 품질을 비교합니다.")

        # 고도화 모델 학습 및 평가
        upgrade_pipeline = train_upgrade_model(X_train, n_cluster=3, seed=42)
        upgrade_result = evaluate_upgrade(upgrade_pipeline, X_train, X_test)

        # 비교 결과 데이터프레임
        comparison = compare_before_after(validation_result, upgrade_result)

        # Score 메트릭
        col1, col2 = st.columns(2)
        with col1:
            st.metric("기존 Test Score", f"{validation_result['test_score']:.4f}")
        with col2:
            st.metric("고도화 Test Score", f"{upgrade_result['test_score']:.4f}")

        st.markdown("---")
        st.subheader("고도화 전 / 후 Silhouette Score")
        st.dataframe(comparison.style.format("{:.4f}"), use_container_width=True)
        st.pyplot(plot_before_after(comparison))
        st.markdown("### 📌 분석 요약")
        st.write("고도화된 모델이 기존 방식보다 더 높은 군집 품질을 보여, 전처리 방식 개선의 효과가 확인되었습니다.")

        # 고도화 모델 군집 분석
        upgrade_train_result = make_cluster_result(X_train, y_train, upgrade_result["train_cluster"])
        upgrade_test_result = make_cluster_result(X_test, y_test, upgrade_result["test_cluster"])
        
        st.markdown("---")
        st.subheader("고도화 후 Cluster별 고객 특성")
        upgrade_cluster_mean = get_cluster_mean(upgrade_train_result, cluster_features)
        upgrade_cluster_mean = upgrade_cluster_mean.rename(columns=short_names)
        st.dataframe(upgrade_cluster_mean.style.format("{:.2f}"), use_container_width=True)
        st.pyplot(plot_cluster_feature(upgrade_cluster_mean))
        st.markdown("### 📌 분석 요약")
        st.write("Cluster 0은 '장기 충성 고객', 1은 '고계약 집중 고객', 2는 '신규·라이트 유저'로 요약할 수 있습니다.")

        st.markdown("---")
        st.subheader("고도화 후 Cluster별 이탈률")
        upgrade_churn = evaluate_cluster_churn(upgrade_train_result, upgrade_test_result)
        st.dataframe(upgrade_churn.style.format("{:.2%}"), use_container_width=True)
        st.pyplot(plot_cluster_churn(upgrade_churn))
        st.markdown("### 📌 분석 요약")
        st.write("고도화 이후에도 이탈 위험이 높은 클러스터가 유지되고 있어, 고객군별 이탈 패턴을 더 명확하게 포착하고 있습니다.")

        st.markdown("---")
        st.subheader("고도화 후 PCA 군집 분포")
        st.pyplot(plot_cluster_pca(upgrade_pipeline, X_train, upgrade_result["train_cluster"]))
        st.markdown("### 📌 분석 요약")
        st.write("고도화 모델의 PCA 분포도는 군집 경계가 더 분명해진 방향으로 개선된 것으로 해석할 수 있습니다.")

else:
    st.warning("데이터를 불러올 수 없습니다.")