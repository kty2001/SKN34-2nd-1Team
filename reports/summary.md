# 분석 1
# 인공지능 데이터 전처리 결과서

* 전체 데이터 **4,000명 × 14개 변수**
* 결측치 및 중복 데이터 **없음**
* 군집 분석 변수: `Lifetime`, `Age`, `Avg_class_frequency_current_month`, `Contract_period`
* `Churn`은 군집별 **이탈률 비교**에 활용
* `StandardScaler`, `RobustScaler`를 적용하여 K-Means 전처리 진행

---

# 인공지능 학습 결과서

## 기본 모델

**StandardScaler → K-Means**

![엘보우 방법 및 실루엣 점수](images/analysis1/03_k_validation.png)

* 최적 K: **4**
* Test Silhouette Score: **0.2669**
* 이용 기간과 방문 빈도가 낮은 고객군에서 **이탈 위험이 높게 나타남**

![클러스터별 이탈률](images/analysis1/06_cluster_churn.png)

## 고도화 모델

**RobustScaler → K-Means**

![RobustScaler 기반 K 검증 결과](images/analysis1/08_k_validation_upgrade.png)

* 최적 K: **3**
* Test Silhouette Score: **0.3130**
* 기본 모델보다 군집 품질 **향상**
* 고도화 모델에서도 이용 기간과 방문 빈도가 낮은 고객군의 **높은 이탈 위험이 유지됨**

![RobustScaler 기반 클러스터별 이탈률](images/analysis1/10_cluster_churn_upgrade.png)

---

# 학습된 인공지능 모델

| 구분              | 기본 모델          | 고도화 모델       |
| --------------- | -------------- | ------------ |
| 전처리             | StandardScaler | RobustScaler |
| 모델              | K-Means        | K-Means      |
| 최적 K            | 4              | **3**        |
| Test Silhouette | 0.2669         | **0.3130**   |

최종 모델은 **RobustScaler + K-Means(K=3)**으로 선정하고 `joblib` 형식으로 저장하여 Streamlit 예측에 활용하였다.

```text
models/
└─ analysis1/
   └─ churn_cluster_model.joblib
```

-----------------------------------------------------


# 분석 2
# 인공지능 데이터 전처리 결과서

* 데이터: `gym_churn_us.csv` **4,000명 × 14개 컬럼**
* 결측치 및 중복 데이터 **없음**
* 전체 이탈률: **26.5%**
* 모든 피처가 숫자형으로 구성되어 **별도 인코딩 불필요**
* `Train/Test = 8:2`, `stratify=y`를 적용하여 이탈 비율 유지
* `Contract_period`와 `Month_to_end_contract`, `Avg_class_frequency_total`과 `Avg_class_frequency_current_month`에서 높은 다중공선성이 확인되었으나 **트리 기반 모델의 특성을 고려해 변수 제거 없이 진행**

![이탈 여부 분포](images/analysis2/01_churn_distribution.png)

![피처 간 상관관계](images/analysis2/02_correlation_heatmap.png)

---

# 인공지능 학습 결과서

## 기본 모델

**Decision Tree / Random Forest**

| 모델            |   Accuracy |     Recall |         F1 |    ROC-AUC |
| ------------- | ---------: | ---------: | ---------: | ---------: |
| Decision Tree |     0.9062 |     0.8066 |     0.8201 |     0.8744 |
| Random Forest | **0.9275** | **0.8349** | **0.8592** | **0.9676** |

![Decision Tree Confusion Matrix](images/analysis2/03_dt_confusion_base.png)

![Random Forest Confusion Matrix](images/analysis2/04_rf_confusion_base.png)

## 모델 고도화

`GridSearchCV`를 활용하여 **Recall을 기준으로 하이퍼파라미터 튜닝**을 진행하였다.

* Decision Tree: `max_depth=5, min_samples_leaf=1`
* Random Forest: `max_depth=10, max_features='sqrt', n_estimators=300`

| 모델            | 구분    |   Accuracy |     Recall |         F1 |    ROC-AUC |
| ------------- | ----- | ---------: | ---------: | ---------: | ---------: |
| Decision Tree | 고도화 전 |     0.9062 |     0.8066 |     0.8201 |     0.8744 |
| Decision Tree | 고도화 후 | **0.9112** | **0.8160** | **0.8297** | **0.9381** |
| Random Forest | 고도화 전 | **0.9275** | **0.8349** | **0.8592** |     0.9676 |
| Random Forest | 고도화 후 |     0.9250 |     0.8208 |     0.8529 | **0.9694** |

## Feature Importance

![Decision Tree Feature Importance](images/analysis2/05_dt_importance_advanced.png)

![Random Forest Feature Importance](images/analysis2/06_rf_importance_advanced.png)

* 두 모델 모두 **`Lifetime`(가입기간)**이 중요도 1위
* 다음으로 **최근 수업 참여빈도**와 **연령**이 주요 변수로 확인
* 인구통계보다 **실제 이용 행동 패턴이 이탈 예측에 더 중요한 요인**으로 나타남

---

# 학습된 인공지능 모델

* **최종 채택 모델: Random Forest (Advanced)**
* 하이퍼파라미터: `max_depth=10`, `max_features='sqrt'`, `n_estimators=300`
* Test 성능: **Accuracy 0.9250 / Recall 0.8208 / F1 0.8529 / ROC-AUC 0.9694**
* 모델은 `joblib` 형식으로 저장하여 Streamlit 예측 페이지에서 활용

```text
models/
└─ analysis2/
   ├─ decision_tree_base.pkl
   ├─ random_forest_base.pkl
   ├─ decision_tree_advanced.pkl
   ├─ random_forest_advanced.pkl
   └─ test_data.pkl
```

**핵심 결과:** `Lifetime`이 가장 중요한 이탈 예측 변수로 확인되었으며, **가입기간이 짧고 최근 수업 참여빈도가 낮은 회원일수록 이탈 위험이 높은 패턴**을 확인하였다.

-----------------------------------------------------

# 분석 3
# 인공지능 데이터 전처리 결과서

* `gym_churn_us.csv` **4,000명 · 14개 컬럼**으로 구성되며, 결측치와 중복 데이터는 없음
* 모든 데이터가 수치형으로 구성되어 별도 인코딩 없이 사용
* `Lifetime`, `Avg_class_frequency_current_month`, `Contract_period`, `Month_to_end_contract`를 주요 이탈 관련 변수로 선정
* Train/Test 데이터를 **8:2 비율**로 분리하고 `stratify`를 적용하여 이탈 비율을 유지
* 로지스틱 회귀 비교를 위해 `StandardScaler`를 적용

![이탈 여부 분포](images/analysis3/01_churn_rate.png)

![피처 간 상관관계](images/analysis3/02_feature_relation.png)

---

# 인공지능 학습 결과서

* Logistic Regression, Random Forest, Gradient Boosting을 비교한 결과 **Gradient Boosting**이 우수한 성능을 보임
* K-Means 군집 분석 결과 **단기계약·저빈도 회원군의 이탈률이 56%**로 가장 높게 나타남
* XGBoost 추가 및 하이퍼파라미터 튜닝을 진행하여 최종 모델의 **F1 0.8942, ROC-AUC 0.9804** 달성
* Feature Importance 분석 결과 **계약 기간, 이용 기간, 계약 잔여 기간**이 이탈 예측에 가장 큰 영향을 미침
* 이탈 회원의 계약 기간·이용 기간·방문 빈도를 개선하는 시뮬레이션을 통해 **장기 계약 전환과 방문 빈도 증가가 주요 리텐션 전략**임을 확인

![모델별 성능 비교](images/analysis3/05_result_by_each_models.png)

![튜닝된 XGBoost 성능](images/analysis3/10_final_output.png)

![튜닝된 XGBoost 피처 중요도](images/analysis3/12_feature_importance_XGB.png)

---

# 학습된 인공지능 모델

* 최종 모델: **XGBoost**
* 테스트 성능: **Accuracy 0.9450 / Recall 0.8774 / F1 0.8942 / ROC-AUC 0.9804**
* 학습된 모델은 회원 데이터를 입력하여 **이탈 가능성을 예측**하고, 고위험 회원을 선별하는 데 활용
* K-Means 군집 분석은 예측 모델의 성능 향상보다는 **회원 세그먼트별 맞춤형 이탈 방지 전략 수립**에 활용

![최종 모델 피처 중요도](images/analysis3/12_feature_importance_XGB.png)

![이탈 방지 시뮬레이션](images/analysis3/13_churn_rate_pred_after_action.png)


-----------------------------------------------------


# 분석 4 
## 1. 인공지능 데이터 전처리 결과서

### 1.1 데이터 개요

헬스장 고객 **4,000명 × 14개 변수** 데이터를 사용하였다.

* 결측치: **0개**
* 중복 데이터: **0개**
* 유지 고객: **2,939명 (73.5%)**
* 이탈 고객: **1,061명 (26.5%)**

### 1.2 데이터 전처리 및 변수 선정

EDA와 통계 검증을 통해 이탈과 관련성이 낮은 `gender`, `Phone`을 제외하고, 이탈 예측에 영향을 주는 `Age`, 계약 기간, 이용 빈도 등의 변수를 중심으로 학습 데이터를 구성하였다.

또한 다중공선성이 높은 변수는 파생변수로 대체하여 모델의 안정성과 해석력을 높였다.

![상관계수 히트맵](images/analysis4/03_correlation.png)

### 1.3 주요 데이터 분석 결과

계약 기간에 따른 이탈률 차이가 가장 크게 나타났다.

![변수별 이탈률](images/analysis4/01_churn_rate_by_feature.png)

* 1개월 계약: **42.32%**
* 6개월 계약: **12.48%**
* 12개월 계약: **2.40%**

또한 이탈 고객은 가입 초기 기간에 집중되어 있었다.

![유지 기간별 이탈률](images/analysis4/02_lifetime_churn.png)

이탈자 1,061명 중 **817명(77.0%)이 가입 1개월 이내에 이탈**한 것으로 나타났다.

→ 따라서 **가입 초기 회원 관리와 단기계약 회원 관리**를 주요 분석 대상으로 선정하였다.

---

## 2. 인공지능 학습 결과서

### 2.1 이탈 예측 모델

XGBoost, Random Forest, Logistic Regression, MLP를 비교하여 고객 이탈 예측 모델을 학습하였다.

![ROC 곡선](images/analysis4/05_roc_curve.png)

| 모델                  |    ROC-AUC |
| ------------------- | ---------: |
| Logistic Regression |     0.9535 |
| XGBoost             | **0.9878** |

XGBoost가 가장 높은 예측 성능을 보여 최종 모델로 선정하였다.

### 2.2 주요 변수 중요도

![변수 중요도](images/analysis4/06_importance.png)

분석 결과 **방문 변화, Lifetime, 계약 기간, 연령** 등이 고객 이탈과 관련된 주요 변수로 확인되었다.

### 2.3 이탈 시점 분석

이탈 시점을 예측하기 위해 분류와 회귀 방법을 비교했으나, 이탈자 내부에서는 이탈 시점을 구분할 수 있는 신호가 부족하였다.

이에 **생존분석(Cox)**을 적용하여 전체 고객의 이탈 위험을 분석하였다.

![Kaplan-Meier 생존곡선](images/analysis4/07_km_curve.png)

생존분석 결과 **C-index 0.8805**를 기록했으며, 계약 기간이 길수록 이탈 위험이 감소하는 경향을 확인하였다.

### 2.4 고객 군집 분석

KMeans를 이용해 고객을 3개 집단으로 분류하였다.

![군집 프로파일](images/analysis4/09_cluster_profile.png)

군집의 실루엣 점수는 **0.1886**으로 군집 간 분리도가 높지는 않았다.

따라서 군집은 이탈 위험 등급을 결정하기보다는 **고객 특성을 파악하고 설명하는 용도**로 활용하였다.

---

## 3. 학습된 인공지능 모델

### 3.1 최종 모델

* 모델: **XGBoost Classifier**
* 목적: 헬스장 고객 이탈 여부 예측
* ROC-AUC: **0.9878**
* 최적 임계값: **0.3943**
* 하이퍼파라미터 최적화: **Optuna 30 Trial**
* SEED: **42**

![고도화 전후 성능](images/analysis4/10_tuning_effect.png)

모델 고도화 후 ROC-AUC는 **0.9849 → 0.9878**로 향상되었으며, Recall 역시 **0.9057 → 0.9132**로 증가하였다.

### 3.2 모델 활용

학습된 모델을 통해 고객별 이탈 확률을 산출하고, 이를 기반으로 고위험 고객을 선별할 수 있도록 구성하였다.

분석 결과를 기반으로 다음과 같은 리텐션 전략을 도출하였다.

| 우선순위  | 리텐션 전략                     |
| ----- | -------------------------- |
| **1** | 가입 1개월 이내 신규 회원 집중 관리      |
| **2** | 단기계약 회원의 장기계약 전환 및 그룹수업 유도 |
| **3** | 친구추천 및 제휴 채널 확대            |

### 3.3 최종 결과

헬스장 고객 이탈은 **가입 초기와 단기계약 회원에게 집중**되는 것으로 확인되었다.

최종 XGBoost 모델은 **ROC-AUC 0.9878**의 높은 이탈 예측 성능을 보였으며, 생존분석을 통해 이탈 위험 요인을 추가적으로 확인하였다.

이를 바탕으로 **가입 초기 집중 관리 → 장기계약 및 그룹수업 유도 → 고객 유입 채널 개선**의 리텐션 전략을 제안하였다.

> 본 분석은 관측 데이터 기반이므로 변수 간 관계를 인과관계로 해석할 수 없으며, 실제 전략의 효과는 A/B 테스트 등을 통해 추가 검증할 필요가 있다.
