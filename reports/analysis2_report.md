# 분석2: 결정트리 & 랜덤포레스트 기반 이탈 예측

## 1. 분석 목적

헬스장 회원 이탈 예측 문제에 트리 기반 모델(결정트리, 랜덤포레스트)을 적용하여
- 회원의 인구통계 및 이용 행동 데이터로부터 이탈 여부를 예측하고
- 하이퍼파라미터 튜닝을 통해 기본 모델 대비 성능을 개선하며
- Feature Importance를 통해 이탈에 가장 크게 영향을 미치는 요인을 파악하여
비즈니스 관점에서 리텐션 전략 수립에 활용 가능한 인사이트를 도출하는 것을 목표로 한다.

## 2. EDA 및 주요 인사이트

- 데이터: `gym_churn_us.csv`, 4,000명 회원, 14개 컬럼 (수치형 13개 + 타겟 `Churn`)
- 결측치 없음, 모든 피처가 이미 숫자형(0/1 인코딩 포함)으로 별도 인코딩 불필요
- 전체 이탈률: `[TODO: get_churn_rate() 실행 결과 기입]`
- 이탈(1) 그룹 vs 잔류(0) 그룹 평균 비교 결과 (`get_key_features_by_churn()` 기준):
  - `[TODO: 실행 후 상위 3~4개 피처와 방향성 기입 — 예상: Lifetime, Avg_class_frequency_current_month가 낮을수록 이탈 위험 높음]`
- 상관관계 히트맵상 다중공선성 이슈: `[TODO: Avg_class_frequency_total vs current_month 상관계수 확인 후 기입]`

## 3. 모델링 및 평가

### 모델
| 모델 | 설명 |
|---|---|
| 결정트리 (Decision Tree) | 기본 파라미터, `random_state=42` |
| 랜덤포레스트 (Random Forest) | 기본 파라미터, `random_state=42` |

### Train/Test 성능 (고도화 전)

| 모델 | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Random Forest | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

> `python -m src.analysis2.train` 실행 후 `evaluate.py`의 `get_metrics_table(stage="base")` 결과를 그대로 표에 채우면 됩니다.

- Confusion Matrix 상 오분류 경향: `[TODO: 이탈자를 잔류로 잘못 예측한 비율(FN) 확인 후 기입]`

## 4. 모델 고도화 및 비교

### 튜닝 방식
- GridSearchCV 사용, `scoring="recall"` 기준 (이탈자를 놓치지 않는 것이 비즈니스적으로 더 중요하다고 판단)
- 결정트리: `max_depth`, `min_samples_leaf` 탐색
- 랜덤포레스트: `n_estimators`, `max_depth`, `max_features` 탐색

### 최적 하이퍼파라미터
| 모델 | Best Params |
|---|---|
| Decision Tree | `[TODO: advanced.py 실행 결과 best_params 기입]` |
| Random Forest | `[TODO]` |

### 고도화 전/후 성능 비교

| 모델 | 구분 | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Decision Tree | 고도화 전 | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Decision Tree | 고도화 후 | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Random Forest | 고도화 전 | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Random Forest | 고도화 후 | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

### Feature Importance (고도화 후 모델 기준)
- `[TODO: plot_feature_importance() 결과 상위 3~5개 피처 나열 및 해석]`

## 5. 최종 결과 및 결론

- 최종 채택 모델: `[TODO: Decision Tree vs Random Forest 중 Recall/F1 기준 우세한 쪽 선택]`
- 선정 근거: `[TODO]`
- 비즈니스 인사이트: 이탈 위험이 높은 회원의 특징을 요약하고, 이를 기반으로 한 리텐션 전략 제안
  (예: 최근 수업 참여 빈도가 급감한 회원 대상 알림/할인 쿠폰 발송 등)
- 한계 및 개선 방향: `[TODO: 데이터 크기, 클래스 불균형 정도, 추가로 시도해볼 수 있는 모델(XGBoost 등) 언급]`