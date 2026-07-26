# 폴더 구조

```text
project_2nd/
├─ .streamlit/                 # Streamlit 설정
│  └─ config.toml              # Streamlit 화면 및 앱 설정 (새로고침 없이 바로 적용)
│
├─ data/                       # 프로젝트 데이터
│  └─ gym_churn_us.csv         # 헬스장 고객 이탈 데이터
│
├─ docs/                       # 프로젝트 관련 문서 (readme에 들어갈 파일 등)
│  └─ sample.md                
│
├─ models/                     # 학습된 모델 저장 (joblib 저장 위치)
│  ├─ analysis1/               # 분석 1 모델
│  │  └─ sample                
│  ├─ analysis2/               # 분석 2 모델
│  │  └─ sample
│  ├─ analysis3/               # 분석 3 모델
│  │  └─ sample
│  └─ analysis4/               # 분석 4 모델
│     └─ sample
│
├─ notebooks/                  # 데이터 분석 및 실험 (ipynb 파일 저장 위치)
│  └─ sample.py                
│
├─ pages/                      # Streamlit 페이지
│  ├─ analysis1.py             # 분석 1 페이지
│  ├─ analysis2.py             # 분석 2 페이지
│  ├─ analysis3.py             # 분석 3 페이지
│  ├─ analysis4.py             # 분석 4 페이지
│  ├─ dashboard.py             # 전체 분석 결과 대시보드
│  └─ sample_pred.py           # 모델 추론 및 예측 페이지
│
├─ reports/                    # 분석 및 프로젝트 결과 보고서
│  ├─ analysis1_report.md      # 분석 1 보고서
│  ├─ analysis2_report.md      # 분석 2 보고서
│  ├─ analysis3_report.md      # 분석 3 보고서
│  ├─ analysis4_report.md      # 분석 4 보고서
│  └─ final_report.md          # 팀 전체 최종 종합 보고서
│
├─ src/                        # 실제 분석 및 머신러닝 코드
│  ├─ analysis1/               # 분석 1 코드
│  │  ├─ __init__.py           # 패키지 초기화 (함수 쉽게 불러오기 위한 설정)
│  │  ├─ advanced.py           # 모델 고도화
│  │  ├─ eda.py                # 탐색적 데이터 분석
│  │  ├─ evaluate.py           # 모델 성능 평가
│  │  └─ train.py              # 모델 학습
│  │
│  ├─ analysis2/               # 분석 2 코드
│  │  ├─ __init__.py
│  │  ├─ advanced.py
│  │  ├─ eda.py
│  │  ├─ evaluate.py
│  │  └─ train.py
│  │
│  ├─ analysis3/               # 분석 3 코드
│  │  ├─ __init__.py
│  │  ├─ advanced.py
│  │  ├─ eda.py
│  │  ├─ evaluate.py
│  │  └─ train.py
│  │
│  ├─ analysis4/               # 분석 4 코드
│  │  ├─ __init__.py
│  │  ├─ advanced.py
│  │  ├─ eda.py
│  │  ├─ evaluate.py
│  │  └─ train.py
│  │
│  ├─ common/                  # 팀원 공통 코드
│  │  ├─ __init__.py
│  │  └─ data_loader.py        # 데이터 불러오기
│  │
│  ├─ prediction/              # 모델 추론 관련 코드
│  │  └─ __init__.py
│  │
│  └─ __init__.py              # src 패키지 초기화
│
├─ .gitignore                  
├─ app.py                      # Streamlit 메인 실행 파일
├─ README.md                   
└─ requirements.txt            
```