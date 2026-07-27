"""분석 4 — 피처 정의 및 전처리 (트랙 A/B/C 공용)

노트북 9절(피처 선택 사전 검증)에서 확정한 방침을 코드로 고정한 모듈.
classification / regression / clustering 세 트랙은 모두 이 모듈을 통해
피처를 구성한다. 각 파일에 상수를 복붙하지 않는다.

사용 예:
    from src.analysis4 import features as ft

    ft.set_seed()                               # 진입점 맨 앞에서 1회
    X, y = ft.get_xy(scenario="S1")             # 파생변수 적용 (기본)
    X, y = ft.get_xy(scenario="S2", derived=False)   # 원본 베이스라인
    cv = ft.get_splitter()
"""

import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.common import SEED, load_data

TARGET = "Churn"

# 9.1절 — 카이제곱 p≈0.99로 이탈과 독립. 제거 시 성능이 오히려 소폭 상승한다.
DROP_ALWAYS = ["gender", "Phone"]

# gender/Phone 제거 후 남는 원본 11개 (9.11절 최종 피처 방침)
BASE_FEATURES = [
    "Contract_period",
    "Near_Location",
    "Partner",
    "Promo_friends",
    "Group_visits",
    "Age",
    "Avg_additional_charges_total",
    "Month_to_end_contract",
    "Lifetime",
    "Avg_class_frequency_total",
    "Avg_class_frequency_current_month",
]

# 9.2절 — 중요도 1위(0.2347)지만 이탈 직전 이용 감소를 반영하는 선행지표 성격.
# 제외 시나리오(S2)를 반드시 함께 보고한다.
LEAK_SUSPECT = "Avg_class_frequency_current_month"

# 10절 트랙 A — 두 시나리오를 나란히 학습해 성능 차이를 보고한다.
SCENARIOS = {
    "S1": list(BASE_FEATURES),
    "S2": [c for c in BASE_FEATURES if c != LEAK_SUSPECT],
}

# 파생변수 (9.4절) — '추가'가 아니라 '대체'로 쓴다. 원본을 남기면 중복 정보가 잡음이 된다.
CONTRACT_USED = "계약소진율"
VISIT_DELTA = "방문변화_차이"

# 파생변수 -> 대체 대상 원본 컬럼
DERIVED_REPLACES = {
    CONTRACT_USED: "Month_to_end_contract",
    VISIT_DELTA: LEAK_SUSPECT,
}

# 10절 트랙 B — Lifetime을 타깃으로 쓰므로 입력 피처에서는 제외해야 한다.
TIMING_TARGET = "Lifetime"

# 9.8절 트랙 C 군집 축 9개 — 정답(Churn) / 무신호(gender, Phone) /
# 다중공선성 쌍의 한쪽(Month_to_end_contract, 직전달 방문빈도) 제외
CLUSTER_FEATURES = [
    "Contract_period",
    "Near_Location",
    "Partner",
    "Promo_friends",
    "Group_visits",
    "Age",
    "Avg_additional_charges_total",
    "Lifetime",
    "Avg_class_frequency_total",
]


# ── 반환 형식 규약 ────────────────────────────────────────────────
# 트랙마다 지표가 다르다 (분류=AUC/F1, 생존=C-index, 군집=실루엣).
# 넓은 표로 만들면 트랙을 합칠 수 없으므로 **long 형식**으로 통일한다.
#   트랙 | 단계 | 구성 | 모델 | 지표 | 값
# 이 규약 덕분에 train.py가 세 트랙 결과를 pd.concat 한 줄로 합칠 수 있다.
SUMMARY_COLUMNS = ["트랙", "단계", "구성", "모델", "지표", "값"]

STAGE_BASE = "기본"
STAGE_TUNED = "고도화"


def make_summary(rows):
    """SUMMARY_COLUMNS 스키마의 DataFrame 생성 (컬럼 누락 시 예외)"""
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    missing = [c for c in SUMMARY_COLUMNS if c not in out.columns]
    if missing:
        raise KeyError(f"요약표에 필요한 컬럼이 없습니다: {missing}")
    return out[SUMMARY_COLUMNS]


def scenario_tag(scenario, derived):
    """구성 표기 통일 — 'S1/derived' 형태"""
    return f"{scenario}/{'derived' if derived else 'raw'}"


def set_seed(seed=SEED):
    """전역 난수 시드 고정 (재현성)

    각 추정기에 random_state=SEED를 넘기면 그것만으로 재현되지만,
    **빠뜨린 곳**은 전역 numpy RNG를 끌어다 쓰기 때문에 실행마다 결과가 달라진다.
    (확인: KMeans·MLPClassifier에서 random_state 누락 시 실행마다 불일치)
    이 함수는 그 안전망이다. random_state를 넘기지 않아도 된다는 뜻은 아니다.

    호출 위치: 학습·튜닝 진입점 맨 앞 (train.py / advanced.py의 main, 노트북 첫 셀).
    import 시점에 자동 호출하지 않는다 — 모듈을 불러오는 것만으로 전역 상태를
    바꾸면 다른 코드의 난수 흐름까지 오염시키기 때문이다.

    주의: PYTHONHASHSEED는 인터프리터 시작 시점에만 적용되므로 여기서 설정해도
    현재 프로세스에는 영향이 없고 이후 생성되는 자식 프로세스에만 적용된다.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def _check_columns(df, cols):
    """필요한 컬럼이 모두 있는지 확인"""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"필요한 컬럼이 없습니다: {missing}")


def add_derived(df):
    """파생변수 2개를 추가한 새 DataFrame 반환 (원본은 건드리지 않음)"""
    _check_columns(df, [
        "Contract_period", "Month_to_end_contract",
        "Avg_class_frequency_total", LEAK_SUSPECT,
    ])
    out = df.copy()

    # 계약 소진율: Contract_period 최솟값이 1이라 0으로 나누기는 발생하지 않는다.
    # 예측 기여는 거의 없으나(9.4절) VIF를 19.18 -> 1.41로 낮춰 계수 해석을 가능하게 한다.
    out[CONTRACT_USED] = 1 - out["Month_to_end_contract"] / out["Contract_period"]

    # 방문 빈도 변화: 비율(÷)이 아니라 차이(−)를 쓴다.
    # Avg_class_frequency_total == 0 인 고객이 88명 있어 비율은 0으로 나누기가 발생한다.
    out[VISIT_DELTA] = out[LEAK_SUSPECT] - out["Avg_class_frequency_total"]

    return out


def get_feature_names(scenario="S1", derived=True):
    """시나리오와 파생변수 적용 여부에 따른 최종 피처 목록"""
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario는 {list(SCENARIOS)} 중 하나여야 합니다: {scenario!r}")

    cols = list(SCENARIOS[scenario])
    if not derived:
        return cols

    # 대체 대상 원본이 시나리오에 포함된 경우에만 파생변수로 치환한다.
    # S2에는 LEAK_SUSPECT가 없으므로 방문변화_차이는 자동으로 제외된다.
    for new, old in DERIVED_REPLACES.items():
        if old in cols:
            cols[cols.index(old)] = new

    # 방어적 확인 — 방문변화_차이는 LEAK_SUSPECT를 포함하므로 S2에서 쓰면 시나리오가 무너진다.
    if scenario == "S2" and VISIT_DELTA in cols:
        raise ValueError(
            f"'{VISIT_DELTA}'는 '{LEAK_SUSPECT}'를 포함하므로 S2에서 사용할 수 없습니다. "
            "(노트북 9.4절 / 10절 트랙 A)"
        )
    return cols


def get_xy(df=None, scenario="S1", derived=True):
    """트랙 A 학습용 (X, y) 반환

    df를 넘기지 않으면 load_data()로 직접 읽는다.
    컬럼 순서는 get_feature_names() 순서로 고정된다 — 저장한 모델과 순서를 맞추기 위함.
    """
    df = load_data() if df is None else df
    cols = get_feature_names(scenario, derived)

    work = add_derived(df) if derived else df
    _check_columns(work, cols + [TARGET])

    return work[cols].copy(), work[TARGET].copy()


def get_timing_features(scenario="S1", derived=True):
    """트랙 B 입력 피처 — 분류용 목록에서 타깃(Lifetime)만 제외

    시나리오 의미는 트랙 A와 동일하다. S2는 Avg_class_frequency_current_month를 뺀다.
    """
    return [c for c in get_feature_names(scenario, derived) if c != TIMING_TARGET]


def get_timing_X(df=None, scenario="S1", derived=True):
    """트랙 B 입력 X 반환 (Lifetime 제외)"""
    df = load_data() if df is None else df
    cols = get_timing_features(scenario, derived)
    work = add_derived(df) if derived else df
    _check_columns(work, cols)
    return work[cols].copy()


def get_cluster_X(df=None):
    """트랙 C 군집용 X 반환 (9축, 스케일링 전)

    스케일링은 clustering.py에서 Pipeline으로 처리한다.
    KMeans는 거리 기반이라 StandardScaler 없이 쓰면
    Avg_additional_charges_total(0~552) 하나로 군집이 갈린다.
    """
    df = load_data() if df is None else df
    _check_columns(df, CLUSTER_FEATURES)
    return df[CLUSTER_FEATURES].copy()


def get_splitter(n_splits=5):
    """세 트랙이 공유하는 교차검증 분할기 (성능 비교가 성립하도록 고정)"""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def get_holdout(df=None, scenario="S1", derived=True, test_size=0.25):
    """최종 검증용 고정 홀드아웃 분할 → (X_train, X_test, y_train, y_test)"""
    X, y = get_xy(df, scenario, derived)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEED)


def describe_scenarios(df=None):
    """시나리오별 피처 구성 요약표 — 설계 확인용"""
    rows = []
    for scenario in SCENARIOS:
        for derived in (False, True):
            cols = get_feature_names(scenario, derived)
            rows.append({
                "시나리오": scenario,
                "파생변수": "적용" if derived else "원본",
                "피처수": len(cols),
                "피처": ", ".join(cols),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # 루트에서 실행: python -m src.analysis4.features
    set_seed()
    df = load_data()
    print(f"원본 데이터: {df.shape}")
    print(f"제거 컬럼: {DROP_ALWAYS}\n")

    print(describe_scenarios().drop(columns="피처").to_string(index=False))
    print()

    for scenario in SCENARIOS:
        X, y = get_xy(df, scenario=scenario)
        print(f"[{scenario}] X={X.shape}, 이탈률={y.mean():.3f}")
        print(f"     {list(X.columns)}")

    print(f"\n군집용 X: {get_cluster_X(df).shape}")
