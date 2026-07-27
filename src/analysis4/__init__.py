"""분석 4 — 헬스장 고객 이탈 분석

모듈 구성
    features        피처 정의·파생변수·시나리오 (트랙 공용 단일 소스)
    eda             탐색적 분석 — 요약표 / Figure 반환
    classification  트랙 A — 이탈 예측 (분류)
    regression      트랙 B — 이탈 타이밍 (생존분석 주력)
    clustering      트랙 C — 군집화 및 등급 부여

사용 예:
    from src.analysis4 import eda, features as ft
    from src.analysis4 import classification as clf

    ft.set_seed()
    fig = eda.plot_target()
    result = clf.evaluate(scenario="S1", derived=True)

서브모듈을 여기서 미리 import하지 않는다.
`python -m src.analysis4.eda` 처럼 모듈을 직접 실행할 때 같은 파일이
`__main__` 과 `src.analysis4.eda` 로 두 번 적재되는 것을 피하기 위함이다.
`from src.analysis4 import eda` 형태는 이 없이도 정상 동작한다.
"""

__all__ = ["features", "eda", "classification", "regression", "clustering"]
