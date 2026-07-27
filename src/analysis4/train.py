"""분석 4 — 학습 진입점

세 트랙을 한 번에 실행하고 결과를 통합 저장한다.
학습 로직은 각 트랙 모듈이 소유하며, 이 파일은 실행 순서·시드·집계만 담당한다.

    classification  트랙 A — 기본 4구성(S1·S2 × raw·derived) + 주력 모델 고도화
    regression      트랙 B — B-1 / B-2 / B-3
    clustering      트랙 C — k 탐색 + k=3 군집

트랙마다 지표가 다르므로(AUC / C-index / 실루엣) 통합 요약은
features.SUMMARY_COLUMNS 의 long 형식을 쓴다.

CLI 실행 (프로젝트 루트에서):
    python -m src.analysis4.train              # 고도화 포함
    python -m src.analysis4.train --no-tune    # 기본 학습만 (빠름)
"""

import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from src.common import SEED, load_data
from src.analysis4 import features as ft
from src.analysis4 import classification as clf
from src.analysis4 import clustering as cl
from src.analysis4 import regression as rg

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "analysis4"
SUMMARY_PATH = MODEL_DIR / "summary.joblib"


def run(df=None, n_trials=30):
    """세 트랙 실행 → dict(summary, tracks)

    n_trials=0 이면 트랙 A 고도화를 건너뛴다.
    """
    ft.set_seed()
    df = load_data() if df is None else df

    tracks = {
        "A": clf.run_all(df, n_trials=n_trials),
        "B": rg.run_all(df),
        "C": cl.run_all(df),
    }
    summary = pd.concat([t["summary"] for t in tracks.values()], ignore_index=True)
    return {"summary": summary, "tracks": tracks}


def save_summary(result):
    """통합 요약 저장 — 보고서·페이지가 재학습 없이 읽을 단일 소스"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "summary": result["summary"],
        "seed": SEED,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }, SUMMARY_PATH)
    return SUMMARY_PATH


def load_summary():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"통합 요약이 없습니다: {SUMMARY_PATH}\n"
            "먼저 학습하세요: python -m src.analysis4.train"
        )
    return joblib.load(SUMMARY_PATH)


def headline(summary):
    """트랙별 대표 지표만 추린 표"""
    picks = {
        ("A 분류", "test_auc"): "max",
        ("B 타이밍", "c_index"): "max",
        ("C 군집", "실루엣"): "max",
    }
    rows = []
    for (track, metric), how in picks.items():
        sub = summary[(summary["트랙"] == track) & (summary["지표"] == metric)]
        if sub.empty:
            continue
        best = sub.loc[sub["값"].idxmax() if how == "max" else sub["값"].idxmin()]
        rows.append({"트랙": track, "지표": metric, "최고값": round(best["값"], 4),
                     "구성": best["구성"], "모델": best["모델"], "단계": best["단계"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    n_trials = 0 if "--no-tune" in sys.argv else 30

    ft.set_seed()
    result = run(n_trials=n_trials)
    path = save_summary(result)

    print("=" * 78)
    print(f"통합 요약 — {len(result['summary'])}행 "
          f"(고도화 {'포함' if n_trials else '건너뜀'})")
    print("=" * 78)
    print(result["summary"].round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("트랙별 대표 지표")
    print("=" * 78)
    print(headline(result["summary"]).to_string(index=False))

    if "stage_compare" in result["tracks"]["A"]:
        print("\n" + "=" * 78)
        print("트랙 A 고도화 전/후")
        print("=" * 78)
        print(result["tracks"]["A"]["stage_compare"].to_string())

    print(f"\n저장: {path}")
    print(f"모델 파일: {len(list(MODEL_DIR.glob('*.joblib')))}개")
