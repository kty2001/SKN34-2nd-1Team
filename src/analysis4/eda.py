"""분석 4 — 탐색적 데이터 분석 (노트북 1~7절 이관)

노트북 `notebooks/analysis4.ipynb`의 시각화·요약을 재사용 가능한 함수로 옮긴 모듈.

설계 원칙
    - Streamlit에 의존하지 않는다. DataFrame 또는 matplotlib Figure만 반환한다.
    - pyplot 전역 레지스트리를 쓰지 않고 Figure 객체를 직접 만든다.
      (Streamlit은 스크립트를 반복 실행하므로 plt.subplots를 쓰면 figure가 계속 쌓인다.
       Figure를 직접 만들면 참조가 끊기는 즉시 회수되어 plt.close()가 필요 없다.)

CLI 실행 (프로젝트 루트에서):
    python -m src.analysis4.eda
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager, rcParams
from matplotlib.figure import Figure
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from src.common import load_data
from src.analysis4 import features as ft

# 노트북과 동일한 색 (유지 / 이탈)
COLOR_STAY = "#4C78A8"
COLOR_CHURN = "#E45756"
PALETTE = [COLOR_STAY, COLOR_CHURN]

# 원본 데이터 기준 변수 분류 (노트북 3절) — gender/Phone 포함. 피처 선택은 features.py 담당.
BINARY = ["gender", "Near_Location", "Partner", "Promo_friends", "Phone", "Group_visits"]
CATEGORICAL = ["Contract_period"]
NUMERIC = [
    "Age",
    "Avg_additional_charges_total",
    "Month_to_end_contract",
    "Lifetime",
    "Avg_class_frequency_total",
    "Avg_class_frequency_current_month",
]

_FONT_CANDIDATES = ["Malgun Gothic", "AppleGothic", "NanumGothic", "NanumBarunGothic"]
_style_applied = False


def set_style(force=False):
    """한글 폰트 및 기본 스타일 설정 → 적용된 폰트명 반환

    OS마다 설치된 한글 폰트가 달라 후보를 순서대로 찾는다.
    하나도 없으면 기본 폰트를 쓰고 경고용으로 None을 반환한다(그래프는 정상, 한글만 깨짐).
    """
    global _style_applied
    if _style_applied and not force:
        return rcParams["font.family"][0]

    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((f for f in _FONT_CANDIDATES if f in available), None)

    sns.set_theme(style="whitegrid")
    if chosen:
        rcParams["font.family"] = chosen
    rcParams["axes.unicode_minus"] = False   # 한글 폰트에서 음수 기호가 깨지는 것 방지
    _style_applied = True
    return chosen


def _new_figure(width, height):
    set_style()
    return Figure(figsize=(width, height), dpi=110, layout="constrained")


# ────────────────────────────────────────────────────────────────
# 1. 기본 구조
# ────────────────────────────────────────────────────────────────
def overview(df=None):
    """컬럼별 자료형 / 결측 / 고유값 수 / 예시값"""
    df = load_data() if df is None else df
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "결측수": df.isna().sum(),
        "고유값수": df.nunique(),
        "예시값": [", ".join(map(str, df[c].unique()[:4])) for c in df.columns],
    })


def basic_stats(df=None):
    """행·열 수, 결측, 중복행 요약"""
    df = load_data() if df is None else df
    return pd.DataFrame([
        {"항목": "행 수", "값": f"{len(df):,}"},
        {"항목": "열 수", "값": f"{df.shape[1]}"},
        {"항목": "결측치", "값": f"{int(df.isna().sum().sum()):,}"},
        {"항목": "중복행", "값": f"{int(df.duplicated().sum()):,}"},
    ])


def describe(df=None):
    df = load_data() if df is None else df
    return df.describe().T.round(3)


# ────────────────────────────────────────────────────────────────
# 2. 타깃 분포
# ────────────────────────────────────────────────────────────────
def target_summary(df=None):
    df = load_data() if df is None else df
    count = df[ft.TARGET].value_counts().sort_index()
    return pd.DataFrame({
        "구분": ["유지(0)", "이탈(1)"],
        "인원": count.values,
        "비율(%)": (count.values / len(df) * 100).round(2),
    })


def plot_target(df=None):
    df = load_data() if df is None else df
    summary = target_summary(df)

    fig = _new_figure(5, 3.5)
    ax = fig.subplots()
    sns.barplot(data=summary, x="구분", y="인원", hue="구분",
                palette=PALETTE, legend=False, ax=ax)
    for i, row in summary.iterrows():
        ax.text(i, row["인원"] + 40, f"{row['인원']:,}명\n({row['비율(%)']:.1f}%)",
                ha="center", va="bottom")
    ax.set_title("타깃 분포")
    ax.set_ylabel("고객 수")
    ax.set_ylim(0, summary["인원"].max() * 1.2)
    return fig


# ────────────────────────────────────────────────────────────────
# 3. 수치형 변수
# ────────────────────────────────────────────────────────────────
def plot_numeric_dist(df=None, cols=None):
    """이탈/유지 분포를 겹쳐 그린 히스토그램"""
    df = load_data() if df is None else df
    cols = cols or NUMERIC

    fig = _new_figure(15, 7)
    axes = fig.subplots(2, 3)
    for ax, col in zip(axes.ravel(), cols):
        sns.histplot(data=df, x=col, hue=ft.TARGET, bins=30, kde=True,
                     palette=PALETTE, alpha=0.5, ax=ax)
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("")
    fig.suptitle("수치형 변수 분포 (파랑=유지, 빨강=이탈)", fontsize=13)
    return fig


def plot_numeric_box(df=None, cols=None):
    """이탈 여부별 박스플롯"""
    df = load_data() if df is None else df
    cols = cols or NUMERIC

    fig = _new_figure(15, 7)
    axes = fig.subplots(2, 3)
    for ax, col in zip(axes.ravel(), cols):
        sns.boxplot(data=df, x=ft.TARGET, y=col, hue=ft.TARGET,
                    palette=PALETTE, legend=False, ax=ax)
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["유지", "이탈"])
    fig.suptitle("이탈 여부별 수치형 변수 분포", fontsize=13)
    return fig


def mean_by_churn(df=None, cols=None):
    df = load_data() if df is None else df
    cols = cols or NUMERIC
    out = df.groupby(ft.TARGET)[cols].mean().T.round(3)
    out.columns = ["유지(0)", "이탈(1)"]
    out["차이"] = (out["이탈(1)"] - out["유지(0)"]).round(3)
    return out


# ────────────────────────────────────────────────────────────────
# 4. 범주형 이탈률
# ────────────────────────────────────────────────────────────────
def churn_rate_table(df=None, cols=None):
    """변수별 값별 이탈률 + 최대-최소 격차"""
    df = load_data() if df is None else df
    cols = cols or (BINARY + CATEGORICAL)

    rows = []
    for col in cols:
        grouped = df.groupby(col)[ft.TARGET].agg(["count", "mean"])
        for value, row in grouped.iterrows():
            rows.append({"변수": col, "값": value, "인원": int(row["count"]),
                         "이탈률(%)": round(row["mean"] * 100, 2)})

    out = pd.DataFrame(rows)
    out["격차(%p)"] = out.groupby("변수")["이탈률(%)"].transform(
        lambda s: round(s.max() - s.min(), 2))
    return out.sort_values(["격차(%p)", "변수"], ascending=[False, True], ignore_index=True)


def plot_churn_rate(df=None, cols=None):
    """변수별 이탈률 막대 (전체 평균 기준선 포함)"""
    df = load_data() if df is None else df
    cols = cols or (BINARY + CATEGORICAL)
    overall = df[ft.TARGET].mean() * 100

    n_cols = 4
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig = _new_figure(4.2 * n_cols, 3.5 * n_rows)
    axes = np.atleast_1d(fig.subplots(n_rows, n_cols)).ravel()

    for ax, col in zip(axes, cols):
        rate = df.groupby(col)[ft.TARGET].agg(["count", "mean"])
        rate["mean"] *= 100
        sns.barplot(x=rate.index.astype(str), y=rate["mean"], hue=rate.index.astype(str),
                    palette="Blues_d", legend=False, ax=ax)
        ax.axhline(overall, color=COLOR_CHURN, linestyle="--", linewidth=1)
        for i, (value, count) in enumerate(zip(rate["mean"], rate["count"])):
            ax.text(i, value + 1, f"{value:.1f}%\n(n={count:,})",
                    ha="center", va="bottom", fontsize=8)
        ax.set_title(col, fontsize=11)
        ax.set_ylabel("이탈률(%)")
        ax.set_xlabel("")
        ax.set_ylim(0, 55)

    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle(f"변수별 이탈률 (빨간 점선 = 전체 평균 {overall:.1f}%)", fontsize=13)
    return fig


# ────────────────────────────────────────────────────────────────
# 5. 상관관계 / 다중공선성
# ────────────────────────────────────────────────────────────────
def correlation(df=None):
    df = load_data() if df is None else df
    return df.corr(numeric_only=True)


def plot_corr_heatmap(df=None):
    corr = correlation(df)
    fig = _new_figure(11, 9)
    ax = fig.subplots()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.5, annot_kws={"size": 8}, ax=ax)
    ax.set_title("상관계수 히트맵", fontsize=13)
    return fig


def churn_correlation(df=None):
    corr = correlation(df)[ft.TARGET].drop(ft.TARGET)
    return corr.sort_values(ascending=False).round(3)


def plot_churn_correlation(df=None):
    corr = churn_correlation(df).sort_values()
    fig = _new_figure(8, 5)
    ax = fig.subplots()
    ax.barh(corr.index, corr.values,
            color=[COLOR_CHURN if v > 0 else COLOR_STAY for v in corr])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Churn과의 상관계수")
    ax.set_xlabel("상관계수")
    return fig


def vif_table(df=None):
    """설명변수별 VIF (분산팽창계수) → Series

    상수항을 넣고 계산한다. 빼면 절편이 맡아야 할 평균이 각 변수에 흡수되어
    겹치지 않는 변수까지 VIF가 부풀려진다(`Phone` 1.00 → 9.36).
    """
    df = load_data() if df is None else df
    X = add_constant(df.drop(columns=[ft.TARGET]))
    vif = [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])]
    return pd.Series(vif, index=X.columns[1:]).round(2)


def multicollinear_pairs(df=None, threshold=0.5):
    """설명변수 간 |r| >= threshold 인 쌍 — 상관계수와 각 변수의 VIF를 함께 낸다.

    상관계수는 두 변수만 보고, VIF는 나머지 변수 전부를 함께 본다.
    둘을 나란히 둬야 '이 쌍이 얼마나 겹치는가'를 두 각도에서 읽을 수 있다.
    """
    df = load_data() if df is None else df
    corr = correlation(df).drop(index=ft.TARGET, columns=ft.TARGET).abs()
    vif = vif_table(df)
    rows = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            if corr.loc[a, b] >= threshold:
                rows.append({"변수1": a, "변수2": b,
                             "상관계수": round(corr.loc[a, b], 3),
                             "변수1 VIF": vif[a], "변수2 VIF": vif[b]})
    return pd.DataFrame(rows).sort_values("상관계수", ascending=False, ignore_index=True)


# ────────────────────────────────────────────────────────────────
# 6. 유지 기간 / 교차 분석
# ────────────────────────────────────────────────────────────────
LIFETIME_BINS = [-0.1, 0.5, 1.5, 3.5, 6.5, np.inf]
LIFETIME_LABELS = ["0개월(신규)", "1개월", "2~3개월", "4~6개월", "7개월+"]


def lifetime_rate(df=None):
    """유지 기간 구간별 이탈률"""
    df = load_data() if df is None else df
    binned = pd.cut(df[ft.TIMING_TARGET], bins=LIFETIME_BINS, labels=LIFETIME_LABELS)
    out = df.groupby(binned, observed=True)[ft.TARGET].agg(["count", "mean"])
    out["mean"] = (out["mean"] * 100).round(1)
    out.columns = ["인원", "이탈률(%)"]
    return out


def plot_lifetime_rate(df=None):
    rate = lifetime_rate(df)
    fig = _new_figure(7, 4)
    ax = fig.subplots()
    sns.barplot(x=rate.index, y=rate["이탈률(%)"], hue=rate.index,
                palette="Reds_r", legend=False, ax=ax)
    for i, value in enumerate(rate["이탈률(%)"]):
        ax.text(i, value + 1, f"{value:.1f}%", ha="center")
    ax.set_title("회원 유지 기간별 이탈률")
    ax.set_ylabel("이탈률(%)")
    ax.set_xlabel("")
    ax.set_ylim(0, 95)
    return fig


def contract_group_cross(df=None, metric="rate"):
    """계약 기간 × 그룹수업 참여 교차표 — metric='rate' 이탈률(%) / 'count' 인원

    이탈률은 칸마다 분모(그 조합의 인원)가 다르다. 분모를 함께 보려면 'count'를 쓴다.
    """
    df = load_data() if df is None else df
    if metric == "count":
        return df.pivot_table(index="Contract_period", columns="Group_visits",
                              values=ft.TARGET, aggfunc="size")
    return (df.pivot_table(index="Contract_period", columns="Group_visits",
                           values=ft.TARGET, aggfunc="mean") * 100).round(1)


def plot_contract_group(df=None):
    cross = contract_group_cross(df)
    fig = _new_figure(6, 4)
    ax = fig.subplots()
    sns.heatmap(cross, annot=True, fmt=".1f", cmap="Reds",
                cbar_kws={"label": "이탈률(%)"}, ax=ax)
    ax.set_title("계약 기간 × 그룹수업 참여별 이탈률(%)")
    ax.set_xlabel("그룹수업 참여")
    ax.set_ylabel("계약 기간(개월)")
    return fig


def age_lifetime_cross(df=None, metric="rate"):
    """연령대 × 유지기간 교차표 — 9.7절 (트랙 D 타깃팅 근거)

    metric='rate' 이탈률(%) / 'count' 인원. contract_group_cross와 같은 규칙이다.
    """
    df = load_data() if df is None else df
    lifetime = pd.cut(df[ft.TIMING_TARGET], [-0.1, 1.5, 3.5, 6.5, np.inf],
                      labels=["0~1개월", "2~3개월", "4~6개월", "7개월+"])
    age = pd.cut(df["Age"], [17, 27, 30, 45], labels=["~27세", "28~30세", "31세+"])
    if metric == "count":
        return df.pivot_table(index=lifetime, columns=age, values=ft.TARGET,
                              aggfunc="size", observed=True)
    return (df.pivot_table(index=lifetime, columns=age, values=ft.TARGET,
                           aggfunc="mean", observed=True) * 100).round(1)


def zero_visit_rate(df=None):
    """직전 달 방문이 0인 비율 — Churn 정의 확인용 (9.5절)"""
    df = load_data() if df is None else df
    rate = df.groupby(ft.TARGET)[ft.LEAK_SUSPECT].apply(lambda s: (s == 0).mean() * 100).round(2)
    return pd.DataFrame({"구분": ["유지(0)", "이탈(1)"], "직전 달 방문 0인 비율(%)": rate.values})


# ────────────────────────────────────────────────────────────────
# 통합
# ────────────────────────────────────────────────────────────────
def all_tables(df=None):
    """페이지에서 쓸 요약표 모음"""
    df = load_data() if df is None else df
    return {
        "기본 정보": basic_stats(df),
        "컬럼 개요": overview(df),
        "기술통계": describe(df),
        "타깃 분포": target_summary(df),
        "이탈 여부별 평균": mean_by_churn(df),
        "변수별 이탈률": churn_rate_table(df),
        "Churn 상관계수": churn_correlation(df).to_frame("상관계수"),
        "다중공선성 쌍": multicollinear_pairs(df),
        "유지기간별 이탈률": lifetime_rate(df),
        "계약×그룹수업": contract_group_cross(df),
        "연령×유지기간": age_lifetime_cross(df),
        "직전달 방문 0 비율": zero_visit_rate(df),
    }


def all_figures(df=None):
    """페이지에서 쓸 그래프 모음"""
    df = load_data() if df is None else df
    return {
        "타깃 분포": plot_target(df),
        "수치형 분포": plot_numeric_dist(df),
        "이탈별 박스플롯": plot_numeric_box(df),
        "변수별 이탈률": plot_churn_rate(df),
        "상관 히트맵": plot_corr_heatmap(df),
        "Churn 상관": plot_churn_correlation(df),
        "유지기간별 이탈률": plot_lifetime_rate(df),
        "계약×그룹수업": plot_contract_group(df),
    }


if __name__ == "__main__":
    font = set_style()
    print(f"적용 폰트: {font or '없음 (한글 깨짐 주의)'}\n")

    df = load_data()
    for name, table in all_tables(df).items():
        print("=" * 70)
        print(name)
        print("=" * 70)
        print(table.to_string())
        print()

    figures = all_figures(df)
    print("=" * 70)
    print(f"Figure {len(figures)}개 생성: {list(figures)}")
