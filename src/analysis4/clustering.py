"""분석 4 트랙 C — 군집화 (이탈군 / 이탈 위험군 / 유지군)

노트북 10절 트랙 C 설계를 구현한다.

피처는 features.CLUSTER_FEATURES 9축을 쓴다 (9.1 / 9.8절).
    - `Churn` 제외        : 정답 누출. 사후 검증에만 사용한다.
    - `gender`, `Phone`   : 무신호. 제거 시 실루엣 0.1538 -> 0.1886 (+23%)
    - 다중공선성 쌍의 한쪽 : Month_to_end_contract, 직전달 방문빈도

⚠ 알려진 한계 (9.8절)
    k=3이 실루엣 최댓값(0.1886)이지만 **절대값이 낮아** 군집 구조가 뚜렷하지 않다.
    특히 이탈군(39.9%)과 위험군(33.6%)의 격차가 6.3%p로 작다.
    → k=4를 비교군으로 병기하고, 실무 등급은 트랙 A 예측확률과 결합해 정한다.

Streamlit에 의존하지 않는다. DataFrame / dict 만 반환한다.

CLI 실행 (프로젝트 루트에서):
    python -m src.analysis4.clustering
"""

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common import SEED
from src.analysis4 import features as ft

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "analysis4"

# 이탈률이 낮은 군집부터 순서대로 붙는 등급명
TIER_NAMES = ["유지군", "이탈 위험군", "이탈군"]

DEFAULT_K = 3


def build_pipeline(k=DEFAULT_K):
    """스케일러 + KMeans

    StandardScaler는 선택이 아니라 필수다. 적용하지 않으면
    Avg_additional_charges_total(0~552)이 나머지(한 자릿수)를 압도해
    그 변수 하나로만 군집이 갈린다.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, random_state=SEED, n_init=10)),
    ])


def search_k(df=None, k_range=range(2, 9)):
    """k 탐색 → 실루엣 / inertia / 군집별 이탈률 (노트북 9.8절 재현)"""
    ft.set_seed()
    df = ft.load_data() if df is None else df
    X = ft.get_cluster_X(df)
    Z = StandardScaler().fit_transform(X)

    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(Z)
        rates = (df.groupby(km.labels_)[ft.TARGET].mean() * 100).round(1)
        rows.append({
            "k": k,
            "실루엣": silhouette_score(Z, km.labels_),
            "inertia": km.inertia_,
            "군집별_이탈률": sorted(rates.tolist()),
            "최소최대_격차": round(rates.max() - rates.min(), 1),
        })
    return pd.DataFrame(rows)


def fit_kmeans(df=None, k=DEFAULT_K):
    """군집 적합 → (labels, pipeline, 실루엣)"""
    ft.set_seed()
    df = ft.load_data() if df is None else df
    X = ft.get_cluster_X(df)

    pipe = build_pipeline(k).fit(X)
    labels = pipe.named_steps["kmeans"].labels_
    sil = silhouette_score(pipe.named_steps["scaler"].transform(X), labels)
    return labels, pipe, float(sil)


def label_tiers(df=None, labels=None, n_tiers=3):
    """군집을 실제 이탈률 순으로 정렬해 등급명 부여 → (등급 Series, 매핑 DataFrame)

    군집 번호는 KMeans가 임의로 매기므로 그대로 쓰면 의미가 없다.
    **실제 이탈률로 정렬한 뒤** 유지군 / 이탈 위험군 / 이탈군을 붙인다.
    k가 3보다 크면 이탈률 순으로 인접 군집을 묶어 3등급으로 만든다.
    """
    df = ft.load_data() if df is None else df
    if labels is None:
        labels, _, _ = fit_kmeans(df)

    rate = df.groupby(labels)[ft.TARGET].agg(["size", "mean"])
    rate.columns = ["인원", "이탈률"]
    rate = rate.sort_values("이탈률")

    # 이탈률 오름차순 군집을 n_tiers 그룹으로 나눠 등급 부여
    groups = np.array_split(rate.index.to_numpy(), n_tiers)
    cluster_to_tier = {c: TIER_NAMES[i] for i, grp in enumerate(groups) for c in grp}

    mapping = (rate.assign(등급=[cluster_to_tier[c] for c in rate.index])
               .reset_index(names="군집")
               .assign(이탈률=lambda t: (t["이탈률"] * 100).round(1)))
    tiers = pd.Series([cluster_to_tier[c] for c in labels], index=df.index, name="등급")
    return tiers, mapping


def profile_clusters(df=None, labels=None):
    """군집별 변수 평균 프로파일 + 이탈률"""
    df = ft.load_data() if df is None else df
    if labels is None:
        labels, _, _ = fit_kmeans(df)

    prof = df.groupby(labels)[ft.CLUSTER_FEATURES].mean().T.round(2)
    prof.columns = [f"군집{c}" for c in prof.columns]

    extra = pd.DataFrame({
        f"군집{c}": [int((labels == c).sum()),
                    round(df.loc[labels == c, ft.TARGET].mean() * 100, 1)]
        for c in sorted(set(labels))
    }, index=["인원", "이탈률(%)"])
    return pd.concat([extra, prof])


def compare_hierarchical(df=None, k=DEFAULT_K):
    """계층적 군집(Ward)과 비교 → KMeans 결과가 안정적인지 검증"""
    ft.set_seed()
    df = ft.load_data() if df is None else df
    X = ft.get_cluster_X(df)
    Z = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(Z)
    hc = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)

    return {
        "ARI": float(adjusted_rand_score(km, hc)),   # 1.0이면 동일 분할
        "kmeans_실루엣": float(silhouette_score(Z, km)),
        "계층적_실루엣": float(silhouette_score(Z, hc)),
        "계층적_이탈률": sorted((df.groupby(hc)[ft.TARGET].mean() * 100).round(1).tolist()),
    }


def pca_2d(df=None, labels=None):
    """PCA 2차원 좌표 → 군집 분리도 시각화용 DataFrame"""
    ft.set_seed()
    df = ft.load_data() if df is None else df
    if labels is None:
        labels, _, _ = fit_kmeans(df)

    X = ft.get_cluster_X(df)
    Z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(Z)

    tiers, _ = label_tiers(df, labels)
    out = pd.DataFrame(coords, columns=["PC1", "PC2"], index=df.index)
    out["군집"] = labels
    out["등급"] = tiers.values
    out["이탈"] = df[ft.TARGET].values
    out.attrs["설명분산비"] = pca.explained_variance_ratio_.round(4).tolist()
    return out


def bundle_path(k=DEFAULT_K):
    return MODEL_DIR / f"cluster_k{k}.joblib"


def train_and_save(df=None, k=DEFAULT_K):
    """군집 적합 후 파이프라인 + 등급 매핑 저장 → 요약 dict"""
    ft.set_seed()
    df = ft.load_data() if df is None else df
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    labels, pipe, sil = fit_kmeans(df, k)
    tiers, mapping = label_tiers(df, labels)

    bundle = {
        "pipeline": pipe,
        "features": list(ft.CLUSTER_FEATURES),
        "k": k,
        "silhouette": sil,
        "tier_map": dict(zip(mapping["군집"], mapping["등급"])),
        "mapping": mapping,
        "seed": SEED,
        "n": int(len(df)),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, bundle_path(k))

    return {"labels": labels, "tiers": tiers, "mapping": mapping,
            "silhouette": sil, "path": bundle_path(k)}


def load_bundle(k=DEFAULT_K):
    path = bundle_path(k)
    if not path.exists():
        raise FileNotFoundError(
            f"군집 모델이 없습니다: {path}\n"
            "먼저 학습하세요: python -m src.analysis4.clustering"
        )
    return joblib.load(path)


def predict_tier(bundle, df):
    """새 데이터에 군집·등급 부여 (피처 순서 검증 포함)"""
    expected = bundle["features"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise KeyError(f"입력에 없는 피처: {missing}")
    labels = bundle["pipeline"].predict(df[expected])
    return pd.Series([bundle["tier_map"][c] for c in labels], index=df.index, name="등급")


def run_all(df=None, k=DEFAULT_K):
    """트랙 C 전체 — k 탐색 + 군집 + 등급 부여 + 검증

    반환: dict — 'summary'(long 형식, 트랙 통합용) + 상세표들
    """
    ft.set_seed()
    df = ft.load_data() if df is None else df

    result = train_and_save(df, k)
    ks = search_k(df)
    hier = compare_hierarchical(df, k)
    mapping = result["mapping"]
    spread = float(mapping["이탈률"].max() - mapping["이탈률"].min())

    # long 형식 요약 (features.SUMMARY_COLUMNS 규약)
    rows = [
        {"트랙": "C 군집", "단계": ft.STAGE_BASE, "구성": f"k={k}",
         "모델": "kmeans", "지표": "실루엣", "값": result["silhouette"]},
        {"트랙": "C 군집", "단계": ft.STAGE_BASE, "구성": f"k={k}",
         "모델": "kmeans", "지표": "이탈률_격차(%p)", "값": spread},
        {"트랙": "C 군집", "단계": ft.STAGE_BASE, "구성": f"k={k}",
         "모델": "ward", "지표": "실루엣", "값": hier["계층적_실루엣"]},
        {"트랙": "C 군집", "단계": ft.STAGE_BASE, "구성": f"k={k}",
         "모델": "kmeans vs ward", "지표": "ARI", "값": hier["ARI"]},
    ]

    return {
        "summary": ft.make_summary(rows),
        "search_k": ks,
        "silhouette": result["silhouette"],
        "mapping": mapping,
        "profile": profile_clusters(df, result["labels"]),
        "hierarchical": hier,
        "pca": pca_2d(df, result["labels"]),
        "tiers": result["tiers"],
    }


if __name__ == "__main__":
    ft.set_seed()
    r = run_all()

    print("=" * 78)
    print("트랙 C — k 탐색 (9.8절 재현)")
    print("=" * 78)
    print(r["search_k"].round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print(f"k={DEFAULT_K} 등급 매핑 (실루엣 {r['silhouette']:.4f})")
    print("=" * 78)
    print(r["mapping"].to_string(index=False))

    print("\n" + "=" * 78)
    print("군집 프로파일")
    print("=" * 78)
    print(r["profile"].to_string())

    print("\n" + "=" * 78)
    print("계층적 군집(Ward)과 비교 — 안정성 검증")
    print("=" * 78)
    for key, val in r["hierarchical"].items():
        print(f"  {key}: {val}")

    print("\nPCA 설명분산비:", r["pca"].attrs["설명분산비"])
    print("등급 분포:")
    print(r["tiers"].value_counts().to_string())

    print("\n" + "=" * 78)
    print(f"통합 요약 (long) — {len(r['summary'])}행")
    print("=" * 78)
    print(r["summary"].round(4).to_string(index=False))
    print(f"\n저장 위치: {bundle_path()}")
