#!/usr/bin/env python
"""Q4: serving & scale analysis for the GBDT re-ranker -- index memory, p99
single-request latency, back-of-envelope cost/QPS at a target SLA.

    python scripts/reranker_scale_analysis.py --dataset all --scale demo

A single "request" here is the full two-stage pipeline for one user: embedding-search
candidate generation (top_k=200, matching Q2's K~100-200 range -- brute-force cosine
sim is what embeddings.py already recommends at this catalog size), Q1 feature
building for those candidates, then GBDT scoring. Measured over N=200 repeated
single-impression runs (not batched -- batching would understate the per-request cost
a live serving path actually pays). Memory is measured via pickle-serialized byte size
(a reasonable proxy for in-memory footprint given Python's real object-graph overhead
is hard to measure exactly) except for the embedding matrix, whose numpy .nbytes is
exact. Writes results/reranker_scale_analysis.csv; the 10x scaling argument itself is
prose, in the notebook's Q4 section, not this script's output.
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from ire_a1.feature_store import UserHistoryIndex  # noqa: E402
from ire_a1.retrieval import eval_utils  # noqa: E402
from ire_a1.retrieval.bm25 import BM25Index  # noqa: E402
from ire_a1.retrieval.embeddings import EmbeddingIndex, mean_pool  # noqa: E402
from ire_a2.features import FEATURE_COLUMNS, MAX_HISTORY, FeatureBuilder  # noqa: E402
from ire_a2.reranker import build_training_frame, train_gbdt  # noqa: E402
from run_eval_harness import resolve_scale  # noqa: E402
from train_reranker import build_sample_with_id  # noqa: E402

N_REQUESTS = 200
RETRIEVE_K = 200
TARGET_SLA_MS = 100.0
ASSUMED_COST_PER_VCPU_HOUR_USD = 0.05  # stated assumption, typical small cloud instance


def pickled_bytes(obj) -> int:
    return len(pickle.dumps(obj))


def measure_memory(bm25: BM25Index, emb_index: EmbeddingIndex, feature_builder: FeatureBuilder) -> dict:
    bm25_bytes = pickled_bytes(bm25.postings) + pickled_bytes(bm25.doc_terms) + pickled_bytes(bm25.idf)
    embedding_bytes = emb_index.matrix.nbytes if emb_index.matrix is not None else 0
    popularity_bytes = pickled_bytes(feature_builder._popularity)
    category_bytes = pickled_bytes(feature_builder._category)
    return {
        "bm25_index_mb": bm25_bytes / 1e6,
        "embedding_matrix_mb": embedding_bytes / 1e6,
        "feature_store_mb": (popularity_bytes + category_bytes) / 1e6,
    }


def time_single_request(
    imp: dict,
    bm25: BM25Index,
    emb_index: EmbeddingIndex,
    feature_builder: FeatureBuilder,
    hist_index: UserHistoryIndex,
    model,
    user_id: str,
    timestamp,
) -> tuple[float, float, float]:
    """Returns (retrieval_ms, feature_build_ms, score_ms) for one simulated request."""
    t0 = time.perf_counter()
    history_vecs = [v for aid in imp["history_ids"] if (v := emb_index.get_embedding(aid)) is not None]
    query_vec = mean_pool(history_vecs)
    candidates = (
        [aid for aid, _score in emb_index.search(query_vec, top_k=RETRIEVE_K)]
        if query_vec is not None
        else imp["candidates"]
    )
    t1 = time.perf_counter()

    history = hist_index.recent(user_id, timestamp, MAX_HISTORY)
    rows = feature_builder.build_rows(user_id, timestamp, candidates, history)
    t2 = time.perf_counter()

    X = [[r[c] for c in FEATURE_COLUMNS] for r in rows]
    model.predict(X)
    t3 = time.perf_counter()

    return (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000


def run(dataset: str, scale: str, csv_rows: list[dict]) -> None:
    d = REPO_ROOT / "data" / "processed" / dataset / scale
    raw_dir = REPO_ROOT / "data" / "raw"
    feat_train_path = d / "reranker_features_train.parquet"
    if not feat_train_path.exists():
        raise FileNotFoundError(f"{feat_train_path} missing -- run scripts/build_reranker_features.py first")

    articles = pl.read_parquet(d / "articles.parquet")
    user_history = pl.read_parquet(d / "user_history.parquet")
    behaviors_val = pl.read_parquet(d / "behaviors_val.parquet")
    behaviors_train = pl.read_parquet(d / "behaviors_train.parquet")
    embeddings_df = pl.read_parquet(d / "article_embeddings.parquet")
    features_train = pl.read_parquet(feat_train_path)

    print(f"\n=== {dataset}/{scale} ===")
    bm25 = BM25Index().build(articles)
    emb_index = EmbeddingIndex().build(embeddings_df)
    hist_index = UserHistoryIndex(user_history, articles)
    feature_builder = FeatureBuilder(dataset, articles, bm25, emb_index, behaviors_train, raw_dir, scale)

    train_frame = build_training_frame(features_train)
    model = train_gbdt(train_frame, FEATURE_COLUMNS)

    mem = measure_memory(bm25, emb_index, feature_builder)
    print(f"memory: BM25 index {mem['bm25_index_mb']:.1f} MB | "
          f"embedding matrix {mem['embedding_matrix_mb']:.1f} MB | "
          f"feature store {mem['feature_store_mb']:.1f} MB")

    sample = build_sample_with_id(behaviors_val, hist_index)
    requests = eval_utils.sample_impressions(sample, sample_size=N_REQUESTS, seed=eval_utils.SEED)
    print(f"timing {len(requests):,} single-request runs (candidate gen -> feature build -> GBDT score)")

    retrieval_ms, feature_ms, score_ms, total_ms = [], [], [], []
    for imp in requests:
        r, f, s = time_single_request(
            imp, bm25, emb_index, feature_builder, hist_index, model,
            imp["user_id"], imp["timestamp"],
        )
        retrieval_ms.append(r)
        feature_ms.append(f)
        score_ms.append(s)
        total_ms.append(r + f + s)

    def pctl(vals, p):
        return float(np.percentile(vals, p))

    print(f"\n{'stage':>16} | {'p50 (ms)':>9} | {'p99 (ms)':>9}")
    for name, vals in (
        ("retrieval", retrieval_ms), ("feature build", feature_ms),
        ("gbdt score", score_ms), ("TOTAL", total_ms),
    ):
        print(f"{name:>16} | {pctl(vals, 50):>9.2f} | {pctl(vals, 99):>9.2f}")

    p99_total_s = pctl(total_ms, 99) / 1000.0
    single_core_qps = 1.0 / p99_total_s if p99_total_s > 0 else float("inf")
    meets_sla = pctl(total_ms, 99) < TARGET_SLA_MS
    cost_per_1k = (1000 / single_core_qps) / 3600 * ASSUMED_COST_PER_VCPU_HOUR_USD

    print(f"\nsingle-core capacity: {single_core_qps:.1f} req/s (at measured p99 latency)")
    print(f"meets {TARGET_SLA_MS:.0f}ms p99 SLA on a single core: {meets_sla}")
    print(f"back-of-envelope cost per 1,000 queries: ${cost_per_1k:.4f} "
          f"(@ ${ASSUMED_COST_PER_VCPU_HOUR_USD}/vCPU-hour, single core, stated assumption)")

    for name, vals in (
        ("retrieval_ms", retrieval_ms), ("feature_build_ms", feature_ms),
        ("gbdt_score_ms", score_ms), ("total_ms", total_ms),
    ):
        csv_rows.append({
            "dataset": dataset, "scale": scale, "metric": name,
            "p50": pctl(vals, 50), "p99": pctl(vals, 99), "n": len(vals),
        })
    for name, value in mem.items():
        csv_rows.append({"dataset": dataset, "scale": scale, "metric": name, "p50": value, "p99": value, "n": 1})
    csv_rows.append({
        "dataset": dataset, "scale": scale, "metric": "single_core_qps",
        "p50": single_core_qps, "p99": single_core_qps, "n": 1,
    })
    csv_rows.append({
        "dataset": dataset, "scale": scale, "metric": "cost_per_1k_queries_usd",
        "p50": cost_per_1k, "p99": cost_per_1k, "n": 1,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    parser.add_argument("--scale", choices=["demo", "small", "large"], default="demo")
    args = parser.parse_args()

    datasets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]
    csv_rows: list[dict] = []
    for dataset in datasets:
        scale = resolve_scale(dataset, args.scale)
        run(dataset, scale, csv_rows)

    out_path = REPO_ROOT / "results" / "reranker_scale_analysis.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(csv_rows).write_csv(out_path)
    print(f"\nwrote {out_path} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
