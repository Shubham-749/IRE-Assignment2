#!/usr/bin/env python
"""Q3: semantic (embedding) candidate generation + recall@K evaluation.

    python scripts/run_embedding_eval.py --dataset all --scale demo

Mirrors run_bm25_eval.py exactly (same sampled impressions, same eligibility rule,
same K's) via retrieval.eval_utils, so the two are directly comparable -- see
compare_retrieval.py. Requires `python scripts/compute_embeddings.py` to have been run
first for the dataset/scale.
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import polars as pl  # noqa: E402

from ire_a1.feature_store import UserHistoryIndex  # noqa: E402
from ire_a1.retrieval import eval_utils  # noqa: E402
from ire_a1.retrieval.embeddings import EmbeddingIndex, mean_pool  # noqa: E402

_MIND_SCALE_FALLBACK = {"demo": "small"}


def resolve_scale(dataset: str, scale: str) -> str:
    if dataset == "mind" and scale in _MIND_SCALE_FALLBACK:
        return _MIND_SCALE_FALLBACK[scale]
    return scale


def evaluate(dataset: str, scale: str) -> tuple[dict, list]:
    d = REPO_ROOT / "data" / "processed" / dataset / scale
    emb_path = d / "article_embeddings.parquet"
    if not emb_path.exists():
        raise FileNotFoundError(f"{emb_path} missing -- run scripts/compute_embeddings.py first")

    articles = pl.read_parquet(d / "articles.parquet")
    user_history = pl.read_parquet(d / "user_history.parquet")
    behaviors = pl.read_parquet(d / "behaviors_val.parquet")
    embeddings_df = pl.read_parquet(emb_path)

    print(f"\n=== {dataset}/{scale} ===")
    t0 = time.time()
    emb_index = EmbeddingIndex().build(embeddings_df)
    print(f"EmbeddingIndex: {len(emb_index.article_ids):,} docs, dim={emb_index.matrix.shape[1]} "
          f"({time.time()-t0:.1f}s)")

    hist_index = UserHistoryIndex(user_history, articles)
    print(f"UserHistoryIndex: {len(hist_index._by_user):,} users ({time.time()-t0:.1f}s total)")

    eligible, n_cold_start, n_with_click = eval_utils.eligible_impressions(
        behaviors, hist_index.recent_article_ids
    )
    print(f"val impressions with >=1 click: {n_with_click:,} / {behaviors.height:,}")
    print(f"eligible (has click + non-empty history): {len(eligible):,}  "
          f"| skipped cold-start (no prior history): {n_cold_start:,}")

    sample = eval_utils.sample_impressions(eligible)
    print(f"evaluating on {len(sample):,} sampled impressions (seed={eval_utils.SEED})")

    def retrieve(article_ids: list[str], top_k: int) -> list[str]:
        vectors = [v for aid in article_ids if (v := emb_index.get_embedding(aid)) is not None]
        query_vec = mean_pool(vectors)
        if query_vec is None:
            return []
        return [aid for aid, _score in emb_index.search(query_vec, top_k=top_k)]

    t0 = time.time()
    means, records = eval_utils.score_recall_at_k(sample, retrieve)
    elapsed = time.time() - t0
    print(f"retrieval for {len(sample):,} impressions took {elapsed:.1f}s "
          f"({elapsed / max(len(sample), 1) * 1000:.1f}ms/impression)")

    print(f"\n{'K':>6} | {'recall@K':>10}")
    for k in eval_utils.KS:
        print(f"{k:>6} | {means[k]:>10.4f}")

    return means, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    parser.add_argument("--scale", choices=["demo", "small", "large"], default="demo")
    args = parser.parse_args()

    datasets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        scale = resolve_scale(dataset, args.scale)
        evaluate(dataset, scale)


if __name__ == "__main__":
    main()
