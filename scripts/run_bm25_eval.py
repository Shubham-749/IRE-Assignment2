#!/usr/bin/env python
"""Q2: BM25 candidate generation + recall@K evaluation.

    python scripts/run_bm25_eval.py --dataset all --scale demo

For each dataset: build an inverted-index BM25 over the article corpus, build a
query per validation impression from the user's point-in-time click history titles,
retrieve top-K candidates from the *full* corpus, and report recall@K -- how often the
article the user actually clicked shows up in those top-K candidates. Impressions with
no prior history (cold-start) can't build a query, so they're counted and reported
separately rather than silently dropped.
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import polars as pl  # noqa: E402

from ire_a1.feature_store import UserHistoryIndex  # noqa: E402
from ire_a1.retrieval.bm25 import BM25Index  # noqa: E402

_MIND_SCALE_FALLBACK = {"demo": "small"}
KS = (50, 100, 200)
MAX_HISTORY_TITLES = 20
SAMPLE_SIZE = 5000
SEED = 42


def resolve_scale(dataset: str, scale: str) -> str:
    if dataset == "mind" and scale in _MIND_SCALE_FALLBACK:
        return _MIND_SCALE_FALLBACK[scale]
    return scale


def evaluate(dataset: str, scale: str) -> None:
    d = REPO_ROOT / "data" / "processed" / dataset / scale
    articles = pl.read_parquet(d / "articles.parquet")
    user_history = pl.read_parquet(d / "user_history.parquet")
    behaviors = pl.read_parquet(d / "behaviors_val.parquet")

    print(f"\n=== {dataset}/{scale} ===")
    t0 = time.time()
    bm25 = BM25Index().build(articles)
    print(f"BM25 index: {bm25.n_docs:,} docs, {len(bm25.postings):,} terms ({time.time()-t0:.1f}s)")

    hist_index = UserHistoryIndex(user_history, articles)
    print(f"UserHistoryIndex: {len(hist_index._by_user):,} users ({time.time()-t0:.1f}s total)")

    has_click = behaviors.filter(
        pl.col("clicked_article_ids").is_not_null() & (pl.col("clicked_article_ids").list.len() > 0)
    )
    print(f"val impressions with >=1 click: {has_click.height:,} / {behaviors.height:,}")

    eligible_rows = []
    n_cold_start = 0
    for row in has_click.iter_rows(named=True):
        titles = hist_index.recent_titles(row["user_id"], row["timestamp"], max_n=MAX_HISTORY_TITLES)
        if not titles:
            n_cold_start += 1
            continue
        eligible_rows.append((row["user_id"], row["timestamp"], row["clicked_article_ids"], titles))

    print(f"eligible (has click + non-empty history): {len(eligible_rows):,}  "
          f"| skipped cold-start (no prior history): {n_cold_start:,}")

    import random

    rng = random.Random(SEED)
    sample = eligible_rows if len(eligible_rows) <= SAMPLE_SIZE else rng.sample(eligible_rows, SAMPLE_SIZE)
    print(f"evaluating on {len(sample):,} sampled impressions (seed={SEED})")

    max_k = max(KS)
    hits = {k: 0.0 for k in KS}
    t0 = time.time()
    for _user_id, _ts, clicked_ids, titles in sample:
        query = " ".join(titles)
        retrieved = [aid for aid, _score in bm25.search(query, top_k=max_k)]
        clicked_set = set(clicked_ids)
        for k in KS:
            topk_set = set(retrieved[:k])
            hits[k] += len(clicked_set & topk_set) / len(clicked_set)
    elapsed = time.time() - t0
    print(f"retrieval for {len(sample):,} impressions took {elapsed:.1f}s "
          f"({elapsed / max(len(sample), 1) * 1000:.1f}ms/impression)")

    print(f"\n{'K':>6} | {'recall@K':>10}")
    for k in KS:
        recall = hits[k] / len(sample) if sample else float("nan")
        print(f"{k:>6} | {recall:>10.4f}")


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
