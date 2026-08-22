#!/usr/bin/env python
"""Q3.5: lexical (BM25) vs semantic (embeddings) retrieval, head to head.

    python scripts/compare_retrieval.py --dataset all --scale demo

Runs both retrievers on the *same* sampled val impressions (same seed, same
eligibility: has a click + non-empty history for both methods) and reports recall@K
side by side, plus one slice preview -- short history (<5 prior clicks) vs. long
history (>=5) -- since the assignment asks "which works better, on which slices."
Full cold-start/warm + bootstrap-CI slicing is Q4's job; this is a lightweight preview.
Requires scripts/compute_embeddings.py to have already been run for the dataset/scale.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import polars as pl  # noqa: E402

from ire_a1.feature_store import UserHistoryIndex  # noqa: E402
from ire_a1.retrieval import eval_utils  # noqa: E402
from ire_a1.retrieval.bm25 import BM25Index  # noqa: E402
from ire_a1.retrieval.embeddings import EmbeddingIndex, mean_pool  # noqa: E402

_MIND_SCALE_FALLBACK = {"demo": "small"}
SHORT_HISTORY_CUTOFF = 5


def resolve_scale(dataset: str, scale: str) -> str:
    if dataset == "mind" and scale in _MIND_SCALE_FALLBACK:
        return _MIND_SCALE_FALLBACK[scale]
    return scale


def compare(dataset: str, scale: str) -> None:
    d = REPO_ROOT / "data" / "processed" / dataset / scale
    emb_path = d / "article_embeddings.parquet"
    if not emb_path.exists():
        raise FileNotFoundError(f"{emb_path} missing -- run scripts/compute_embeddings.py first")

    articles = pl.read_parquet(d / "articles.parquet")
    user_history = pl.read_parquet(d / "user_history.parquet")
    behaviors = pl.read_parquet(d / "behaviors_val.parquet")
    embeddings_df = pl.read_parquet(emb_path)

    print(f"\n=== {dataset}/{scale} ===")
    bm25 = BM25Index().build(articles)
    emb_index = EmbeddingIndex().build(embeddings_df)
    hist_index = UserHistoryIndex(user_history, articles)

    has_click = behaviors.filter(
        pl.col("clicked_article_ids").is_not_null() & (pl.col("clicked_article_ids").list.len() > 0)
    )
    eligible = []
    for row in has_click.iter_rows(named=True):
        titles = hist_index.recent_titles(row["user_id"], row["timestamp"], eval_utils.MAX_HISTORY)
        article_ids = hist_index.recent_article_ids(row["user_id"], row["timestamp"], eval_utils.MAX_HISTORY)
        if titles and article_ids:
            eligible.append((row["user_id"], row["timestamp"], row["clicked_article_ids"], titles, article_ids))

    sample = eval_utils.sample_impressions(eligible)
    print(f"comparing on {len(sample):,} sampled impressions (seed={eval_utils.SEED}, "
          f"same sample for both methods)")

    max_k = max(eval_utils.KS)
    rows = []
    for _user_id, _ts, clicked_ids, titles, article_ids in sample:
        clicked_set = set(clicked_ids)

        bm25_retrieved = [aid for aid, _s in bm25.search(" ".join(titles), top_k=max_k)]
        vectors = [v for aid in article_ids if (v := emb_index.get_embedding(aid)) is not None]
        query_vec = mean_pool(vectors)
        emb_retrieved = [aid for aid, _s in emb_index.search(query_vec, top_k=max_k)] if query_vec is not None else []

        row = {"history_len": len(article_ids)}
        for k in eval_utils.KS:
            row[f"bm25_recall@{k}"] = len(clicked_set & set(bm25_retrieved[:k])) / len(clicked_set)
            row[f"emb_recall@{k}"] = len(clicked_set & set(emb_retrieved[:k])) / len(clicked_set)
        rows.append(row)

    df = pl.DataFrame(rows)
    _print_comparison("overall", df)

    short = df.filter(pl.col("history_len") < SHORT_HISTORY_CUTOFF)
    long_ = df.filter(pl.col("history_len") >= SHORT_HISTORY_CUTOFF)
    print(f"\n-- slice: short history (<{SHORT_HISTORY_CUTOFF} prior clicks), n={short.height:,} --")
    _print_comparison("short", short)
    print(f"\n-- slice: long history (>={SHORT_HISTORY_CUTOFF} prior clicks), n={long_.height:,} --")
    _print_comparison("long", long_)


def _print_comparison(label: str, df: pl.DataFrame) -> None:
    if df.height == 0:
        print(f"  ({label}: no impressions in this slice)")
        return
    print(f"\n{'K':>6} | {'BM25':>10} | {'embeddings':>10} | {'winner':>10}")
    for k in eval_utils.KS:
        bm25_mean = df[f"bm25_recall@{k}"].mean()
        emb_mean = df[f"emb_recall@{k}"].mean()
        winner = "bm25" if bm25_mean > emb_mean else ("embeddings" if emb_mean > bm25_mean else "tie")
        print(f"{k:>6} | {bm25_mean:>10.4f} | {emb_mean:>10.4f} | {winner:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    parser.add_argument("--scale", choices=["demo", "small", "large"], default="demo")
    args = parser.parse_args()

    datasets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        scale = resolve_scale(dataset, args.scale)
        compare(dataset, scale)


if __name__ == "__main__":
    main()
