#!/usr/bin/env python
"""Q5: extended evaluation for the GBDT track -- the full metric set (AUC, MRR,
nDCG@5, nDCG@10, diversity, novelty, coverage) for all four methods (B1 BM25, B2
embeddings, B3 hybrid, GBDT), sliced two ways (cold-start vs. warm, head vs. tail
click), all with bootstrap 95% CIs.

    python scripts/run_reranker_extended_eval.py --dataset all --scale demo

Reuses run_eval_harness.py's attach_beyond_accuracy()/aggregate() (Q4's own
diversity/novelty/coverage + bootstrap machinery, unmodified) and
train_reranker.py's evaluate_hybrid()/evaluate_gbdt()/learn_alpha(), so this is
strictly the Q4 harness's shape extended to two more methods and a second slice --
not a reimplementation. Writes results/reranker_extended_eval.csv.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from ire_a1.feature_store import UserHistoryIndex  # noqa: E402
from ire_a1.retrieval.bm25 import BM25Index  # noqa: E402
from ire_a1.retrieval.embeddings import EmbeddingIndex  # noqa: E402
from ire_a2.features import FEATURE_COLUMNS  # noqa: E402
from ire_a2.reranker import build_training_frame, train_gbdt  # noqa: E402
from run_eval_harness import (  # noqa: E402
    COLD_START_CUTOFF,
    attach_beyond_accuracy,
    aggregate,
    evaluate_bm25,
    evaluate_embeddings,
    print_table,
    resolve_scale,
)
from train_reranker import build_sample_with_id, evaluate_gbdt, evaluate_hybrid, learn_alpha  # noqa: E402


def slice_tags(sample: list[dict], popularity: dict, head_threshold: float) -> list[dict]:
    """One {"cold_start": bool, "tail": bool} dict per impression *kept* by
    _score_impression()'s filter (both classes present among candidates) -- computed
    purely from imp["clicked"]/imp["candidates"]/imp["history_len"], the same fields
    _score_impression()'s own inclusion rule depends on, so this list aligns
    positionally with any evaluate_*() records list built from the same `sample`
    (same filter, same order, both entirely independent of which retriever scored it).
    """
    tags = []
    for imp in sample:
        clicked_set = set(imp["clicked"])
        n_candidates = len(imp["candidates"])
        n_pos = sum(1 for aid in imp["candidates"] if aid in clicked_set)
        if n_pos == 0 or n_pos == n_candidates:
            continue
        clicked_article = imp["clicked"][0]
        pop = popularity.get(clicked_article, 0)
        tags.append({"cold_start": imp["history_len"] < COLD_START_CUTOFF, "tail": pop < head_threshold})
    return tags


def run(dataset: str, scale: str, csv_rows: list[dict]) -> None:
    d = REPO_ROOT / "data" / "processed" / dataset / scale
    feat_train_path = d / "reranker_features_train.parquet"
    feat_val_path = d / "reranker_features_val.parquet"
    if not feat_train_path.exists() or not feat_val_path.exists():
        raise FileNotFoundError(
            f"{feat_train_path} / {feat_val_path} missing -- run scripts/build_reranker_features.py first"
        )

    articles = pl.read_parquet(d / "articles.parquet")
    user_history = pl.read_parquet(d / "user_history.parquet")
    behaviors_train = pl.read_parquet(d / "behaviors_train.parquet")
    behaviors_val = pl.read_parquet(d / "behaviors_val.parquet")
    embeddings_df = pl.read_parquet(d / "article_embeddings.parquet")
    features_train = pl.read_parquet(feat_train_path)
    features_val = pl.read_parquet(feat_val_path)

    print(f"\n=== {dataset}/{scale} ===")
    bm25 = BM25Index().build(articles)
    emb_index = EmbeddingIndex().build(embeddings_df)
    hist_index = UserHistoryIndex(user_history, articles)
    catalog_size = articles.height

    pop_df = (
        behaviors_train.filter(pl.col("clicked_article_ids").is_not_null())
        .select(pl.col("clicked_article_ids").explode().alias("article_id"))
        .group_by("article_id")
        .agg(pl.len().alias("clicks"))
    )
    popularity = dict(zip(pop_df["article_id"].to_list(), pop_df["clicks"].to_list()))
    total_clicks = pop_df["clicks"].sum()
    head_threshold = float(np.percentile(pop_df["clicks"].to_list(), 75)) if pop_df.height else 0.0
    print(f"popularity: {len(popularity):,} distinct clicked articles, head/tail threshold={head_threshold:.0f} clicks")

    train_frame = build_training_frame(features_train)
    model = train_gbdt(train_frame, FEATURE_COLUMNS)

    train_sample = build_sample_with_id(behaviors_train, hist_index)
    alpha = learn_alpha(train_sample, bm25, emb_index)

    sample = build_sample_with_id(behaviors_val, hist_index)
    print(f"evaluating on {len(sample):,} sampled impressions (B3 alpha={alpha:.2f})")

    tags = slice_tags(sample, popularity, head_threshold)

    method_records = {
        "bm25": evaluate_bm25(sample, bm25),
        "embeddings": evaluate_embeddings(sample, emb_index),
        "hybrid": evaluate_hybrid(sample, bm25, emb_index, alpha),
        "gbdt_reranker": evaluate_gbdt(sample, model, features_val),
    }

    for method_name, records in method_records.items():
        assert len(records) == len(tags), (
            f"{method_name} produced {len(records)} records but {len(tags)} slice tags -- "
            "slice_tags() and _score_impression() must apply the identical inclusion filter"
        )
        attach_beyond_accuracy(records, emb_index, popularity, total_clicks, catalog_size)

        overall = aggregate(records, catalog_size)
        print_table(f"{method_name}: overall", overall)
        for name, (mean, lo, hi, n) in overall.items():
            csv_rows.append({"dataset": dataset, "scale": scale, "method": method_name,
                              "slice": "overall", "metric": name, "mean": mean, "ci_low": lo, "ci_high": hi, "n": n})

        for slice_name, key, want in (
            ("cold_start", "cold_start", True), ("warm", "cold_start", False),
            ("tail", "tail", True), ("head", "tail", False),
        ):
            sliced = [r for r, tag in zip(records, tags) if tag[key] == want]
            agg = aggregate(sliced, catalog_size)
            print_table(f"{method_name}: {slice_name} (n={len(sliced):,})", agg)
            for name, (mean, lo, hi, n) in agg.items():
                csv_rows.append({"dataset": dataset, "scale": scale, "method": method_name,
                                  "slice": slice_name, "metric": name, "mean": mean, "ci_low": lo, "ci_high": hi, "n": n})


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

    out_path = REPO_ROOT / "results" / "reranker_extended_eval.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(csv_rows).write_csv(out_path)
    print(f"\nwrote {out_path} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
