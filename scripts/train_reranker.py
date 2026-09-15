#!/usr/bin/env python
"""Q2 (Option A): train a LightGBM re-ranker over Q1's behavioural features and report
AUC/MRR/nDCG@5/@10 before (plain BM25 / plain embeddings) vs. after (GBDT) re-ranking.

    python scripts/train_reranker.py --dataset all --scale demo

Evaluated on the same val-split sample run_eval_harness.py (Q4) already reports on --
same eligibility, same eval_utils.sample_impressions(seed=42) -- so "before" numbers
here are directly comparable to results/eval_results.csv. Reads
data/processed/<dataset>/<scale>/reranker_features_{train,val}.parquet -- run
scripts/build_reranker_features.py first if missing.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import polars as pl  # noqa: E402

from ire_a1.eval.metrics import bootstrap_ci  # noqa: E402
from ire_a1.feature_store import UserHistoryIndex  # noqa: E402
from ire_a1.retrieval import eval_utils  # noqa: E402
from ire_a1.retrieval.bm25 import BM25Index  # noqa: E402
from ire_a1.retrieval.embeddings import EmbeddingIndex  # noqa: E402
from ire_a2.features import FEATURE_COLUMNS  # noqa: E402
from ire_a2.reranker import build_training_frame, feature_importance, train_gbdt  # noqa: E402
from run_eval_harness import _score_impression, evaluate_bm25, evaluate_embeddings, resolve_scale  # noqa: E402


def build_sample_with_id(behaviors: pl.DataFrame, hist_index: UserHistoryIndex) -> list[dict]:
    """Identical to run_eval_harness.build_sample() (same eligibility, same
    eval_utils.sample_impressions() seed) plus an impression_id field, needed here to
    join each sampled candidate against its precomputed GBDT feature row. Same pattern
    run_leakage_ablation.py already uses (build_sample_with_user) to extend the Q4
    sample without duplicating its selection logic.
    """
    has_click = behaviors.filter(
        pl.col("clicked_article_ids").is_not_null() & (pl.col("clicked_article_ids").list.len() > 0)
    )
    eligible = []
    for row in has_click.iter_rows(named=True):
        titles = hist_index.recent_titles(row["user_id"], row["timestamp"], eval_utils.MAX_HISTORY)
        article_ids = hist_index.recent_article_ids(row["user_id"], row["timestamp"], eval_utils.MAX_HISTORY)
        if titles and article_ids:
            eligible.append(
                {
                    "impression_id": row["impression_id"],
                    "candidates": row["candidate_article_ids"],
                    "clicked": row["clicked_article_ids"],
                    "titles": titles,
                    "history_ids": article_ids,
                    "history_len": len(article_ids),
                }
            )
    return eval_utils.sample_impressions(eligible)


def evaluate_gbdt(sample: list[dict], model, features_val: pl.DataFrame) -> list[dict]:
    imp_ids = [imp["impression_id"] for imp in sample]
    feats = features_val.filter(pl.col("impression_id").is_in(imp_ids))
    feat_lookup = {(r["impression_id"], r["article_id"]): r for r in feats.iter_rows(named=True)}

    records = []
    for imp in sample:
        feat_rows = [feat_lookup.get((imp["impression_id"], aid)) for aid in imp["candidates"]]
        if any(r is None for r in feat_rows):
            continue  # a served candidate missing its precomputed feature row (shouldn't
            # happen -- build_reranker_features.py uses the same eligibility -- skip
            # rather than guess a score for it)
        X = [[r[c] for c in FEATURE_COLUMNS] for r in feat_rows]
        scores = model.predict(X)
        scores_map = dict(zip(imp["candidates"], scores))
        rec = _score_impression(imp["candidates"], imp["clicked"], scores_map, imp["history_len"])
        if rec:
            records.append(rec)
    return records


def aggregate(records: list[dict]) -> dict:
    groups = {
        "AUC": [r["auc"] for r in records if r["auc"] is not None],
        "MRR": [r["mrr"] for r in records],
        "nDCG@5": [r["ndcg5"] for r in records],
        "nDCG@10": [r["ndcg10"] for r in records],
    }
    return {name: bootstrap_ci(vals) + (len(vals),) for name, vals in groups.items()}


def print_table(label: str, agg: dict) -> None:
    print(f"\n-- {label} --")
    print(f"{'metric':>10} | {'mean':>8} | {'95% CI':>17} | {'n':>6}")
    for name, (mean, lo, hi, n) in agg.items():
        print(f"{name:>10} | {mean:>8.4f} | [{lo:.4f}, {hi:.4f}] | {n:>6,}")


def run(dataset: str, scale: str, csv_rows: list[dict]) -> None:
    d = REPO_ROOT / "data" / "processed" / dataset / scale
    feat_train_path = d / "reranker_features_train.parquet"
    feat_val_path = d / "reranker_features_val.parquet"
    if not feat_train_path.exists() or not feat_val_path.exists():
        raise FileNotFoundError(
            f"{feat_train_path} / {feat_val_path} missing -- run scripts/build_reranker_features.py first"
        )
    emb_path = d / "article_embeddings.parquet"

    articles = pl.read_parquet(d / "articles.parquet")
    user_history = pl.read_parquet(d / "user_history.parquet")
    behaviors_val = pl.read_parquet(d / "behaviors_val.parquet")
    embeddings_df = pl.read_parquet(emb_path)
    features_train = pl.read_parquet(feat_train_path)
    features_val = pl.read_parquet(feat_val_path)

    print(f"\n=== {dataset}/{scale} ===")
    bm25 = BM25Index().build(articles)
    emb_index = EmbeddingIndex().build(embeddings_df)
    hist_index = UserHistoryIndex(user_history, articles)

    train_frame = build_training_frame(features_train)
    print(
        f"training on {train_frame.height:,} rows "
        f"({train_frame['impression_id'].n_unique():,} impressions)"
    )
    model = train_gbdt(train_frame, FEATURE_COLUMNS)

    sample = build_sample_with_id(behaviors_val, hist_index)
    print(f"evaluating on {len(sample):,} sampled impressions (seed={eval_utils.SEED}, same as Q4)")

    bm25_records = evaluate_bm25(sample, bm25)
    emb_records = evaluate_embeddings(sample, emb_index)
    gbdt_records = evaluate_gbdt(sample, model, features_val)

    for method_name, records in (
        ("bm25", bm25_records),
        ("embeddings", emb_records),
        ("gbdt_reranker", gbdt_records),
    ):
        agg = aggregate(records)
        print_table(method_name, agg)
        for name, (mean, lo, hi, n) in agg.items():
            csv_rows.append(
                {
                    "dataset": dataset,
                    "scale": scale,
                    "method": method_name,
                    "metric": name,
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": n,
                }
            )

    print("\ntop feature importances (gain):")
    for name, gain in feature_importance(model, FEATURE_COLUMNS)[:8]:
        print(f"  {name:>28} {gain:>10.1f}")


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

    out_path = REPO_ROOT / "results" / "reranker_eval.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(csv_rows).write_csv(out_path)
    print(f"\nwrote {out_path} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
