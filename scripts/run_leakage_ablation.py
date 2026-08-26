#!/usr/bin/env python
"""Q9 (anti-gaming): metrics with vs. without a feature unavailable at serving time.

    python scripts/run_leakage_ablation.py --dataset all --scale demo

The rest of this codebase is built so a leaky feature like this can't happen by
accident (Q1's get_user_history()/UserHistoryIndex point-in-time cutoff, tested in
test_no_leakage.py). This script does the opposite on purpose: it deliberately
introduces one concrete serving-time-unavailable feature, measures how much it moves
the official metrics, and reports both conditions side by side -- the assignment asks
for this comparison to be reported, not just designed around.

The leaked feature: each user's clicks from their *other* validation-split
impressions, added to their query alongside their real (point-in-time-safe) history.
At serving time you never know what a user is about to click in their next few
impressions -- this is exactly that information, deliberately smuggled in. The
impression's own answer is excluded from its own leaked set (using the literal label
as a "feature" would be a trivial, uninteresting ablation; this is the more realistic
and more instructive form of leakage). Sample construction, scoring, and metrics are
identical to Q4's real run_eval_harness.py, reused directly -- the "without leak"
condition here is expected to exactly reproduce Q4's published results.
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
from run_eval_harness import evaluate_bm25, evaluate_embeddings, resolve_scale  # noqa: E402


def build_sample_with_user(behaviors: pl.DataFrame, hist_index: UserHistoryIndex) -> list[dict]:
    """Identical to run_eval_harness.build_sample() (same eligibility, same
    eval_utils.sample_impressions() call, same seed) plus a user_id field, needed here
    to look up that user's leaked extra clicks. Not a different sample -- Q4's.
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
                    "user_id": row["user_id"],
                    "candidates": row["candidate_article_ids"],
                    "clicked": row["clicked_article_ids"],
                    "titles": titles,
                    "history_ids": article_ids,
                    "history_len": len(article_ids),
                }
            )
    return eval_utils.sample_impressions(eligible)


def build_leaked_extras(behaviors: pl.DataFrame, articles: pl.DataFrame) -> dict:
    """user_id -> [(article_id, title), ...] for articles that user clicked in OTHER
    val impressions (across the whole val split). The current impression's own answer
    is excluded per-impression in build_leaky_sample(), not here.
    """
    id_to_title = dict(zip(articles["article_id"].to_list(), articles["title"].to_list()))
    exploded = (
        behaviors.filter(pl.col("clicked_article_ids").is_not_null() & (pl.col("clicked_article_ids").list.len() > 0))
        .select("user_id", "clicked_article_ids")
        .explode("clicked_article_ids")
        .rename({"clicked_article_ids": "article_id"})
        .unique()
    )
    out = {}
    for user_id, aids in exploded.group_by("user_id").agg(pl.col("article_id")).iter_rows():
        pairs = [(a, id_to_title[a]) for a in aids if a in id_to_title]
        if pairs:
            out[user_id] = pairs
    return out


def aggregate_accuracy(records: list[dict]) -> dict:
    """Same accuracy metrics as run_eval_harness.aggregate() (AUC/MRR/nDCG, same
    bootstrap_ci), minus the beyond-accuracy fields -- this ablation is about ranking
    accuracy under a leaked feature, not diversity/novelty, and those need popularity
    stats this script doesn't build.
    """
    groups = {
        "AUC": [r["auc"] for r in records if r["auc"] is not None],
        "MRR": [r["mrr"] for r in records],
        "nDCG@5": [r["ndcg5"] for r in records],
        "nDCG@10": [r["ndcg10"] for r in records],
    }
    return {name: bootstrap_ci(vals) for name, vals in groups.items()}


def build_leaky_sample(safe_sample: list[dict], leaked_extras: dict) -> list[dict]:
    leaky = []
    for imp in safe_sample:
        pairs = leaked_extras.get(imp["user_id"], [])
        own_clicks = set(imp["clicked"])
        extra_pairs = [(a, t) for a, t in pairs if a not in own_clicks]
        leaky_imp = dict(imp)
        leaky_imp["titles"] = imp["titles"] + [t for _, t in extra_pairs]
        leaky_imp["history_ids"] = imp["history_ids"] + [a for a, _ in extra_pairs]
        leaky.append(leaky_imp)
    return leaky


def print_compare(label: str, safe_agg: dict, leaky_agg: dict) -> None:
    print(f"\n-- {label} --")
    print(f"{'metric':>12} | {'without leak':>13} | {'with leak':>13} | {'delta':>9}")
    for name in ("AUC", "MRR", "nDCG@5", "nDCG@10"):
        safe_mean = safe_agg[name][0]
        leaky_mean = leaky_agg[name][0]
        print(f"{name:>12} | {safe_mean:>13.4f} | {leaky_mean:>13.4f} | {leaky_mean - safe_mean:>+9.4f}")


def run(dataset: str, scale: str, csv_rows: list[dict]) -> None:
    d = REPO_ROOT / "data" / "processed" / dataset / scale
    emb_path = d / "article_embeddings.parquet"
    if not emb_path.exists():
        raise FileNotFoundError(f"{emb_path} missing -- run scripts/compute_embeddings.py first")

    articles = pl.read_parquet(d / "articles.parquet")
    user_history = pl.read_parquet(d / "user_history.parquet")
    behaviors_val = pl.read_parquet(d / "behaviors_val.parquet")
    embeddings_df = pl.read_parquet(emb_path)

    print(f"\n=== {dataset}/{scale} ===")
    bm25 = BM25Index().build(articles)
    emb_index = EmbeddingIndex().build(embeddings_df)
    hist_index = UserHistoryIndex(user_history, articles)

    safe_sample = build_sample_with_user(behaviors_val, hist_index)
    leaked_extras = build_leaked_extras(behaviors_val, articles)
    n_leaky_users = sum(1 for imp in safe_sample if leaked_extras.get(imp["user_id"]))
    print(
        f"impressions evaluated: {len(safe_sample):,} | impressions with >=1 leak-eligible "
        f"extra click: {n_leaky_users:,} ({n_leaky_users / len(safe_sample):.1%})"
    )

    leaky_sample = build_leaky_sample(safe_sample, leaked_extras)

    for retriever_name, evaluate_fn, index in (
        ("bm25", evaluate_bm25, bm25),
        ("embeddings", evaluate_embeddings, emb_index),
    ):
        safe_records = evaluate_fn(safe_sample, index)
        leaky_records = evaluate_fn(leaky_sample, index)
        safe_agg = aggregate_accuracy(safe_records)
        leaky_agg = aggregate_accuracy(leaky_records)
        print_compare(retriever_name, safe_agg, leaky_agg)

        for name in ("AUC", "MRR", "nDCG@5", "nDCG@10"):
            safe_mean, safe_lo, safe_hi = safe_agg[name]
            leaky_mean, leaky_lo, leaky_hi = leaky_agg[name]
            csv_rows.append(
                {
                    "dataset": dataset,
                    "scale": scale,
                    "retriever": retriever_name,
                    "metric": name,
                    "without_leak_mean": safe_mean,
                    "without_leak_ci_low": safe_lo,
                    "without_leak_ci_high": safe_hi,
                    "with_leak_mean": leaky_mean,
                    "with_leak_ci_low": leaky_lo,
                    "with_leak_ci_high": leaky_hi,
                    "delta": leaky_mean - safe_mean,
                    "n": len(safe_records),
                }
            )


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

    out_path = REPO_ROOT / "results" / "leakage_ablation.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(csv_rows).write_csv(out_path)
    print(f"\nwrote {out_path} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
