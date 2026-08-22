#!/usr/bin/env python
"""Q4: offline evaluation harness -- the official ranking metrics, on both retrievers.

    python scripts/run_eval_harness.py --dataset all --scale demo

Unlike Q2/Q3 (full-corpus top-K search), this scores and ranks each impression's own
shown candidate list (candidate_article_ids) -- the same thing MIND's and EB-NeRD's
Codabench leaderboards actually compute AUC/MRR/nDCG over. Both BM25 and embeddings
get evaluated this way, on the same sampled impressions eval_utils already uses for
Q2/Q3 (same seed=42), plus beyond-accuracy metrics and a cold-start-vs-warm slice, all
with bootstrap 95% CIs. Results are printed and written to results/eval_results.csv
(small enough to commit, unlike data/ -- it's the one processed-data-shaped file this
repo keeps in git).
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import polars as pl  # noqa: E402

from ire_a1.eval.beyond_accuracy import bootstrap_ci_coverage, intra_list_diversity, novelty  # noqa: E402
from ire_a1.eval.metrics import auc_score, bootstrap_ci, mrr_score, ndcg_score  # noqa: E402
from ire_a1.feature_store import UserHistoryIndex  # noqa: E402
from ire_a1.retrieval import eval_utils  # noqa: E402
from ire_a1.retrieval.bm25 import BM25Index  # noqa: E402
from ire_a1.retrieval.embeddings import EmbeddingIndex, mean_pool  # noqa: E402

_MIND_SCALE_FALLBACK = {"demo": "small"}
COLD_START_CUTOFF = 5
METRIC_NAMES = ("AUC", "MRR", "nDCG@5", "nDCG@10", "Diversity@5", "Novelty@5")


def resolve_scale(dataset: str, scale: str) -> str:
    if dataset == "mind" and scale in _MIND_SCALE_FALLBACK:
        return _MIND_SCALE_FALLBACK[scale]
    return scale


def build_sample(behaviors: pl.DataFrame, hist_index: UserHistoryIndex) -> list[dict]:
    """Same eligibility as Q2/Q3 (has a click + non-empty history for both title-based
    and id-based lookups) and the same sample_impressions() seed=42, so this Q4 sample
    is the same population Q2/Q3 already reported on.
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
                    "candidates": row["candidate_article_ids"],
                    "clicked": row["clicked_article_ids"],
                    "titles": titles,
                    "history_ids": article_ids,
                    "history_len": len(article_ids),
                }
            )
    return eval_utils.sample_impressions(eligible)


def _score_impression(candidates: list[str], clicked: list[str], scores_map: dict, history_len: int) -> dict | None:
    clicked_set = set(clicked)
    labels = [1 if aid in clicked_set else 0 for aid in candidates]
    scores = [scores_map.get(aid, 0.0) for aid in candidates]
    if sum(labels) == len(labels) or sum(labels) == 0:
        return None  # AUC/nDCG need both classes present -- shouldn't happen here, but never silently mis-score
    ranked = [aid for aid, _s in sorted(zip(candidates, scores), key=lambda x: -x[1])]
    return {
        "history_len": history_len,
        "auc": auc_score(labels, scores),
        "mrr": mrr_score(labels, scores),
        "ndcg5": ndcg_score(labels, scores, 5),
        "ndcg10": ndcg_score(labels, scores, 10),
        "top5": ranked[:5],
    }


def evaluate_bm25(sample: list[dict], bm25: BM25Index) -> list[dict]:
    records = []
    for imp in sample:
        query = " ".join(imp["titles"])
        scores_map = bm25.score_candidates(query, imp["candidates"])
        rec = _score_impression(imp["candidates"], imp["clicked"], scores_map, imp["history_len"])
        if rec:
            records.append(rec)
    return records


def evaluate_embeddings(sample: list[dict], emb_index: EmbeddingIndex) -> list[dict]:
    records = []
    for imp in sample:
        vectors = [v for aid in imp["history_ids"] if (v := emb_index.get_embedding(aid)) is not None]
        query_vec = mean_pool(vectors)
        if query_vec is None:
            continue
        scores_map = emb_index.score_candidates(query_vec, imp["candidates"])
        rec = _score_impression(imp["candidates"], imp["clicked"], scores_map, imp["history_len"])
        if rec:
            records.append(rec)
    return records


def attach_beyond_accuracy(records: list[dict], emb_index: EmbeddingIndex, popularity: dict, total_clicks: int, catalog_size: int) -> None:
    for rec in records:
        rec["diversity"] = intra_list_diversity(rec["top5"], emb_index)
        rec["novelty"] = novelty(rec["top5"], popularity, total_clicks, catalog_size)


def aggregate(records: list[dict], catalog_size: int) -> dict:
    groups = {
        "AUC": [r["auc"] for r in records if r["auc"] is not None],
        "MRR": [r["mrr"] for r in records],
        "nDCG@5": [r["ndcg5"] for r in records],
        "nDCG@10": [r["ndcg10"] for r in records],
        "Diversity@5": [r["diversity"] for r in records if r["diversity"] is not None],
        "Novelty@5": [r["novelty"] for r in records if r["novelty"] is not None],
    }
    result = {}
    for name, vals in groups.items():
        mean, lo, hi = bootstrap_ci(vals)
        result[name] = (mean, lo, hi, len(vals))
    top5_lists = [r["top5"] for r in records]
    result["Coverage@5"] = bootstrap_ci_coverage(top5_lists, catalog_size) + (len(top5_lists),)
    return result


def print_table(label: str, agg: dict) -> None:
    print(f"\n-- {label} --")
    print(f"{'metric':>12} | {'mean':>8} | {'95% CI':>17} | {'n':>6}")
    for name, (mean, lo, hi, n) in agg.items():
        print(f"{name:>12} | {mean:>8.4f} | [{lo:.4f}, {hi:.4f}] | {n:>6,}")


def run(dataset: str, scale: str, csv_rows: list[dict]) -> None:
    d = REPO_ROOT / "data" / "processed" / dataset / scale
    emb_path = d / "article_embeddings.parquet"
    if not emb_path.exists():
        raise FileNotFoundError(f"{emb_path} missing -- run scripts/compute_embeddings.py first")

    articles = pl.read_parquet(d / "articles.parquet")
    user_history = pl.read_parquet(d / "user_history.parquet")
    behaviors_train = pl.read_parquet(d / "behaviors_train.parquet")
    behaviors_val = pl.read_parquet(d / "behaviors_val.parquet")
    embeddings_df = pl.read_parquet(emb_path)

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
    print(f"popularity: {len(popularity):,} distinct clicked articles, {total_clicks:,} total train clicks")

    sample = build_sample(behaviors_val, hist_index)
    print(f"evaluating on {len(sample):,} sampled impressions (seed={eval_utils.SEED}, same as Q2/Q3)")

    t0 = time.time()
    bm25_records = evaluate_bm25(sample, bm25)
    print(f"BM25 scoring took {time.time()-t0:.1f}s ({len(bm25_records):,} usable impressions)")
    t0 = time.time()
    emb_records = evaluate_embeddings(sample, emb_index)
    print(f"Embedding scoring took {time.time()-t0:.1f}s ({len(emb_records):,} usable impressions)")

    for retriever_name, records in (("bm25", bm25_records), ("embeddings", emb_records)):
        attach_beyond_accuracy(records, emb_index, popularity, total_clicks, catalog_size)

        overall = aggregate(records, catalog_size)
        print_table(f"{retriever_name}: overall", overall)
        for name, (mean, lo, hi, n) in overall.items():
            csv_rows.append({"dataset": dataset, "scale": scale, "retriever": retriever_name,
                              "slice": "overall", "metric": name, "mean": mean, "ci_low": lo, "ci_high": hi, "n": n})

        for slice_name, pred in (
            ("cold_start", lambda r: r["history_len"] < COLD_START_CUTOFF),
            ("warm", lambda r: r["history_len"] >= COLD_START_CUTOFF),
        ):
            sliced = [r for r in records if pred(r)]
            agg = aggregate(sliced, catalog_size)
            print_table(f"{retriever_name}: {slice_name} (n={len(sliced):,})", agg)
            for name, (mean, lo, hi, n) in agg.items():
                csv_rows.append({"dataset": dataset, "scale": scale, "retriever": retriever_name,
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

    out_path = REPO_ROOT / "results" / "eval_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(csv_rows).write_csv(out_path)
    print(f"\nwrote {out_path} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
