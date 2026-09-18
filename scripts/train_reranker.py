#!/usr/bin/env python
"""Q2 (Option A) + Q3: train a LightGBM re-ranker over Q1's behavioural features,
report AUC/MRR/nDCG@5/@10 for three baselines (B1 BM25, B2 embeddings, B3 a
learned-weight BM25+embedding hybrid) vs. the GBDT re-ranker, and ship a paired
bootstrap 95% CI on the GBDT's improvement over each baseline (Q3's "claimed gains
must ship a paired bootstrap 95% CI that excludes zero").

    python scripts/train_reranker.py --dataset all --scale demo

Evaluated on the same val-split sample run_eval_harness.py (Q4) already reports on --
same eligibility, same eval_utils.sample_impressions(seed=42) -- so B1/B2 numbers here
are directly comparable to results/eval_results.csv. Reads
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

from ire_a1.eval.metrics import bootstrap_ci, paired_bootstrap_ci  # noqa: E402
from ire_a1.feature_store import UserHistoryIndex  # noqa: E402
from ire_a1.retrieval import eval_utils  # noqa: E402
from ire_a1.retrieval.bm25 import BM25Index  # noqa: E402
from ire_a1.retrieval.embeddings import EmbeddingIndex, mean_pool  # noqa: E402
from ire_a2.baselines import hybrid_score, learn_hybrid_weight  # noqa: E402
from ire_a2.features import FEATURE_COLUMNS  # noqa: E402
from ire_a2.reranker import build_training_frame, feature_importance, train_gbdt  # noqa: E402
from run_eval_harness import _score_impression, evaluate_bm25, evaluate_embeddings, resolve_scale  # noqa: E402

METRIC_KEYS = {"AUC": "auc", "MRR": "mrr", "nDCG@5": "ndcg5", "nDCG@10": "ndcg10"}


def build_sample_with_id(behaviors: pl.DataFrame, hist_index: UserHistoryIndex) -> list[dict]:
    """Identical to run_eval_harness.build_sample() (same eligibility, same
    eval_utils.sample_impressions() seed) plus impression_id/user_id/timestamp fields,
    needed here (impression_id, to join each sampled candidate against its
    precomputed GBDT feature row) and by reranker_scale_analysis.py (user_id/timestamp,
    to re-derive point-in-time history for a fresh single-request latency simulation).
    Same pattern run_leakage_ablation.py already uses (build_sample_with_user) to
    extend the Q4 sample without duplicating its selection logic.
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
                    "user_id": row["user_id"],
                    "timestamp": row["timestamp"],
                    "candidates": row["candidate_article_ids"],
                    "clicked": row["clicked_article_ids"],
                    "titles": titles,
                    "history_ids": article_ids,
                    "history_len": len(article_ids),
                }
            )
    return eval_utils.sample_impressions(eligible)


def per_candidate_bm25_embedding_scores(imp: dict, bm25: BM25Index, emb_index: EmbeddingIndex):
    """(bm25_scores, embedding_scores, labels), all aligned to imp['candidates']
    order -- the per-candidate triples both learn_hybrid_weight() (B3's alpha fit) and
    evaluate_hybrid() need. Mirrors evaluate_bm25()/evaluate_embeddings()'s own scoring
    exactly (same score_candidates() calls, same mean-pooled query vector) so B3's
    inputs are the same numbers B1/B2 are separately evaluated on.
    """
    query = " ".join(imp["titles"])
    bm25_map = bm25.score_candidates(query, imp["candidates"])
    vectors = [v for aid in imp["history_ids"] if (v := emb_index.get_embedding(aid)) is not None]
    query_vec = mean_pool(vectors)
    emb_map = (
        emb_index.score_candidates(query_vec, imp["candidates"])
        if query_vec is not None
        else dict.fromkeys(imp["candidates"], 0.0)
    )
    clicked_set = set(imp["clicked"])
    bm25_scores = [bm25_map[aid] for aid in imp["candidates"]]
    embedding_scores = [emb_map[aid] for aid in imp["candidates"]]
    labels = [1 if aid in clicked_set else 0 for aid in imp["candidates"]]
    return bm25_scores, embedding_scores, labels


def learn_alpha(train_sample: list[dict], bm25: BM25Index, emb_index: EmbeddingIndex) -> float:
    """Fits B3's fusion weight on a *train*-split sample, not val -- val is what B1/B2/
    B3/GBDT all get evaluated on, so fitting alpha there would let B3 see the same data
    its own eval score is computed from.
    """
    impressions = []
    for imp in train_sample:
        bm25_scores, embedding_scores, labels = per_candidate_bm25_embedding_scores(imp, bm25, emb_index)
        if 0 < sum(labels) < len(labels):
            impressions.append(
                {"bm25_scores": bm25_scores, "embedding_scores": embedding_scores, "labels": labels}
            )
    return learn_hybrid_weight(impressions)


def evaluate_hybrid(sample: list[dict], bm25: BM25Index, emb_index: EmbeddingIndex, alpha: float) -> list[dict]:
    records = []
    for imp in sample:
        bm25_scores, embedding_scores, _labels = per_candidate_bm25_embedding_scores(imp, bm25, emb_index)
        scores = [hybrid_score(b, e, alpha) for b, e in zip(bm25_scores, embedding_scores)]
        scores_map = dict(zip(imp["candidates"], scores))
        rec = _score_impression(imp["candidates"], imp["clicked"], scores_map, imp["history_len"])
        if rec:
            records.append(rec)
    return records


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


def compute_ablation(
    dataset: str,
    scale: str,
    baseline_name: str,
    baseline_records: list[dict],
    gbdt_records: list[dict],
    ablation_rows: list[dict],
) -> None:
    """Q3: paired bootstrap 95% CI on the GBDT's per-impression improvement over one
    baseline. Requires baseline_records and gbdt_records to be the same length, in the
    same impression order -- both come from iterating the identical `sample` and
    applying the identical (candidate/clicked-based, score-independent)
    _score_impression() filter, so this holds by construction; asserted rather than
    assumed, since a silent misalignment would produce a meaningless paired delta.
    """
    assert len(baseline_records) == len(gbdt_records), (
        f"{baseline_name} produced {len(baseline_records)} records but gbdt produced "
        f"{len(gbdt_records)} -- paired comparison requires the same impressions in "
        f"the same order (see this function's docstring)"
    )
    print(f"\n-- GBDT vs {baseline_name} (paired bootstrap ablation) --")
    for metric_name, key in METRIC_KEYS.items():
        baseline_vals = [r[key] for r in baseline_records]
        gbdt_vals = [r[key] for r in gbdt_records]
        delta_mean, lo, hi = paired_bootstrap_ci(baseline_vals, gbdt_vals)
        significant = lo > 0 or hi < 0
        flag = "SIGNIFICANT" if significant else "not significant"
        print(f"  {metric_name:>10}: delta={delta_mean:+.4f} [{lo:+.4f}, {hi:+.4f}]  {flag}")
        ablation_rows.append(
            {
                "dataset": dataset,
                "scale": scale,
                "baseline": baseline_name,
                "metric": metric_name,
                "delta_mean": delta_mean,
                "delta_ci_low": lo,
                "delta_ci_high": hi,
                "significant": significant,
                "n": len(gbdt_vals),
            }
        )


def run(dataset: str, scale: str, csv_rows: list[dict], ablation_rows: list[dict]) -> None:
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
    behaviors_train = pl.read_parquet(d / "behaviors_train.parquet")
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

    train_sample = build_sample_with_id(behaviors_train, hist_index)
    alpha = learn_alpha(train_sample, bm25, emb_index)
    print(f"B3 hybrid: learned alpha={alpha:.2f} on {len(train_sample):,} train-sample impressions")

    sample = build_sample_with_id(behaviors_val, hist_index)
    print(f"evaluating on {len(sample):,} sampled impressions (seed={eval_utils.SEED}, same as Q4)")

    bm25_records = evaluate_bm25(sample, bm25)
    emb_records = evaluate_embeddings(sample, emb_index)
    hybrid_records = evaluate_hybrid(sample, bm25, emb_index, alpha)
    gbdt_records = evaluate_gbdt(sample, model, features_val)

    for method_name, records in (
        ("bm25", bm25_records),
        ("embeddings", emb_records),
        ("hybrid", hybrid_records),
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

    for baseline_name, baseline_records in (
        ("bm25", bm25_records),
        ("embeddings", emb_records),
        ("hybrid", hybrid_records),
    ):
        compute_ablation(dataset, scale, baseline_name, baseline_records, gbdt_records, ablation_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    parser.add_argument("--scale", choices=["demo", "small", "large"], default="demo")
    args = parser.parse_args()

    datasets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]
    csv_rows: list[dict] = []
    ablation_rows: list[dict] = []
    for dataset in datasets:
        scale = resolve_scale(dataset, args.scale)
        run(dataset, scale, csv_rows, ablation_rows)

    out_path = REPO_ROOT / "results" / "reranker_eval.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(csv_rows).write_csv(out_path)
    print(f"\nwrote {out_path} ({len(csv_rows)} rows)")

    ablation_path = REPO_ROOT / "results" / "reranker_ablation.csv"
    pl.DataFrame(ablation_rows).write_csv(ablation_path)
    print(f"wrote {ablation_path} ({len(ablation_rows)} rows)")


if __name__ == "__main__":
    main()
