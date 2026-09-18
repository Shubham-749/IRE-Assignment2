#!/usr/bin/env python
"""Q9 (anti-gaming) for the GBDT track: metrics with vs. without a feature unavailable
at serving time.

    python scripts/run_reranker_leakage_ablation.py --dataset all --scale demo

The boundary-enforcement test itself ("include a test asserting this") is
tests/test_a2_features.py::test_user_history_index_recent_excludes_clicks_at_or_after_cutoff
-- this script does A1's run_leakage_ablation.py's other job: deliberately introduce
one concrete serving-time-unavailable feature and measure how much it moves the
official metrics, reported as a paired comparison (not just two separate numbers).

The leaked feature: `candidate_popularity` computed with train **+ val** click
counts (hindsight) instead of the safe train-only default
(ire_a2.features.FeatureBuilder's `popularity_override` hook exists for exactly this).
At serving time on the val period you don't yet know how popular an article will turn
out to be over that same period -- this is exactly that information, deliberately
smuggled in. Trains two small GBDT models (on a train-split sample, not the full
cached features -- fast enough that a smaller from-scratch build is simpler than
maintaining a second full cached parquet) that are identical except for this one
feature, evaluates both on the same val sample, and reports a paired bootstrap 95% CI
on the metric deltas (safe -> leaky) via ire_a1.eval.metrics.paired_bootstrap_ci.
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
from ire_a1.retrieval.embeddings import EmbeddingIndex  # noqa: E402
from ire_a2.features import FEATURE_COLUMNS, MAX_HISTORY, FeatureBuilder  # noqa: E402
from ire_a2.reranker import build_training_frame, train_gbdt  # noqa: E402
from run_eval_harness import _score_impression, resolve_scale  # noqa: E402
from train_reranker import METRIC_KEYS, build_sample_with_id  # noqa: E402

TRAIN_SAMPLE_SIZE = 2000


def hindsight_popularity(behaviors_train: pl.DataFrame, behaviors_val: pl.DataFrame) -> dict:
    """Train + val click counts -- deliberately including the val split's own
    outcomes, which is exactly what makes this a leak (it's val's own answer key,
    aggregated into a feature instead of used directly as a label).
    """
    both = pl.concat(
        [
            behaviors_train.select("clicked_article_ids"),
            behaviors_val.select("clicked_article_ids"),
        ]
    )
    pop = (
        both.filter(pl.col("clicked_article_ids").is_not_null())
        .select(pl.col("clicked_article_ids").explode().alias("article_id"))
        .group_by("article_id")
        .agg(pl.len().alias("clicks"))
    )
    return dict(zip(pop["article_id"].to_list(), pop["clicks"].to_list()))


def build_frame_from_sample(
    sample: list[dict], hist_index: UserHistoryIndex, feature_builder: FeatureBuilder
) -> pl.DataFrame:
    """Same per-impression feature-building + labeling as
    build_reranker_features.build_split(), but over an already-sampled impression list
    instead of a full behaviors split -- fast enough for this script's small samples
    without needing a second cached parquet per variant.
    """
    rows = []
    for imp in sample:
        history = hist_index.recent(imp["user_id"], imp["timestamp"], MAX_HISTORY)
        if not history:
            continue
        clicked = set(imp["clicked"])
        for fr in feature_builder.build_rows(imp["user_id"], imp["timestamp"], imp["candidates"], history):
            fr["impression_id"] = imp["impression_id"]
            fr["label"] = 1 if fr["article_id"] in clicked else 0
            rows.append(fr)
    return pl.DataFrame(rows)


def evaluate(sample: list[dict], model, features: pl.DataFrame) -> list[dict]:
    feat_lookup = {(r["impression_id"], r["article_id"]): r for r in features.iter_rows(named=True)}
    records = []
    for imp in sample:
        feat_rows = [feat_lookup.get((imp["impression_id"], aid)) for aid in imp["candidates"]]
        if any(r is None for r in feat_rows):
            continue
        X = [[r[c] for c in FEATURE_COLUMNS] for r in feat_rows]
        scores = model.predict(X)
        scores_map = dict(zip(imp["candidates"], scores))
        rec = _score_impression(imp["candidates"], imp["clicked"], scores_map, imp["history_len"])
        if rec:
            records.append(rec)
    return records


def run(dataset: str, scale: str, csv_rows: list[dict]) -> None:
    d = REPO_ROOT / "data" / "processed" / dataset / scale
    raw_dir = REPO_ROOT / "data" / "raw"
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

    leaky_popularity = hindsight_popularity(behaviors_train, behaviors_val)

    fb_safe = FeatureBuilder(dataset, articles, bm25, emb_index, behaviors_train, raw_dir, scale)
    fb_leaky = FeatureBuilder(
        dataset, articles, bm25, emb_index, behaviors_train, raw_dir, scale,
        popularity_override=leaky_popularity,
    )

    train_sample_full = build_sample_with_id(behaviors_train, hist_index)
    train_sample = eval_utils.sample_impressions(train_sample_full, sample_size=TRAIN_SAMPLE_SIZE)
    val_sample = build_sample_with_id(behaviors_val, hist_index)
    print(f"train sample: {len(train_sample):,} impressions | eval sample: {len(val_sample):,} impressions")

    models, features_val = {}, {}
    for variant, fb in (("safe", fb_safe), ("leaky", fb_leaky)):
        train_features = build_frame_from_sample(train_sample, hist_index, fb)
        train_frame = build_training_frame(train_features)
        models[variant] = train_gbdt(train_frame, FEATURE_COLUMNS)
        features_val[variant] = build_frame_from_sample(val_sample, hist_index, fb)
        print(f"  {variant}: trained on {train_frame.height:,} rows "
              f"({train_frame['impression_id'].n_unique():,} impressions)")

    safe_records = evaluate(val_sample, models["safe"], features_val["safe"])
    leaky_records = evaluate(val_sample, models["leaky"], features_val["leaky"])
    assert len(safe_records) == len(leaky_records), (
        f"safe produced {len(safe_records)} records but leaky produced {len(leaky_records)} -- "
        "both variants use the identical val_sample and the identical (feature-independent) "
        "_score_impression() filter, so this should never happen"
    )

    def aggregate(records):
        groups = {name: [r[key] for r in records] for name, key in METRIC_KEYS.items()}
        return {name: bootstrap_ci(vals) + (len(vals),) for name, vals in groups.items()}

    print(f"\n{'metric':>10} | {'without leak':>13} | {'with leak':>13} | {'delta':>9} | significant")
    safe_agg, leaky_agg = aggregate(safe_records), aggregate(leaky_records)
    for metric_name, key in METRIC_KEYS.items():
        safe_vals = [r[key] for r in safe_records]
        leaky_vals = [r[key] for r in leaky_records]
        delta_mean, lo, hi = paired_bootstrap_ci(safe_vals, leaky_vals)
        significant = lo > 0 or hi < 0
        print(f"{metric_name:>10} | {safe_agg[metric_name][0]:>13.4f} | {leaky_agg[metric_name][0]:>13.4f} | "
              f"{delta_mean:>+9.4f} | {'YES' if significant else 'no'}")
        csv_rows.append({
            "dataset": dataset, "scale": scale, "metric": metric_name,
            "without_leak_mean": safe_agg[metric_name][0], "with_leak_mean": leaky_agg[metric_name][0],
            "delta_mean": delta_mean, "delta_ci_low": lo, "delta_ci_high": hi,
            "significant": significant, "n": len(safe_records),
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

    out_path = REPO_ROOT / "results" / "reranker_leakage_ablation.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(csv_rows).write_csv(out_path)
    print(f"\nwrote {out_path} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
