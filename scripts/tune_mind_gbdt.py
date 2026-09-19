#!/usr/bin/env python
"""Hyperparameter search for the MIND large-scale GBDT re-ranker.

    python scripts/tune_mind_gbdt.py [--n-trials N] [--tune-sample-size N] [--eval-sample-size N]

train_gbdt() (src/ire_a2/reranker.py) uses fixed, untuned hyperparameters everywhere
else in this repo -- deliberately, to keep the demo/small-scale results simple and
reproducible. This script is the one place that actually searches: random search
(not exhaustive grid -- the grid below is 4^4=256 combinations, too many to train at
this scale) over num_leaves/learning_rate/n_estimators/min_child_samples, each trial
trained on a smaller tuning-sample (for speed) and scored on a held-out sample of
MIND-large's real validation split (behaviors_val.parquet -- real dev-split labels,
never touched by the submission-generation script, which only trains on train and
scores unlabeled test).

Reuses generate_mind_gbdt_submission.py's already-built feature store, cached
embeddings, and indices -- no new data processing, only new search/eval logic.
"""

import argparse
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import polars as pl  # noqa: E402

from generate_mind_gbdt_submission import build_frame_from_sample, build_train_sample  # noqa: E402
from ire_a1.eval.metrics import bootstrap_ci  # noqa: E402
from ire_a1.feature_store import UserHistoryIndex  # noqa: E402
from ire_a1.retrieval import eval_utils  # noqa: E402
from ire_a1.retrieval.bm25 import BM25Index  # noqa: E402
from ire_a1.retrieval.embeddings import EmbeddingIndex  # noqa: E402
from ire_a2.features import FEATURE_COLUMNS, FeatureBuilder  # noqa: E402
from ire_a2.reranker import build_training_frame, train_gbdt  # noqa: E402
from run_eval_harness import _score_impression  # noqa: E402

PROC_DIR = REPO_ROOT / "data" / "processed" / "mind" / "large"
TUNE_SAMPLE_SIZE = 15_000
EVAL_SAMPLE_SIZE = 5_000
N_TRIALS = 12
SEED = 42

PARAM_GRID = {
    "num_leaves": [15, 31, 63, 127],
    "learning_rate": [0.02, 0.05, 0.1, 0.15],
    "n_estimators": [100, 150, 200, 300],
    "min_child_samples": [3, 5, 10, 20],
}


def sample_params(rng: random.Random) -> dict:
    return {name: rng.choice(values) for name, values in PARAM_GRID.items()}


def evaluate_on_val(sample: list[dict], model, feature_builder: FeatureBuilder) -> dict:
    records = []
    for imp in sample:
        rows = feature_builder.build_rows(imp["user_id"], imp["timestamp"], imp["candidates"], imp["history"])
        X = [[r[c] for c in FEATURE_COLUMNS] for r in rows]
        scores = model.predict(X)
        scores_map = dict(zip(imp["candidates"], scores))
        rec = _score_impression(imp["candidates"], imp["clicked"], scores_map, len(imp["history"]))
        if rec:
            records.append(rec)
    groups = {
        "AUC": [r["auc"] for r in records if r["auc"] is not None],
        "MRR": [r["mrr"] for r in records],
        "nDCG@5": [r["ndcg5"] for r in records],
        "nDCG@10": [r["ndcg10"] for r in records],
    }
    return {name: bootstrap_ci(vals)[0] for name, vals in groups.items()} | {"n": len(records)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--tune-sample-size", type=int, default=TUNE_SAMPLE_SIZE)
    parser.add_argument("--eval-sample-size", type=int, default=EVAL_SAMPLE_SIZE)
    args = parser.parse_args()

    t_start = time.time()
    print("=== Loading already-built MIND-large data ===")
    articles = pl.read_parquet(PROC_DIR / "articles.parquet")
    user_history = pl.read_parquet(PROC_DIR / "user_history.parquet")
    behaviors_train = pl.read_parquet(PROC_DIR / "behaviors_train.parquet")
    behaviors_val = pl.read_parquet(PROC_DIR / "behaviors_val.parquet")
    embeddings_df = pl.read_parquet(PROC_DIR / "article_embeddings.parquet")

    bm25 = BM25Index().build(articles)
    emb_index = EmbeddingIndex().build(embeddings_df)
    hist_index = UserHistoryIndex(user_history, articles)
    feature_builder = FeatureBuilder("mind", articles, bm25, emb_index, behaviors_train)

    print(f"\n=== Building tuning-train sample ({args.tune_sample_size:,}) and eval sample ({args.eval_sample_size:,}) ===")
    tune_sample = build_train_sample(behaviors_train, hist_index, args.tune_sample_size)
    tune_features = build_frame_from_sample(tune_sample, feature_builder)
    tune_frame = build_training_frame(tune_features)
    print(f"tuning-train: {tune_frame.height:,} rows ({tune_frame['impression_id'].n_unique():,} impressions)")

    eval_sample_full = build_train_sample(behaviors_val, hist_index, args.eval_sample_size)
    print(f"eval sample: {len(eval_sample_full):,} impressions from behaviors_val (real dev-split labels)")

    print(f"\n=== Random search: {args.n_trials} trials ===")
    rng = random.Random(SEED)
    results = []
    for trial in range(args.n_trials):
        params = sample_params(rng)
        t0 = time.time()
        model = train_gbdt(tune_frame, FEATURE_COLUMNS, **params)
        metrics = evaluate_on_val(eval_sample_full, model, feature_builder)
        elapsed = time.time() - t0
        results.append({"trial": trial, "params": params, "metrics": metrics, "seconds": elapsed})
        print(
            f"trial {trial:>2}: {params} -> AUC={metrics['AUC']:.4f} MRR={metrics['MRR']:.4f} "
            f"nDCG@10={metrics['nDCG@10']:.4f} ({elapsed:.1f}s, n={metrics['n']:,})"
        )

    results.sort(key=lambda r: r["metrics"]["AUC"], reverse=True)
    best = results[0]
    total_elapsed = time.time() - t_start

    print("\n=== Results, sorted by AUC ===")
    print(f"{'trial':>5} | {'AUC':>7} | {'MRR':>7} | {'nDCG@5':>7} | {'nDCG@10':>8} | params")
    for r in results:
        m = r["metrics"]
        print(f"{r['trial']:>5} | {m['AUC']:>7.4f} | {m['MRR']:>7.4f} | {m['nDCG@5']:>7.4f} | "
              f"{m['nDCG@10']:>8.4f} | {r['params']}")

    print(f"\nBEST: trial {best['trial']} -- {best['params']}")
    print(f"  AUC={best['metrics']['AUC']:.4f} MRR={best['metrics']['MRR']:.4f} "
          f"nDCG@5={best['metrics']['nDCG@5']:.4f} nDCG@10={best['metrics']['nDCG@10']:.4f}")
    print(f"\nTotal search time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min) "
          f"for {args.n_trials} trials + setup")

    out_path = REPO_ROOT / "results" / "mind_gbdt_hyperparam_search.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [
            {"trial": r["trial"], **r["params"], **{k: v for k, v in r["metrics"].items() if k != "n"},
             "n": r["metrics"]["n"], "seconds": r["seconds"]}
            for r in results
        ]
    ).write_csv(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
