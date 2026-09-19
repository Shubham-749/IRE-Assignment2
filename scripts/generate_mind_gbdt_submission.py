#!/usr/bin/env python
"""Q5/Q6 (GBDT track): generate the MIND Codabench submission using a GBDT re-ranker
trained at large scale, not the demo/small-scale model used for local evaluation.

    python scripts/generate_mind_gbdt_submission.py [--limit N] [--train-sample-size N]

Two phases:
1. Train: sample --train-sample-size (default 50,000) impressions from MIND-large's
   real train split, build Q1 features for them, train a fresh GBDT. Not the
   demo/small model -- candidate_popularity (this repo's #1-importance feature on
   every run so far) is a raw click count; training on large-scale counts and scoring
   on large-scale counts keeps that feature internally consistent, which reusing the
   small-scale model would not.
2. Score: run the trained GBDT over all 2,370,727 unlabeled test impressions,
   batched (BATCH_SIZE impressions/batch, GBDT scored once per batch not once per
   impression -- avoids multiplying LightGBM's per-call overhead by 2.37M), with
   periodic gc.collect() -- generate_ebnerd_submission.py's hard-won lesson on this
   exact machine (repeated OOM kills on a simpler, non-ML scoring path) applies at
   least as much here, since FeatureBuilder allocates far more per impression.

Reuses ire_a1.feature_store.build_feature_store() for the raw-to-clean-parquet step
(scale="large" -- same function demo/small already use, just a bigger scale) and
ire_a2's FeatureBuilder/train_gbdt/build_training_frame unmodified -- this script is
new orchestration, not new feature/training logic.

Codabench's required filename inside the zip (confirmed the hard way in A1, not
guessed): prediction.txt.
"""

import argparse
import gc
import sys
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from ire_a1.download import download_mind, load_config  # noqa: E402
from ire_a1.eval.submission import rank_and_group  # noqa: E402
from ire_a1.feature_store import UserHistoryIndex, build_feature_store  # noqa: E402
from ire_a1.retrieval import eval_utils  # noqa: E402
from ire_a1.retrieval.bm25 import BM25Index  # noqa: E402
from ire_a1.retrieval.embeddings import EmbeddingIndex, compute_embeddings  # noqa: E402
from ire_a2.features import FEATURE_COLUMNS, MAX_HISTORY, FeatureBuilder  # noqa: E402
from ire_a2.reranker import build_training_frame, feature_importance, train_gbdt  # noqa: E402

RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PROC_DIR = PROCESSED_DIR / "mind" / "large"
TRAIN_SAMPLE_SIZE = 50_000
BATCH_SIZE = 20_000  # matches generate_mind_submission.py's batch size


def build_train_sample(behaviors_train: pl.DataFrame, hist_index: UserHistoryIndex, n: int) -> list[dict]:
    has_click = behaviors_train.filter(
        pl.col("clicked_article_ids").is_not_null() & (pl.col("clicked_article_ids").list.len() > 0)
    )
    eligible = []
    for row in has_click.iter_rows(named=True):
        history = hist_index.recent(row["user_id"], row["timestamp"], MAX_HISTORY)
        if not history:
            continue
        eligible.append(
            {
                "impression_id": row["impression_id"],
                "user_id": row["user_id"],
                "timestamp": row["timestamp"],
                "candidates": row["candidate_article_ids"],
                "clicked": row["clicked_article_ids"],
                "history": history,
            }
        )
    return eval_utils.sample_impressions(eligible, sample_size=n)


def build_frame_from_sample(sample: list[dict], feature_builder: FeatureBuilder) -> pl.DataFrame:
    rows = []
    for imp in sample:
        clicked = set(imp["clicked"])
        for fr in feature_builder.build_rows(imp["user_id"], imp["timestamp"], imp["candidates"], imp["history"]):
            fr["impression_id"] = imp["impression_id"]
            fr["label"] = 1 if fr["article_id"] in clicked else 0
            rows.append(fr)
    return pl.DataFrame(rows)


def score_and_write(
    behaviors_test: pl.DataFrame,
    hist_index: UserHistoryIndex,
    feature_builder: FeatureBuilder,
    model,
    out_path: Path,
    batch_size: int = BATCH_SIZE,
) -> int:
    total_rows = behaviors_test.height
    written = 0
    t0 = time.time()
    with open(out_path, "w") as f:
        for start in range(0, total_rows, batch_size):
            batch = behaviors_test.slice(start, batch_size)

            rows = []
            for row in batch.iter_rows(named=True):
                history = hist_index.recent(row["user_id"], row["timestamp"], MAX_HISTORY)
                for pos, fr in enumerate(
                    feature_builder.build_rows(row["user_id"], row["timestamp"], row["candidate_article_ids"], history)
                ):
                    fr["raw_impression_id"] = row["raw_impression_id"]
                    fr["pos"] = pos
                    rows.append(fr)

            if rows:
                batch_frame = pl.DataFrame(rows)
                X = batch_frame.select(FEATURE_COLUMNS).to_numpy()
                scores = model.predict(X)
                scored = rank_and_group(
                    batch_frame.select("raw_impression_id", "pos").with_columns(pl.Series("score", scores)),
                    id_col="raw_impression_id",
                )
                for imp_id, ranks in zip(scored["raw_impression_id"].to_list(), scored["rank_order"].to_list()):
                    f.write(f"{imp_id} [{','.join(map(str, ranks))}]\n")
                written += scored.height

            del rows
            batch_num = start // batch_size + 1
            if batch_num % 5 == 0:
                gc.collect()
                f.flush()

            n_batches = (total_rows + batch_size - 1) // batch_size
            elapsed = time.time() - t0
            print(
                f"  batch {batch_num}/{n_batches}: {written:,}/{total_rows:,} written "
                f"({elapsed:.1f}s elapsed, {elapsed / batch_num:.2f}s/batch avg)",
                flush=True,
            )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only score the first N test impressions")
    parser.add_argument("--train-sample-size", type=int, default=TRAIN_SAMPLE_SIZE)
    parser.add_argument("--num-leaves", type=int, default=None, dest="num_leaves",
                         help="override train_gbdt()'s default, e.g. from tune_mind_gbdt.py's search")
    parser.add_argument("--learning-rate", type=float, default=None, dest="learning_rate")
    parser.add_argument("--n-estimators", type=int, default=None, dest="n_estimators")
    parser.add_argument("--min-child-samples", type=int, default=None, dest="min_child_samples")
    args = parser.parse_args()

    print("=== Step 1: raw data ===")
    config = load_config()
    download_mind("large", RAW_DIR, config)

    print("\n=== Step 2: clean + feature store (scale=large) ===")
    build_feature_store("mind", "large", RAW_DIR, PROCESSED_DIR)

    articles = pl.read_parquet(PROC_DIR / "articles.parquet")
    user_history = pl.read_parquet(PROC_DIR / "user_history.parquet")
    behaviors_train = pl.read_parquet(PROC_DIR / "behaviors_train.parquet")
    behaviors_test = pl.read_parquet(PROC_DIR / "behaviors_test.parquet")
    print(f"articles={articles.height:,} train_impressions={behaviors_train.height:,} "
          f"test_impressions={behaviors_test.height:,}")

    print("\n=== Step 3: embeddings + indices ===")
    emb_path = PROC_DIR / "article_embeddings.parquet"
    if emb_path.exists():
        embeddings_df = pl.read_parquet(emb_path)
        print(f"[skip] loaded cached embeddings ({embeddings_df.height:,} rows)")
    else:
        t0 = time.time()
        embeddings_df = compute_embeddings(articles)
        embeddings_df.write_parquet(emb_path)
        print(f"computed embeddings for {embeddings_df.height:,} articles in {time.time() - t0:.1f}s")

    bm25 = BM25Index().build(articles)
    emb_index = EmbeddingIndex().build(embeddings_df)
    hist_index = UserHistoryIndex(user_history, articles)
    feature_builder = FeatureBuilder("mind", articles, bm25, emb_index, behaviors_train)

    print(f"\n=== Step 4: train GBDT on a {args.train_sample_size:,}-impression train sample ===")
    train_sample = build_train_sample(behaviors_train, hist_index, args.train_sample_size)
    print(f"sampled {len(train_sample):,} train impressions")
    train_features = build_frame_from_sample(train_sample, feature_builder)
    train_frame = build_training_frame(train_features)
    print(f"training on {train_frame.height:,} rows ({train_frame['impression_id'].n_unique():,} impressions)")
    gbdt_params = {}
    if args.num_leaves is not None:
        gbdt_params["num_leaves"] = args.num_leaves
    if args.learning_rate is not None:
        gbdt_params["learning_rate"] = args.learning_rate
    if args.n_estimators is not None:
        gbdt_params["n_estimators"] = args.n_estimators
    if args.min_child_samples is not None:
        gbdt_params["min_child_samples"] = args.min_child_samples
    if gbdt_params:
        print(f"GBDT hyperparameter overrides: {gbdt_params}")
    model = train_gbdt(train_frame, FEATURE_COLUMNS, **gbdt_params)
    print("top feature importances (gain):")
    for name, gain in feature_importance(model, FEATURE_COLUMNS)[:8]:
        print(f"  {name:>28} {gain:>10.1f}")

    print("\n=== Step 5: score test set ===")
    if args.limit:
        behaviors_test = behaviors_test.head(args.limit)
        print(f"--limit set: only scoring first {args.limit:,} impressions")
    print(f"{behaviors_test.height:,} impressions to score")

    out_name = f"mind_a2_gbdt_predictions{'_sample' if args.limit else ''}.txt"
    out_path = REPO_ROOT / out_name
    written = score_and_write(behaviors_test, hist_index, feature_builder, model, out_path)
    print(f"\nwrote {written:,} predictions to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    zip_path = out_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_path, arcname="prediction.txt")
    print(f"zipped -> {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
