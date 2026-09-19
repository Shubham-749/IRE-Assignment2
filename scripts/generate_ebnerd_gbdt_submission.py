#!/usr/bin/env python
"""Q5/Q6 (GBDT track): generate the EB-NeRD Codabench submission using a GBDT
re-ranker trained at large scale. Same two-phase shape as
generate_mind_gbdt_submission.py (train on a large-train sample, score the full
unlabeled test set) -- see that script's docstring for the full rationale.

    python scripts/generate_ebnerd_gbdt_submission.py [--limit N] [--train-sample-size N]

EB-NeRD's large history.parquet OOM'd this exact 17GB-RAM machine repeatedly, in two
different places, both traced to the same root cause -- exploding every user's full
click-history list at once (`.explode(["article_id_fixed", "impression_time_fixed"])`
over history.parquet):

1. ire_a1.clean_ebnerd.clean_user_history() -- fixed in that module (per-user local
   dedup instead of one global explode+unique), but even the fixed version still
   OOM'd once tried against EB-NeRD-large's real train+validation history: the
   python-list accumulation across *every* user (potentially millions), not just the
   explode itself, was still too much for this machine.
2. ire_a2.features._load_ebnerd_dwell_lookup() -- same explode pattern, over the same
   file, never reached in testing because (1) already crashed first, but would
   almost certainly hit the same wall.

This script's actual fix, applied here rather than upstream, since it's specific to
what a large-scale *submission* run needs (not a general-purpose correctness fix like
the two above): never build history for every user. Training only ever needs the
~50,000 sampled impressions' own users' history -- build_targeted_history_lookup()
filters history.parquet down to just those user_ids *before* exploding anything, so
memory scales with the training sample, not the full user base. The EB-NeRD-only
session/dwell FeatureBuilder enrichment (points 2 above) is skipped for this
large-scale run entirely (FeatureBuilder built without raw_dir/scale) -- those three
features (session_ordinal, session_click_count_so_far, hist_avg_dwell_seconds) are
neutral/zero here, same as they already are for MIND everywhere. The other ten
features (bm25/embedding score, position, click-history recency/category, popularity,
freshness) are fully computed, for both training and test scoring.

Codabench's required filename (confirmed directly by the user on the Submission
Guidelines page, per generate_ebnerd_submission.py): predictions.txt.
"""

import argparse
import gc
import sys
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import polars as pl  # noqa: E402

from ire_a1.clean_ebnerd import clean_articles, clean_behaviors  # noqa: E402
from ire_a1.download import download_ebnerd, load_config  # noqa: E402
from ire_a1.eval.submission import rank_and_group  # noqa: E402
from ire_a1.retrieval import eval_utils  # noqa: E402
from ire_a1.retrieval.bm25 import BM25Index  # noqa: E402
from ire_a1.retrieval.embeddings import EmbeddingIndex, compute_embeddings  # noqa: E402
from ire_a2.features import FEATURE_COLUMNS, MAX_HISTORY, FeatureBuilder  # noqa: E402
from ire_a2.reranker import build_training_frame, feature_importance, train_gbdt  # noqa: E402

RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PROC_DIR = PROCESSED_DIR / "ebnerd" / "large"
LARGE_DIR = RAW_DIR / "ebnerd" / "large"
TEST_BEHAVIORS_DIR = LARGE_DIR / "ebnerd_testset" / "test"
TRAIN_SAMPLE_SIZE = 50_000
TRAIN_OVERSAMPLE_FACTOR = 1.5  # candidate pool before the has-history filter narrows it --
# kept small on purpose: this machine's memory ceiling is the binding constraint here
# (see module docstring), not sample quality, and a bigger pool means a bigger
# needed_users set for build_targeted_history_lookup() below
BATCH_SIZE = 10_000


def build_targeted_history_lookup(history_path: Path, user_ids: set, articles: pl.DataFrame) -> dict:
    """user_id -> full (not truncated) list of (click_ts, article_id, title) tuples,
    most-recent-first, for exactly `user_ids` -- filtering to just the users we need
    *before* exploding anything is what keeps this bounded (see module docstring).
    Full history (not truncated to MAX_HISTORY) because training needs real
    point-in-time cutoff filtering, unlike the unconditional-full-history test path.

    Reads only the *matching* split's own history.parquet (train history for a train
    sample, validation history for a validation sample), not both combined -- a
    split's history.parquet already holds each of its own users' full history up to
    that point, and users needed here come from that same split's own behaviors, so
    the other split's file is largely redundant for this purpose. Skipping it roughly
    halves the data this function ever reads, which is what actually fixed the
    remaining OOM risk (the per-split-then-merge version still ran this machine out of
    memory building history for a 50,000-impression training sample's ~140K users).
    """
    title_map = dict(zip(articles["article_id"].to_list(), articles["title"].to_list()))
    if not history_path.exists():
        return {}
    raw = (
        pl.read_parquet(history_path)
        .select(pl.col("user_id").cast(pl.Utf8), "article_id_fixed", "impression_time_fixed")
        .filter(pl.col("user_id").is_in(user_ids))
    )
    lookup: dict = {}
    for user_id, aids, tss in raw.iter_rows():
        deduped = dict.fromkeys(zip(aids, tss))
        items = sorted(deduped.keys(), key=lambda p: p[1], reverse=True)
        lookup[user_id] = [(ts, str(aid), title_map.get(str(aid))) for aid, ts in items]
    return lookup


def recent_from_lookup(lookup: dict, user_id: str, cutoff_ts, max_n: int = MAX_HISTORY) -> list:
    items = lookup.get(user_id, [])
    return [t for t in items if t[0] < cutoff_ts][:max_n]  # already sorted most-recent-first


def build_train_sample(
    behaviors: pl.DataFrame, articles: pl.DataFrame, n: int, split_dirname: str = "train"
) -> list[dict]:
    """`split_dirname` picks which split's own history.parquet to build the targeted
    lookup from -- "train" for a training sample (the normal case), "validation" when
    reused to build an eval sample from the real validation split (tune_ebnerd_gbdt.py).
    """
    has_click = behaviors.filter(
        pl.col("clicked_article_ids").is_not_null() & (pl.col("clicked_article_ids").list.len() > 0)
    )
    pool_size = int(min(n * TRAIN_OVERSAMPLE_FACTOR, has_click.height))
    candidates = has_click.sample(n=pool_size, seed=eval_utils.SEED, shuffle=True)
    print(f"  candidate pool: {candidates.height:,} impressions (before has-history filter)")

    needed_users = set(candidates["user_id"].unique().to_list())
    history_lookup = build_targeted_history_lookup(
        LARGE_DIR / split_dirname / "history.parquet", needed_users, articles
    )
    print(f"  targeted history built for {len(history_lookup):,} users")

    eligible = []
    for row in candidates.iter_rows(named=True):
        history = recent_from_lookup(history_lookup, row["user_id"], row["timestamp"])
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
    print(f"  {len(eligible):,} candidates had non-empty point-in-time history")
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


def build_test_history_lookup(articles: pl.DataFrame) -> dict:
    """user_id -> up to MAX_HISTORY (click_ts, article_id, title) tuples, most-recent
    first. Bounded to (test users x MAX_HISTORY), not (test users x full history) --
    safe unconditionally since every test impression for a user shares the same given
    history (no point-in-time cutoff needed; see module docstring / A1 precedent).
    """
    title_map = dict(zip(articles["article_id"].to_list(), articles["title"].to_list()))
    t0 = time.time()
    raw = pl.read_parquet(TEST_BEHAVIORS_DIR / "history.parquet")
    print(f"  {raw.height:,} test users with history ({time.time() - t0:.1f}s to load)")

    lookup = {}
    for user_id, article_ids, timestamps in raw.select(
        "user_id", "article_id_fixed", "impression_time_fixed"
    ).iter_rows():
        deduped = dict.fromkeys(zip(article_ids, timestamps))
        top = sorted(deduped.keys(), key=lambda p: p[1], reverse=True)[:MAX_HISTORY]
        lookup[str(user_id)] = [(ts, str(aid), title_map.get(str(aid))) for aid, ts in top]
    print(f"  built history lookup for {len(lookup):,} users in {time.time() - t0:.1f}s total")
    return lookup


def score_and_write(
    behaviors_test: pl.DataFrame,
    history_lookup: dict,
    feature_builder: FeatureBuilder,
    model,
    out_path: Path,
    batch_size: int = BATCH_SIZE,
) -> int:
    """EB-NeRD's real large test set has 200,000 rows (out of 13,536,710) that all
    share raw_impression_id == "0" -- a genuine, verified data characteristic (not a
    bug in our own cleaning), concentrated in one region of the file. rank_and_group()
    groups by id_col, so grouping directly on raw_impression_id would silently
    collapse all "0" rows sharing a batch into a single output line -- verified this
    actually happens (toy repro: three rows sharing an id become one 3-element
    rank_order, not three separate lines), which would desync every following line
    from Codabench's positionally-compared ground truth. Every row gets a synthetic
    row_uid (global row index, always unique) to group by instead; raw_impression_id
    is only reattached afterward, for the line actually written.
    """
    total_rows = behaviors_test.height
    written = 0
    t0 = time.time()
    with open(out_path, "w") as f:
        for start in range(0, total_rows, batch_size):
            batch = behaviors_test.slice(start, batch_size)

            rows = []
            for local_idx, row in enumerate(batch.iter_rows(named=True)):
                row_uid = start + local_idx
                history = history_lookup.get(row["user_id"], [])
                for pos, fr in enumerate(
                    feature_builder.build_rows(row["user_id"], row["timestamp"], row["candidate_article_ids"], history)
                ):
                    fr["row_uid"] = row_uid
                    fr["raw_impression_id"] = row["raw_impression_id"]
                    fr["pos"] = pos
                    rows.append(fr)

            if rows:
                batch_frame = pl.DataFrame(rows)
                id_map = dict(
                    zip(batch_frame["row_uid"].to_list(), batch_frame["raw_impression_id"].to_list())
                )
                X = batch_frame.select(FEATURE_COLUMNS).to_numpy()
                scores = model.predict(X)
                scored = rank_and_group(
                    batch_frame.select("row_uid", "pos").with_columns(pl.Series("score", scores)),
                    id_col="row_uid",
                )
                for row_uid, ranks in zip(scored["row_uid"].to_list(), scored["rank_order"].to_list()):
                    imp_id = id_map[row_uid]
                    f.write(f"{imp_id} [{','.join(map(str, ranks))}]\n")
                written += scored.height

            del rows
            batch_num = start // batch_size + 1
            if batch_num % 10 == 0:
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
    parser.add_argument("--num-leaves", type=int, default=None, dest="num_leaves")
    parser.add_argument("--learning-rate", type=float, default=None, dest="learning_rate")
    parser.add_argument("--n-estimators", type=int, default=None, dest="n_estimators")
    parser.add_argument("--min-child-samples", type=int, default=None, dest="min_child_samples")
    args = parser.parse_args()

    print("=== Step 1: raw data ===")
    config = load_config()
    download_ebnerd("large", RAW_DIR, config)

    print("\n=== Step 2: articles + train/validation behaviors (no full history load) ===")
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    articles = clean_articles(LARGE_DIR)
    behaviors_train = clean_behaviors(LARGE_DIR / "train", "train")
    print(f"articles={articles.height:,} train_impressions={behaviors_train.height:,}")

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
        # Only relevant when compute_embeddings() actually ran (loads a
        # sentence-transformer onto MPS/CUDA) -- an earlier version of this script
        # called this unconditionally, including on cache-hit runs that never touched
        # MPS at all, and a later, unrelated LightGBM call then crashed with SIGSEGV
        # at the exact same spot on every retry. Scoping it to only the branch that
        # actually used MPS removes that unconditional call from cache-hit runs.
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()

    bm25 = BM25Index().build(articles)
    emb_index = EmbeddingIndex().build(embeddings_df)
    # No raw_dir/scale here -- skips EB-NeRD's session/dwell enrichment, which reads
    # (and explodes) the same history.parquet files that OOM'd at this scale. Those
    # three features stay neutral/zero for this large-scale run; see module docstring.
    feature_builder = FeatureBuilder("ebnerd", articles, bm25, emb_index, behaviors_train)

    print(f"\n=== Step 4: train GBDT on a {args.train_sample_size:,}-impression train sample ===")
    train_sample = build_train_sample(behaviors_train, articles, args.train_sample_size)
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

    del behaviors_train, train_sample, train_features, train_frame
    gc.collect()

    print("\n=== Step 5: score test set ===")
    print("cleaning test behaviors...")
    behaviors_test = clean_behaviors(TEST_BEHAVIORS_DIR, split="test")
    if args.limit:
        behaviors_test = behaviors_test.head(args.limit)
        print(f"--limit set: only scoring first {args.limit:,} impressions")
    print(f"{behaviors_test.height:,} impressions to score")

    print("building test history lookup (bounded, local per-user dedup + truncation)...")
    history_lookup = build_test_history_lookup(articles)

    out_name = f"ebnerd_a2_gbdt_predictions{'_sample' if args.limit else ''}.txt"
    out_path = REPO_ROOT / out_name
    written = score_and_write(behaviors_test, history_lookup, feature_builder, model, out_path)
    print(f"\nwrote {written:,} predictions to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    zip_path = out_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_path, arcname="predictions.txt")
    print(f"zipped -> {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
