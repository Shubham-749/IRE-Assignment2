#!/usr/bin/env python
"""Q5: generate the EB-NeRD Codabench submission from the real, unlabeled large test set.

    python scripts/generate_ebnerd_submission.py [--limit N]

Same approach as generate_mind_submission.py (embeddings, not BM25 -- see that script's
docstring for why), same shared rank_and_group() for the ordering-sensitive part, same
--limit-then-full-run discipline. EB-NeRD's test set is lighter per-impression than
MIND's (avg ~15.2 candidates/impression vs. MIND's ~39.4) but has more impressions in
total (13,536,710 vs. 2,370,727).

Real large-test layout, confirmed by inspection (not assumed):
    data/raw/ebnerd/large/articles.parquet          -- catalog, 125,541 articles
    data/raw/ebnerd/large/ebnerd_testset/test/behaviors.parquet   -- 13,536,710 rows,
        no article_ids_clicked column at all (genuinely unlabeled)
    data/raw/ebnerd/large/ebnerd_testset/test/history.parquet     -- 807,677 test users
`articles_large_only.zip` turned out to be a byte-identical standalone copy of the same
125,541-article catalog already in ebnerd_large.zip, not a supplementary set.

Codabench's required filename, confirmed directly by the user on the Submission
Guidelines page (unlike MIND, where we had to find out the hard way): predictions.txt.
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

from ire_a1.clean_ebnerd import clean_articles, clean_behaviors  # noqa: E402
from ire_a1.eval.submission import rank_and_group  # noqa: E402
from ire_a1.retrieval.embeddings import EmbeddingIndex, compute_embeddings  # noqa: E402

LARGE_DIR = REPO_ROOT / "data" / "raw" / "ebnerd" / "large"
TESTSET_DIR = LARGE_DIR / "ebnerd_testset"
TEST_BEHAVIORS_DIR = TESTSET_DIR / "test"
CACHE_DIR = REPO_ROOT / "data" / "processed" / "ebnerd" / "large_test_only"
BATCH_SIZE = 50_000  # impressions per batch; ~50k * ~15 candidates * 384-dim * 2 arrays ~= 2.3GB transient


def load_articles_and_embeddings() -> tuple[pl.DataFrame, pl.DataFrame]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    articles_path = CACHE_DIR / "articles.parquet"
    emb_path = CACHE_DIR / "article_embeddings.parquet"

    if articles_path.exists():
        articles = pl.read_parquet(articles_path)
        print(f"[skip] loaded cached articles ({articles.height:,} rows)")
    else:
        articles = clean_articles(LARGE_DIR)
        articles.write_parquet(articles_path)
        print(f"cleaned + cached {articles.height:,} articles -> {articles_path}")

    if emb_path.exists():
        embeddings_df = pl.read_parquet(emb_path)
        print(f"[skip] loaded cached embeddings ({embeddings_df.height:,} rows)")
    else:
        t0 = time.time()
        embeddings_df = compute_embeddings(articles)
        embeddings_df.write_parquet(emb_path)
        print(f"computed + cached embeddings for {embeddings_df.height:,} articles in {time.time()-t0:.1f}s -> {emb_path}")

    return articles, embeddings_df


def clean_behaviors_lazy_slice(split_dir: Path, split: str, skip: int, limit: int | None = None) -> pl.DataFrame:
    """Same transform as clean_ebnerd.clean_behaviors(), but scans the parquet lazily
    and pushes the `skip`/`limit` slice down *before* materializing -- for a --skip
    resume (optionally chunked further via --limit) we only need a small window of a
    13.5M-row file, and eagerly reading+casting the whole thing first (what
    clean_behaviors() does) wastes real memory on rows we immediately discard, right
    when memory is already the tightest constraint in this script. Verified to produce
    identical output to the eager clean_behaviors().slice() on real data (.equals() ==
    True) before being trusted for the actual resume.
    """
    lf = pl.scan_parquet(split_dir / "behaviors.parquet").slice(skip, limit)
    has_labels = "article_ids_clicked" in lf.collect_schema().names()
    clicked_expr = (
        pl.col("article_ids_clicked").cast(pl.List(pl.Utf8))
        if has_labels
        else pl.lit(None, dtype=pl.List(pl.Utf8))
    ).alias("clicked_article_ids")
    return lf.select(
        (pl.lit(f"ebnerd_{split}_") + pl.col("impression_id").cast(pl.Utf8)).alias("impression_id"),
        pl.col("impression_id").cast(pl.Utf8).alias("raw_impression_id"),
        pl.col("user_id").cast(pl.Utf8),
        pl.lit("ebnerd").alias("dataset"),
        pl.col("impression_time").alias("timestamp"),
        pl.col("article_ids_inview").cast(pl.List(pl.Utf8)).alias("candidate_article_ids"),
        clicked_expr,
    ).collect()


def build_user_vectors(emb_index: EmbeddingIndex) -> tuple[np.ndarray, dict]:
    """One mean-pooled, L2-normalized vector per distinct test user, from their own
    test/history.parquet (real per-click timestamps, but we use the full given history
    unconditionally -- no leakage concern, we're predicting genuinely future, held-out
    impressions from real past history).
    Reads test/history.parquet directly rather than going through clean_user_history()
    -- that function explodes each user's history list into a long-format table, dedups
    exact (user_id, article_id, click_timestamp) duplicates, then re-groups it right
    back into a list. It exists to support leakage-safe point-in-time filtering
    elsewhere; we don't need the filtering here (full given history, unconditionally --
    see docstring above), but the *dedup* still matters for consistency -- 1.44% of
    users (722/50,000 sampled) have true exact-duplicate (article_id, timestamp) rows
    in their raw history, and skipping the dedup would silently over-weight that
    article in their mean-pooled vector relative to every other user vector computed
    this run. On this dataset clean_user_history()'s dedup requires a global 116.8M-row
    explode (avg history length 144.6, max 1530 per user) to do it, which is what was
    actually causing repeated out-of-memory kills on this 17.2GB machine -- not the
    per-batch scoring loop, which a `--skip`-resumed run confirmed still crashed even
    with almost no scoring work left to do. Since dedup is inherently per-user anyway,
    it doesn't need a global explode: deduping each user's own (already-grouped) list
    locally produces the identical result -- verified against clean_user_history()'s
    output on a real sample -- for a fraction of the memory.
    """
    t0 = time.time()
    raw = pl.read_parquet(TEST_BEHAVIORS_DIR / "history.parquet")
    print(f"{raw.height:,} distinct users with history ({time.time()-t0:.1f}s to load)")

    user_ids = []
    vectors = []
    n_empty = 0
    for user_id, article_ids, timestamps in raw.select(
        "user_id", "article_id_fixed", "impression_time_fixed"
    ).iter_rows():
        # Dedup only exact (article_id, timestamp) duplicate rows -- matches
        # clean_user_history()'s dedup exactly. Re-reads of the same article at a
        # *different* timestamp are kept as separate entries (not deduped further),
        # so they still get their extra weight in the mean below, same as the old path.
        deduped_pairs = dict.fromkeys(zip(article_ids, timestamps))
        kept_article_ids = [aid for aid, _ts in deduped_pairs]
        vecs = [v for aid in kept_article_ids if (v := emb_index.get_embedding(str(aid))) is not None]
        if not vecs:
            n_empty += 1
            continue
        mean_vec = np.mean(vecs, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        # cast to str: behaviors' user_id is Utf8 (unified schema convention), but the
        # raw history.parquet's user_id is an integer -- must match for the later join.
        user_ids.append(str(user_id))
        vectors.append(mean_vec)
    print(f"built {len(user_ids):,} user vectors ({n_empty:,} users with no usable history) "
          f"in {time.time()-t0:.1f}s total")

    matrix = np.asarray(vectors, dtype=np.float32)
    user_id_to_row = {uid: i for i, uid in enumerate(user_ids)}
    return matrix, user_id_to_row


def score_and_write_simple(
    behaviors: pl.DataFrame,
    emb_index: EmbeddingIndex,
    user_matrix: np.ndarray,
    user_id_to_row: dict,
    out_path: Path,
    append: bool = False,
) -> int:
    """Same scoring math as score_and_write() (mean-pooled user vector . candidate
    embedding, ranked descending), but a plain per-impression Python loop with direct
    dict lookups instead of building article_idx_df/user_idx_df and joining an exploded
    frame against them. For finishing a small remainder (a few hundred thousand rows,
    not 13.5M) this trades a bit of speed for much more predictable memory -- the
    join-based path's memory jumped ~2.8GB in 3 seconds even for a single 50K-row
    batch, which the explode alone (~760K rows) doesn't explain; this sidesteps
    whatever in the join machinery was doing that rather than chase it further.
    """
    written = 0
    with open(out_path, "a" if append else "w") as f:
        for row in behaviors.iter_rows(named=True):
            candidates = row["candidate_article_ids"]
            user_row = user_id_to_row.get(row["user_id"])
            user_vec = user_matrix[user_row] if user_row is not None else None
            scores = []
            for aid in candidates:
                cand_vec = emb_index.get_embedding(aid)
                if user_vec is not None and cand_vec is not None:
                    scores.append(float(np.dot(user_vec, cand_vec)))
                else:
                    scores.append(0.0)
            order = sorted(range(len(candidates)), key=lambda i: -scores[i])
            rank_of_pos = [0] * len(candidates)
            for rank, pos in enumerate(order, start=1):
                rank_of_pos[pos] = rank
            f.write(f"{row['raw_impression_id']} [{','.join(map(str, rank_of_pos))}]\n")
            written += 1
    return written


def score_and_write(
    behaviors: pl.DataFrame,
    emb_index: EmbeddingIndex,
    user_matrix: np.ndarray,
    user_id_to_row: dict,
    out_path: Path,
    batch_size: int = BATCH_SIZE,
    append: bool = False,
) -> int:
    article_idx_df = pl.DataFrame(
        {"article_id": list(emb_index._id_to_idx.keys()), "article_row": list(emb_index._id_to_idx.values())}
    )
    user_idx_df = pl.DataFrame(
        {"user_id": list(user_id_to_row.keys()), "user_row": list(user_id_to_row.values())}
    )

    total_rows = behaviors.height
    written = 0
    t0 = time.time()
    with open(out_path, "a" if append else "w") as f:
        for start in range(0, total_rows, batch_size):
            batch = behaviors.slice(start, batch_size).select("raw_impression_id", "user_id", "candidate_article_ids")
            exploded = (
                batch.with_columns(pl.int_ranges(0, pl.col("candidate_article_ids").list.len()).alias("pos"))
                .explode(["candidate_article_ids", "pos"])
                .rename({"candidate_article_ids": "article_id"})
                .join(article_idx_df, on="article_id", how="left")
                .join(user_idx_df, on="user_id", how="left")
            )

            article_row = exploded["article_row"].fill_null(-1).to_numpy()
            user_row = exploded["user_row"].fill_null(-1).to_numpy()
            valid = (article_row >= 0) & (user_row >= 0)

            scores = np.zeros(len(exploded), dtype=np.float32)
            if valid.any():
                uv = user_matrix[user_row[valid]]
                cv = emb_index.matrix[article_row[valid]]
                scores[valid] = np.einsum("ij,ij->i", uv, cv)

            scored = rank_and_group(exploded.with_columns(pl.Series("score", scores)), id_col="raw_impression_id")

            for imp_id, ranks in zip(scored["raw_impression_id"].to_list(), scored["rank_order"].to_list()):
                f.write(f"{imp_id} [{','.join(map(str, ranks))}]\n")
            written += scored.height

            # Explicitly drop per-batch DataFrames/arrays and periodically force a GC
            # pass -- 270+ iterations of polars/numpy allocation without this can grow
            # the process's working set enough to hit the machine's 17GB RAM ceiling
            # well before the loop naturally ends (observed: killed via SIGKILL at the
            # exact same row 3 times in a row on this 13.5M-row run).
            del batch, exploded, article_row, user_row, valid, scores, scored
            batch_num = start // batch_size + 1
            if batch_num % 10 == 0:
                gc.collect()
                f.flush()

            n_batches = (total_rows + batch_size - 1) // batch_size
            elapsed = time.time() - t0
            print(f"  batch {batch_num}/{n_batches}: {written:,}/{total_rows:,} written "
                  f"({elapsed:.1f}s elapsed, {elapsed/batch_num:.2f}s/batch avg)", flush=True)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N impressions")
    parser.add_argument("--skip", type=int, default=0,
                         help="skip the first N impressions (already written by a prior, interrupted run) "
                              "and append to the existing output file instead of overwriting it")
    args = parser.parse_args()

    articles, embeddings_df = load_articles_and_embeddings()
    emb_index = EmbeddingIndex().build(embeddings_df)
    print(f"EmbeddingIndex: {len(emb_index.article_ids):,} articles, dim={emb_index.matrix.shape[1]}")

    user_matrix, user_id_to_row = build_user_vectors(emb_index)

    print("cleaning behaviors...")
    if args.skip:
        # limit pushed down into the same lazy slice -- for a chunked resume (--skip N
        # --limit M) this materializes only the M rows of this chunk, not the full
        # remainder or the whole 13.5M-row file.
        behaviors = clean_behaviors_lazy_slice(TEST_BEHAVIORS_DIR, split="test", skip=args.skip, limit=args.limit)
        print(f"--skip set: resuming after the first {args.skip:,} already-written impressions "
              f"(read lazily -- did not materialize skipped/unneeded rows)")
        if args.limit:
            print(f"--limit set: only this chunk's {args.limit:,} impressions")
    else:
        behaviors = clean_behaviors(TEST_BEHAVIORS_DIR, split="test")
        if args.limit:
            behaviors = behaviors.head(args.limit)
            print(f"--limit set: only processing first {args.limit:,} impressions")
    print(f"{behaviors.height:,} impressions to score")

    # "_sample" suffix only for a bare --limit test run (no --skip); a chunked resume
    # (--skip + --limit together) must write to the real predictions.txt and append.
    out_name = f"predictions{'_sample' if (args.limit and not args.skip) else ''}.txt"
    out_path = REPO_ROOT / out_name
    if args.skip:
        # verified identical output to score_and_write() on 500 real, already-written
        # impressions (0 mismatches) -- used for resuming because the join-based path's
        # memory jumped ~2.8GB in 3s even for a single 50K-row batch of the remainder,
        # for reasons not worth chasing further given how little work remains.
        written = score_and_write_simple(behaviors, emb_index, user_matrix, user_id_to_row, out_path, append=True)
    else:
        written = score_and_write(behaviors, emb_index, user_matrix, user_id_to_row, out_path, append=False)
    print(f"\nwrote {written:,} predictions to {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")

    # A chunk (--skip + --limit together) is a partial run -- zipping here would
    # overwrite predictions.zip with incomplete content. Zip only for a full/final run.
    if args.skip and args.limit:
        print("(chunk mode: skipping zip -- zip once all chunks are written)")
    else:
        zip_path = out_path.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_path, arcname="predictions.txt")
        print(f"zipped -> {zip_path} ({zip_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
