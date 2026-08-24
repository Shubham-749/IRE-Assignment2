#!/usr/bin/env python
"""Q5: generate the MIND Codabench submission from the real, unlabeled large test set.

    python scripts/generate_mind_submission.py [--limit N] [--force-embeddings]

Uses embeddings, not BM25, for this run: BM25's per-query postings-list walk (the
inverted index from Q2) is fundamentally per-impression and, at ~170ms/impression
measured on MIND-small, would take on the order of days across 2.37M impressions.
Embedding similarity vectorizes instead -- a batched row-wise dot product over
(impression, candidate) pairs, following the exact same batched
explode -> join -> rank(ordinal).over(impression) -> regroup -> write pattern the
earlier popularity-baseline notebook (notebooks/mind_analysis.ipynb) already built and
validated against the official sample format -- just with a personalized score
(mean-pooled user history vs. candidate embedding cosine similarity) instead of a
global click-count lookup.

--limit N processes only the first N impressions -- for verifying the format/pipeline
before committing to the full 2.37M-impression run.
"""

import argparse
import sys
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from ire_a1.clean_mind import BEHAVIOR_COLS, clean_behaviors, clean_news  # noqa: E402
from ire_a1.eval.submission import rank_and_group  # noqa: E402
from ire_a1.retrieval.embeddings import EmbeddingIndex, compute_embeddings  # noqa: E402

TEST_DIR = REPO_ROOT / "data" / "raw" / "mind" / "MINDlarge_test" / "MINDlarge_test"
CACHE_DIR = REPO_ROOT / "data" / "processed" / "mind" / "large_test_only"
BATCH_SIZE = 20_000  # impressions per batch; ~20k * ~40 candidates * 384-dim * 2 arrays ~= 2.4GB transient


def load_articles_and_embeddings() -> tuple[pl.DataFrame, pl.DataFrame]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    articles_path = CACHE_DIR / "articles.parquet"
    emb_path = CACHE_DIR / "article_embeddings.parquet"

    if articles_path.exists():
        articles = pl.read_parquet(articles_path)
        print(f"[skip] loaded cached articles ({articles.height:,} rows)")
    else:
        articles = clean_news(TEST_DIR / "news.tsv")
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


def build_user_vectors(emb_index: EmbeddingIndex) -> tuple[np.ndarray, dict]:
    """One mean-pooled, L2-normalized vector per distinct user, from their raw MIND
    `history` field (fixed per user across this test file). Not point-in-time filtered
    -- there's no leakage concern here, we're using a user's full given history to
    predict clicks on genuinely future, held-out impressions.
    """
    raw = pl.read_csv(
        TEST_DIR / "behaviors.tsv",
        separator="\t",
        quote_char=None,
        has_header=False,
        new_columns=BEHAVIOR_COLS,
    )
    users = raw.select("user_id", "history").unique(subset=["user_id"], keep="first")
    print(f"{users.height:,} distinct users")

    t0 = time.time()
    user_ids = []
    vectors = []
    n_no_history = 0
    for user_id, history in users.iter_rows():
        if not history:
            n_no_history += 1
            continue
        article_ids = history.split(" ")
        vecs = [v for aid in article_ids if (v := emb_index.get_embedding(aid)) is not None]
        if not vecs:
            n_no_history += 1
            continue
        mean_vec = np.mean(vecs, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        user_ids.append(user_id)
        vectors.append(mean_vec)
    print(f"built {len(user_ids):,} user vectors ({n_no_history:,} users with no usable history) "
          f"in {time.time()-t0:.1f}s")

    matrix = np.asarray(vectors, dtype=np.float32)
    user_id_to_row = {uid: i for i, uid in enumerate(user_ids)}
    return matrix, user_id_to_row


def score_and_write(
    behaviors: pl.DataFrame,
    emb_index: EmbeddingIndex,
    user_matrix: np.ndarray,
    user_id_to_row: dict,
    out_path: Path,
    batch_size: int = BATCH_SIZE,
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
    with open(out_path, "w") as f:
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

            batch_num = start // batch_size + 1
            n_batches = (total_rows + batch_size - 1) // batch_size
            elapsed = time.time() - t0
            print(f"  batch {batch_num}/{n_batches}: {written:,}/{total_rows:,} written "
                  f"({elapsed:.1f}s elapsed, {elapsed/batch_num:.2f}s/batch avg)")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N impressions")
    args = parser.parse_args()

    articles, embeddings_df = load_articles_and_embeddings()
    emb_index = EmbeddingIndex().build(embeddings_df)
    print(f"EmbeddingIndex: {len(emb_index.article_ids):,} articles, dim={emb_index.matrix.shape[1]}")

    user_matrix, user_id_to_row = build_user_vectors(emb_index)

    print("cleaning behaviors...")
    behaviors = clean_behaviors(TEST_DIR / "behaviors.tsv", split="test")
    if args.limit:
        behaviors = behaviors.head(args.limit)
        print(f"--limit set: only processing first {args.limit:,} impressions")
    print(f"{behaviors.height:,} impressions to score")

    out_name = f"mind_prediction{'_sample' if args.limit else ''}.txt"
    out_path = REPO_ROOT / out_name
    written = score_and_write(behaviors, emb_index, user_matrix, user_id_to_row, out_path)
    print(f"\nwrote {written:,} predictions to {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")

    # Codabench's grader hardcodes the expected filename inside the submission zip --
    # it must be exactly "prediction.txt", regardless of what we name the local file.
    zip_path = out_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_path, arcname="prediction.txt")
    print(f"zipped -> {zip_path} ({zip_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
