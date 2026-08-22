"""Unified schema shared by MIND and EB-NeRD after cleaning.

Both datasets are projected onto the same three tables so that downstream code
(BM25, embeddings, eval harness) never needs to know which dataset it's looking at.
Fields that only exist in one dataset are nullable in the other.
"""

import polars as pl

ARTICLES_SCHEMA = {
    "article_id": pl.Utf8,
    "dataset": pl.Utf8,  # "mind" | "ebnerd"
    "title": pl.Utf8,
    "abstract": pl.Utf8,
    "body": pl.Utf8,  # null for MIND (no body text provided)
    "category": pl.Utf8,
    "subcategory": pl.Utf8,
    "published_time": pl.Datetime,  # null for MIND
    "entities": pl.Utf8,  # raw JSON string; shape differs by dataset
    "sentiment_score": pl.Float32,  # null for MIND
    "total_pageviews": pl.Int64,  # null for MIND
    "total_inviews": pl.Int64,  # null for MIND
    "embedding": pl.List(pl.Float32),  # null if not provided/computed
}

BEHAVIORS_SCHEMA = {
    "impression_id": pl.Utf8,  # globally unique: f"{dataset}_{split}_{raw_impression_id}"
    "raw_impression_id": pl.Utf8,  # original id, as-is from the raw file (needed for Codabench submission format)
    "user_id": pl.Utf8,
    "dataset": pl.Utf8,
    "timestamp": pl.Datetime,
    "candidate_article_ids": pl.List(pl.Utf8),
    "clicked_article_ids": pl.List(pl.Utf8),  # null for the unlabeled test split
}

# Long-format click log: the single source of truth for user history.
# click_timestamp is real for EB-NeRD; for MIND (no per-click timestamps) it is a
# synthetic per-user monotonically increasing rank derived from history order -- see
# clean_mind.py. Point-in-time filtering (click_timestamp < cutoff) is what keeps
# feature_store.get_user_history() leakage-safe.
USER_HISTORY_SCHEMA = {
    "user_id": pl.Utf8,
    "dataset": pl.Utf8,
    "article_id": pl.Utf8,
    "click_timestamp": pl.Datetime,
}


def empty_frame(schema: dict) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)
