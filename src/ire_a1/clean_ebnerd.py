"""Parse raw EB-NeRD parquet files into the unified schema (schema.py).

Raw layout (per scale, e.g. data/raw/ebnerd/demo/):
    articles.parquet
    train/behaviors.parquet, train/history.parquet
    validation/behaviors.parquet, validation/history.parquet
    test/behaviors.parquet                              (large scale only; no click labels)
"""

from pathlib import Path

import polars as pl

# EB-NeRD's "validation" fold is our val split; "test" only exists in the large bundle.
_SPLIT_DIRNAMES = {"train": "train", "val": "validation", "test": "test"}


def clean_articles(raw_scale_dir: Path) -> pl.DataFrame:
    raw = pl.read_parquet(raw_scale_dir / "articles.parquet")
    return raw.select(
        pl.col("article_id").cast(pl.Utf8),
        pl.lit("ebnerd").alias("dataset"),
        pl.col("title"),
        pl.col("subtitle").alias("abstract"),
        pl.col("body"),
        pl.col("category_str").alias("category"),
        pl.col("subcategory").cast(pl.List(pl.Utf8)).list.join(",").alias("subcategory"),
        pl.col("published_time"),
        pl.col("ner_clusters").cast(pl.List(pl.Utf8)).list.join("|").alias("entities"),
        pl.col("sentiment_score"),
        pl.col("total_pageviews").cast(pl.Int64),
        pl.col("total_inviews").cast(pl.Int64),
        pl.lit(None, dtype=pl.List(pl.Float32)).alias("embedding"),
    )


def clean_behaviors(split_dir: Path, split: str) -> pl.DataFrame:
    raw = pl.read_parquet(split_dir / "behaviors.parquet")
    has_labels = "article_ids_clicked" in raw.columns
    clicked_expr = (
        pl.col("article_ids_clicked").cast(pl.List(pl.Utf8))
        if has_labels
        else pl.lit(None, dtype=pl.List(pl.Utf8))
    ).alias("clicked_article_ids")
    return raw.select(
        (pl.lit(f"ebnerd_{split}_") + pl.col("impression_id").cast(pl.Utf8)).alias("impression_id"),
        pl.col("impression_id").cast(pl.Utf8).alias("raw_impression_id"),
        pl.col("user_id").cast(pl.Utf8),
        pl.lit("ebnerd").alias("dataset"),
        pl.col("impression_time").alias("timestamp"),
        pl.col("article_ids_inview").cast(pl.List(pl.Utf8)).alias("candidate_article_ids"),
        clicked_expr,
    )


def clean_user_history(raw_scale_dir: Path) -> pl.DataFrame:
    """Combine train+validation(+test, if present) history.parquet into one
    long-format, deduplicated click log -- the single source of truth used for
    point-in-time (leakage-safe) history lookups.
    """
    frames = []
    for split_dirname in ("train", "validation", "test"):
        history_path = raw_scale_dir / split_dirname / "history.parquet"
        if not history_path.exists():
            continue
        raw = pl.read_parquet(history_path)
        exploded = raw.select(
            pl.col("user_id").cast(pl.Utf8),
            pl.col("article_id_fixed"),
            pl.col("impression_time_fixed"),
        ).explode(["article_id_fixed", "impression_time_fixed"])
        frames.append(
            exploded.select(
                pl.col("user_id"),
                pl.lit("ebnerd").alias("dataset"),
                pl.col("article_id_fixed").cast(pl.Utf8).alias("article_id"),
                pl.col("impression_time_fixed").alias("click_timestamp"),
            )
        )
    combined = pl.concat(frames)
    return combined.unique(subset=["user_id", "article_id", "click_timestamp"])


def clean_ebnerd(raw_dir: Path, scale: str) -> dict:
    raw_scale_dir = raw_dir / "ebnerd" / scale
    out = {"articles": clean_articles(raw_scale_dir)}

    for split, dirname in _SPLIT_DIRNAMES.items():
        split_dir = raw_scale_dir / dirname
        if (split_dir / "behaviors.parquet").exists():
            out[f"behaviors_{split}"] = clean_behaviors(split_dir, split)

    out["user_history"] = clean_user_history(raw_scale_dir)
    return out
