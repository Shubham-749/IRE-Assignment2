"""Parse raw MIND TSVs into the unified schema (schema.py).

Raw layout (per split, e.g. data/raw/mind/MINDsmall_train/MINDsmall_train/):
    behaviors.tsv           impression_id, user_id, time, history, impressions
    news.tsv                news_id, category, subcategory, title, abstract, url,
                             title_entities, abstract_entities

MIND gives no per-click timestamp for history -- only chronological order. To keep
`click_timestamp` meaningful for point-in-time filtering, each user's history list is
assigned synthetic timestamps strictly before this split's earliest impression
(1-second steps counting back from that anchor, most-recent-history-item first).
This is a real data limitation, not a modelling choice -- called out again in the
Q6 design note.
"""

from pathlib import Path

import polars as pl

BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]
NEWS_COLS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]
_TIME_FMT = "%m/%d/%Y %I:%M:%S %p"

# zip name (without .zip) -> (split name, nested subfolder name from download.py)
_SPLIT_DIRS = {
    "MINDsmall_train": "train",
    "MINDsmall_dev": "val",
    "MINDlarge_train": "train",
    "MINDlarge_dev": "val",
    "MINDlarge_test": "test",
}


def _find_split_dir(raw_mind_dir: Path, zip_stem: str) -> Path | None:
    # download.py extracts to raw_mind_dir/<zip_stem>/, but the MIND zips themselves
    # contain one more nested <zip_stem>/ folder -- unwrap it.
    outer = raw_mind_dir / zip_stem
    if not outer.exists():
        return None
    inner = outer / zip_stem
    return inner if inner.exists() else outer


def clean_news(news_path: Path) -> pl.DataFrame:
    raw = pl.read_csv(
        news_path,
        separator="\t",
        quote_char=None,
        has_header=False,
        new_columns=NEWS_COLS,
        schema_overrides={"title_entities": pl.Utf8, "abstract_entities": pl.Utf8},
    )
    return raw.select(
        pl.col("news_id").alias("article_id"),
        pl.lit("mind").alias("dataset"),
        pl.col("title"),
        pl.col("abstract"),
        pl.lit(None, dtype=pl.Utf8).alias("body"),
        pl.col("category"),
        pl.col("subcategory"),
        pl.lit(None, dtype=pl.Datetime).alias("published_time"),
        (pl.col("title_entities").fill_null("") + pl.lit("|") + pl.col("abstract_entities").fill_null("")).alias(
            "entities"
        ),
        pl.lit(None, dtype=pl.Float32).alias("sentiment_score"),
        pl.lit(None, dtype=pl.Int64).alias("total_pageviews"),
        pl.lit(None, dtype=pl.Int64).alias("total_inviews"),
        pl.lit(None, dtype=pl.List(pl.Float32)).alias("embedding"),
    )


def _read_behaviors_raw(behaviors_path: Path) -> pl.DataFrame:
    return pl.read_csv(
        behaviors_path,
        separator="\t",
        quote_char=None,
        has_header=False,
        new_columns=BEHAVIOR_COLS,
    ).with_columns(pl.col("time").str.strptime(pl.Datetime, _TIME_FMT))


def clean_behaviors(behaviors_path: Path, split: str) -> pl.DataFrame:
    raw = _read_behaviors_raw(behaviors_path)
    impressions = pl.col("impressions").str.split(" ")
    if split == "test":
        # test impressions are bare news_ids, no "-label" suffix, no click labels
        candidates = impressions
        clicked = pl.lit(None, dtype=pl.List(pl.Utf8))
    else:
        pairs = impressions.list.eval(pl.element().str.split("-"))
        candidates = pairs.list.eval(pl.element().list.get(0))
        clicked = pairs.list.eval(
            pl.element().filter(pl.element().list.get(1) == "1").list.get(0)
        )
    return raw.select(
        (pl.lit(f"mind_{split}_") + pl.col("impression_id").cast(pl.Utf8)).alias("impression_id"),
        pl.col("impression_id").cast(pl.Utf8).alias("raw_impression_id"),
        pl.col("user_id"),
        pl.lit("mind").alias("dataset"),
        pl.col("time").alias("timestamp"),
        candidates.alias("candidate_article_ids"),
        clicked.alias("clicked_article_ids"),
    )


def clean_user_history(behaviors_path: Path, split: str) -> pl.DataFrame:
    raw = _read_behaviors_raw(behaviors_path)
    anchor = raw["time"].min()

    with_hist = raw.filter(pl.col("history").is_not_null()).select(
        pl.col("user_id"),
        pl.col("history").str.split(" "),
        pl.col("history").str.split(" ").list.len().alias("hlen"),
    )
    # a user's history string should be constant across their rows in this split file;
    # keep the longest one observed as a safety net against any inconsistency.
    best = with_hist.sort("hlen", descending=True).unique(subset=["user_id"], keep="first")

    exploded = (
        best.select("user_id", pl.col("history").list.reverse().alias("history_rev"))
        .with_row_index("row_id")
        .explode("history_rev")
        .rename({"history_rev": "article_id"})
    )
    exploded = exploded.with_columns(pl.col("article_id").cum_count().over("row_id").alias("pos"))
    return exploded.select(
        pl.col("user_id"),
        pl.lit("mind").alias("dataset"),
        pl.col("article_id"),
        (pl.lit(anchor) - pl.duration(seconds=pl.col("pos"))).alias("click_timestamp"),
    )


def clean_mind(raw_dir: Path, scale: str) -> dict:
    raw_mind_dir = raw_dir / "mind"
    zip_stems = [stem for stem, s in _SPLIT_DIRS.items() if stem.lower().startswith(f"mind{scale}")]

    out = {}
    articles_frames = []
    history_frames = []
    for stem in zip_stems:
        split = _SPLIT_DIRS[stem]
        split_dir = _find_split_dir(raw_mind_dir, stem)
        if split_dir is None:
            continue
        articles_frames.append(clean_news(split_dir / "news.tsv"))
        out[f"behaviors_{split}"] = clean_behaviors(split_dir / "behaviors.tsv", split)
        if split != "test":
            history_frames.append(clean_user_history(split_dir / "behaviors.tsv", split))

    out["articles"] = pl.concat(articles_frames).unique(subset=["article_id"], keep="first")
    # A user's history can appear (with different synthetic anchors) in more than one
    # split file; keep the earliest timestamp seen -- the safest bound for a
    # point-in-time cutoff check.
    out["user_history"] = (
        pl.concat(history_frames)
        .group_by(["user_id", "dataset", "article_id"])
        .agg(pl.col("click_timestamp").min())
    )
    return out
