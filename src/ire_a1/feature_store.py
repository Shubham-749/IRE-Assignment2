"""Assemble the final feature store from cleaned tables, and expose the one function
downstream code (BM25, embeddings, eval) should use to read a user's click history:
get_user_history(). It is the single leakage-safe chokepoint -- always filters to
click_timestamp < cutoff_ts, per the point-in-time design in schema.py.
"""

from pathlib import Path

import polars as pl

from .clean_ebnerd import clean_ebnerd
from .clean_mind import clean_mind
from .temporal_split import verify_split_integrity

_CLEANERS = {"mind": clean_mind, "ebnerd": clean_ebnerd}


def build_feature_store(dataset: str, scale: str, raw_dir: Path, processed_dir: Path, force: bool = False) -> dict:
    out_dir = processed_dir / dataset / scale
    articles_path = out_dir / "articles.parquet"

    if articles_path.exists() and not force:
        print(f"  [skip] feature store for {dataset}/{scale} already built at {out_dir}")
        return {p.stem: p for p in out_dir.glob("*.parquet")}

    tables = _CLEANERS[dataset](raw_dir, scale)

    behavior_splits = {
        split: tables[f"behaviors_{split}"]
        for split in ("train", "val", "test")
        if f"behaviors_{split}" in tables
    }
    verify_split_integrity(behavior_splits, ts_col="timestamp", id_col="impression_id")

    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    tables["articles"].write_parquet(articles_path)
    written["articles"] = articles_path
    tables["user_history"].write_parquet(out_dir / "user_history.parquet")
    written["user_history"] = out_dir / "user_history.parquet"
    for split, df in behavior_splits.items():
        p = out_dir / f"behaviors_{split}.parquet"
        df.write_parquet(p)
        written[f"behaviors_{split}"] = p

    print(f"  built feature store for {dataset}/{scale} -> {out_dir}")
    return written


def get_user_history(user_history: pl.DataFrame, user_id: str, cutoff_ts) -> pl.DataFrame:
    """Return the click history for `user_id` strictly before `cutoff_ts`, most
    recent first. This is the only sanctioned way to read a user's history -- always
    go through this cutoff filter, never join user_history in unfiltered.
    """
    return (
        user_history.filter((pl.col("user_id") == user_id) & (pl.col("click_timestamp") < cutoff_ts))
        .sort("click_timestamp", descending=True)
    )
