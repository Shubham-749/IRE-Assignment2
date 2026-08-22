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


class UserHistoryIndex:
    """Same point-in-time contract as get_user_history() (click_timestamp < cutoff_ts),
    but grouped by user once up front so a per-impression lookup is a short scan over
    one user's own history instead of a full table scan. Needed because BM25/embedding
    retrieval must build a query for every impression -- tens of thousands of calls,
    where get_user_history()'s O(table size) filter per call doesn't finish in
    reasonable time. tests/test_bm25.py checks this class agrees with the (slower,
    already-tested) get_user_history() on a real-data sample.
    """

    def __init__(self, user_history: pl.DataFrame, articles: pl.DataFrame):
        joined = (
            user_history.join(articles.select("article_id", "title"), on="article_id", how="left")
            .sort(["user_id", "click_timestamp"], descending=[False, True])
        )
        grouped = joined.group_by("user_id", maintain_order=True).agg(
            pl.col("click_timestamp"), pl.col("article_id"), pl.col("title")
        )
        self._by_user = {
            row["user_id"]: list(zip(row["click_timestamp"], row["article_id"], row["title"]))
            for row in grouped.iter_rows(named=True)
        }

    def _recent(self, user_id: str, cutoff_ts, max_n: int) -> list[tuple]:
        """Up to `max_n` (click_timestamp, article_id, title) tuples strictly before
        `cutoff_ts`, most recent first. Shared by recent_titles() (Q2/BM25) and
        recent_article_ids() (Q3/embeddings) so both retrieval methods see exactly the
        same point-in-time history.
        """
        out = []
        for row in self._by_user.get(user_id, ()):
            if row[0] >= cutoff_ts:
                continue
            out.append(row)
            if len(out) >= max_n:
                break
        return out

    def recent_titles(self, user_id: str, cutoff_ts, max_n: int = 20) -> list[str]:
        """Titles of up to `max_n` most-recent clicks strictly before `cutoff_ts`."""
        return [title for _ts, _article_id, title in self._recent(user_id, cutoff_ts, max_n) if title]

    def recent_article_ids(self, user_id: str, cutoff_ts, max_n: int = 20) -> list[str]:
        """Article ids of up to `max_n` most-recent clicks strictly before `cutoff_ts`."""
        return [article_id for _ts, article_id, _title in self._recent(user_id, cutoff_ts, max_n)]
