"""Time-based splitting + verification. Never use a random split for interaction data.

Both MIND and EB-NeRD already ship official train/val(/test) folders that are
themselves temporal splits (MIND: train=Nov11-14, dev=Nov15; EB-NeRD: train/validation
by date). clean_mind.py / clean_ebnerd.py preserve that structure rather than
re-splitting, and `verify_split_integrity` below is the sanity check that this is
really true. `split_by_time` is kept generic and reusable for anyone who wants to carve
an additional internal dev-of-dev slice out of "train" later (e.g. for early stopping).
"""

from datetime import datetime

import polars as pl

_SPLIT_ORDER = ["train", "val", "test"]


def split_by_time(df: pl.DataFrame, ts_col: str, cutoffs: dict) -> dict:
    """cutoffs: e.g. {"train_end": t1, "val_end": t2} -> {"train": ts<=t1,
    "val": t1<ts<=t2, "test": ts>t2}. Any subset of {"train_end","val_end"} is allowed.
    """
    train_end = cutoffs.get("train_end")
    val_end = cutoffs.get("val_end")

    out = {}
    remaining = df
    if train_end is not None:
        out["train"] = remaining.filter(pl.col(ts_col) <= train_end)
        remaining = remaining.filter(pl.col(ts_col) > train_end)
    if val_end is not None:
        out["val"] = remaining.filter(pl.col(ts_col) <= val_end)
        remaining = remaining.filter(pl.col(ts_col) > val_end)
    out["test"] = remaining
    return out


def split_by_time_days(df: pl.DataFrame, ts_col: str, val_days: int, test_days: int = 0) -> dict:
    """Convenience wrapper: last `test_days` as test, preceding `val_days` as val."""
    max_ts: datetime = df[ts_col].max()
    val_end = max_ts - pl.duration(days=test_days) if test_days else None
    train_end = (val_end or max_ts) - pl.duration(days=val_days)
    cutoffs = {"train_end": train_end}
    if val_end is not None:
        cutoffs["val_end"] = val_end
    return split_by_time(df, ts_col, cutoffs)


def verify_split_integrity(splits: dict, ts_col: str = "timestamp", id_col: str = "impression_id") -> None:
    """Raise AssertionError if the split is not a valid temporal split:
    - no id overlap between any two present splits
    - max(ts) of an earlier split must not exceed min(ts) of a later split
      (checked in canonical order train -> val -> test)
    """
    present = [s for s in _SPLIT_ORDER if s in splits and splits[s].height > 0]

    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            a, b = present[i], present[j]
            ids_a = set(splits[a][id_col].to_list())
            ids_b = set(splits[b][id_col].to_list())
            overlap = ids_a & ids_b
            assert not overlap, f"{len(overlap)} {id_col} values appear in both '{a}' and '{b}'"

    for i in range(len(present) - 1):
        a, b = present[i], present[i + 1]
        max_a = splits[a][ts_col].max()
        min_b = splits[b][ts_col].min()
        assert max_a <= min_b, (
            f"temporal split violated: max({ts_col}) of '{a}' ({max_a}) is after "
            f"min({ts_col}) of '{b}' ({min_b})"
        )
