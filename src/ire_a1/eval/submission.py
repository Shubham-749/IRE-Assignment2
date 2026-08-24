"""Shared logic for building Codabench submission files (MIND and, later, EB-NeRD use
the identical impression_id [rank_order] format).
"""

import polars as pl


def rank_and_group(scored: pl.DataFrame, id_col: str, score_col: str = "score") -> pl.DataFrame:
    """Given an exploded (id_col, pos, score_col, ...) frame already in the correct
    row order -- impressions in the ground truth's original sequential order, `pos`
    ascending within each impression, straight out of explode() -- assign 1-indexed
    descending ranks per impression and regroup into `rank_order` lists, preserving
    that same order.

    Never sort by id_col first: Codabench's grader compares predictions to ground
    truth *positionally*, line by line, in the ground truth's original order. If
    id_col is a string column (impression ids usually are), sorting by it reorders
    lexicographically ("1", "10", "100", ... instead of "1", "2", "3", ...) and the
    grader rejects the submission with a line-by-line id mismatch.
    """
    return (
        scored.with_columns(pl.col(score_col).rank(method="ordinal", descending=True).over(id_col).alias("rank"))
        .group_by(id_col, maintain_order=True)
        .agg(pl.col("rank").alias("rank_order"))
    )
