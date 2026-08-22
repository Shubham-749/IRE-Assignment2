"""Anti-leakage tests (Q9): the behaviour-window boundary must never be crossed.

Two layers:
1. Synthetic unit tests for temporal_split.verify_split_integrity and for MIND's
   test-split label-blinding -- these always run, no data download needed.
2. Real-data checks against the actual built feature store (get_user_history must
   never return a click that happens at/after the impression it's being used for) --
   skipped automatically if `make data` hasn't been run yet.
"""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from ire_a1.clean_mind import clean_behaviors as clean_mind_behaviors
from ire_a1.feature_store import get_user_history
from ire_a1.temporal_split import verify_split_integrity

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def _toy_behaviors(ids, timestamps):
    return pl.DataFrame(
        {
            "impression_id": ids,
            "timestamp": timestamps,
            "user_id": ["u1"] * len(ids),
        }
    )


def test_verify_split_integrity_passes_on_valid_split():
    base = datetime(2024, 1, 1)
    train = _toy_behaviors(["t1", "t2"], [base, base + timedelta(hours=1)])
    val = _toy_behaviors(["v1", "v2"], [base + timedelta(days=1), base + timedelta(days=1, hours=1)])
    verify_split_integrity({"train": train, "val": val})  # must not raise


def test_verify_split_integrity_rejects_id_overlap():
    base = datetime(2024, 1, 1)
    train = _toy_behaviors(["dup", "t2"], [base, base + timedelta(hours=1)])
    val = _toy_behaviors(["dup", "v2"], [base + timedelta(days=1), base + timedelta(days=1, hours=1)])
    with pytest.raises(AssertionError, match="appear in both"):
        verify_split_integrity({"train": train, "val": val})


def test_verify_split_integrity_rejects_out_of_order_time():
    base = datetime(2024, 1, 1)
    train = _toy_behaviors(["t1"], [base + timedelta(days=5)])  # train "after" val -- invalid
    val = _toy_behaviors(["v1"], [base])
    with pytest.raises(AssertionError, match="temporal split violated"):
        verify_split_integrity({"train": train, "val": val})


def test_mind_test_split_has_no_click_labels(tmp_path):
    behaviors_tsv = tmp_path / "behaviors.tsv"
    behaviors_tsv.write_text(
        "1\tU1\t11/20/2019 9:00:00 AM\tN1 N2\tN3 N4 N5\n"
    )
    df = clean_mind_behaviors(behaviors_tsv, split="test")
    assert df["clicked_article_ids"].null_count() == df.height
    assert df["candidate_article_ids"][0].to_list() == ["N3", "N4", "N5"]


_HAS_MIND_DATA = (PROCESSED_DIR / "mind" / "small" / "user_history.parquet").exists()
_HAS_EBNERD_DATA = (PROCESSED_DIR / "ebnerd" / "demo" / "user_history.parquet").exists()


@pytest.mark.skipif(not _HAS_MIND_DATA, reason="run `make data` first")
def test_mind_get_user_history_never_leaks_future_clicks():
    _assert_no_leakage("mind", "small")


@pytest.mark.skipif(not _HAS_EBNERD_DATA, reason="run `make data` first")
def test_ebnerd_get_user_history_never_leaks_future_clicks():
    _assert_no_leakage("ebnerd", "demo")


def _assert_no_leakage(dataset: str, scale: str):
    d = PROCESSED_DIR / dataset / scale
    user_history = pl.read_parquet(d / "user_history.parquet")
    behaviors = pl.read_parquet(d / "behaviors_train.parquet")

    sample = behaviors.filter(pl.col("user_id").is_in(user_history["user_id"].implode())).sample(
        n=min(200, behaviors.height), seed=0
    )
    for row in sample.iter_rows(named=True):
        hist = get_user_history(user_history, row["user_id"], row["timestamp"])
        if hist.height:
            assert hist["click_timestamp"].max() < row["timestamp"], (
                f"leakage: user {row['user_id']} history includes a click at/after "
                f"impression timestamp {row['timestamp']}"
            )
