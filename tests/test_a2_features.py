"""Q1: FeatureBuilder correctness (toy corpus) + the Q9-style leakage assertion for
UserHistoryIndex.recent(), the method every A2 feature is built on top of.
"""

import math
from datetime import datetime, timedelta

import polars as pl
import pytest

from ire_a1.feature_store import UserHistoryIndex
from ire_a1.retrieval.bm25 import BM25Index
from ire_a1.retrieval.embeddings import EmbeddingIndex
from ire_a2.features import HALF_LIFE_HOURS, FeatureBuilder


def _toy_articles():
    return pl.DataFrame(
        {
            "article_id": ["A1", "A2", "A3", "A4"],
            "title": [
                "Election results: president wins re-election",
                "Local weather: sunny skies this weekend",
                "President signs new election reform bill",
                "Weather forecast: storm warning for the weekend",
            ],
            "abstract": [
                "The president secured a second term after a close election.",
                "Expect clear skies and mild temperatures across the region.",
                "New legislation reforms how elections are run nationwide.",
                "A storm system is expected to bring heavy rain this weekend.",
            ],
            "category": ["politics", "weather", "politics", "weather"],
            "published_time": [None, None, None, None],
        }
    )


def _toy_embeddings():
    # A1/A3 (politics) close together, A2/A4 (weather) close together.
    return pl.DataFrame(
        {
            "article_id": ["A1", "A2", "A3", "A4"],
            "embedding": [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]],
        }
    )


def _builder(dataset: str = "mind", behaviors_train: pl.DataFrame | None = None) -> FeatureBuilder:
    articles = _toy_articles()
    bm25 = BM25Index().build(articles)
    emb_index = EmbeddingIndex().build(_toy_embeddings())
    if behaviors_train is None:
        behaviors_train = pl.DataFrame({"clicked_article_ids": [["A1"], ["A1"], ["A3"]]})
    return FeatureBuilder(dataset, articles, bm25, emb_index, behaviors_train)


def test_hist_recency_weighted_count_decays_with_age():
    now = datetime(2024, 1, 10, 12, 0, 0)
    recent = [(now - timedelta(hours=1), "A1", "Election results")]
    one_half_life_ago = [(now - timedelta(hours=HALF_LIFE_HOURS), "A1", "Election results")]

    b = _builder()
    recent_rows = b.build_rows("u1", now, ["A2"], recent)
    half_life_rows = b.build_rows("u1", now, ["A2"], one_half_life_ago)

    assert recent_rows[0]["hist_recency_weighted_count"] > half_life_rows[0]["hist_recency_weighted_count"]
    assert half_life_rows[0]["hist_recency_weighted_count"] == pytest.approx(0.5, rel=0.05)


def test_hist_category_match_counts_same_category_clicks():
    now = datetime(2024, 1, 10, 12, 0, 0)
    history = [
        (now - timedelta(hours=1), "A1", "Election results"),  # politics
        (now - timedelta(hours=2), "A2", "Weather sunny"),  # weather
    ]
    b = _builder()
    rows = b.build_rows("u1", now, ["A3", "A4"], history)  # A3=politics, A4=weather
    by_id = {r["article_id"]: r for r in rows}
    assert by_id["A3"]["hist_category_match"] == 1
    assert by_id["A4"]["hist_category_match"] == 1


def test_candidate_position_matches_input_order():
    now = datetime(2024, 1, 10, 12, 0, 0)
    b = _builder()
    rows = b.build_rows("u1", now, ["A4", "A2", "A1"], [])
    assert [r["candidate_position"] for r in rows] == [0, 1, 2]
    assert [r["article_id"] for r in rows] == ["A4", "A2", "A1"]


def test_candidate_popularity_uses_only_train_split_counts():
    behaviors_train = pl.DataFrame({"clicked_article_ids": [["A1"], ["A1"], ["A1"]]})
    b = _builder(behaviors_train=behaviors_train)
    now = datetime(2024, 1, 10, 12, 0, 0)
    rows = b.build_rows("u1", now, ["A1", "A2"], [])
    by_id = {r["article_id"]: r for r in rows}
    assert by_id["A1"]["candidate_popularity"] == pytest.approx(math.log1p(3))
    assert by_id["A2"]["candidate_popularity"] == pytest.approx(0.0)  # never clicked in train


def test_mind_has_no_freshness_or_session_signal():
    b = _builder(dataset="mind")
    now = datetime(2024, 1, 10, 12, 0, 0)
    rows = b.build_rows("u1", now, ["A1"], [])
    assert rows[0]["has_freshness"] == 0
    assert rows[0]["freshness_hours"] == 0.0
    assert rows[0]["session_ordinal"] == 0
    assert rows[0]["session_click_count_so_far"] == 0


def test_user_history_index_recent_excludes_clicks_at_or_after_cutoff():
    """Q9: the behaviour-window boundary every A2 feature relies on. FeatureBuilder
    itself trusts whatever `history` it's handed (tested above); this pins that the
    thing that hands it history -- UserHistoryIndex.recent() -- never includes a click
    at or after the cutoff, same contract as recent_titles()/recent_article_ids().
    """
    user_history = pl.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "dataset": ["mind"] * 3,
            "article_id": ["A1", "A2", "A3"],
            "click_timestamp": [
                datetime(2024, 1, 1, 10),
                datetime(2024, 1, 2, 10),
                datetime(2024, 1, 3, 10),  # at the cutoff below -- must be excluded
            ],
        }
    )
    idx = UserHistoryIndex(user_history, _toy_articles())
    cutoff = datetime(2024, 1, 3, 10)

    recent = idx.recent("u1", cutoff, max_n=20)
    seen_ids = [aid for _ts, aid, _title in recent]
    assert "A3" not in seen_ids
    assert seen_ids == ["A2", "A1"]  # most-recent-first, strictly before cutoff
