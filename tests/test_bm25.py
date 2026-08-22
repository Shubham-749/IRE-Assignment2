"""Q2: BM25Index correctness (toy corpus) + UserHistoryIndex agreement with the
already-tested get_user_history() (real-data sample, skipped if `make data` hasn't
been run yet).
"""

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from ire_a1.feature_store import UserHistoryIndex, get_user_history
from ire_a1.retrieval.bm25 import BM25Index
from ire_a1.retrieval.tokenize import tokenize

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


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
        }
    )


def test_tokenize_lowercases_and_splits_on_words():
    assert tokenize("Trump's White-House Visit!") == ["trump", "s", "white", "house", "visit"]
    assert tokenize(None) == []
    assert tokenize("") == []


def test_bm25_ranks_topically_relevant_docs_first():
    idx = BM25Index().build(_toy_articles())
    results = idx.search("president election", top_k=4)
    ranked_ids = [aid for aid, _ in results]
    # A1 and A3 are about the election/president; A2 and A4 (weather, zero term
    # overlap) must not be scored/returned at all.
    assert ranked_ids == ["A1", "A3"] or ranked_ids == ["A3", "A1"]


def test_bm25_top_k_is_respected():
    idx = BM25Index().build(_toy_articles())
    results = idx.search("election weather weekend president storm", top_k=2)
    assert len(results) == 2


def test_bm25_query_with_no_matching_terms_returns_empty():
    idx = BM25Index().build(_toy_articles())
    assert idx.search("nonexistent xyzzy quux", top_k=10) == []


def test_bm25_scores_are_sorted_descending():
    idx = BM25Index().build(_toy_articles())
    results = idx.search("election president weather storm", top_k=4)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_user_history_index_agrees_with_get_user_history():
    user_history = pl.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2"],
            "dataset": ["mind"] * 4,
            "article_id": ["A1", "A2", "A3", "A4"],
            "click_timestamp": [
                datetime(2024, 1, 1, 10),
                datetime(2024, 1, 2, 10),
                datetime(2024, 1, 3, 10),
                datetime(2024, 1, 1, 10),
            ],
        }
    )
    articles = _toy_articles()

    fast = UserHistoryIndex(user_history, articles)
    cutoff = datetime(2024, 1, 3, 10)  # excludes A3, includes A1/A2 for u1

    fast_titles = fast.recent_titles("u1", cutoff, max_n=20)
    slow = get_user_history(user_history, "u1", cutoff)
    slow_titles = [
        articles.filter(pl.col("article_id") == aid)["title"][0] for aid in slow["article_id"].to_list()
    ]
    assert fast_titles == slow_titles == [
        "Local weather: sunny skies this weekend",
        "Election results: president wins re-election",
    ]


_HAS_MIND_DATA = (PROCESSED_DIR / "mind" / "small" / "user_history.parquet").exists()
_HAS_EBNERD_DATA = (PROCESSED_DIR / "ebnerd" / "demo" / "user_history.parquet").exists()


@pytest.mark.skipif(not _HAS_MIND_DATA, reason="run `make data` first")
def test_user_history_index_agrees_on_real_mind_data():
    _cross_check_real_data("mind", "small")


@pytest.mark.skipif(not _HAS_EBNERD_DATA, reason="run `make data` first")
def test_user_history_index_agrees_on_real_ebnerd_data():
    _cross_check_real_data("ebnerd", "demo")


def _cross_check_real_data(dataset: str, scale: str):
    d = PROCESSED_DIR / dataset / scale
    articles = pl.read_parquet(d / "articles.parquet")
    user_history = pl.read_parquet(d / "user_history.parquet")
    behaviors = pl.read_parquet(d / "behaviors_train.parquet")

    fast = UserHistoryIndex(user_history, articles)
    sample = behaviors.filter(pl.col("user_id").is_in(user_history["user_id"].implode())).sample(
        n=min(100, behaviors.height), seed=0
    )
    for row in sample.iter_rows(named=True):
        fast_titles = fast.recent_titles(row["user_id"], row["timestamp"], max_n=20)
        slow = get_user_history(user_history, row["user_id"], row["timestamp"]).head(20)
        slow_titles = [
            articles.filter(pl.col("article_id") == aid)["title"][0] for aid in slow["article_id"].to_list()
        ]
        slow_titles = [t for t in slow_titles if t]
        assert fast_titles == slow_titles, f"mismatch for user {row['user_id']}"
