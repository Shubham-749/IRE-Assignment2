"""Q3: EmbeddingIndex correctness (toy vectors) + UserHistoryIndex.recent_article_ids()
agreement with the already-tested recent_titles() (same underlying point-in-time
window, just projected to ids instead of titles).
"""

from datetime import datetime

import numpy as np
import polars as pl

from ire_a1.feature_store import UserHistoryIndex
from ire_a1.retrieval.embeddings import EmbeddingIndex, mean_pool


def _toy_embeddings():
    # A1/A3 near-parallel ("politics" direction), A2/A4 near-parallel ("weather"),
    # and roughly orthogonal to each other.
    return pl.DataFrame(
        {
            "article_id": ["A1", "A2", "A3", "A4"],
            "embedding": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 0.95, 0.05],
            ],
        }
    )


def test_embedding_index_normalizes_and_self_similarity_is_one():
    idx = EmbeddingIndex().build(_toy_embeddings())
    vec = idx.get_embedding("A1")
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-6
    results = dict(idx.search(vec, top_k=4))
    assert abs(results["A1"] - 1.0) < 1e-6


def test_embedding_index_ranks_by_cosine_similarity():
    idx = EmbeddingIndex().build(_toy_embeddings())
    results = idx.search(idx.get_embedding("A1"), top_k=4)
    ranked_ids = [aid for aid, _score in results]
    # A3 is near-parallel to A1 (politics direction); A2/A4 (weather) should rank last
    assert ranked_ids[0] == "A1"
    assert ranked_ids[1] == "A3"
    assert set(ranked_ids[2:]) == {"A2", "A4"}


def test_embedding_index_top_k_is_respected():
    idx = EmbeddingIndex().build(_toy_embeddings())
    results = idx.search(idx.get_embedding("A1"), top_k=2)
    assert len(results) == 2


def test_embedding_index_unknown_article_returns_none():
    idx = EmbeddingIndex().build(_toy_embeddings())
    assert idx.get_embedding("does-not-exist") is None


def test_embedding_index_zero_query_vector_returns_empty():
    idx = EmbeddingIndex().build(_toy_embeddings())
    assert idx.search(np.zeros(3), top_k=5) == []


def test_mean_pool():
    vectors = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    pooled = mean_pool(vectors)
    assert np.allclose(pooled, [0.5, 0.5])
    assert mean_pool([]) is None


def test_recent_article_ids_matches_recent_titles_window():
    user_history = pl.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "dataset": ["mind"] * 3,
            "article_id": ["A1", "A2", "A3"],
            "click_timestamp": [
                datetime(2024, 1, 1, 10),
                datetime(2024, 1, 2, 10),
                datetime(2024, 1, 3, 10),
            ],
        }
    )
    articles = pl.DataFrame(
        {"article_id": ["A1", "A2", "A3"], "title": ["title A1", "title A2", "title A3"]}
    )
    hist_index = UserHistoryIndex(user_history, articles)
    cutoff = datetime(2024, 1, 3, 10)  # excludes A3

    titles = hist_index.recent_titles("u1", cutoff, max_n=20)
    article_ids = hist_index.recent_article_ids("u1", cutoff, max_n=20)

    assert article_ids == ["A2", "A1"]  # most recent first
    assert titles == ["title A2", "title A1"]
    assert len(titles) == len(article_ids)


def test_recent_article_ids_max_n_is_respected():
    user_history = pl.DataFrame(
        {
            "user_id": ["u1"] * 5,
            "dataset": ["mind"] * 5,
            "article_id": [f"A{i}" for i in range(5)],
            "click_timestamp": [datetime(2024, 1, i + 1, 10) for i in range(5)],
        }
    )
    articles = pl.DataFrame({"article_id": [f"A{i}" for i in range(5)], "title": [f"t{i}" for i in range(5)]})
    hist_index = UserHistoryIndex(user_history, articles)
    ids = hist_index.recent_article_ids("u1", datetime(2024, 1, 6), max_n=3)
    assert ids == ["A4", "A3", "A2"]
