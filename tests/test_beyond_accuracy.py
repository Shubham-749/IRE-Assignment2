"""Q4: beyond-accuracy metrics (diversity, novelty, coverage) against hand-computed
examples.
"""

import math

import polars as pl
import pytest

from ire_a1.eval.beyond_accuracy import bootstrap_ci_coverage, coverage, intra_list_diversity, novelty
from ire_a1.retrieval.embeddings import EmbeddingIndex


def _toy_emb_index():
    return EmbeddingIndex().build(
        pl.DataFrame(
            {
                "article_id": ["A", "B", "C"],
                "embedding": [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],  # A==B (parallel), C orthogonal to both
            }
        )
    )


def test_diversity_zero_for_identical_items():
    idx = _toy_emb_index()
    assert intra_list_diversity(["A", "B"], idx) == pytest.approx(0.0, abs=1e-9)


def test_diversity_one_for_orthogonal_items():
    idx = _toy_emb_index()
    assert intra_list_diversity(["A", "C"], idx) == pytest.approx(1.0, abs=1e-9)


def test_diversity_undefined_below_two_items():
    idx = _toy_emb_index()
    assert intra_list_diversity(["A"], idx) is None
    assert intra_list_diversity([], idx) is None


def test_novelty_rarer_items_score_higher():
    popularity = {"A": 9}  # A is popular; B is unseen in training clicks
    total_clicks, catalog_size = 10, 10
    novelty_a = novelty(["A"], popularity, total_clicks, catalog_size)
    novelty_b = novelty(["B"], popularity, total_clicks, catalog_size)
    assert novelty_a == pytest.approx(1.0, abs=1e-6)  # -log2((9+1)/20) = -log2(0.5) = 1.0
    assert novelty_b == pytest.approx(-math.log2(1 / 20), abs=1e-6)
    assert novelty_b > novelty_a  # rarer item is more novel


def test_novelty_empty_list_is_none():
    assert novelty([], {}, total_clicks=10, catalog_size=10) is None


def test_coverage_is_fraction_of_catalog_ever_recommended():
    lists = [["A", "B"], ["B", "C"]]
    assert coverage(lists, catalog_size=4) == pytest.approx(0.75)  # {A,B,C} / 4


def test_coverage_zero_catalog_is_zero():
    assert coverage([["A"]], catalog_size=0) == 0.0


def test_bootstrap_ci_coverage_constant_case():
    lists = [["A", "B"]] * 10
    mean, lo, hi = bootstrap_ci_coverage(lists, catalog_size=2, n_boot=200)
    assert mean == lo == hi == 1.0


def test_bootstrap_ci_coverage_empty_returns_nan():
    mean, lo, hi = bootstrap_ci_coverage([], catalog_size=10)
    assert math.isnan(mean) and math.isnan(lo) and math.isnan(hi)
