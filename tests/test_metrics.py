"""Q4: ranking metrics (AUC, MRR, nDCG@k) against hand-computed examples, plus
bootstrap_ci sanity checks. Every expected value below was worked out by hand from the
metric definitions, not just re-derived from the implementation.
"""

import math

from ire_a1.eval.metrics import auc_score, bootstrap_ci, mrr_score, ndcg_score


def test_perfect_ranking_gives_max_scores():
    labels, scores = [1, 0, 0, 0], [4, 3, 2, 1]  # clicked item ranked first
    assert auc_score(labels, scores) == 1.0
    assert mrr_score(labels, scores) == 1.0
    assert ndcg_score(labels, scores, k=4) == 1.0


def test_worst_ranking_gives_min_scores():
    labels, scores = [1, 0, 0, 0], [1, 2, 3, 4]  # clicked item ranked last
    assert auc_score(labels, scores) == 0.0
    assert mrr_score(labels, scores) == 0.25  # rank 4
    assert abs(ndcg_score(labels, scores, k=4) - 0.4307) < 1e-3


def test_auc_with_tied_scores_is_a_half():
    # one clicked, one not, both score identically -- no information, chance-level AUC
    assert auc_score([1, 0], [1, 1]) == 0.5


def test_auc_undefined_for_single_class_impressions():
    assert auc_score([1, 1, 1], [3, 2, 1]) is None  # all clicked
    assert auc_score([0, 0, 0], [3, 2, 1]) is None  # none clicked


def test_mrr_is_reciprocal_of_first_click_rank():
    labels, scores = [0, 1, 0, 0], [5, 4, 3, 2]  # click is 2nd-highest score
    assert mrr_score(labels, scores) == 0.5


def test_mrr_zero_when_no_click_present():
    assert mrr_score([0, 0, 0], [3, 2, 1]) == 0.0


def test_ndcg_excludes_click_ranked_below_k():
    labels, scores = [0, 0, 1], [3, 2, 1]  # clicked item is ranked 3rd
    assert ndcg_score(labels, scores, k=2) == 0.0
    assert ndcg_score(labels, scores, k=3) == 1.0 / math.log2(4)  # ideal@3 = 1, actual has gain at rank 3


def test_ndcg_perfect_with_multiple_clicks():
    labels, scores = [1, 1, 0, 0], [4, 3, 2, 1]  # both clicks ranked at the top
    assert ndcg_score(labels, scores, k=4) == 1.0
    assert auc_score(labels, scores) == 1.0


def test_bootstrap_ci_constant_values_has_zero_width():
    mean, lo, hi = bootstrap_ci([5.0, 5.0, 5.0], n_boot=200)
    assert mean == lo == hi == 5.0


def test_bootstrap_ci_empty_returns_nan():
    mean, lo, hi = bootstrap_ci([])
    assert math.isnan(mean) and math.isnan(lo) and math.isnan(hi)


def test_bootstrap_ci_bounds_sample_mean():
    values = [0.1, 0.9, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.5]
    mean, lo, hi = bootstrap_ci(values, n_boot=1000)
    assert lo <= mean <= hi
    assert abs(mean - (sum(values) / len(values))) < 1e-9
