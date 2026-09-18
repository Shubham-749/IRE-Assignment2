"""Q3 (baseline B3): learn_hybrid_weight()/hybrid_score() correctness."""

import pytest

from ire_a1.eval.metrics import auc_score
from ire_a2.baselines import hybrid_score, learn_hybrid_weight


def _mean_auc(impressions: list[dict], alpha: float) -> float:
    aucs = []
    for imp in impressions:
        blended = [alpha * b + (1 - alpha) * e for b, e in zip(imp["bm25_scores"], imp["embedding_scores"])]
        aucs.append(auc_score(imp["labels"], blended))
    return sum(aucs) / len(aucs)


def test_hybrid_score_endpoints_are_pure_bm25_and_pure_embedding():
    assert hybrid_score(bm25_score=5.0, embedding_score=0.2, alpha=1.0) == pytest.approx(5.0)
    assert hybrid_score(bm25_score=5.0, embedding_score=0.2, alpha=0.0) == pytest.approx(0.2)
    assert hybrid_score(bm25_score=4.0, embedding_score=2.0, alpha=0.5) == pytest.approx(3.0)


def test_learn_hybrid_weight_prefers_the_perfectly_ranking_signal():
    # bm25_scores are uninformative noise; embedding_scores perfectly rank the click
    # first in every impression -- the chosen alpha must reach perfect mean AUC and
    # lean toward embeddings (low alpha). Not asserting an exact alpha: on a 3-item,
    # 3-impression grid the AUC-vs-alpha curve plateaus (a merely-constant signal on
    # one side can't change ranking for a range of alpha), so the meaningful
    # invariant is "reaches the optimum, on the expected side" -- not the precise
    # tie-broken value within a plateau.
    impressions = [
        {"bm25_scores": [0.5, 0.5, 0.5], "embedding_scores": [0.1, 0.9, 0.2], "labels": [0, 1, 0]},
        {"bm25_scores": [0.5, 0.5, 0.5], "embedding_scores": [0.8, 0.1, 0.3], "labels": [1, 0, 0]},
        {"bm25_scores": [0.5, 0.5, 0.5], "embedding_scores": [0.2, 0.1, 0.95], "labels": [0, 0, 1]},
    ]
    alpha = learn_hybrid_weight(impressions)
    assert alpha <= 0.5
    assert _mean_auc(impressions, alpha) == pytest.approx(1.0)


def test_learn_hybrid_weight_prefers_bm25_when_embeddings_are_actively_misleading():
    # bm25_scores perfectly rank the click first; embedding_scores rank a *wrong*
    # candidate first in every impression (adversarial, not just uninformative -- a
    # merely-constant embedding signal wouldn't move AUC for any alpha > 0, since
    # adding the same constant to every candidate in an impression can't change its
    # ranking, so that wouldn't actually test which side the optimizer favors).
    impressions = [
        {"bm25_scores": [0.1, 0.9, 0.2], "embedding_scores": [0.9, 0.1, 0.2], "labels": [0, 1, 0]},
        {"bm25_scores": [0.8, 0.1, 0.3], "embedding_scores": [0.1, 0.8, 0.3], "labels": [1, 0, 0]},
        {"bm25_scores": [0.2, 0.1, 0.95], "embedding_scores": [0.95, 0.1, 0.2], "labels": [0, 0, 1]},
    ]
    alpha = learn_hybrid_weight(impressions)
    assert alpha >= 0.5
    assert _mean_auc(impressions, alpha) == pytest.approx(1.0)
