"""The official ranking metrics (AUC, MRR, nDCG@k), implemented from scratch --
same spirit as building BM25's inverted index from scratch in Q2, and these are
simple, well-defined formulas at the ~20-candidates-per-impression scale these
datasets actually operate at. Each takes one impression's (labels, scores) -- binary
click labels aligned with retrieval scores over that impression's own candidate list
-- and returns a single scalar for that impression; callers average across impressions
and use bootstrap_ci() below for a confidence interval on that average.
"""

import numpy as np


def _average_ranks(scores: np.ndarray) -> np.ndarray:
    """1-based ranks (ascending order), tied scores get the average of their ranks --
    ties are common for BM25 (many candidates share zero query terms and score
    exactly 0.0), so this can't be skipped without biasing AUC.
    """
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    n = len(scores)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def auc_score(labels, scores) -> float | None:
    """Probability a random clicked candidate outranks a random non-clicked one, via
    the Mann-Whitney U / rank-sum formula. None if the impression has only one class
    (all-clicked or all-non-clicked) -- AUC is undefined there, never silently 0 or 1.
    """
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _average_ranks(scores)
    sum_ranks_pos = ranks[labels == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def mrr_score(labels, scores) -> float:
    """1 / rank of the first clicked candidate in the score-descending order."""
    labels = np.asarray(labels)
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    hits = np.nonzero(labels[order] == 1)[0]
    return float(1.0 / (hits[0] + 1)) if len(hits) else 0.0


def _dcg(labels_ranked, k: int) -> float:
    labels_ranked = np.asarray(labels_ranked[:k], dtype=float)
    gains = 2.0**labels_ranked - 1.0
    discounts = np.log2(np.arange(2, len(labels_ranked) + 2))
    return float(np.sum(gains / discounts))


def ndcg_score(labels, scores, k: int) -> float:
    """Normalized DCG@k with binary relevance: actual DCG (candidates ranked by our
    score) divided by ideal DCG (candidates ranked by true relevance)."""
    labels = np.asarray(labels, dtype=float)
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    actual = _dcg(labels[order], k)
    ideal = _dcg(np.sort(labels)[::-1], k)
    return actual / ideal if ideal > 0 else 0.0


def bootstrap_ci(values, n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    """Non-parametric bootstrap over a list of per-impression scalar metric values.
    Returns (mean, ci_low, ci_high).
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(values.mean()), float(lo), float(hi)


def paired_bootstrap_ci(
    values_a, values_b, n_boot: int = 1000, ci: float = 0.95, seed: int = 42
) -> tuple[float, float, float]:
    """Bootstrap CI on the paired difference (b - a), e.g. "improved" minus "baseline"
    per-impression metric values from the *same* sampled impressions. Unlike calling
    bootstrap_ci() on each side separately (whose CIs can overlap even when b reliably
    beats a on every impression), this resamples impressions once and differences
    within each resample -- the CI a claimed gain must exclude zero to be significant
    (Q3/Q9). Returns (mean_delta, ci_low, ci_high).
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"paired arrays must be the same length, got {len(a)} and {len(b)}")
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan")
    delta = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    boot_means = delta[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(delta.mean()), float(lo), float(hi)
