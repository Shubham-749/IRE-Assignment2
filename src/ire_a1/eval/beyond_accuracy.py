"""Beyond-accuracy metrics: diversity, novelty, coverage. Computed on each
impression's top-5 ranked candidates (matching nDCG@5's cutoff).
"""

import math

import numpy as np


def intra_list_diversity(ranked_ids: list[str], emb_index) -> float | None:
    """Mean pairwise cosine *distance* (1 - similarity) among the ranked items, using
    Q3's embeddings regardless of which retriever produced the ranking -- diversity is
    inherently a semantic notion, not tied to whichever method is being scored.
    None if fewer than 2 items have embeddings (distance is undefined for <2 items).
    """
    vectors = [v for aid in ranked_ids if (v := emb_index.get_embedding(aid)) is not None]
    if len(vectors) < 2:
        return None
    matrix = np.asarray(vectors)  # rows already L2-normalized
    sims = matrix @ matrix.T
    iu = np.triu_indices(len(vectors), k=1)
    return float(np.mean(1.0 - sims[iu]))


def novelty(ranked_ids: list[str], popularity: dict, total_clicks: int, catalog_size: int) -> float | None:
    """Mean self-information of the ranked items, Laplace-smoothed:
    -log2((clicks_on_item + 1) / (total_clicks + catalog_size)). `popularity` is a
    fixed click-count dict computed once from behaviors_train -- an evaluation-only
    statistic, not a ranking feature, so this doesn't touch the Q9 leakage boundary.
    """
    if not ranked_ids:
        return None
    denom = total_clicks + catalog_size
    vals = [-math.log2((popularity.get(aid, 0) + 1) / denom) for aid in ranked_ids]
    return float(np.mean(vals))


def coverage(all_ranked_lists: list[list[str]], catalog_size: int) -> float:
    """Fraction of the full catalog that appears in *any* impression's ranked list
    across the evaluated sample -- one aggregate number per run, not per-impression.
    """
    if catalog_size == 0:
        return 0.0
    seen = set()
    for ids in all_ranked_lists:
        seen.update(ids)
    return len(seen) / catalog_size


def bootstrap_ci_coverage(
    all_ranked_lists: list[list[str]], catalog_size: int, n_boot: int = 1000, ci: float = 0.95, seed: int = 42
) -> tuple[float, float, float]:
    """Coverage isn't a per-impression scalar (it's a set-union over the whole sample),
    so it needs its own bootstrap: resample *which impressions* are included, recompute
    coverage each time. Returns (point_estimate, ci_low, ci_high).

    Known limitation: resampling n impressions with replacement from n only covers
    ~63% of the *distinct* original impressions on average (1 - 1/e, the standard
    bootstrap coverage fraction) -- so for a distinct-count statistic like this, the
    bootstrap replicates are systematically biased low relative to the exact
    point_estimate computed on the full, non-resampled sample. The point_estimate
    itself is still exact; treat the CI as showing the direction/scale of sampling
    variability rather than a band centered on the point estimate.
    """
    n = len(all_ranked_lists)
    if n == 0 or catalog_size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    lists_arr = np.empty(n, dtype=object)
    lists_arr[:] = all_ranked_lists
    boot_vals = np.empty(n_boot)
    for b in range(n_boot):
        seen = set()
        for ids in lists_arr[idx[b]]:
            seen.update(ids)
        boot_vals[b] = len(seen) / catalog_size
    lo, hi = np.percentile(boot_vals, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return coverage(all_ranked_lists, catalog_size), float(lo), float(hi)
