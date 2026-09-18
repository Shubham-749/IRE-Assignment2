"""The "three baselines" (B1 BM25, B2 embeddings, B3 learned-weight hybrid) A2's Q3
re-ranker comparison is measured against. B1/B2 already exist as
run_eval_harness.evaluate_bm25()/evaluate_embeddings() -- this module only adds B3.
"""

import numpy as np

from ire_a1.eval.metrics import auc_score

_ALPHA_GRID = [round(a, 2) for a in np.linspace(0.0, 1.0, 21)]  # 0.00, 0.05, ..., 1.00


def hybrid_score(bm25_score: float, embedding_score: float, alpha: float) -> float:
    """alpha=1.0 is pure BM25, alpha=0.0 is pure embeddings."""
    return alpha * bm25_score + (1 - alpha) * embedding_score


def learn_hybrid_weight(impressions: list[dict]) -> float:
    """`impressions`: a list of {"bm25_scores": [...], "embedding_scores": [...],
    "labels": [...]}, all three lists aligned per-candidate within an impression (same
    order, same length). Grid searches alpha in [0, 1] (step 0.05) maximizing *mean
    per-impression* AUC of the blended score -- the same per-impression-then-average
    shape every other metric in this pipeline uses (eval/metrics.py's auc_score is
    itself defined per-impression), not one AUC pooled across every candidate from
    every impression, which would conflate impressions whose raw score scales differ.
    A small, auditable fit (21 candidate values) rather than a trained model, in
    keeping with this repo's "from scratch, not a black box" style for retrieval
    scoring (bm25.py, embeddings.py).
    """
    best_alpha, best_mean_auc = 0.5, -1.0
    for alpha in _ALPHA_GRID:
        aucs = []
        for imp in impressions:
            blended = [
                alpha * b + (1 - alpha) * e for b, e in zip(imp["bm25_scores"], imp["embedding_scores"])
            ]
            auc = auc_score(imp["labels"], blended)
            if auc is not None:
                aucs.append(auc)
        if aucs:
            mean_auc = float(np.mean(aucs))
            if mean_auc > best_mean_auc:
                best_alpha, best_mean_auc = alpha, mean_auc
    return best_alpha
