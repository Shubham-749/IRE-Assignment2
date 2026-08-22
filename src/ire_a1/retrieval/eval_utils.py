"""Shared evaluation plumbing for retrieval methods (BM25, embeddings): sampling
eligible impressions and scoring recall@K. Kept retriever-agnostic on purpose --
`run_bm25_eval.py` and `run_embedding_eval.py` both call this with their own history
lookup + retrieve function, so they evaluate on exactly the same sampled impressions
under exactly the same rules. That's what makes their recall@K numbers directly
comparable in `compare_retrieval.py`.
"""

import random

import polars as pl

KS = (50, 100, 200)
MAX_HISTORY = 20
SAMPLE_SIZE = 5000
SEED = 42


def eligible_impressions(behaviors: pl.DataFrame, history_fn, max_history: int = MAX_HISTORY) -> tuple[list, int, int]:
    """history_fn(user_id, cutoff_ts, max_n) -> list of history "keys" (titles for
    BM25, article_ids for embeddings). Returns (eligible_rows, n_cold_start,
    n_with_click), where each eligible row is
    (user_id, timestamp, clicked_article_ids, history_keys).
    """
    has_click = behaviors.filter(
        pl.col("clicked_article_ids").is_not_null() & (pl.col("clicked_article_ids").list.len() > 0)
    )
    eligible = []
    n_cold_start = 0
    for row in has_click.iter_rows(named=True):
        keys = history_fn(row["user_id"], row["timestamp"], max_history)
        if not keys:
            n_cold_start += 1
            continue
        eligible.append((row["user_id"], row["timestamp"], row["clicked_article_ids"], keys))
    return eligible, n_cold_start, has_click.height


def sample_impressions(eligible: list, sample_size: int = SAMPLE_SIZE, seed: int = SEED) -> list:
    if len(eligible) <= sample_size:
        return eligible
    return random.Random(seed).sample(eligible, sample_size)


def score_recall_at_k(sample: list, retrieve_fn, ks: tuple = KS) -> tuple[dict, list]:
    """retrieve_fn(history_keys, top_k) -> ranked list of article_ids. Returns
    ({k: mean_recall}, per_impression_records) -- the per-impression records (with
    history_len + recall@k for each k) are what compare_retrieval.py slices by
    history length.
    """
    max_k = max(ks)
    hits = {k: 0.0 for k in ks}
    records = []
    for user_id, ts, clicked_ids, keys in sample:
        retrieved = retrieve_fn(keys, max_k)
        clicked_set = set(clicked_ids)
        record = {"user_id": user_id, "timestamp": ts, "history_len": len(keys)}
        for k in ks:
            recall = len(clicked_set & set(retrieved[:k])) / len(clicked_set)
            hits[k] += recall
            record[f"recall@{k}"] = recall
        records.append(record)
    means = {k: (hits[k] / len(sample) if sample else float("nan")) for k in ks}
    return means, records
