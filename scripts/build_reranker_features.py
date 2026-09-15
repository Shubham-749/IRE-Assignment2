#!/usr/bin/env python
"""Q1: build the A2 re-ranker's candidate-level feature table (train + val), one row
per (impression, candidate_article_id), for both the GBDT re-ranker (Q2 Option A) and
any other re-ranker consuming the same features.

    python scripts/build_reranker_features.py --dataset all --scale demo

Every feature comes from ire_a2.features.FeatureBuilder, fed history strictly before
each impression's own timestamp (via UserHistoryIndex.recent()) -- the Q9
behaviour-window boundary. Output is cached to data/processed/<dataset>/<scale>/
reranker_features_{train,val}.parquet (gitignored, same as the rest of
data/processed) so scripts/train_reranker.py doesn't recompute it every run.
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import polars as pl  # noqa: E402

from ire_a1.feature_store import UserHistoryIndex  # noqa: E402
from ire_a1.retrieval.bm25 import BM25Index  # noqa: E402
from ire_a1.retrieval.embeddings import EmbeddingIndex  # noqa: E402
from ire_a2.features import FEATURE_COLUMNS, MAX_HISTORY, FeatureBuilder  # noqa: E402

_MIND_SCALE_FALLBACK = {"demo": "small"}


def resolve_scale(dataset: str, scale: str) -> str:
    if dataset == "mind" and scale in _MIND_SCALE_FALLBACK:
        return _MIND_SCALE_FALLBACK[scale]
    return scale


def _session_lookup(dataset: str, raw_dir: Path, scale: str) -> dict:
    """raw_impression_id -> session_id, EB-NeRD only (MIND has no session_id)."""
    if dataset != "ebnerd":
        return {}
    out = {}
    for split_dirname in ("train", "validation"):
        p = raw_dir / "ebnerd" / scale / split_dirname / "behaviors.parquet"
        if not p.exists():
            continue
        raw = pl.read_parquet(p).select(pl.col("impression_id").cast(pl.Utf8), pl.col("session_id"))
        out.update(dict(zip(raw["impression_id"].to_list(), raw["session_id"].to_list())))
    return out


_EMPTY_SCHEMA = {"impression_id": pl.Utf8, "article_id": pl.Utf8, "label": pl.Int64}
_EMPTY_SCHEMA.update({c: pl.Float64 for c in FEATURE_COLUMNS})


def build_split(
    behaviors: pl.DataFrame,
    hist_index: UserHistoryIndex,
    feature_builder: FeatureBuilder,
    session_lookup: dict,
) -> pl.DataFrame:
    has_click = behaviors.filter(
        pl.col("clicked_article_ids").is_not_null() & (pl.col("clicked_article_ids").list.len() > 0)
    )
    rows = []
    for row in has_click.iter_rows(named=True):
        history = hist_index.recent(row["user_id"], row["timestamp"], MAX_HISTORY)
        if not history:
            continue  # cold-start: no query can be built without any prior history
        session_id = session_lookup.get(row["raw_impression_id"])
        clicked = set(row["clicked_article_ids"])
        feature_rows = feature_builder.build_rows(
            row["user_id"], row["timestamp"], row["candidate_article_ids"], history, session_id
        )
        for fr in feature_rows:
            fr["impression_id"] = row["impression_id"]
            fr["label"] = 1 if fr["article_id"] in clicked else 0
            rows.append(fr)
    if not rows:
        return pl.DataFrame(schema=_EMPTY_SCHEMA)
    return pl.DataFrame(rows)


def run(dataset: str, scale: str) -> None:
    d = REPO_ROOT / "data" / "processed" / dataset / scale
    raw_dir = REPO_ROOT / "data" / "raw"
    emb_path = d / "article_embeddings.parquet"
    if not emb_path.exists():
        raise FileNotFoundError(f"{emb_path} missing -- run scripts/compute_embeddings.py first")

    articles = pl.read_parquet(d / "articles.parquet")
    user_history = pl.read_parquet(d / "user_history.parquet")
    behaviors_train = pl.read_parquet(d / "behaviors_train.parquet")
    behaviors_val = pl.read_parquet(d / "behaviors_val.parquet")
    embeddings_df = pl.read_parquet(emb_path)

    print(f"\n=== {dataset}/{scale} ===")
    bm25 = BM25Index().build(articles)
    emb_index = EmbeddingIndex().build(embeddings_df)
    hist_index = UserHistoryIndex(user_history, articles)
    feature_builder = FeatureBuilder(dataset, articles, bm25, emb_index, behaviors_train, raw_dir, scale)
    session_lookup = _session_lookup(dataset, raw_dir, scale)

    for split_name, behaviors in (("train", behaviors_train), ("val", behaviors_val)):
        t0 = time.time()
        frame = build_split(behaviors, hist_index, feature_builder, session_lookup)
        out_path = d / f"reranker_features_{split_name}.parquet"
        frame.write_parquet(out_path)
        n_impressions = frame["impression_id"].n_unique() if frame.height else 0
        print(
            f"{split_name}: {frame.height:,} rows ({n_impressions:,} impressions) "
            f"-> {out_path} ({time.time() - t0:.1f}s)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    parser.add_argument("--scale", choices=["demo", "small", "large"], default="demo")
    args = parser.parse_args()

    datasets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        scale = resolve_scale(dataset, args.scale)
        run(dataset, scale)


if __name__ == "__main__":
    main()
