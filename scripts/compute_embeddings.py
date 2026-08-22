#!/usr/bin/env python
"""Q3: compute article embeddings for semantic retrieval.

    python scripts/compute_embeddings.py --dataset all --scale demo

Encodes `title + abstract` per article with a multilingual sentence-transformer model
(covers English/MIND and Danish/EB-NeRD with one model) and writes
data/processed/<dataset>/<scale>/article_embeddings.parquet (article_id, embedding).
Additive to the Q1 feature store -- articles.parquet itself is untouched.
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import polars as pl  # noqa: E402

from ire_a1.retrieval.embeddings import DEFAULT_MODEL, compute_embeddings  # noqa: E402

_MIND_SCALE_FALLBACK = {"demo": "small"}


def resolve_scale(dataset: str, scale: str) -> str:
    if dataset == "mind" and scale in _MIND_SCALE_FALLBACK:
        return _MIND_SCALE_FALLBACK[scale]
    return scale


def run(dataset: str, scale: str, force: bool = False) -> None:
    d = REPO_ROOT / "data" / "processed" / dataset / scale
    out_path = d / "article_embeddings.parquet"

    if out_path.exists() and not force:
        print(f"  [skip] {out_path} already exists")
        return

    articles = pl.read_parquet(d / "articles.parquet")
    print(f"\n=== {dataset}/{scale}: embedding {articles.height:,} articles ({DEFAULT_MODEL}) ===")
    t0 = time.time()
    emb_df = compute_embeddings(articles)
    print(f"encoded {emb_df.height:,} articles, dim={len(emb_df['embedding'][0])} in {time.time()-t0:.1f}s")

    emb_df.write_parquet(out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    parser.add_argument("--scale", choices=["demo", "small", "large"], default="demo")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    datasets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        scale = resolve_scale(dataset, args.scale)
        run(dataset, scale, force=args.force)


if __name__ == "__main__":
    main()
