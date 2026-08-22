#!/usr/bin/env python
"""One-command rebuild: raw files -> feature store, for MIND and/or EB-NeRD.

    python scripts/build_pipeline.py --dataset all --scale demo

MIND has no "demo" bundle -- when --scale demo is requested for MIND, "small" is
used instead (the smallest MIND offers), and this is printed so it's never silent.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import polars as pl  # noqa: E402

from ire_a1.download import download, load_config  # noqa: E402
from ire_a1.feature_store import build_feature_store  # noqa: E402

_MIND_SCALE_FALLBACK = {"demo": "small"}


def resolve_scale(dataset: str, scale: str) -> str:
    if dataset == "mind" and scale in _MIND_SCALE_FALLBACK:
        fallback = _MIND_SCALE_FALLBACK[scale]
        print(f"[note] MIND has no '{scale}' bundle -- using '{fallback}' instead")
        return fallback
    return scale


def print_summary(dataset: str, scale: str, written: dict) -> None:
    print(f"\n=== {dataset}/{scale} summary ===")
    for name, path in sorted(written.items()):
        df = pl.read_parquet(path)
        print(f"  {name:20s} {df.height:>10,} rows  {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    parser.add_argument("--scale", choices=["demo", "small", "large"], default="demo")
    parser.add_argument("--force", action="store_true", help="rebuild even if output already exists")
    args = parser.parse_args()

    if args.scale == "large":
        print("[warning] 'large' bundles are several GB and are only required for Codabench "
              "submission (Q5) -- consider 'demo'/'small' for development.")

    datasets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]
    config = load_config()
    raw_dir = REPO_ROOT / config["paths"]["raw_dir"]
    processed_dir = REPO_ROOT / config["paths"]["processed_dir"]

    for dataset in datasets:
        scale = resolve_scale(dataset, args.scale)
        print(f"\n--- {dataset}/{scale}: download ---")
        download(dataset, scale, raw_dir, config, force=args.force)
        print(f"--- {dataset}/{scale}: clean + split-verify + feature store ---")
        written = build_feature_store(dataset, scale, raw_dir, processed_dir, force=args.force)
        print_summary(dataset, scale, written)


if __name__ == "__main__":
    main()
