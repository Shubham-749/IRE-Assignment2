# CS4.406 Assignment 1 — Lexical & Semantic Retrieval (MIND + EB-NeRD)

## Quickstart

```bash
make venv                     # create .venv, install requirements.txt
make data DATASET=all SCALE=demo   # download -> clean -> split-verify -> feature store
make test                     # anti-leakage tests (Q9)
```

`DATASET` is one of `mind` / `ebnerd` / `all`. `SCALE` is `demo` / `small` / `large`
(MIND has no `demo` bundle — `small` is used automatically, and the CLI prints a note
when it does). `large` bundles are several GB and are only required for the Codabench
submission (Q5); use `demo`/`small` for everything else.

The MIND dataset on HuggingFace (`yjw1029/MIND`) is gated — run `hf auth login` and
accept access on https://huggingface.co/datasets/yjw1029/MIND before the first MIND
download.

## Layout

```
config/datasets.yaml     dataset URLs / HF repo / paths, per scale
src/ire_a1/
  schema.py              unified articles / behaviors / user_history schema
  download.py            fetch + unzip raw files (idempotent)
  clean_mind.py          raw MIND TSVs -> unified schema
  clean_ebnerd.py        raw EB-NeRD parquet -> unified schema
  temporal_split.py      time-based split + integrity verification (Q9 hook)
  feature_store.py       assembles final parquet outputs + get_user_history()
scripts/build_pipeline.py   the one-command CLI (what `make data` calls)
tests/test_no_leakage.py    Q9: behaviour-window boundary tests
notebooks/                  original EDA notebooks (kept as reference)
data/                        gitignored: raw/ + processed/ (feature store output)
```

## Design notes (Q1)

- **Splits are not re-derived.** Both datasets already ship official time-based
  train/val folders (MIND: train=Nov11-14 / dev=Nov15; EB-NeRD: train/validation by
  date). The pipeline preserves these and calls `verify_split_integrity()` to confirm
  they really are temporally ordered with no id overlap, rather than performing its
  own random or ad-hoc split.
- **User history is a single long-format table**, not pre-joined onto each impression.
  `feature_store.get_user_history(user_id, cutoff_ts)` is the only sanctioned read path
  and always filters to `click_timestamp < cutoff_ts` — this is the leakage boundary
  Q9's tests check.
- **MIND has no per-click timestamps**, only chronological order in the `history`
  field. `clean_mind.py` assigns synthetic per-user timestamps (1-second steps,
  anchored strictly before that split's earliest impression) so the same cutoff-based
  accessor works for both datasets. This is a real data limitation, not a modelling
  choice.
- **`impression_id` is made globally unique** (`f"{dataset}_{split}_{raw_impression_id}"`)
  because MIND's raw impression ids are just per-file row indices and collide across
  splits. The original id is kept in `raw_impression_id` for producing Codabench
  submission files later (Q5).

## Status

- [x] Q1 — reproducible data pipeline
- [ ] Q2 — BM25 lexical retrieval
- [ ] Q3 — embedding-based semantic retrieval
- [ ] Q4 — offline evaluation harness
- [ ] Q5 — Codabench submissions
- [ ] Q6 — design note
