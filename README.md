# CS4.406 Assignments 1 & 2 — News Recommendation (MIND + EB-NeRD)

This is the A2 repo. It carries A1's commit history and code forward (A2's re-ranker
runs on top of A1's retrieval pipeline, so the code has to be here), but the *design
writeup* for A1's own questions — the "why" behind each choice, not just the commands
— stays in [`IRE-Assignment1`](https://github.com/Shubham-749/IRE-Assignment1), where
it was actually submitted. Below: enough of the A1 Quickstart to run the pipeline this
repo depends on, then A2's own work.

## A1 Quickstart — retrieval foundation

```bash
make venv                     # create .venv, install requirements.txt
make data DATASET=all SCALE=demo   # download -> clean -> split-verify -> feature store
make eval-bm25 DATASET=all SCALE=demo   # Q2: BM25 index + recall@K
make embeddings DATASET=all SCALE=demo  # Q3: compute article embeddings
make eval-embeddings DATASET=all SCALE=demo   # Q3: embedding index + recall@K
make compare-retrieval DATASET=all SCALE=demo # Q3.5: BM25 vs embeddings, head to head
make eval-harness DATASET=all SCALE=demo      # Q4: official metrics + slicing + CIs
make leakage-ablation DATASET=all SCALE=demo  # Q9: metrics with vs. without a leaky feature
make test                     # Q1/Q9 leakage tests + Q2/Q3/Q4 retrieval + metrics tests
```

`DATASET` is one of `mind` / `ebnerd` / `all`. `SCALE` is `demo` / `small` / `large`
(MIND has no `demo` bundle — `small` is used automatically, and the CLI prints a note
when it does). `large` bundles are several GB and are only required for the Codabench
submission (Q5); use `demo`/`small` for everything else.

## A2 Quickstart — GBDT re-ranker (Q1–Q5, Q9)

Builds on the A1 pipeline above (`make data`/`make embeddings` must already have run
for the dataset/scale you're targeting).

```bash
make reranker-features DATASET=all SCALE=demo          # Q1: behavioural features (train+val)
make train-reranker DATASET=all SCALE=demo             # Q2/Q3: 3 baselines + GBDT + paired ablation
make reranker-scale DATASET=all SCALE=demo              # Q4: index memory + p99 latency + cost/QPS
make reranker-extended-eval DATASET=all SCALE=demo      # Q5: full metrics, 2 slices, all 4 methods
make reranker-leakage-ablation DATASET=all SCALE=demo   # Q9: with/without-leak paired ablation
```

- `reranker-features` writes `data/processed/<dataset>/<scale>/reranker_features_{train,val}.parquet`
  (gitignored, cached so later steps don't recompute them).
- `train-reranker` evaluates B1 (BM25), B2 (embeddings), B3 (a learned-weight
  BM25+embedding hybrid — `ire_a2/baselines.py`) and the GBDT re-ranker on the same
  val-split sample `eval-harness` (A1's Q4) already reports on, trains the GBDT, then
  reports Q3's paired-bootstrap-95%-CI ablation (GBDT vs. each baseline). Writes
  `results/reranker_eval.csv` and `results/reranker_ablation.csv`.
- `reranker-scale` times a single end-to-end request (candidate gen → feature build →
  GBDT score) and measures index/feature-store memory. Writes
  `results/reranker_scale_analysis.csv`.
- `reranker-extended-eval` adds diversity/novelty/coverage and a head/tail slice
  (by the clicked article's train-split popularity) alongside the existing
  cold-start/warm slice, for all four methods. Writes `results/reranker_extended_eval.csv`.
- `reranker-leakage-ablation` trains a second GBDT with one feature swapped for a
  deliberately leaky (train+val hindsight) version and reports the paired delta.
  Writes `results/reranker_leakage_ablation.csv`.

`notebooks/02_a2_pipeline_gbdt.ipynb` runs all of the above end to end and is a
presentation layer only — every cell imports and calls the same functions the scripts
above call, so there is exactly one source of truth for the GBDT track. Re-execute
with `jupyter nbconvert --to notebook --execute --inplace notebooks/02_a2_pipeline_gbdt.ipynb`.

**Q6 (Codabench resubmission)** is deliberately not built yet — pending a decision
between this track and the teammate's NRMS track once both are complete.

**Q5's Codabench submissions from A1** (unrelated to A2's own Q5 above — A1's Q5 was
the Codabench prediction-file step) operate on the large, unlabeled test bundles
directly (no `demo`/`small` equivalent — real submissions need real scale) and aren't
wired into the `make`/`DATASET`/`SCALE` pattern above:

```bash
python scripts/generate_mind_submission.py     # -> mind_prediction.zip (2,370,727 rows)
python scripts/generate_ebnerd_submission.py   # -> predictions.zip (13,536,710 rows)
```

Each downloads/caches what it needs (large test set, article embeddings) on first run.
`--limit N` runs a small slice first to sanity-check the format before committing to
the full run; `--skip N` resumes an interrupted EB-NeRD run by appending past row N.

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
  feature_store.py       assembles final parquet outputs + get_user_history() +
                         UserHistoryIndex (fast bulk point-in-time history lookups)
src/ire_a1/retrieval/
  tokenize.py            shared word tokenizer (indexing + queries)
  bm25.py                BM25Index: dict-based inverted index, Okapi BM25 scoring
  embeddings.py           compute_embeddings() + EmbeddingIndex: brute-force cosine sim
                         + score_candidates() (both indices): score a fixed candidate
                         list for Q4, sharing the same scoring formula as search()
  eval_utils.py           shared sampling + recall@K scoring, used by both eval scripts
src/ire_a1/eval/
  metrics.py              AUC / MRR / nDCG@k / bootstrap_ci, all from scratch
  beyond_accuracy.py       diversity / novelty / coverage + its own set-based bootstrap
  submission.py            rank_and_group(): shared, tested, ordering-safe scoring ->
                           rank_order logic used by both Q5 submission scripts
scripts/build_pipeline.py   the one-command CLI (what `make data` calls)
scripts/run_bm25_eval.py    Q2: BM25 recall@K evaluation (what `make eval-bm25` calls)
scripts/compute_embeddings.py  Q3: encode articles -> article_embeddings.parquet
scripts/run_embedding_eval.py  Q3: embedding recall@K evaluation
scripts/compare_retrieval.py   Q3.5: BM25 vs embeddings, same sample, side by side
scripts/run_eval_harness.py    Q4: official metrics on both retrievers, both slices
scripts/generate_mind_submission.py    Q5: MIND large-test predictions -> Codabench zip
scripts/generate_ebnerd_submission.py  Q5: EB-NeRD large-test predictions -> Codabench zip
scripts/run_leakage_ablation.py        Q9: metrics with vs. without a leaked feature
tests/test_no_leakage.py    Q9: behaviour-window boundary tests
tests/test_bm25.py          Q2: BM25 + UserHistoryIndex correctness
tests/test_embeddings.py    Q3: EmbeddingIndex + recent_article_ids() correctness
tests/test_metrics.py       Q4: AUC/MRR/nDCG/bootstrap_ci vs. hand-computed examples
tests/test_beyond_accuracy.py  Q4: diversity/novelty/coverage vs. hand-computed examples
tests/test_download.py      Q5: _unzip() per-zip idempotency (regression: 3 zips sharing
                             one extract dir silently skipped 2 of them)
tests/test_submission.py    Q5: rank_and_group() ordering (regression: a real MIND
                             submission was rejected for scrambled row order)
notebooks/                  original EDA notebooks (kept as reference)
data/                        gitignored: raw/ + processed/ (feature store output)
results/eval_results.csv    Q4 output, small enough to commit -- the one exception to
                             the data/ gitignore rule
results/leakage_ablation.csv  Q9 output, same committed-CSV pattern as Q4
mind_prediction*.txt/.zip, predictions*.txt/.zip   Q5 submission files (gitignored --
                             large, regenerable via the two generate_*_submission.py scripts)

src/ire_a2/
  features.py              A2 Q1: FeatureBuilder -- click-history/session/article
                            features, built on UserHistoryIndex.recent(); EB-NeRD-only
                            session/dwell signal via EbnerdSessionIndex + a raw-parquet
                            dwell lookup (MIND has no session_id or per-click dwell time);
                            popularity_override hook used by Q9's leak ablation
  baselines.py               A2 Q3 (baseline B3): learn_hybrid_weight()/hybrid_score()
                             -- grid-searched BM25+embedding fusion weight
  reranker.py                A2 Q2 Option A: LightGBM LambdaRank training/scoring
scripts/build_reranker_features.py    A2 Q1: candidate-level feature table (train+val)
scripts/train_reranker.py             A2 Q2/Q3: 3 baselines + GBDT + paired ablation
scripts/reranker_scale_analysis.py    A2 Q4: index memory + p99 latency + cost/QPS
scripts/run_reranker_extended_eval.py A2 Q5: full metrics, 2 slices, all 4 methods
scripts/run_reranker_leakage_ablation.py  A2 Q9: with/without-leak paired ablation
                             (popularity feature, via FeatureBuilder's popularity_override)
tests/test_a2_features.py   A2 Q1/Q9: FeatureBuilder correctness + UserHistoryIndex.recent()
                             leakage boundary test
tests/test_a2_reranker.py   A2 Q2: GBDT ranks the true click above distractors (synthetic)
tests/test_a2_baselines.py  A2 Q3: learn_hybrid_weight()/hybrid_score() correctness
notebooks/02_a2_pipeline_gbdt.ipynb  A2 presentation layer -- runs the above end to end,
                             real executed output, no logic of its own
results/reranker_eval.csv              A2 Q2/Q3 per-method metrics (same pattern as eval_results.csv)
results/reranker_ablation.csv          A2 Q3 paired-bootstrap deltas (GBDT vs. each baseline)
results/reranker_scale_analysis.csv    A2 Q4 memory/latency/cost numbers
results/reranker_extended_eval.csv     A2 Q5 full metrics x 4 methods x 5 slices
results/reranker_leakage_ablation.csv  A2 Q9 with/without-leak paired deltas
```

## Status

- [x] Q1 — click-history/session/article features (`src/ire_a2/features.py`)
- [x] Q2 Option A — GBDT re-ranker, before/after metrics vs. B1/B2/B3 baselines
- [ ] Q2 Option B — NRMS (teammate's track, separate from this repo's history so far)
- [x] Q3 — 3 baselines (B1 BM25, B2 embeddings, B3 hybrid) reproduced, GBDT beats them
      with a paired-bootstrap-95%-CI ablation (`scripts/train_reranker.py`)
- [x] Q4 — serving & scale analysis (`scripts/reranker_scale_analysis.py`)
- [x] Q5 — extended evaluation: diversity/novelty/coverage, cold-start/warm +
      head/tail slices (`scripts/run_reranker_extended_eval.py`)
- [ ] Q5's Codabench resubmission — deferred pending a decision between tracks
- [ ] Q6 — design note
- [x] Q9 — boundary test (live-checked in the notebook) + with/without-leak paired
      ablation on `candidate_popularity` (`scripts/run_reranker_leakage_ablation.py`)
