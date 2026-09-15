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

## A2 Quickstart — GBDT re-ranker (Q1/Q2 Option A)

Builds on the A1 pipeline above (`make data`/`make embeddings` must already have run
for the dataset/scale you're targeting).

```bash
make reranker-features DATASET=all SCALE=demo  # A2 Q1: behavioural features (train+val)
make train-reranker DATASET=all SCALE=demo     # A2 Q2: train GBDT, before/after metrics
```

`reranker-features` writes `data/processed/<dataset>/<scale>/reranker_features_{train,val}.parquet`
(gitignored, cached so `train-reranker` doesn't recompute them). `train-reranker` trains
a LightGBM LambdaRank model and prints AUC/MRR/nDCG@5/@10 for plain BM25, plain
embeddings, and the GBDT re-ranker on the same val-split sample `eval-harness` (Q4)
already reports on, writing `results/reranker_eval.csv`.

**Q5 (Codabench submissions)** operate on the large, unlabeled test bundles directly
(no `demo`/`small` equivalent — real submissions need real scale) and aren't wired
into the `make`/`DATASET`/`SCALE` pattern above:

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
                            dwell lookup (MIND has no session_id or per-click dwell time)
  reranker.py               A2 Q2 Option A: LightGBM LambdaRank training/scoring
scripts/build_reranker_features.py  A2 Q1: candidate-level feature table (train+val)
scripts/train_reranker.py           A2 Q2: train GBDT, before/after metrics vs. Q4
tests/test_a2_features.py   A2 Q1: FeatureBuilder correctness + UserHistoryIndex.recent()
                             leakage boundary test
tests/test_a2_reranker.py   A2 Q2: GBDT ranks the true click above distractors (synthetic)
results/reranker_eval.csv   A2 Q2 output, same committed-CSV pattern as eval_results.csv
```

## Status

- [x] Q1 — click-history/session/article features (`src/ire_a2/features.py`)
- [x] Q2 Option A — GBDT re-ranker, before/after metrics vs. Q4's BM25/embedding baselines
- [ ] Q2 Option B — NRMS (teammate's track, separate from this repo's history so far)
- [ ] Q3 — baseline reproduced, then beaten + ablation with paired bootstrap CI
- [ ] Q4 — serving & scale analysis
- [ ] Q5 — extended evaluation (diversity/novelty/coverage, slices) + Codabench resubmission
- [ ] Q6 — design note
- [ ] Q9 — with/without-leak ablation for the new re-ranker features
