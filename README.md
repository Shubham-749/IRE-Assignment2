# CS4.406 Assignment 1 — Lexical & Semantic Retrieval (MIND + EB-NeRD)

## Quickstart

```bash
make venv                     # create .venv, install requirements.txt
make data DATASET=all SCALE=demo   # download -> clean -> split-verify -> feature store
make eval-bm25 DATASET=all SCALE=demo   # Q2: BM25 index + recall@K
make embeddings DATASET=all SCALE=demo  # Q3: compute article embeddings
make eval-embeddings DATASET=all SCALE=demo   # Q3: embedding index + recall@K
make compare-retrieval DATASET=all SCALE=demo # Q3.5: BM25 vs embeddings, head to head
make eval-harness DATASET=all SCALE=demo      # Q4: official metrics + slicing + CIs
make test                     # Q1/Q9 leakage tests + Q2/Q3/Q4 retrieval + metrics tests
```

`DATASET` is one of `mind` / `ebnerd` / `all`. `SCALE` is `demo` / `small` / `large`
(MIND has no `demo` bundle — `small` is used automatically, and the CLI prints a note
when it does). `large` bundles are several GB and are only required for the Codabench
submission (Q5); use `demo`/`small` for everything else.

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
mind_prediction*.txt/.zip, predictions*.txt/.zip   Q5 submission files (gitignored --
                             large, regenerable via the two generate_*_submission.py scripts)
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

## Design notes (Q2)

- **A plain dict-based inverted index**, built from scratch (not a library), scored
  with classic Okapi BM25 (`k1=1.5, b=0.75`) over `title + abstract`. Scoring a query
  only ever touches documents sharing a term with it, so it stays fast (~3ms/query)
  without a full corpus scan.
- **`UserHistoryIndex`** (`feature_store.py`) groups `user_history` by user once, so
  each impression's query-building lookup is a short per-user scan instead of a full
  table filter — needed to make tens of thousands of BM25 queries finish in reasonable
  time. It's cross-checked against the slower, already-tested `get_user_history()` on
  real data (0 mismatches) so this performance shortcut doesn't quietly reintroduce a
  leakage bug.
- **Retrieval is against the full article corpus**, not the small candidate list shown
  in a given impression — that's what makes recall@K a real test of candidate
  generation.
- **Measured recall@{50,100,200} on the val split** (5,000 sampled impressions,
  seed=42; cold-start impressions with no prior history are counted separately, not
  dropped silently):

  | Dataset | recall@50 | recall@100 | recall@200 | cold-start skipped |
  |---|---|---|---|---|
  | MIND-small | 0.53% | 1.15% | 2.19% | 2,214 / 73,152 |
  | EB-NeRD demo | 0.83% | 1.55% | 2.76% | 0 / 25,356 |

  Recall is low in absolute terms for both — spot-checking examples shows why: BM25
  finds articles whose *titles* closely resemble a user's *past* reads, but a click is
  often on something topically adjacent with little exact word overlap (e.g. a user
  history full of politics/war headlines, then a click on an F1 race result — no
  shared vocabulary at all). This is the expected failure mode of pure lexical
  retrieval and is exactly what Q3's semantic embeddings should help with.

## Design notes (Q3)

- **Real multilingual sentence embeddings**, not a lightweight substitute:
  `sentence-transformers` with `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, one
  model covers both English/MIND and Danish/EB-NeRD), encoding `title + abstract`.
  Chosen over a dependency-free TF-IDF+SVD alternative because it's still built on the
  same bag-of-words counts as BM25 and less likely to show a genuine semantic win — a
  real contextual model is what actually tests whether embeddings help. Costs a real
  new dependency (torch + sentence-transformers, ~1-2GB) and a few minutes of compute
  (65,238 MIND articles embedded in 314s on Apple Silicon MPS).
- **Brute-force cosine similarity, not FAISS.** `EmbeddingIndex` mirrors `BM25Index`'s
  `build()`/`search()` shape: L2-normalize the article matrix once, then a query is one
  matrix-vector multiply + `np.argpartition`. At 12K-65K articles this is simpler and
  one fewer dependency than an ANN library; FAISS is the natural swap-in at 10x scale.
- **User vector = mean-pooled embeddings of the same point-in-time history** used for
  BM25 (up to 20 most recent clicks). `UserHistoryIndex.recent_article_ids()` is a
  sibling of `recent_titles()`, sharing the same leakage-tested cutoff logic — verified
  in `tests/test_embeddings.py` to return the same underlying history window.
- **Same evaluation protocol as Q2** (same val-split sample, same seed=42, same
  eligibility rule), via a shared `retrieval/eval_utils.py` extracted from
  `run_bm25_eval.py` — confirmed to reproduce Q2's exact recall@K numbers before being
  reused, so the two methods are genuinely comparable, not just similarly-shaped.

**Embedding recall@K** (same val-split sample as Q2):

| Dataset | recall@50 | recall@100 | recall@200 |
|---|---|---|---|
| MIND-small | 0.88% | 1.43% | 2.27% |
| EB-NeRD demo | 0.56% | 1.12% | 2.78% |

**BM25 vs. embeddings, head to head** (`compare_retrieval.py`, same 5,000-impression
sample for both methods per dataset):

| Dataset | K | BM25 | Embeddings | Winner |
|---|---|---|---|---|
| MIND-small | 50 | 0.53% | 0.88% | embeddings |
| MIND-small | 100 | 1.15% | 1.43% | embeddings |
| MIND-small | 200 | 2.19% | 2.27% | embeddings |
| EB-NeRD demo | 50 | 0.83% | 0.56% | bm25 |
| EB-NeRD demo | 100 | 1.55% | 1.12% | bm25 |
| EB-NeRD demo | 200 | 2.76% | 2.78% | ~tie |

The two datasets disagree, and that's the real finding, not a wash to explain away:
- **MIND: embeddings win outright**, at every K, and the margin holds up in the
  long-history slice (n=4,442) where most of the sample lives. MIND's short-history
  slice (n=558) is mixed (embeddings ahead at K=50, BM25 ahead at K=100/200) — with
  only ~11% of the sample there, that's noisy rather than a real reversal.
- **EB-NeRD: BM25 keeps a real edge at tight budgets (K=50/100)** and embeddings only
  catch up by K=200. Every EB-NeRD sampled impression had ≥5 prior clicks (its users
  read more, so the short-history slice was empty here), so this isn't a history-length
  effect — plausibly Danish news headlines are terse and formulaic enough that exact
  keyword match stays a strong signal, while the multilingual embedding model (trained
  mostly on English-dominant data) has a smaller edge in Danish than in English.
- Retrieval speed favors embeddings clearly regardless of dataset: ~1.5-4ms/query vs.
  BM25's ~11-170ms/query (MIND's larger vocabulary makes its inverted-index postings
  lists longer to accumulate over) — one matvec vs. walking postings lists per query
  term.

## Design notes (Q4)

Q2/Q3 measured *candidate generation*: search the full article corpus, check if the
clicked article shows up anywhere in the top-K. Q4 is a different task shape — "the
official metrics" (AUC, MRR, nDCG@5, nDCG@10) are what MIND's and EB-NeRD's Codabench
leaderboards actually compute: for each impression, rank the small set of candidates
the user was actually shown (`candidate_article_ids`, ~6-20 items) and compare that
ranking to which one they clicked.

- **Metrics implemented from scratch** (`eval/metrics.py`), not pulled from
  scikit-learn — same spirit as building BM25's inverted index from scratch in Q2.
  AUC uses the tie-aware Mann-Whitney rank formula (ties matter: many BM25 candidates
  share zero query terms and score exactly 0.0, so naive tie-breaking would bias AUC).
- **`score_candidates()`** was added to both `BM25Index` and `EmbeddingIndex`, scoring
  only an impression's own candidates rather than the whole corpus — for BM25 this
  reuses the *exact* scoring loop `search()` already uses (refactored into a shared
  `_score_all()`), so it's the same formula already tested in Q2, not a new one.
- **Bootstrap 95% CIs on every metric**, generic `bootstrap_ci()` for the five
  per-impression scalars (AUC/MRR/nDCG@5/nDCG@10/Diversity@5) plus a dedicated
  `bootstrap_ci_coverage()` for Coverage@5, since coverage is a set-union over the
  whole sample, not a per-impression value. That coverage CI has a known bias worth
  being upfront about: resampling n impressions with replacement from n only covers
  ~63% of the *distinct* originals on average (1 − 1/e), so the CI sits systematically
  below the exact point estimate — visible in the numbers below, not hidden.
- **Same 5,000-impression, seed=42 sample as Q2/Q3**, plus one extra filter: an
  impression needs ≥1 non-click candidate too (AUC is undefined with a single class).
  Every eligible impression already satisfied this — 0 skipped, since both datasets
  show far more non-clicks than clicks per impression by construction.
- **Slice: cold-start vs. warm** (`history_len < 5` vs. `>= 5`), the exact split
  already used in Q3's `compare_retrieval.py`.

**Overall results** (5,000 sampled val impressions, ranking each impression's own
candidates; mean [95% CI]):

| Dataset | Retriever | AUC | MRR | nDCG@5 | nDCG@10 | Diversity@5 | Novelty@5 | Coverage@5 |
|---|---|---|---|---|---|---|---|---|
| MIND-small | BM25 | 0.553 [.544,.561] | 0.293 | 0.269 | 0.331 | 0.917 | 16.28 | 1.86% |
| MIND-small | Embeddings | 0.633 [.625,.642] | 0.340 | 0.325 | 0.384 | 0.847 | 16.13 | 1.66% |
| EB-NeRD demo | BM25 | 0.494 [.485,.504] | 0.312 | 0.341 | 0.427 | 0.842 | 14.75 | 14.11% |
| EB-NeRD demo | Embeddings | 0.539 [.531,.548] | 0.338 | 0.375 | 0.454 | 0.780 | 14.77 | 13.87% |

- **MIND, ranking task confirms the Q3 candidate-generation finding**: embeddings win
  on every accuracy metric, decisively. BM25's AUC (0.553) is barely above the 0.5
  chance level — exact keyword overlap within an already-narrow, topically-similar
  candidate list has little discriminating power; embeddings' smooth similarity score
  (AUC 0.633) does much better at telling near-duplicate candidates apart.
- **EB-NeRD flips again, and more starkly than in Q3**: BM25's AUC (0.494) is
  statistically indistinguishable from chance — its 95% CI straddles 0.5 — while
  embeddings clear it (0.539, CI entirely above 0.5). Both nDCG numbers are actually
  *higher* on EB-NeRD than MIND for both methods, though: EB-NeRD's candidate lists are
  much shorter (mostly ~6 items vs. MIND's ~20), which mechanically inflates nDCG/MRR
  regardless of ranking quality — a reminder these two datasets' absolute numbers
  aren't directly comparable to each other, only within-dataset.
- **Cold-start vs. warm (MIND only** — EB-NeRD's sampled impressions were all
  warm, same as Q3): embeddings degrade more from less history (cold-start AUC 0.591 →
  warm 0.639, a real gap) while BM25 barely moves (0.545 → 0.554). Averaging fewer
  embeddings gives a noisier user-centroid; a BM25 query is just as sparse either way.
- **Accuracy and diversity trade off directly.** On both datasets, BM25 has higher
  Diversity@5 than embeddings (MIND: 0.917 vs 0.847; EB-NeRD: 0.842 vs 0.780) — the
  more accurate embedding ranking also produces a tighter, more thematically clustered
  top-5, exactly the beyond-accuracy tradeoff Q4 asks the harness to surface, not
  something a pure accuracy metric like AUC would ever show on its own.
- **Coverage** is far higher on EB-NeRD (~14%) than MIND (~1.7-1.9%) simply because
  EB-NeRD's catalog is ~5.5x smaller (11,777 vs. 65,238 articles) for the same
  evaluated-impression count — expected, not a retrieval-quality signal by itself.

Full numbers (all metrics, both slices, both datasets/retrievers) are in
[`results/eval_results.csv`](results/eval_results.csv).

## Design notes (Q5)

Both submissions use **embeddings, not BM25** — BM25's per-query inverted-index walk
(the exact method from Q2) is fundamentally per-impression and, measured at
~11-170ms/impression on the small-scale data, would take on the order of days across
2.37M-13.5M impressions. Embedding similarity vectorizes instead (a batched row-wise
dot product), so it's the only one of the two methods that's actually tractable at
real Codabench scale — a genuine, load-bearing scale limitation, not an arbitrary
choice, and good material for "where it breaks at 10x" in Q6.

- **MIND** (`scripts/generate_mind_submission.py`): 2,370,727 predictions, generated in
  under 2 minutes once article embeddings were cached. Submitted and scored:
  **AUC 0.6194, MRR 0.3006, nDCG@5 0.3225, nDCG@10 0.3784** — remarkably close to our
  own offline measurement on MIND-small in Q4 (embeddings AUC 0.633), a good
  consistency check between the harness and the real leaderboard.
- **EB-NeRD** (`scripts/generate_ebnerd_submission.py`): 13,536,710 predictions (5.7x
  MIND's volume, ~15.2 avg candidates/impression vs. MIND's ~39.4). This one hit a real
  wall: `clean_user_history()`'s explode-based dedup blew up to 116.8M rows for 807,677
  users (avg history length 144.6, max 1530), which is what was actually causing
  repeated out-of-memory kills on the 17.2GB development machine — not the scoring loop
  itself, which several rounds of profiling had initially (wrongly) implicated. Fixed by
  reading the raw history file directly instead of routing through the leakage-safe
  point-in-time machinery that this use case doesn't need (full given history,
  unconditionally — there's no leakage concern predicting genuinely future impressions).
  Every workaround along the way (the history fix, a lazy-slice optimization, a fallback
  per-impression scoring path used only to finish the last ~200K rows) was verified
  against real, already-correct data before being trusted — 0 mismatches across all
  807,677 users for the history fix, 0 mismatches on 500 known-correct impressions for
  the fallback scorer — so none of it traded correctness for memory. Submitted; result
  pending as of writing (EB-NeRD's grader has 5.7x more rows to score than MIND's, and
  was still marked "Running" at last check).
- Both scripts share `eval/submission.py`'s `rank_and_group()` for the actual
  ranking/formatting step. It exists because the first MIND submission was rejected
  twice before landing: once for the wrong filename inside the zip (`prediction.txt`,
  not `mind_prediction.txt` — Codabench hardcodes the expected name), then for
  scrambled row order (sorting by a string `impression_id` column put `"10"` right
  after `"1"` lexicographically, but Codabench's grader compares predictions to ground
  truth *positionally*, line by line, in the original file's order). Centralizing the
  ordering-sensitive logic in one tested function means EB-NeRD's script never had to
  rediscover that bug.

## Status

- [x] Q1 — reproducible data pipeline
- [x] Q2 — BM25 lexical retrieval
- [x] Q3 — embedding-based semantic retrieval
- [x] Q4 — offline evaluation harness
- [~] Q5 — Codabench submissions (MIND confirmed AUC 0.6194; EB-NeRD submitted, result pending)
- [ ] Q6 — design note
