# CS4.406 Assignment 1 — Lexical & Semantic Retrieval (MIND + EB-NeRD)

## Quickstart

```bash
make venv                     # create .venv, install requirements.txt
make data DATASET=all SCALE=demo   # download -> clean -> split-verify -> feature store
make eval-bm25 DATASET=all SCALE=demo   # Q2: BM25 index + recall@K
make embeddings DATASET=all SCALE=demo  # Q3: compute article embeddings
make eval-embeddings DATASET=all SCALE=demo   # Q3: embedding index + recall@K
make compare-retrieval DATASET=all SCALE=demo # Q3.5: BM25 vs embeddings, head to head
make test                     # Q1/Q9 leakage tests + Q2/Q3 retrieval tests
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
  feature_store.py       assembles final parquet outputs + get_user_history() +
                         UserHistoryIndex (fast bulk point-in-time history lookups)
src/ire_a1/retrieval/
  tokenize.py            shared word tokenizer (indexing + queries)
  bm25.py                BM25Index: dict-based inverted index, Okapi BM25 scoring
  embeddings.py           compute_embeddings() + EmbeddingIndex: brute-force cosine sim
  eval_utils.py           shared sampling + recall@K scoring, used by both eval scripts
scripts/build_pipeline.py   the one-command CLI (what `make data` calls)
scripts/run_bm25_eval.py    Q2: BM25 recall@K evaluation (what `make eval-bm25` calls)
scripts/compute_embeddings.py  Q3: encode articles -> article_embeddings.parquet
scripts/run_embedding_eval.py  Q3: embedding recall@K evaluation
scripts/compare_retrieval.py   Q3.5: BM25 vs embeddings, same sample, side by side
tests/test_no_leakage.py    Q9: behaviour-window boundary tests
tests/test_bm25.py          Q2: BM25 + UserHistoryIndex correctness
tests/test_embeddings.py    Q3: EmbeddingIndex + recent_article_ids() correctness
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

## Status

- [x] Q1 — reproducible data pipeline
- [x] Q2 — BM25 lexical retrieval
- [x] Q3 — embedding-based semantic retrieval
- [ ] Q4 — offline evaluation harness
- [ ] Q5 — Codabench submissions
- [ ] Q6 — design note
