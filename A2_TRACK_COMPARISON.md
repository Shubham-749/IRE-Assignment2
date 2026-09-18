# A2 Track Comparison: GBDT vs. NRMS

Side-by-side results from the two independently-built re-ranker tracks, for deciding
what goes into the report and which predictions (if either, as-is) get resubmitted to
Codabench. **Read the caveats section first** — the two tracks don't share a BM25/
embedding implementation, an evaluation sample size, or a cold-start threshold, so
several of these numbers are not a clean controlled comparison of "GBDT vs. NRMS" on
their own; they're two separately-built systems that happen to both target the same
task.

- **GBDT track**: this repo (`IRE-Assignment2`), `src/ire_a1`/`src/ire_a2` +
  `scripts/`, real output in `notebooks/02_a2_pipeline_gbdt.ipynb` and
  `results/reranker_*.csv`.
- **NRMS track**: teammate's `02_a2_pipeline.ipynb`, a self-contained from-scratch
  notebook (not in this repo's history), real executed output.

## Caveats — what's *not* apples-to-apples

| | GBDT track | NRMS track |
|---|---|---|
| BM25 | from-scratch dict-based inverted index (`ire_a1/retrieval/bm25.py`) | `rank_bm25` (`BM25Okapi`) |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers) | `all-MiniLM-L6-v2` (EB-NeRD) / `all-mpnet-base-v2`-family (MIND) via sentence-transformers |
| Tokenization | hand-rolled regex tokenizer | NLTK (Porter/Snowball stemmer, stopword removal) |
| Eval sample (val) | **5,000** sampled impressions (seed=42), matching A1's own `run_eval_harness.py` convention | **full val split** — 25,356 (EB-NeRD), 73,152 (MIND) |
| Cold-start threshold | fixed `history_len < 5` | percentile-derived — EB-NeRD "Cold threshold: 97", i.e. a much higher, dataset-specific cutoff |
| Q1 feature count | 13 features (see below) | 12 features |

So: **B1 (BM25) and B2 (embeddings) numbers differ between the two notebooks for
reasons that have nothing to do with the re-ranker** — different retrievers, different
sample sizes. Treat the *within-track* comparisons (does each track's re-ranker beat
its own baselines?) as the reliable signal, and the *across-track* raw numbers as
indicative, not a controlled experiment.

## Q1 — Feature Engineering

| GBDT (13 features) | NRMS (12 features) |
|---|---|
| `bm25_score`, `embedding_score` | `bm25_score`, `emb_sim`, `emb_sim_decay` |
| `candidate_position` | `position`, `n_candidates` |
| `hist_click_count`, `hist_recency_weighted_count` | `hist_len` |
| `hist_category_match`, `hist_category_match_weighted` | `cat_overlap`, `cat_overlap_frac`, `user_cat_entropy` |
| `candidate_popularity` | `popularity` |
| `freshness_hours`, `has_freshness` (EB-NeRD only) | — |
| `hist_avg_dwell_seconds`, `session_ordinal`, `session_click_count_so_far` (EB-NeRD only) | — |
| — | `title_len`, `cat_freq` |

Both explicitly excluded a leaky feature: GBDT's `candidate_popularity` is frozen to
train-split-only counts by design (see Q9 below); NRMS's code has a comment
`# REMOVED click_ratio_in_imp to fix label leakage` — they had it, found the leak,
and removed it rather than keeping it in for a quantified before/after (see Q9).
GBDT is the only track with EB-NeRD session/dwell signal (freshness, session
position, per-click dwell time) — genuinely new information NRMS's feature set
doesn't have access to at all.

## Q2/Q3 — Baselines, Re-Ranker, Ablation

Both tracks' own B1/B2/B3 numbers are shown — remember these come from *different*
BM25/embedding implementations (see Caveats), so a same-row comparison (e.g. "GBDT
track's B1" vs. "NRMS track's B1") is comparing two different retrievers, not the
same one measured twice.

### EB-NeRD (demo)

| Method | GBDT-track AUC | MRR | nDCG@5 | nDCG@10 | NRMS-track AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|
| B1 BM25 | 0.494 | 0.312 | 0.341 | 0.427 | 0.505 | 0.320 | 0.351 | 0.435 |
| B2 Embeddings | 0.539 | 0.338 | 0.375 | 0.454 | 0.496 | 0.321 | 0.350 | 0.435 |
| B3 Hybrid | 0.539 (α=0.00 → pure embeddings) | 0.338 | 0.375 | 0.454 | 0.505 | 0.321 | 0.352 | 0.436 |
| **Re-ranker (GBDT / NRMS)** | **0.559** | **0.324** | **0.372** | **0.455** | **0.563** | **0.349** | **0.390** | **0.466** |

### MIND (small)

| Method | GBDT-track AUC | MRR | nDCG@5 | nDCG@10 | NRMS-track AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|
| B1 BM25 | 0.553 | 0.293 | 0.269 | 0.331 | 0.547 | 0.297 | 0.273 | 0.334 |
| B2 Embeddings | 0.633 | 0.340 | 0.325 | 0.384 | 0.637 | 0.351 | 0.334 | 0.394 |
| B3 Hybrid | 0.633 (α=0.00 → pure embeddings) | 0.340 | 0.325 | 0.384 | 0.638 | 0.353 | 0.336 | 0.395 |
| **Re-ranker (GBDT / NRMS)** | **0.594** | **0.340** | **0.318** | **0.373** | **0.656** | **0.363** | **0.347** | **0.410** |

*(GBDT numbers from `results/reranker_eval.csv`, rounded to 3dp. NRMS numbers
transcribed from their notebook's executed output. GBDT's learned hybrid weight was
α=0.00 on both datasets — B3 collapses to B2 because plain embeddings already
maximized mean per-impression AUC on the train-split fit sample; the NRMS track's B3
is a distinct, non-degenerate blend.)*

### Q3 ablation (paired, "does the re-ranker beat its own baselines?")

**GBDT vs. its own three baselines** (paired bootstrap 95% CI, `results/reranker_ablation.csv`):

| Dataset | vs. | ΔAUC | ΔMRR | Significant? |
|---|---|---|---|---|
| EB-NeRD | BM25 | +0.065 [0.053, 0.076] | +0.012 [0.003, 0.020] | both YES |
| EB-NeRD | Embeddings/Hybrid | +0.020 [0.008, 0.031] | **−0.014** [−0.023, −0.004] | AUC YES (gain), MRR YES (**loss**) |
| MIND | BM25 | +0.041 [0.029, 0.052] | +0.046 [0.037, 0.056] | both YES |
| MIND | Embeddings/Hybrid | **−0.040** [−0.049, −0.031] | −0.001 [−0.009, 0.009] | AUC YES (**loss**), MRR not significant |

**NRMS vs. B2 only** (their notebook only ran the ablation against B2, not all three baselines):

| Dataset | ΔMRR (NRMS − B2) | Significant? |
|---|---|---|
| EB-NeRD | +0.0277 [0.0224, 0.0326] | YES |
| MIND | +0.0119 [0.0097, 0.0142] | YES |

**Honest read:** GBDT beats BM25 significantly and consistently on both datasets. It
does **not** beat plain embeddings/hybrid cleanly — it *loses* on MRR on EB-NeRD and
on AUC on MIND (both statistically significant losses, not noise). NRMS, on the
metric it reported, beats its embedding baseline significantly on both datasets.
Given B2 differs between the two notebooks (different embedding model), this isn't
proof NRMS is strictly better than GBDT — but on the numbers as reported, NRMS shows
a cleaner, more consistent win over its own baseline than GBDT does over its own.

## Q4 — Serving & Scale

| | GBDT (this repo) | NRMS (teammate) |
|---|---|---|
| EB-NeRD index memory | BM25 5.9MB + embed 18.1MB + features 0.3MB | FAISS 14.1MB + BM25 ~4.7MB + model ~1.9MB |
| MIND index memory | BM25 42.5MB + embed 100.2MB + features 1.2MB | FAISS 100.2MB + BM25 ~26.1MB + model ~3.0MB |
| p99 latency (single request) | EB-NeRD ~3.6ms, MIND ~7.7ms (retrieval+features+score, measured) | ~2ms (estimated, "NRMS inference: ~2.0 ms/request (200 candidates)") |
| Single-core QPS | EB-NeRD 274, MIND 129 | ~437 |
| Cost / 1k queries | EB-NeRD $0.00005, MIND $0.00011 (@ $0.05/vCPU-hr) | ~$0.0003 |
| 10x bottleneck identified | brute-force embedding search (O(catalog size)/query) | BM25 `get_scores` O(V)/query — recommends Pyserini |

Both tracks independently landed on the same conclusion: **BM25/candidate-retrieval
is the first thing to break at 10x scale**, not the re-ranker itself. Absolute
latency/cost numbers aren't directly comparable (different hardware, different
"single request" definitions — NRMS's is model-inference-only and stated as an
estimate; GBDT's measures the full retrieval→feature→score path end to end).

## Q5 — Extended Evaluation

| Metric (overall) | GBDT EB-NeRD | NRMS EB-NeRD | GBDT MIND | NRMS MIND |
|---|---|---|---|---|
| Diversity@5 | 0.844 | 0.036 (ILD, different scale/definition) | 0.894 | 0.914 |
| Novelty@5 | 14.39 | 31.18 | 14.94 | 24.06 |
| Coverage@5 | 0.137 | 0.213 | 0.012 | 0.047 |

Diversity numbers aren't comparable at all — GBDT's is mean pairwise cosine
*distance* (`ire_a1/eval/beyond_accuracy.intra_list_diversity`, range ~0–1, higher =
more diverse); NRMS's "ILD" appears to use a different scale/normalization (0.036 on
EB-NeRD vs. 0.914 on MIND is a much bigger swing). Novelty/coverage use similar
definitions but different popularity denominators (different train splits/samples
feeding them), so treat these as directionally suggestive only.

**Slicing:** GBDT reports both cold-start/warm (fixed threshold) *and* head/tail (by
clicked-article popularity) — NRMS reports only cold-start/warm, on a very different,
percentile-derived threshold ("Cold threshold: 97" vs. GBDT's fixed `< 5`), so even
the shared slice isn't measuring the same population.

## Q9 — Anti-Leakage

| | GBDT | NRMS |
|---|---|---|
| Boundary test (temporal ordering) | `tests/test_a2_features.py` (pytest) + re-verified live in the notebook | inline assertions in their notebook's test cell |
| With/without-leak **quantified** ablation | **Yes** — `candidate_popularity` computed with train+val hindsight vs. train-only, paired bootstrap CI, both datasets | **No** — they found a leaky feature (`click_ratio_in_imp`), removed it, and left a printed note recommending the ablation be run; no actual before/after numbers in the notebook |
| Leak effect size (when measured) | Large and clearly significant: e.g. EB-NeRD AUC +0.149 [0.139, 0.159] with the leak | not measured |

This is the one front where completeness genuinely differs, not just methodology:
Q9 explicitly asks to "report metrics with and without features unavailable at
serving time," and only the GBDT track actually did that as a quantified,
paired-CI'd comparison.

## Completeness Checklist

| | GBDT | NRMS |
|---|---|---|
| Q1 features | ✅ | ✅ |
| Q2 re-ranker | ✅ GBDT | ✅ NRMS |
| Q3 ablation vs. baseline(s) | ✅ vs. all 3 baselines | ✅ vs. B2 only |
| Q4 scale analysis | ✅ | ✅ |
| Q5 extended eval + slices | ✅ 2 slices | ✅ 1 slice |
| Q9 boundary test | ✅ | ✅ |
| Q9 with/without-leak ablation | ✅ quantified | ⚠️ flagged, not measured |
| Q6 Codabench submission | ⏸️ deferred (this doc) | prediction files generated; **upload to Codabench itself unconfirmed** |

## Open Questions for the Team

1. **Which re-ranker goes in the report?** On the numbers as reported, neither
   dominates: GBDT wins on BM25 comparisons and is faster/cheaper to serve; NRMS
   shows a cleaner win over its own embedding baseline. Worth deciding whether the
   report presents one, or both as a comparative study (which the assignment doesn't
   forbid — Q2 lists GBDT/neural as *options*, not a required choice of exactly one).
2. **Before trusting a cross-track comparison further**, the biggest fix would be
   running both tracks' re-rankers over the *same* B1/B2 candidate scores and the
   *same* eval sample — right now differences in embedding model and sample size are
   entangled with the re-ranker comparison.
3. **Confirm with the teammate** whether their generated prediction files were
   actually submitted to Codabench, since that's an external-platform action the
   notebook execution alone doesn't prove.
