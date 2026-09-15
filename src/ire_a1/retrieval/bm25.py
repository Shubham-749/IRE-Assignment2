"""A plain dict-based inverted index + classic Okapi BM25 scoring.

Scoring a query only ever touches documents that share a term with it (accumulated
over each query term's postings list), so this stays fast against a 65K-document
corpus without a full corpus scan per query. At 10x scale this dict-based index is the
first thing to swap for something like `bm25s` (numpy-vectorized) or a real search
engine (Lucene/tantivy) -- see the design note.
"""

import heapq
import math
from collections import Counter

import polars as pl

from .tokenize import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.article_ids: list[str] = []
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0
        self.n_docs: int = 0
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.idf: dict[str, float] = {}
        self._id_to_idx: dict[str, int] = {}
        self.doc_terms: list[Counter] = []  # doc_idx -> {term: tf}, for score_candidates()

    def build(self, articles_df: pl.DataFrame) -> "BM25Index":
        texts = (
            articles_df.select(
                pl.col("article_id"),
                (pl.col("title").fill_null("") + " " + pl.col("abstract").fill_null("")).alias("text"),
            )
        ).iter_rows()

        postings: dict[str, list[tuple[int, int]]] = {}
        for doc_idx, (article_id, text) in enumerate(texts):
            tokens = tokenize(text)
            self.article_ids.append(article_id)
            self.doc_len.append(len(tokens))
            term_counts = Counter(tokens)
            self.doc_terms.append(term_counts)
            for term, tf in term_counts.items():
                postings.setdefault(term, []).append((doc_idx, tf))

        self.postings = postings
        self.n_docs = len(self.article_ids)
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.idf = {
            term: math.log(1 + (self.n_docs - len(plist) + 0.5) / (len(plist) + 0.5))
            for term, plist in postings.items()
        }
        self._id_to_idx = {aid: i for i, aid in enumerate(self.article_ids)}
        return self

    def _score_all(self, query_text: str) -> dict[int, float]:
        """doc_idx -> BM25 score, for every doc sharing >=1 term with the query.
        Shared by search() (Q2: full-corpus top-K) and score_candidates() (Q4:
        score a fixed candidate list) so both use the identical scoring formula.
        """
        query_terms = set(tokenize(query_text))
        scores: dict[int, float] = {}
        for term in query_terms:
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf[term]
            for doc_idx, tf in plist:
                dl = self.doc_len[doc_idx]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * (tf * (self.k1 + 1)) / denom
        return scores

    def search(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
        scores = self._score_all(query_text)
        top = heapq.nlargest(top_k, scores.items(), key=lambda kv: kv[1])
        return [(self.article_ids[doc_idx], score) for doc_idx, score in top]

    def score_candidates(self, query_text: str, candidate_ids: list[str]) -> dict[str, float]:
        """Score exactly these candidates (e.g. an impression's shown article list),
        not a full-corpus search. Candidates sharing no term with the query score 0.0
        -- never dropped, since ranking metrics (Q4) need every candidate accounted for.

        Deliberately doesn't reuse _score_all(): that walks every posting for every
        query term (i.e. every doc in the corpus sharing a term with the query), which
        is the right cost for search()'s full-corpus top-K but wasteful here, where the
        answer is only ever needed for a handful of already-known candidates. Instead
        this looks up each candidate's own (small) term-frequency map directly --
        O(candidates x query terms) instead of O(query terms x corpus docs per term).
        Identical scores to the old _score_all()-based version (same BM25 formula, same
        idf/avgdl), just not computed for the ~entire corpus first. Matters once a
        caller scores every labeled impression rather than a bounded eval sample (see
        A2's scripts/build_reranker_features.py, which made this the dominant cost).
        """
        query_terms = set(tokenize(query_text))
        scores = {}
        for aid in candidate_ids:
            doc_idx = self._id_to_idx.get(aid)
            if doc_idx is None:
                scores[aid] = 0.0
                continue
            dl = self.doc_len[doc_idx]
            doc_terms = self.doc_terms[doc_idx]
            s = 0.0
            for term in query_terms:
                tf = doc_terms.get(term)
                if not tf:
                    continue
                idf = self.idf[term]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += idf * (tf * (self.k1 + 1)) / denom
            scores[aid] = s
        return scores
