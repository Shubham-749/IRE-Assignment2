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
            for term, tf in Counter(tokens).items():
                postings.setdefault(term, []).append((doc_idx, tf))

        self.postings = postings
        self.n_docs = len(self.article_ids)
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.idf = {
            term: math.log(1 + (self.n_docs - len(plist) + 0.5) / (len(plist) + 0.5))
            for term, plist in postings.items()
        }
        return self

    def search(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
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

        top = heapq.nlargest(top_k, scores.items(), key=lambda kv: kv[1])
        return [(self.article_ids[doc_idx], score) for doc_idx, score in top]
