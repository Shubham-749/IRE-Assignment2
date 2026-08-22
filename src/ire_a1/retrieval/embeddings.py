"""Semantic article embeddings: a multilingual sentence-transformer model (covers
both English/MIND and Danish/EB-NeRD) + brute-force cosine-similarity retrieval.
`EmbeddingIndex` mirrors `bm25.BM25Index`'s build()/search() shape for consistency.
At 12K-65K articles a normalized matrix + matvec per query is simpler and one fewer
dependency than FAISS/ScaNN -- the assignment explicitly allows brute-force at this
scale; FAISS is the natural next step at 10x+.
"""

import numpy as np
import polars as pl

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def compute_embeddings(
    articles_df: pl.DataFrame, model_name: str = DEFAULT_MODEL, batch_size: int = 256
) -> pl.DataFrame:
    """Encode `title + abstract` per article. Returns (article_id, embedding) --
    embedding is a plain python list[float] so it round-trips through parquet.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name, device=device)

    texts = (articles_df["title"].fill_null("") + " " + articles_df["abstract"].fill_null("")).to_list()
    article_ids = articles_df["article_id"].to_list()
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

    return pl.DataFrame({"article_id": article_ids, "embedding": vectors.astype(np.float32).tolist()})


class EmbeddingIndex:
    def __init__(self):
        self.article_ids: list[str] = []
        self.matrix: np.ndarray | None = None  # (n_docs, dim), L2-normalized rows
        self._id_to_idx: dict[str, int] = {}

    def build(self, embeddings_df: pl.DataFrame) -> "EmbeddingIndex":
        self.article_ids = embeddings_df["article_id"].to_list()
        matrix = np.asarray(embeddings_df["embedding"].to_list(), dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = matrix / norms
        self._id_to_idx = {aid: i for i, aid in enumerate(self.article_ids)}
        return self

    def get_embedding(self, article_id: str) -> np.ndarray | None:
        """Returns the (already L2-normalized) stored vector for an article, or None
        if unknown. Fine to mean-pool these directly -- search() re-normalizes the
        resulting query vector regardless, so cosine similarity stays correct.
        """
        idx = self._id_to_idx.get(article_id)
        return None if idx is None else self.matrix[idx]

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        norm = float(np.linalg.norm(query_vec))
        if norm == 0.0 or self.matrix is None:
            return []
        q = query_vec / norm
        scores = self.matrix @ q  # cosine similarity, since both sides are unit vectors

        if top_k >= len(scores):
            top_idx = np.argsort(-scores)
        else:
            top_idx = np.argpartition(-scores, top_k)[:top_k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self.article_ids[i], float(scores[i])) for i in top_idx]

    def score_candidates(self, query_vec: np.ndarray, candidate_ids: list[str]) -> dict[str, float]:
        """Score exactly these candidates (e.g. an impression's shown article list),
        not a full-corpus search. A candidate missing from the index (shouldn't happen
        -- every catalog article gets an embedding -- but defended anyway) scores 0.0,
        a neutral (orthogonal) similarity, never silently dropped.
        """
        norm = float(np.linalg.norm(query_vec))
        if norm == 0.0:
            return dict.fromkeys(candidate_ids, 0.0)
        q = query_vec / norm
        return {aid: float(np.dot(vec, q)) if (vec := self.get_embedding(aid)) is not None else 0.0
                for aid in candidate_ids}


def mean_pool(vectors: list[np.ndarray]) -> np.ndarray | None:
    return np.mean(vectors, axis=0) if vectors else None
