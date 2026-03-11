"""
Vector store module.
Creates FAISS index, saves/loads index, and performs similarity search.
Loaded once at startup and reused (singleton).
"""

import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import faiss

from .embeddings import generate_embeddings
from .config import FAISS_INDEX_PATH, CHUNKS_PATH, TOP_K_RETRIEVAL, MMR_FETCH_K

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages FAISS vector store for semantic search.
    """

    def __init__(self, index_path: Path = FAISS_INDEX_PATH, chunks_path: Path = CHUNKS_PATH):
        """
        Initialize vector store.

        Args:
            index_path: path to FAISS index
            chunks_path: path to stored text chunks
        """

        self.index_path = index_path
        self.chunks_path = chunks_path

        self.index: Optional[faiss.Index] = None
        self.chunks: List[str] = []
        self.dimension: int = 0

    def create_index(self, texts: List[str]) -> None:
        """
        Create FAISS index from text chunks.

        Args:
            texts: list of text chunks
        """

        if not texts:
            raise ValueError("Cannot create index from empty text list")

        logger.info("Creating embeddings for %s chunks...", len(texts))

        embeddings = generate_embeddings(texts).astype("float32")

        self.dimension = embeddings.shape[1]

        logger.info("Embedding dimension: %s", self.dimension)

        # Use inner product index for cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)

        # Normalize vectors to enable cosine similarity
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)

        self.chunks = texts

        logger.info("FAISS index created with %s vectors", self.index.ntotal)

    def save(self) -> None:
        """
        Save FAISS index and chunks to disk.
        """

        if self.index is None:
            raise RuntimeError("No index to save")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.chunks_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(self.index_path))

        with open(self.chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

        logger.info("Index saved to: %s", self.index_path)
        logger.info("Chunks saved to: %s", self.chunks_path)

    def load(self) -> bool:
        """
        Load FAISS index and chunks from disk. No-op if already loaded (load once).

        Returns:
            True if loaded successfully
        """
        if self.is_loaded():
            return True

        if not self.index_path.exists() or not self.chunks_path.exists():
            logger.warning("Vector index files not found")
            return False

        try:

            self.index = faiss.read_index(str(self.index_path))

            with open(self.chunks_path, "rb") as f:
                self.chunks = pickle.load(f)

            self.dimension = self.index.d

            logger.info("Loaded FAISS index with %s vectors", self.index.ntotal)
            logger.info("Loaded %s chunks", len(self.chunks))

            return True

        except Exception as e:

            logger.exception("Error loading index: %s", e)
            return False

    def search(self, query_embedding: np.ndarray, k: int = TOP_K_RETRIEVAL) -> List[Tuple[str, float]]:
        """
        Search for similar chunks using query embedding.

        Args:
            query_embedding: embedding vector of query
            k: number of results

        Returns:
            list of (chunk, score)
        """

        if self.index is None:
            raise RuntimeError("Vector index not loaded")

        query_vector = query_embedding.reshape(1, -1).astype("float32")

        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        return results

    def mmr_search(
        self,
        query_embedding: np.ndarray,
        k: int = TOP_K_RETRIEVAL,
        fetch_k: int = MMR_FETCH_K,
        lambda_param: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        Max-marginal relevance search: fetch fetch_k candidates, re-rank by
        relevance vs diversity to return k diverse, relevant chunks.
        """
        if self.index is None:
            raise RuntimeError("Vector index not loaded")

        fetch_k = min(fetch_k, self.index.ntotal)
        k = min(k, fetch_k)

        query_vector = query_embedding.reshape(1, -1).astype("float32")
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, fetch_k)
        # Build candidates: (idx, chunk, score, embedding)
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            emb = self.index.reconstruct(int(idx))
            emb = np.array(emb, dtype="float32").reshape(1, -1)
            faiss.normalize_L2(emb)
            candidates.append((idx, self.chunks[idx], float(score), emb))

        if not candidates or k <= 0:
            return []

        selected: List[Tuple[int, str, float, np.ndarray]] = []

        while len(selected) < k and candidates:
            best_score = -np.inf
            best_idx = -1
            for i, (idx, chunk, sim_q, emb) in enumerate(candidates):
                if (idx, chunk) in [(s[0], s[1]) for s in selected]:
                    continue
                if not selected:
                    mmr_score = sim_q
                else:
                    max_sim_to_sel = max(
                        float(np.dot(emb, s[3].T)[0, 0]) for s in selected
                    )
                    mmr_score = (1 - lambda_param) * sim_q - lambda_param * max_sim_to_sel
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            if best_idx < 0:
                break
            chosen = candidates.pop(best_idx)
            selected.append(chosen)

        return [(chunk, score) for _, chunk, score, _ in selected]

    def is_loaded(self) -> bool:
        """
        Check if vector store is loaded.
        """

        return self.index is not None and len(self.chunks) > 0


# ---------------------------------------------------
# Global Vector Store Instance (Singleton Pattern)
# FAISS index is loaded only once and reused for all queries.
# ---------------------------------------------------

_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """
    Get global vector store instance. Creates store once; index is loaded
    once (via load() or load_vector_store()) and reused—never reloaded per query.
    """
    global _vector_store

    if _vector_store is None:
        _vector_store = VectorStore()

    return _vector_store


def load_vector_store() -> VectorStore:
    """
    Load vector store from disk. Call at startup for warmup so first query is fast.
    """
    store = get_vector_store()
    if not store.is_loaded():
        store.load()
    return store


def search_similar_chunks(query_embedding: np.ndarray, k: int = TOP_K_RETRIEVAL) -> List[Tuple[str, float]]:
    """
    Convenience search function.

    Args:
        query_embedding: query embedding
        k: number of results

    Returns:
        similar chunks
    """

    store = get_vector_store()

    return store.search(query_embedding, k)