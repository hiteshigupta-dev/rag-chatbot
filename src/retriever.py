"""
Retriever module.
Retrieves relevant chunks from the vector store.
"""

from typing import List, Tuple
import logging
import time

from .vector_store import get_vector_store
from .embeddings import generate_query_embedding
from .config import (
    TOP_K_RETRIEVAL,
    MMR_FETCH_K,
    SIMILARITY_THRESHOLD,
    USE_SIMILARITY_SEARCH_BELOW_CHUNKS,
)


logger = logging.getLogger(__name__)


class Retriever:
    """
    Handles retrieval of relevant document chunks.
    """

    def __init__(self, k: int = TOP_K_RETRIEVAL):
        """
        Initialize retriever.

        Args:
            k: number of chunks to retrieve
        """

        self.k = k
        self.vector_store = get_vector_store()
        if not self.vector_store.is_loaded():
            logger.info("Loading vector store...")
            self.vector_store.load()

    def retrieve(self, query: str, k: int | None = None) -> List[Tuple[str, float]]:
        """
        Retrieve relevant chunks for query (MMR search).

        Args:
            query: user query
            k: number of chunks to return (default: self.k)

        Returns:
            list of (chunk, similarity score)
        """
        if not query.strip():
            return []

        n = k if k is not None else self.k
        start_time = time.time()
        query_embedding = generate_query_embedding(query)
        # Use faster similarity_search for small datasets; MMR for larger ones
        chunk_count = len(self.vector_store.chunks) if self.vector_store.is_loaded() else 0
        if chunk_count > 0 and chunk_count < USE_SIMILARITY_SEARCH_BELOW_CHUNKS:
            results = self.vector_store.search(query_embedding, k=n)
        else:
            results = self.vector_store.mmr_search(
                query_embedding, k=n, fetch_k=MMR_FETCH_K
            )
        filtered_results = [
            (chunk, score)
            for chunk, score in results
            if score >= SIMILARITY_THRESHOLD
        ]
        if not filtered_results and results:
            logger.info(
                "No chunks above similarity threshold (%.2f); using top %s results",
                SIMILARITY_THRESHOLD, len(results),
            )
            filtered_results = results

        retrieval_time = time.time() - start_time
        logger.info("Retrieved %s documents in %.3fs", len(filtered_results), retrieval_time)
        return filtered_results

    def get_context(self, query: str) -> str:
        """
        Build context string from retrieved chunks.

        Args:
            query: user query

        Returns:
            formatted context
        """

        results = self.retrieve(query)

        if not results:
            return ""

        context_parts = []

        for chunk, score in results:
            context_parts.append(chunk)

        context = "\n\n".join(context_parts)

        return context


# ---------------------------------------------------
# Global Singleton Retriever
# ---------------------------------------------------

_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    """
    Get global retriever instance.
    """

    global _retriever

    if _retriever is None:
        _retriever = Retriever()

    return _retriever


# ---------------------------------------------------
# Convenience function
# ---------------------------------------------------

def retrieve_chunks(query: str, k: int | None = None) -> List[Tuple[str, float]]:
    """
    Retrieve chunks directly (MMR search).

    Args:
        query: user query
        k: number of chunks to return (default: TOP_K_RETRIEVAL)

    Returns:
        list of (chunk, score)
    """
    return get_retriever().retrieve(query, k=k)