"""
Embeddings module.
Initializes and generates embeddings using SentenceTransformers.
"""

from typing import List
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE


logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Handles embedding generation using SentenceTransformers.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, device: str = EMBEDDING_DEVICE):
        """
        Initialize embedding generator.

        Args:
            model_name: sentence-transformer model
            device: cpu or cuda
        """

        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """
        Lazy load embedding model.
        """

        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Embedding model loaded on device: %s", self.device)

        return self._model

    def get_embedding_dimension(self) -> int:
        """
        Get embedding vector dimension.
        """

        return self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for single text.
        """

        embedding = self.model.encode(
            text,
            convert_to_numpy=True
        )

        return embedding.astype("float32")

    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        """

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            batch_size=32
        )

        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for user query.
        """

        embedding = self.model.encode(
            query,
            convert_to_numpy=True
        )

        return embedding.astype("float32")


# ---------------------------------------------------
# Global Singleton Instance
# ---------------------------------------------------

_embedding_generator: EmbeddingGenerator | None = None


def get_embedding_generator() -> EmbeddingGenerator:
    """
    Get global embedding generator. Model is lazy-loaded once on first use.
    """
    global _embedding_generator

    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()

    return _embedding_generator


def warmup_embedding_model() -> None:
    """
    Load the embedding model once at startup so first query does not pay init cost.
    """
    gen = get_embedding_generator()
    _ = gen.model  # force lazy load
    logger.info("Embedding model warmed up")


# ---------------------------------------------------
# Convenience Functions
# ---------------------------------------------------

def generate_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings for list of texts.
    """

    generator = get_embedding_generator()

    return generator.embed_texts(texts)


def generate_query_embedding(query: str) -> np.ndarray:
    """
    Generate embedding for query.
    """

    generator = get_embedding_generator()

    return generator.embed_query(query)