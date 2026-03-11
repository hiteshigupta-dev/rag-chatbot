"""
Context guardrail module.
Implements Layer 3: Context validation to ensure retrieved context is relevant.
"""

import logging
from typing import List, Tuple

from .config import RELEVANCE_THRESHOLD, NO_CONTEXT_FALLBACK, MAX_CONTEXT_CHUNKS


logger = logging.getLogger(__name__)


class ContextGuardrail:
    """
    Validates retrieved context for relevance and quality.
    """

    def __init__(self, relevance_threshold: float = RELEVANCE_THRESHOLD):
        """
        Initialize context guardrail.

        Args:
            relevance_threshold: minimum similarity score required
        """
        self.relevance_threshold = relevance_threshold

    def has_context(self, retrieved_chunks: List[Tuple[str, float]]) -> bool:
        """
        Check if any chunks were retrieved.
        """

        return len(retrieved_chunks) > 0

    def is_relevant(self, retrieved_chunks: List[Tuple[str, float]]) -> bool:
        """
        Check if retrieved chunks are relevant.
        """

        if not retrieved_chunks:
            return False

        top_score = retrieved_chunks[0][1]

        logger.info("Top retrieval score: %s", top_score)

        if top_score < self.relevance_threshold:
            logger.warning(
                "Context rejected due to low similarity score (%s)", top_score
            )
            return False

        return True

    def validate(self, retrieved_chunks: List[Tuple[str, float]]) -> Tuple[bool, str]:
        """
        Validate retrieved context.
        """

        if not self.has_context(retrieved_chunks):
            logger.warning("No context retrieved from vector store")
            return False, "No context retrieved"

        if not self.is_relevant(retrieved_chunks):
            return False, "Retrieved context is not relevant enough"

        return True, ""

    def get_fallback_response(self) -> str:
        """
        Fallback response when no relevant context is retrieved.
        """
        return NO_CONTEXT_FALLBACK

    def format_context(self, retrieved_chunks: List[Tuple[str, float]]) -> str:
        """
        Convert retrieved chunks into context string for LLM.
        Limited to MAX_CONTEXT_CHUNKS to reduce latency and token usage.
        """
        if not retrieved_chunks:
            return ""

        context_parts = []
        limited = retrieved_chunks[:MAX_CONTEXT_CHUNKS]

        for chunk, score in limited:
            if score < self.relevance_threshold:
                continue
            context_parts.append(chunk)

        context = "\n\n".join(context_parts)
        logger.info("Formatted context: %s chunks", len(context_parts))
        return context


# ---------------------------------------------------
# Global singleton
# ---------------------------------------------------

_context_guardrail = None


def get_context_guardrail() -> ContextGuardrail:
    """
    Get global context guardrail instance.
    """

    global _context_guardrail

    if _context_guardrail is None:
        _context_guardrail = ContextGuardrail()

    return _context_guardrail


def validate_context(retrieved_chunks: List[Tuple[str, float]]) -> Tuple[bool, str]:
    """
    Convenience function for validation.
    """

    guardrail = get_context_guardrail()

    return guardrail.validate(retrieved_chunks)