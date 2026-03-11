"""
RAG Pipeline module.
Orchestrates the full retrieval-augmented generation pipeline.
"""

import logging
import time
from typing import Dict, Any

from .guardrails import check_guardrails
from .retriever import retrieve_chunks
from .context_guardrail import get_context_guardrail, validate_context
from .llm import get_llm_generator, generate_with_context_cached
from .cache import get_cached_response, cache_response
from .config import USE_LANGCHAIN, MAX_CONTEXT_CHUNKS


logger = logging.getLogger(__name__)


def _log_latency_summary(retrieval_time: float, llm_time: float, total_time: float) -> None:
    """Log latency in the format expected for production (console)."""
    logger.info(
        "Latency Summary → Retrieval: %.3fs | LLM: %.3fs | Total: %.3fs",
        retrieval_time, llm_time, total_time,
    )
    # Console-friendly one-line summary for reviewers
    print(
        f"Retrieval Time: {retrieval_time:.2f}s\n"
        f"LLM Time: {llm_time:.2f}s\n"
        f"Total Pipeline Time: {total_time:.2f}s"
    )


# Responses that indicate LLM/API errors — do not cache these so the user can retry.
_LLM_ERROR_RESPONSES = frozenset({
    "Response blocked due to safety policy.",
    "Response blocked due to recitation policy.",
    "I couldn't generate a response.",
    "The language model took too long to respond.",
    "I'm having trouble connecting to the language model.",
    "Something went wrong while generating the response.",
})


def _should_cache_result(result: Dict[str, Any]) -> bool:
    """Return False if the result is an error or failure so we don't cache it."""
    if not result.get("success", True):
        return False
    response = (result.get("response") or "").strip()
    return response not in _LLM_ERROR_RESPONSES


class RAGPipeline:
    """
    Orchestrates the full RAG pipeline.
    """

    def __init__(self):
        """Initialize pipeline components."""

        self.llm = get_llm_generator()
        self.context_guardrail = get_context_guardrail()

    def process(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the full RAG pipeline.
        """

        if not query.strip():
            return {
                "response": "Please enter a valid question.",
                "success": False
            }

        logger.info("Processing query: %s", query)

        # --------------------------------------------------
        # Step 0: Cache check
        # --------------------------------------------------

        cache_start = time.time()
        cached = get_cached_response(query)

        if cached:
            latency_seconds = time.time() - cache_start
            logger.info("Cache hit (latency: %.3fs)", latency_seconds)
            return {
                **cached,
                "latency_seconds": latency_seconds,
                "from_cache": True,
            }

        # --------------------------------------------------
        # Step 1: Guardrails
        # --------------------------------------------------

        is_safe, reason = check_guardrails(query)

        if not is_safe:
            logger.warning("Guardrails blocked query: %s", reason)

            return {
                "response": "I can't help with that request.",
                "source": "guardrails",
                "error": reason,
                "success": False
            }

        # --------------------------------------------------
        # Step 2: Retrieval
        # --------------------------------------------------

        pipeline_start = time.time()
        retrieval_time = 0.0
        llm_time = 0.0

        retrieval_start = time.time()
        retrieved_chunks = retrieve_chunks(query)
        retrieval_time = time.time() - retrieval_start

        logger.info("Retrieved %s documents", len(retrieved_chunks))

        # Limit to top chunks for context (reduces tokens and latency)
        context_chunks = retrieved_chunks[:MAX_CONTEXT_CHUNKS]

        # --------------------------------------------------
        # Step 3: Context validation
        # --------------------------------------------------

        is_valid, message = validate_context(context_chunks)

        if not is_valid:

            fallback = self.context_guardrail.get_fallback_response()

            total_time = time.time() - pipeline_start
            result = {
                "response": fallback,
                "source": "context_guardrail",
                "error": message,
                "success": True,
                "latency_seconds": total_time,
            }
            _log_latency_summary(retrieval_time, llm_time, total_time)
            if _should_cache_result(result):
                cache_response(query, result)
            return result

        # --------------------------------------------------
        # Step 4: Format context
        # --------------------------------------------------

        context = self.context_guardrail.format_context(context_chunks)

        # --------------------------------------------------
        # Step 5: LLM generation (LangChain or custom LLM)
        # --------------------------------------------------

        llm_start = time.time()
        if USE_LANGCHAIN:
            from .langchain_rag import invoke_rag_with_context
            response = invoke_rag_with_context(query, context)
            result = {
                "response": response,
                "source": "langchain",
                "context_used": len(context_chunks),
                "success": True
            }
        else:
            response = generate_with_context_cached(query, context)
            result = {
                "response": response,
                "source": "llm",
                "context_used": len(context_chunks),
                "success": True
            }
        llm_time = time.time() - llm_start

        total_time = time.time() - pipeline_start
        result["latency_seconds"] = total_time
        logger.info("Total pipeline time: %.3f seconds", total_time)
        _log_latency_summary(retrieval_time, llm_time, total_time)

        # --------------------------------------------------
        # Step 6: Cache response (skip caching error responses so user can retry)
        # --------------------------------------------------

        if _should_cache_result(result):
            cache_response(query, result)
        else:
            logger.info("Skipping cache for error/failure response")

        return result

    def chat(self, query: str) -> str:
        """
        Simple chat interface returning only the response text.
        """

        result = self.process(query)

        return result.get("response", "I couldn't process your request.")


# ---------------------------------------------------
# Global singleton pipeline
# ---------------------------------------------------

_rag_pipeline = None


def get_rag_pipeline() -> RAGPipeline:
    """
    Get global RAG pipeline instance.
    """

    global _rag_pipeline

    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()

    return _rag_pipeline


# ---------------------------------------------------
# Convenience functions
# ---------------------------------------------------

def process_query(query: str) -> Dict[str, Any]:
    """
    Run full RAG pipeline.
    """

    pipeline = get_rag_pipeline()

    return pipeline.process(query)


def chat(query: str) -> str:
    """
    Simple chat wrapper.
    """

    pipeline = get_rag_pipeline()

    return pipeline.chat(query)