"""
LangChain RAG integration.
Uses LangChain's LCEL to build a RAG chain with our retriever + Gemini.
"""

import logging
from functools import lru_cache
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from .retriever import retrieve_chunks
from .config import (
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    TOP_K_RETRIEVAL,
    MAX_CONTEXT_CHUNKS,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------
# Custom retriever: wraps our FAISS retriever as LangChain Retriever
# ---------------------------------------------------


class FAISSRetriever(BaseRetriever):
    """LangChain retriever that uses our existing FAISS + sentence-transformers retrieval."""

    k: int = TOP_K_RETRIEVAL

    def _get_relevant_documents(self, query: str) -> List[Document]:
        chunks_with_scores = retrieve_chunks(query, k=self.k)
        return [
            Document(page_content=chunk, metadata={"score": score})
            for chunk, score in chunks_with_scores
        ]


# ---------------------------------------------------
# RAG chain
# ---------------------------------------------------

RAG_SYSTEM = (
    "You are a Tonton FAQ support assistant. "
    "Answer ONLY from the provided context. "
    "If the answer is not in the context, respond with exactly: "
    "I'm sorry, I couldn't find this information in the FAQ. "
    "Otherwise answer clearly and concisely."
)

RAG_TEMPLATE = """Context:
{context}

Question:
{question}

Answer based only on the context above. If not in context, say: I'm sorry, I couldn't find this information in the FAQ.
"""


def _format_docs(docs: List[Document]) -> str:
    limited = docs[:MAX_CONTEXT_CHUNKS]
    return "\n\n".join(doc.page_content for doc in limited)


def _get_llm():
    """Shared Gemini LLM for LangChain (optimized for FAQ latency)."""
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        google_api_key=GEMINI_API_KEY or None,
        temperature=DEFAULT_LLM_TEMPERATURE,
        max_output_tokens=DEFAULT_LLM_MAX_TOKENS,
    )


def _get_prompt():
    """Shared RAG prompt template."""
    return ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM),
        ("human", RAG_TEMPLATE),
    ])


def get_rag_chain():
    """
    Build and return the LangChain RAG chain:
    retriever -> format_docs -> prompt -> llm -> parse
    """
    retriever = FAISSRetriever(k=TOP_K_RETRIEVAL)
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | _get_prompt()
        | _get_llm()
        | StrOutputParser()
    )
    return chain


def get_context_chain():
    """
    Chain that takes pre-formatted context + question (no retriever).
    Used when pipeline already did retrieval and context validation.
    """
    return _get_prompt() | _get_llm() | StrOutputParser()


# ---------------------------------------------------
# Singleton chain and public API
# ---------------------------------------------------

_rag_chain = None
_context_chain = None


def get_rag_chain_cached():
    """Return the global RAG chain (lazy init)."""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = get_rag_chain()
    return _rag_chain


def get_context_chain_cached():
    """Return the global context-only chain (lazy init)."""
    global _context_chain
    if _context_chain is None:
        _context_chain = get_context_chain()
    return _context_chain


def invoke_rag(query: str) -> str:
    """
    Run the full LangChain RAG chain (retriever + LLM) for a single query.
    Returns the model response string.
    """
    chain = get_rag_chain_cached()
    try:
        response = chain.invoke(query)
        return (response or "").strip()
    except Exception as e:
        logger.exception("LangChain RAG invocation failed: %s", e)
        return "Something went wrong while generating the response."


@lru_cache(maxsize=100)
def _invoke_rag_with_context_impl(query: str, context: str) -> str:
    """Cached LLM call for repeated (query, context) pairs."""
    chain = get_context_chain_cached()
    try:
        response = chain.invoke({"context": context, "question": query})
        return (response or "").strip()
    except Exception as e:
        logger.exception("LangChain RAG (with context) failed: %s", e)
        return "Something went wrong while generating the response."


def invoke_rag_with_context(query: str, context: str) -> str:
    """
    Run LangChain with pre-formatted context (no retrieval).
    Used by the pipeline after it has already retrieved and validated context.
    Results are cached for repeated (query, context) to reduce LLM latency.
    """
    return _invoke_rag_with_context_impl(query, context)


def clear_langchain_llm_cache() -> None:
    """Clear LangChain LLM response cache (e.g. when user clears cache)."""
    _invoke_rag_with_context_impl.cache_clear()
    logger.info("LangChain LLM cache cleared")
