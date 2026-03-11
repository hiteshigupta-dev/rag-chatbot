"""
Configuration module for RAG Chatbot.
Contains all configuration settings and constants.
Uses relative paths for Streamlit Cloud compatibility.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from .utils import get_project_root

load_dotenv()

# Project paths (relative to project root; safe for Streamlit Cloud)
PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "index"
CACHE_DIR = PROJECT_ROOT / "cache"

# Create directories if they don't exist (skip if read-only, e.g. cloud)
try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass

# File paths (use faq.pdf; build_index also checks FAQ.pdf for compatibility)
FAQ_PDF_PATH = DATA_DIR / "faq.pdf"
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index"
CHUNKS_PATH = INDEX_DIR / "chunks.pkl"
SUGGESTED_QUESTIONS_PATH = INDEX_DIR / "suggested_questions.json"

# Embedding model settings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"  # Use "cuda" if GPU available

# Chunking (LangChain RecursiveCharacterTextSplitter)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Vector store settings
TOP_K_RETRIEVAL = 3  # Number of chunks to retrieve (faster retrieval)
MMR_FETCH_K = 5  # Candidates fetched for MMR re-ranking (lower = faster retrieval)
USE_SIMILARITY_SEARCH_BELOW_CHUNKS = 100  # Use simple similarity_search instead of MMR when chunks < this
MAX_CONTEXT_CHUNKS = 3  # Max chunks sent to LLM (reduces context size and latency)

# LLM settings
try:
    import streamlit as st
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
GEMINI_MODEL_NAME = "gemini-2.5-flash"
DEFAULT_LLM_TEMPERATURE = 0.2  # Low for deterministic FAQ-style answers
DEFAULT_LLM_MAX_TOKENS = 768  # Enough for full FAQ answers (contact info, steps, etc.)

# Use LangChain for RAG (True) or custom LLM (False). Default False for lighter deps.
USE_LANGCHAIN = os.getenv("USE_LANGCHAIN", "false").lower() in ("true", "1", "yes")

# Guardrail settings
MALICIOUS_KEYWORDS = [
    "hack", "illegal", "bypass", "exploit", "malware", "phishing",
    "unauthorized", "breach", "steal", "fraud", "harmful", "attack"
]

# Caching settings
QUERY_CACHE_TTL = 3600  # 1 hour in seconds
MAX_CACHE_SIZE = 1000

# Streamlit settings
PAGE_TITLE = "Tonton FAQ Chatbot"
PAGE_ICON = "💬"

# Context relevance threshold (for guardrail)
RELEVANCE_THRESHOLD = 0.3

# Similarity threshold for filtering retrieved chunks (lower = more permissive)
SIMILARITY_THRESHOLD = 0.35

# Fallback when no context is retrieved (guardrail: answer only from FAQ)
NO_CONTEXT_FALLBACK = (
    "I'm sorry, I couldn't find this information in the FAQ. "
    "Please try rephrasing or ask another question about the Tonton platform."
)

