"""
Shared utilities for RAG Chatbot.
Provides path resolution and helpers safe for Streamlit Cloud (no absolute local paths).
"""

from pathlib import Path


def get_project_root() -> Path:
    """
    Resolve project root (directory containing app.py).
    Uses __file__ so it works on Streamlit Cloud and local runs.
    """
    # src/utils.py -> parent -> parent = project root
    return Path(__file__).resolve().parent.parent


def resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to project root. Use forward slashes, e.g. 'data/faq.pdf'."""
    root = get_project_root()
    return (root / relative_path).resolve()
