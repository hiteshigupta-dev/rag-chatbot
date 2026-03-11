"""
Build Index Script.
Runs offline indexing: load PDF, chunk with RecursiveCharacterTextSplitter,
embed each chunk, and store in FAISS. Run this after adding or updating FAQ.pdf.
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.ingest import load_faq_document, extract_full_text_from_pdf, extract_qa_questions
from src.vector_store import VectorStore
from src.config import (
    INDEX_DIR,
    DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SUGGESTED_QUESTIONS_PATH,
    FAQ_PDF_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_index() -> bool:
    """
    Build FAISS index from FAQ PDF:
    1. Load PDF and extract text (UTF-8 safe)
    2. Split into chunks (RecursiveCharacterTextSplitter)
    3. Create embeddings for each chunk
    4. Store in FAISS and save to disk.
    """
    logger.info("=" * 60)
    logger.info("Starting offline indexing (chunk_size=%s, chunk_overlap=%s)", CHUNK_SIZE, CHUNK_OVERLAP)
    logger.info("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Support both faq.pdf and FAQ.pdf
    pdf_path = FAQ_PDF_PATH if FAQ_PDF_PATH.exists() else DATA_DIR / "FAQ.pdf"
    if not pdf_path.exists():
        logger.error("FAQ PDF not found. Place faq.pdf or FAQ.pdf in the data/ directory.")
        return False

    try:
        logger.info("[1/3] Loading and chunking FAQ document from %s", pdf_path)
        chunks = load_faq_document(pdf_path)

        if not chunks:
            logger.error("No chunks generated. Check PDF content and format.")
            return False

        logger.info("Chunks generated: %s", len(chunks))
        if len(chunks) < 3:
            logger.warning("Very few chunks; consider checking PDF structure or chunk settings.")

        logger.info("[2/3] Creating FAISS vector store (embedding each chunk)")
        vector_store = VectorStore()
        vector_store.create_index(chunks)

        logger.info("[3/3] Saving index to disk")
        vector_store.save()

        # Save suggested questions from FAQ so app shows only questions that have answers
        full_text = extract_full_text_from_pdf(pdf_path)
        qa_questions = extract_qa_questions(full_text)
        if qa_questions:
            SUGGESTED_QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(SUGGESTED_QUESTIONS_PATH, "w", encoding="utf-8") as f:
                json.dump(qa_questions, f, ensure_ascii=False, indent=0)
            logger.info("Saved %s suggested questions to %s", len(qa_questions), SUGGESTED_QUESTIONS_PATH)

        logger.info("=" * 60)
        logger.info("Indexing complete. Total chunks indexed: %s", len(chunks))
        logger.info("Index directory: %s", INDEX_DIR)
        logger.info("=" * 60)
        return True

    except Exception:
        logger.exception("Error during indexing")
        return False


if __name__ == "__main__":
    success = build_index()
    sys.exit(0 if success else 1)