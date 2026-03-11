"""
Document ingestion module.

Implements a hybrid chunking strategy for RAG:
1. Q&A chunks — extract "Question: ... Answer: ..." pairs from the PDF.
2. Semantic chunks — split full text with RecursiveCharacterTextSplitter.

Both chunk types are embedded and stored in FAISS for improved retrieval.
"""

import re
import logging
from pathlib import Path
from typing import List

import PyPDF2

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from .config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

# Regex to detect FAQ pairs: "Question: ... Answer: ..." until next Question or end.
QA_PATTERN = re.compile(
    r"Question:\s*(.*?)\s*Answer:\s*(.*?)(?=Question:|$)",
    re.DOTALL | re.IGNORECASE,
)


def _safe_decode(text: str) -> str:
    """Normalize and clean text for UTF-8 safety."""
    if not text or not isinstance(text, str):
        return ""
    replacements = {
        "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
        "\u2013": "-", "\u2014": "--", "\u2026": "...",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.strip()


def extract_full_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract and concatenate text from all PDF pages with UTF-8 safe handling.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Single string of all page text joined by newlines.

    Raises:
        FileNotFoundError: If the PDF does not exist.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PyPDF2.PdfReader(str(pdf_path))
    text_parts: List[str] = []

    for page in reader.pages:
        try:
            raw = page.extract_text()
            if raw:
                text_parts.append(_safe_decode(raw))
        except Exception as e:
            logger.warning("Page extraction failed: %s", e)

    full_text = "\n".join(text_parts)
    logger.info("PDF text extracted successfully (%s pages, %s chars)", len(reader.pages), len(full_text))
    return full_text


def extract_qa_chunks(text: str) -> List[str]:
    """
    Detect FAQ pairs and return one chunk per pair in the format:
    "Question: <question text>\nAnswer: <answer text>"

    Args:
        text: Full document text.

    Returns:
        List of Q&A chunks (each is one full Q&A pair).
    """
    text = re.sub(r"\r", "", text)
    text = re.sub(r"\n+", "\n", text)

    chunks: List[str] = []
    for match in QA_PATTERN.finditer(text):
        question = match.group(1).strip()
        answer = match.group(2).strip()
        if question and answer:
            chunk = f"Question: {question}\nAnswer: {answer}"
            chunks.append(chunk)

    logger.info("Q&A chunks extracted: %s", len(chunks))
    return chunks


def extract_qa_questions(text: str) -> List[str]:
    """
    Extract only the question text from each "Question: ... Answer: ..." pair.
    Used to build suggested questions that have answers in the FAQ data.
    """
    text = re.sub(r"\r", "", text)
    text = re.sub(r"\n+", "\n", text)
    questions: List[str] = []
    for match in QA_PATTERN.finditer(text):
        question = match.group(1).strip()
        if question and match.group(2).strip():
            questions.append(question)
    return questions


def get_semantic_chunks(full_text: str) -> List[str]:
    """
    Split full text into semantic chunks using RecursiveCharacterTextSplitter.

    Args:
        full_text: Full document text.

    Returns:
        List of text chunks (chunk_size=500, chunk_overlap=100).
    """
    if not full_text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(full_text)
    logger.info("Semantic chunks created: %s", len(chunks))
    return chunks


def load_faq_document(pdf_path: Path) -> List[str]:
    """
    Hybrid ingestion: extract full text, then produce both Q&A and semantic chunks.

    Pipeline:
    1. Extract full text from PDF (UTF-8 safe).
    2. Extract Q&A chunks (one chunk per "Question: ... Answer: ..." pair).
    3. Build semantic chunks from full text (RecursiveCharacterTextSplitter).
    4. Return qa_chunks + semantic_chunks for embedding and FAISS.

    Args:
        pdf_path: Path to the FAQ PDF.

    Returns:
        Combined list of chunks ready for embedding; compatible with VectorStore.
    """
    logger.info("Loading PDF from: %s", pdf_path)

    full_text = extract_full_text_from_pdf(pdf_path)
    if not full_text.strip():
        logger.warning("PDF is empty")
        return []

    qa_chunks = extract_qa_chunks(full_text)
    semantic_chunks = get_semantic_chunks(full_text)

    chunks = qa_chunks + semantic_chunks
    logger.info("Total chunks for embedding: %s", len(chunks))

    if chunks:
        sample = chunks[0][:200] if len(chunks[0]) > 200 else chunks[0]
        logger.info("Sample chunk: %s", sample)
    else:
        logger.warning("No chunks produced; check document content")

    return chunks
