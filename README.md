# RAG-based Tonton FAQ Support Chatbot

A production-oriented **Retrieval-Augmented Generation (RAG)** chatbot that answers frequently asked questions about the **Tonton** streaming platform. It combines semantic search over FAQ content with a large language model to deliver accurate, context-grounded support answers through a modern Streamlit chat interface.

---

## Overview

The chatbot is designed to reduce support load by letting users ask natural-language questions and receive answers drawn directly from Tonton’s FAQ documentation. Instead of relying only on the LLM’s training data, the system:

1. **Retrieves** relevant FAQ chunks using a FAISS vector index and semantic similarity.
2. **Augments** the LLM prompt with this retrieved context.
3. **Generates** answers that are grounded in the provided context, improving accuracy and reducing hallucination.

RAG is used so that responses stay aligned with the official FAQ, can be updated by changing the source documents and rebuilding the index, and remain traceable to the underlying content.

---

## Features

| Feature | Description |
|--------|-------------|
| **Retrieval-Augmented Generation (RAG)** | Answers are generated from retrieved FAQ chunks, not from the model’s memory alone. |
| **FAISS semantic search** | Dense vector search over Sentence Transformer embeddings for relevant passage retrieval. |
| **Gemini LLM integration** | Uses Google’s Gemini 2.5 Flash API for fast, high-quality answer generation. |
| **Streamlit chat interface** | ChatGPT-style UI with custom bubbles, quick-question buttons, and sidebar conversation history. |
| **Latency monitoring and logging** | Logs retrieval time, LLM time, and total pipeline time for observability. |
| **Context-grounded responses** | Only retrieved, relevant chunks are sent to the LLM as context. |
| **Response caching** | In-memory and disk caching for repeated queries to reduce latency and API usage. |
| **Clean modular architecture** | Separate modules for retrieval, LLM, cache, guardrails, and pipeline orchestration. |

Additional capabilities include **guardrails**: the model is instructed to answer only from the retrieved FAQ context; if the answer is not in the context, it returns a fallback message (e.g. *"I'm sorry, I couldn't find this information in the FAQ."*). Safety guardrails block unsafe or off-topic queries; context validation rejects weak retrieval matches.

---

## Architecture

End-to-end flow from user query to response:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAG Pipeline                                       │
└─────────────────────────────────────────────────────────────────────────────┘

  User Query
       │
       ▼
  ┌─────────────────┐
  │ Cache lookup    │  ← If hit: return cached response (skip retrieval + LLM)
  └────────┬────────┘
           │ miss
           ▼
  ┌─────────────────┐
  │ Guardrails      │  ← Block unsafe or off-topic queries
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Embedding       │  ← Sentence Transformers encode the query
  │ Generation      │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ FAISS Retrieval │  ← Similarity / MMR search over indexed chunks
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Context         │  ← Validate relevance, format top-k chunks
  │ Assembly        │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Prompt          │  ← Build system + user message with context
  │ Construction    │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Gemini LLM      │  ← Generate answer from context
  │ Generation      │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Cache response  │  ← Store for future identical queries
  └────────┬────────┘
           │
           ▼
  Streamlit UI Response
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.x |
| **Vector store** | FAISS (CPU) |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) |
| **LLM** | Google Gemini 2.5 Flash API |
| **Web UI** | Streamlit |
| **RAG / chains** | LangChain (optional; custom pipeline also supported) |
| **PDF ingestion** | PyPDF2 |
| **Caching** | Custom in-memory + disk (JSON files in `cache/`) |
| **Config & env** | `python-dotenv`, `src/config.py` |

---

## Project Structure

```
rag_chatbot/
│
├── app.py                 # Streamlit chat application entry point
├── build_index.py         # Script to build FAISS index from FAQ PDF
├── requirements.txt       # Python dependencies
├── README.md
├── Makefile
├── .env.example           # Template for env vars (copy to .env)
├── .env                   # Your Gemini API key (create from .env.example)
│
├── data/
│   └── faq.pdf            # Source FAQ (faq.pdf or FAQ.pdf)
│
├── index/                 # Generated at build time
│   ├── faiss_index        # FAISS index
│   ├── chunks.pkl         # Chunk metadata
│   └── suggested_questions.json
│
├── cache/                 # Runtime response cache (created automatically)
│   └── *.json
│
└── src/
    ├── config.py          # Paths, model names, and app settings
    ├── utils.py           # Project root and path helpers (Streamlit-safe)
    ├── rag_pipeline.py    # End-to-end RAG orchestration
    ├── retriever.py       # FAISS retrieval and MMR
    ├── vector_store.py    # FAISS index load/save and similarity search
    ├── embeddings.py      # Sentence Transformer embedding model
    ├── ingest.py          # PDF loading, chunking, Q&A extraction
    ├── llm.py             # Gemini LLM client and prompt logic
    ├── langchain_rag.py   # LangChain-based RAG (optional)
    ├── cache.py           # Query response cache (memory + disk)
    ├── guardrails.py      # Query safety and scope checks
    ├── context_guardrail.py # Context validation and fallback messages
    └── __init__.py
```

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd rag_chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

- **Windows:** `venv\Scripts\activate`
- **macOS/Linux:** `source venv/bin/activate`

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or use the Makefile (requires `make`, e.g. from Git for Windows or WSL):

```bash
make install
```

---

## Running the Project

### Step 1: Build the FAISS index

Place your FAQ document at `data/faq.pdf` (or `data/FAQ.pdf`), then run:

```bash
python build_index.py
```

Or:

```bash
make build-index
```

This will:

- Load and chunk the PDF (RecursiveCharacterTextSplitter)
- Embed chunks with Sentence Transformers
- Build and save the FAISS index under `index/`
- Generate `suggested_questions.json` for the UI

### Step 2: Run the Streamlit application

```bash
streamlit run app.py
```

Or:

```bash
make run
```

Then open the URL shown in the terminal (typically `http://localhost:8501`) to use the chat UI.

**Without Make:** run `python build_index.py` then `streamlit run app.py` from the project root.

---

## Environment Variables

**You must create your own `.env` file** — the app does not ship with API keys.

1. Copy the example file:  
   `cp .env.example .env` (Linux/Mac) or copy `.env.example` to `.env` (Windows).
2. Edit `.env` and set your Gemini API key:
   ```env
   GEMINI_API_KEY=your_key_here
   ```
3. Get an API key from [Google AI Studio](https://aistudio.google.com/apikey).

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | **Yes** | Your Google Gemini API key for the LLM. |

Optional:

- `USE_LANGCHAIN` — Set to `true` to use LangChain for RAG; `false` (default) for the custom LLM path. If `true`, install LangChain dependencies (see `requirements.txt`).

---

## Logging and Monitoring

The application logs to the console (and can be extended to files or a logging service). When running `streamlit run app.py`, you will see:

| Metric | Description |
|--------|-------------|
| **Total pipeline time** | End-to-end time for a query (including retrieval and LLM). |
| **Retrieval latency** | Time to embed the query and run FAISS search. |
| **LLM latency** | Time for the Gemini API call. |
| **Cache hits** | Logged when a response is served from cache (memory or disk). |

Example log output:

```
INFO - Processing query: How do I subscribe to Tonton?
INFO - Retrieved 3 documents
INFO - Total pipeline time: 1.234 seconds
INFO - Latency Summary → Retrieval: 0.045s | LLM: 1.180s | Total: 1.234s
Retrieval Time: 0.05s
LLM Time: 1.18s
Total Pipeline Time: 1.23s
```

Cache hits look like:

```
INFO - Cache hit (latency: 0.001s)
```

Log level is set in `app.py` (e.g. `logging.basicConfig(level=logging.INFO, ...)`).

---

## Deployment

### Streamlit Cloud deployment checklist

Follow these steps to deploy the app on [Streamlit Cloud](https://share.streamlit.io):

**Step 1:** Push the repository to GitHub (include `app.py`, `requirements.txt`, `src/`, `data/faq.pdf` or `data/FAQ.pdf`, and a prebuilt `index/` if you do not use a build step).

**Step 2:** Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.

**Step 3:** Click **New app** and connect your GitHub repository. Choose the branch (e.g. `main`).

**Step 4:** Configure the app:
- **Main file path:** `app.py`
- **App root:** the directory that contains `app.py` (e.g. repo root or `rag_chatbot` if the repo root is the parent folder).
- In **Settings → Secrets**, add your Gemini API key (required). Use the format:
  ```
  GEMINI_API_KEY = "your_api_key_here"
  ```
  On Streamlit Cloud, secrets are injected as environment variables; no `.env` file is needed in the repo.

**Step 5:** Deploy. Streamlit Cloud will install dependencies from `requirements.txt` and run `streamlit run app.py`.

**Notes:**
- The FAISS index must be present at `index/faiss_index` (and `index/chunks.pkl`). Either commit a prebuilt `index/` after running `python build_index.py` locally, or add a build step that runs `build_index.py` before the app starts.
- The app uses relative paths and is compatible with Streamlit Cloud; avoid hard-coded absolute paths.

---

## Future Improvements

- **Hybrid search** — Combine BM25 (keyword) with FAISS (semantic) for better recall on mixed queries.
- **MMR retrieval** — Expand or tune Maximal Marginal Relevance for more diverse chunks.
- **Conversation memory** — Multi-turn context (e.g. last N turns) for follow-up questions.
- **Stronger guardrails** — PII detection, stricter topic scoping, and configurable blocklists.
- **Observability dashboards** — Metrics (latency, cache hit rate, errors) in Grafana, Datadog, or similar.
- **A/B testing** — Compare different chunk sizes, models, or prompts.
- **Feedback loop** — Thumbs up/down or “was this helpful?” to improve retrieval and prompts over time.
