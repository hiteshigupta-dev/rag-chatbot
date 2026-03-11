"""
Streamlit Chatbot Application.
Tonton FAQ Assistant — ChatGPT-style layout: fixed header (after sidebar), full-width chat, custom bubbles, fixed input.
"""

import html
import json
import logging
import streamlit as st

# Show INFO logs (e.g. latency metrics) in the console when running: streamlit run app.py
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

from src.config import PAGE_TITLE, PAGE_ICON, SUGGESTED_QUESTIONS_PATH
from src.vector_store import get_vector_store
from src.embeddings import warmup_embedding_model, get_embedding_generator
from src.rag_pipeline import process_query
from src.cache import clear_cache

try:
    import markdown as md_lib
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False


# ------------------------------
# Session state
# ------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store_loaded" not in st.session_state:
    st.session_state.vector_store_loaded = False

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# ------------------------------
# Cached resource loading (heavy resources load only once at startup)
# ------------------------------
# Caching is important because:
# - Prevents repeated model loading on every user query (SentenceTransformer is large).
# - Improves latency: first query and all later queries use the same in-memory model/index.
# - Improves Streamlit Cloud deployment performance and avoids redundant disk/network load.
# Uses relative paths from config (PROJECT_ROOT) so it works on Streamlit Cloud.

@st.cache_resource
def load_embedding_model():
    """
    Load the SentenceTransformer embedding model once. Cached for the app lifetime.
    Returns the model so the pipeline uses it via the shared embedding generator.
    """
    warmup_embedding_model()
    return get_embedding_generator().model


@st.cache_resource
def load_faiss_index():
    """
    Load the FAISS index and chunks from disk once. Cached for the app lifetime.
    Uses paths from config (index/faiss_index, index/chunks.pkl) — relative to project root.
    Returns (index, chunks) or (None, None) if load fails.
    """
    vector_store = get_vector_store()
    success = vector_store.load()
    if not success:
        return None, None
    return vector_store.index, vector_store.chunks


# ------------------------------
# Custom CSS
# ------------------------------

def inject_custom_css():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        /* Base */
        .stApp { background: #fafafa; }
        .stApp [data-testid="stHeader"] { background: transparent; }
        html { font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important; }

        /* Prevent horizontal scrolling */
        html, body { overflow-x: hidden !important; }
        [data-testid="stAppViewContainer"] {
            overflow-x: hidden !important;
            position: relative !important;
        }

        /* Main area: scrollable */
        .stApp section.main {
            overflow-y: auto !important;
            overflow-x: hidden !important;
            -webkit-overflow-scrolling: touch !important;
            padding-bottom: 1rem !important;
        }
        /* Main container controls layout — everything follows this */
        .main .block-container,
        .stApp section.main .block-container {
            max-width: 1200px !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            padding-top: 0 !important;
            padding-bottom: 1rem !important;
        }

        /* Prevent sidebar overlap — sidebar on top when open */
        section[data-testid="stSidebar"] {
            z-index: 100 !important;
        }

        /* ---- HEADER: in flow, aligns with main content (no position: fixed) ---- */
        .chat-header-wrapper {
            position: relative !important;
        }
        .chat-header {
            width: 100% !important;
            position: relative !important;
            padding: 1rem 2rem !important;
            background: #22c55e !important;
            color: white;
            border-radius: 0 0 12px 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        .chat-header h1 {
            color: white !important;
            font-weight: 700 !important;
            font-size: 1.4rem !important;
            margin: 0 !important;
        }

        /* ---- CHAT INPUT: in flow, respects sidebar (no position: fixed) ---- */
        .stApp div:has(> [data-testid="stChatInput"]) {
            position: relative !important;
            background: #f8fafc !important;
            padding: 0.75rem 0 1rem !important;
        }
        [data-testid="stChatInput"] {
            width: 100% !important;
        }
        [data-testid="stChatInput"] > div {
            margin-bottom: 10px !important;
        }
        [data-testid="stChatInput"] textarea {
            border-radius: 12px !important;
            border: 2px solid #e2e8f0 !important;
            font-size: 0.95rem !important;
        }
        [data-testid="stChatInput"] textarea:focus {
            border-color: #22c55e !important;
            box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.2) !important;
        }

        /* ---- CUSTOM CHAT BUBBLES (no Streamlit avatars) ---- */
        .chat-bubble-container { width: 100%; margin-bottom: 0.5rem; }
        .chat-bubble-container.user { display: flex; justify-content: flex-end; }
        .chat-bubble-container.assistant { display: flex; justify-content: flex-start; }
        .chat-bubble {
            max-width: 78%;
            padding: 0.75rem 1rem;
            border-radius: 18px;
            font-size: 0.95rem;
            line-height: 1.5;
        }
        .chat-bubble.user {
            background: #DCF8C6 !important;
            color: #1a1a1a;
        }
        .chat-bubble.assistant {
            background: #F1F0F0 !important;
            color: #1a1a1a;
        }
        .chat-bubble .bubble-content p { margin: 0 0 0.5em 0; }
        .chat-bubble .bubble-content p:last-child { margin-bottom: 0; }
        .chat-bubble .bubble-content ul, .chat-bubble .bubble-content ol { margin: 0.5em 0; padding-left: 1.25em; }
        .chat-bubble-latency { font-size: 0.75rem; color: #64748b; margin-top: 0.35rem; }

        /* Quick question buttons: horizontal row, rounded cards */
        section.main [data-testid="column"] button[kind="secondary"] {
            width: 100% !important;
            border-radius: 12px !important;
            padding: 0.65rem 1rem !important;
            font-size: 0.875rem !important;
            border: 1px solid #e2e8f0 !important;
            background: white !important;
            color: #334155 !important;
            transition: all 0.2s ease !important;
        }
        section.main [data-testid="column"] button[kind="secondary"]:hover {
            background: #f0fdf4 !important;
            border-color: #22c55e !important;
            color: #166534 !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%); }
        [data-testid="stSidebar"] .stButton button { border-radius: 10px; font-weight: 500; }
        .sidebar-section-title {
            font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; margin-bottom: 0.5rem;
        }
        .sidebar-history-item {
            padding: 0.55rem 0.75rem; margin: 0.3rem 0; background: white; border-radius: 10px;
            font-size: 0.8rem; color: #334155; border: 1px solid #e2e8f0; line-height: 1.4;
        }

        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Header in main content flow — aligns with block container, never overlaps sidebar."""
    st.markdown("""
    <div class="chat-header-wrapper">
        <div class="chat-header">
            <h1>💬 Tonton FAQ Assistant</h1>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Default suggestions when no FAQ-derived list exists
DEFAULT_SUGGESTIONS = [
    "How do I subscribe to Tonton?",
    "What payment methods are accepted?",
    "How can I cancel my subscription?",
    "How do I contact support?",
]

# Max quick-question buttons to show
MAX_SUGGESTIONS = 4


def _normalize_question(text: str) -> str:
    """Collapse newlines and extra spaces so questions display as single lines."""
    if not isinstance(text, str):
        return str(text).strip()
    return " ".join(text.split())


def get_suggested_questions():
    """Load suggested questions from FAQ data so each has a good answer; fallback to defaults."""
    try:
        if SUGGESTED_QUESTIONS_PATH.exists():
            with open(SUGGESTED_QUESTIONS_PATH, "r", encoding="utf-8") as f:
                questions = json.load(f)
            if isinstance(questions, list) and questions:
                normalized = [_normalize_question(q) for q in questions[:MAX_SUGGESTIONS]]
                return normalized
    except Exception:
        pass
    return DEFAULT_SUGGESTIONS


def _text_to_html(text: str) -> str:
    """Convert message content to safe HTML (optional markdown)."""
    if HAS_MARKDOWN:
        return md_lib.markdown(text, extensions=["nl2br"])
    # Fallback: escape and newlines to <br>
    return "<br>".join(html.escape(line) for line in text.split("\n"))


def render_message_bubble(role: str, content: str, latency_seconds: float | None = None):
    """Render one chat bubble with custom HTML (no avatars). Optionally show latency."""
    html_content = _text_to_html(content)
    latency_html = ""
    if role == "assistant" and latency_seconds is not None:
        latency_html = f'<div class="chat-bubble-latency">⏱ {latency_seconds:.2f}s</div>'
    st.markdown(
        f'<div class="chat-bubble-container {role}">'
        f'<div class="chat-bubble {role}"><div class="bubble-content">{html_content}</div>{latency_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def display_chat_history():
    """Render all messages as custom bubbles: user right (#DCF8C6), assistant left (#F1F0F0)."""
    for message in st.session_state.messages:
        render_message_bubble(
            message["role"],
            message["content"],
            latency_seconds=message.get("latency_seconds"),
        )


def render_quick_questions():
    """Horizontal row of clickable quick-question buttons (from FAQ data when available)."""
    suggestions = get_suggested_questions()
    n = len(suggestions)
    cols = st.columns(n if n else 4)
    for i, suggestion in enumerate(suggestions):
        with cols[i]:
            if st.button(
                suggestion,
                key=f"quick_q_{i}",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state.pending_query = suggestion
                st.rerun()


# ------------------------------
# App init
# ------------------------------

def initialize_app():
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()


def handle_user_input(user_input: str):
    """Append user message, get RAG response, append assistant message."""
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        try:
            result = process_query(user_input)
            response = result.get("response", "I couldn't process your request.")
            msg = {"role": "assistant", "content": response}
            if result.get("latency_seconds") is not None:
                msg["latency_seconds"] = result["latency_seconds"]
            st.session_state.messages.append(msg)
        except Exception as e:
            logging.exception("RAG pipeline error")
            fallback = "Something went wrong. Please try again or rephrase your question."
            st.session_state.messages.append({
                "role": "assistant",
                "content": fallback,
            })


# ------------------------------
# Main
# ------------------------------

def main():
    initialize_app()

    # HEADER (in flow — aligns with main content, no sidebar overlap)
    render_header()

    # Load heavy resources once at application startup (not inside the query loop).
    # Both load_embedding_model() and load_faiss_index() are @st.cache_resource — they run
    # only once per app session; subsequent calls reuse the cached result.
    if not st.session_state.vector_store_loaded:
        with st.spinner("Loading knowledge base..."):
            try:
                model = load_embedding_model()
                index, chunks = load_faiss_index()
                if index is not None and chunks is not None:
                    st.session_state.vector_store_loaded = True
                else:
                    st.error("Knowledge base not found. Please run `python build_index.py` first.")
                    return
            except Exception as e:
                st.error(f"Failed to load knowledge base: {str(e)}")
                logging.exception("Load error")
                return

    # Welcome screen only when no messages
    if len(st.session_state.messages) == 0:
        st.markdown("##### 👋 Start a conversation")
        st.markdown("Choose a question below or type your own.")
        render_quick_questions()
        st.markdown("")

    # Handle pending query from quick-question click
    if st.session_state.pending_query:
        q = st.session_state.pending_query
        st.session_state.pending_query = None
        handle_user_input(q)
        st.rerun()

    # CHAT CONTAINER (scrollable): custom bubbles, full width
    chat_container = st.container()
    with chat_container:
        display_chat_history()

    # INPUT BAR (in flow — aligns with chat messages)
    if prompt := st.chat_input("Ask a question about Tonton..."):
        handle_user_input(prompt)
        st.rerun()

    # Sidebar
    with st.sidebar:
        st.markdown("### 💬 Chat")
        st.markdown("")
        user_messages = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
        if user_messages:
            st.markdown('<p class="sidebar-section-title">This conversation</p>', unsafe_allow_html=True)
            to_show = user_messages[-15:] if len(user_messages) > 15 else user_messages
            for i, q in enumerate(to_show):
                preview = (q[:60] + "…") if len(q) > 60 else q
                st.markdown(f'<div class="sidebar-history-item">{i + 1}. {preview}</div>', unsafe_allow_html=True)
            if len(user_messages) > 15:
                st.caption(f"+ {len(user_messages) - 15} earlier")
            st.markdown("")
        else:
            st.caption("No messages yet. Ask something to see history here.")
            st.markdown("")
        st.markdown("---")
        st.markdown("### ⚙️ Actions")
        st.markdown("")
        if st.button("🗑️ Clear chat", use_container_width=True, help="Clear messages and start fresh"):
            st.session_state.messages = []
            st.rerun()
        if st.button("🔄 Clear cache", use_container_width=True, help="Clear cached answers"):
            try:
                clear_cache()
                st.success("Cache cleared")
            except Exception as e:
                st.warning(f"Could not clear cache: {e}")
        st.markdown("---")
        with st.expander("ℹ️ About", expanded=False):
            st.markdown("**Tonton FAQ Assistant**")
            st.markdown("Answers are based only on the FAQ. If your question isn't in the FAQ, you'll see a fallback message.")
            st.markdown("**Powered by**")
            st.markdown("- FAISS · Vector search")
            st.markdown("- SentenceTransformers · Embeddings")
            st.markdown("- Gemini · Language model")


if __name__ == "__main__":
    main()
