"""
LawAI India - Retrieval-Augmented Generation for Indian Law
Main Streamlit Application
"""

import os
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from utils.loader import DocumentLoader
from utils.embedder import EmbeddingManager
from utils.retriever import LegalRetriever
from utils.llm import LegalLLM
from utils.ingestion import IngestionPipeline
from utils.prompt_templates import PromptTemplates

# ─── Load environment variables ────────────────────────────────────────────────
load_dotenv()

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("LawAI")

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LawAI India",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Import Playfair Display + Source Sans 3 */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+3:wght@300;400;600&display=swap');

:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-card: #1c2230;
    --accent-gold: #c9a84c;
    --accent-gold-light: #e8c97a;
    --accent-crimson: #8b1a1a;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --text-muted: #484f58;
    --border: #30363d;
    --border-accent: #c9a84c40;
    --success: #3fb950;
    --radius: 8px;
}

/* Global reset */
html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* App container */
.main .block-container {
    max-width: 1100px;
    padding: 1.5rem 2rem;
}

/* Header */
.law-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 0.25rem;
    border-bottom: 1px solid var(--border-accent);
    padding-bottom: 1rem;
}
.law-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent-gold);
    margin: 0;
    letter-spacing: 0.02em;
}
.law-header .tagline {
    font-size: 0.8rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 0;
}

/* Chat messages */
.stChatMessage {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    margin-bottom: 0.75rem !important;
}

/* User message */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [aria-label="user"]) {
    border-left: 3px solid var(--accent-gold) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Playfair Display', serif;
    color: var(--accent-gold);
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Metric cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.metric-label {
    font-size: 0.75rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.metric-value {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--accent-gold-light);
}

/* Citation panel */
.citation-box {
    background: var(--bg-card);
    border: 1px solid var(--border-accent);
    border-left: 3px solid var(--accent-gold);
    border-radius: var(--radius);
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
}
.citation-score {
    display: inline-block;
    background: var(--accent-gold);
    color: #000;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 3px;
    margin-left: 8px;
}

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--accent-gold) !important;
    color: var(--accent-gold) !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--accent-gold) !important;
    color: #000 !important;
}

/* Expander */
.st-expander {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--bg-card);
    border: 1px dashed var(--border-accent);
    border-radius: var(--radius);
    padding: 1rem;
}

/* Input */
[data-testid="stChatInput"] {
    border-color: var(--border-accent) !important;
    background: var(--bg-card) !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Status indicator */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-green { background: var(--success); }
.status-yellow { background: #f0a500; }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "chroma_db"))
NEW_LAWS_DIR = DATA_DIR / "new_laws"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

# Ensure dirs exist
DATA_DIR.mkdir(exist_ok=True)
NEW_LAWS_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "additional_laws").mkdir(exist_ok=True)


# ─── Cached resource initialisation ────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def init_system():
    """Initialise embedding manager and retriever (cached across sessions)."""
    embedder = EmbeddingManager(
        model_name=EMBEDDING_MODEL,
        chroma_dir=str(CHROMA_DIR),
    )
    retriever = LegalRetriever(
        embedding_manager=embedder,
        top_k=TOP_K,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )
    llm = LegalLLM(
        model_name=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    return embedder, retriever, llm


@st.cache_data(show_spinner=False)
def get_db_stats(_embedder: EmbeddingManager):
    """Return document/chunk counts from ChromaDB (cached, invalidated on new ingestion)."""
    return _embedder.get_stats()


# ─── Session state defaults ────────────────────────────────────────────────────
def init_session():
    defaults = {
        "messages": [],
        "retrieved_contexts": [],
        "ingestion_done": False,
        "last_query": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()

# ─── Load system ───────────────────────────────────────────────────────────────
with st.spinner("Initialising LawAI India …"):
    try:
        embedder, retriever, llm = init_system()
        system_ready = True
    except Exception as exc:
        st.error(f"System initialisation failed: {exc}")
        logger.exception("System init error")
        system_ready = False

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center;padding:1rem 0 0.5rem'>
            <div style='font-size:2.5rem'>⚖️</div>
            <div style='font-family:"Playfair Display",serif;font-size:1.2rem;
                        color:#c9a84c;font-weight:700;letter-spacing:0.04em'>
                LawAI India
            </div>
            <div style='font-size:0.7rem;color:#8b949e;text-transform:uppercase;
                        letter-spacing:0.1em;margin-top:2px'>
                Legal Intelligence System
            </div>
        </div>
        <hr/>
        """,
        unsafe_allow_html=True,
    )

    # ── System status ─────────────────────────────────────────────────────────
    st.markdown("### 📊 System Status")
    if system_ready:
        stats = get_db_stats(embedder)
        cols = st.columns(2)
        cols[0].metric("Documents", stats.get("num_documents", 0))
        cols[1].metric("Chunks", stats.get("num_chunks", 0))
        st.markdown(
            f"""
            <div class='metric-card'>
                <span class='metric-label'>Model</span>
                <span class='metric-value'>{OLLAMA_MODEL}</span>
            </div>
            <div class='metric-card'>
                <span class='metric-label'>Embeddings</span>
                <span class='metric-value'>{stats.get("num_embeddings", 0)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # List loaded acts
        acts = stats.get("acts", [])
        if acts:
            st.markdown("### 📚 Loaded Acts")
            for act in acts:
                st.markdown(
                    f"<div style='font-size:0.8rem;color:#8b949e;padding:2px 0'>• {act}</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.error("System not ready")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── Ingest documents ──────────────────────────────────────────────────────
    st.markdown("### 📥 Ingest Documents")

    if st.button("🔄 Ingest All PDFs in /data", use_container_width=True):
        pdf_files = list(DATA_DIR.rglob("*.pdf"))
        if not pdf_files:
            st.warning("No PDFs found in /data directory.")
        else:
            pipeline = IngestionPipeline(embedder=embedder, data_dir=str(DATA_DIR))
            with st.spinner(f"Ingesting {len(pdf_files)} PDF(s)…"):
                result = pipeline.ingest_all()
            st.success(
                f"✅ Ingested {result['total_chunks']} chunks from {result['total_files']} files."
            )
            get_db_stats.clear()
            st.rerun()

    st.markdown("**Upload New Legal PDF**")
    uploaded = st.file_uploader(
        "Drop PDF here", type=["pdf"], label_visibility="collapsed"
    )
    if uploaded:
        act_name = st.text_input("Act Name (optional)", placeholder="e.g. Motor Vehicles Act")
        if st.button("📤 Upload & Index", use_container_width=True):
            save_path = NEW_LAWS_DIR / uploaded.name
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            pipeline = IngestionPipeline(embedder=embedder, data_dir=str(DATA_DIR))
            with st.spinner("Indexing uploaded PDF…"):
                result = pipeline.ingest_single(
                    str(save_path),
                    act_name=act_name or uploaded.name.replace(".pdf", ""),
                )
            st.success(f"✅ Indexed {result['chunks_added']} chunks.")
            get_db_stats.clear()
            st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── Controls ──────────────────────────────────────────────────────────────
    st.markdown("### ⚙️ Controls")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.retrieved_contexts = []
        st.rerun()

    if st.session_state.messages:
        chat_export = json.dumps(
            [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            indent=2,
            ensure_ascii=False,
        )
        st.download_button(
            "💾 Download Chat History",
            data=chat_export,
            file_name=f"lawai_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.7rem;color:#484f58;text-align:center'>"
        "LawAI India v1.0 — For educational purposes only.<br/>"
        "Not a substitute for legal advice."
        "</div>",
        unsafe_allow_html=True,
    )

# ─── Main area ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class='law-header'>
        <span style='font-size:2rem'>⚖️</span>
        <div>
            <h1>LawAI India</h1>
            <p class='tagline'>AI-Powered Indian Legal Research Assistant</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Welcome message
if not st.session_state.messages:
    st.markdown(
        """
        <div style='background:#1c2230;border:1px solid #c9a84c40;border-radius:8px;
                    padding:1.25rem 1.5rem;margin-bottom:1.5rem'>
            <p style='margin:0 0 0.5rem;font-family:"Playfair Display",serif;
                      color:#c9a84c;font-size:1rem'>Welcome to LawAI India</p>
            <p style='margin:0;font-size:0.88rem;color:#8b949e;line-height:1.6'>
                Ask questions about Indian law in <strong style='color:#e6edf3'>English</strong>,
                <strong style='color:#e6edf3'>Hindi</strong>, or
                <strong style='color:#e6edf3'>Telugu</strong>.
                The system retrieves relevant legal provisions and answers based only on
                indexed documents. Ingest PDFs using the sidebar before querying.
            </p>
            <div style='margin-top:0.75rem;display:flex;gap:8px;flex-wrap:wrap'>
                <span style='background:#0d1117;border:1px solid #30363d;border-radius:4px;
                             padding:3px 10px;font-size:0.75rem;color:#8b949e'>
                    Constitution of India
                </span>
                <span style='background:#0d1117;border:1px solid #30363d;border-radius:4px;
                             padding:3px 10px;font-size:0.75rem;color:#8b949e'>
                    BNS / BNSS / BSA
                </span>
                <span style='background:#0d1117;border:1px solid #30363d;border-radius:4px;
                             padding:3px 10px;font-size:0.75rem;color:#8b949e'>
                    IT Act · GST · Income Tax
                </span>
                <span style='background:#0d1117;border:1px solid #30363d;border-radius:4px;
                             padding:3px 10px;font-size:0.75rem;color:#8b949e'>
                    Companies Act · Labour Laws
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Render chat history ─────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander("📎 Legal Citations & Retrieved Context", expanded=False):
                for i, ctx in enumerate(msg["citations"], 1):
                    score_pct = int((1 - ctx.get("score", 0)) * 100)
                    section_str = f" · §{ctx.get('section', '')}" if ctx.get("section") else ""
                    chapter_str = f" · Ch.{ctx.get('chapter', '')}" if ctx.get("chapter") else ""
                    st.markdown(
                        f"""
                        <div class='citation-box'>
                            <div style='display:flex;justify-content:space-between;
                                        align-items:center;margin-bottom:4px'>
                                <strong style='color:#c9a84c'>
                                    [{i}] {ctx.get("act_name", "Unknown Act")}
                                </strong>
                                <span class='citation-score'>Relevance {score_pct}%</span>
                            </div>
                            <div style='font-size:0.75rem;color:#8b949e;margin-bottom:6px'>
                                📄 Page {ctx.get("page_number","?")}{section_str}{chapter_str}
                            </div>
                            <div style='font-size:0.82rem;color:#c9d1d9;
                                        border-top:1px solid #30363d;padding-top:6px;
                                        max-height:120px;overflow-y:auto'>
                                {ctx.get("text","")[:500]}…
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# ── Chat input ──────────────────────────────────────────────────────────────────
if prompt := st.chat_input(
    "Ask about Indian law… (English / हिंदी / తెలుగు)"
):
    if not system_ready:
        st.error("System not initialised. Please check the logs.")
        st.stop()

    stats = get_db_stats(embedder)
    if stats.get("num_chunks", 0) == 0:
        st.warning(
            "No documents indexed yet. Please ingest PDFs from the sidebar first."
        )
        st.stop()

    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("Searching legal database…"):
            try:
                results = retriever.retrieve(prompt)
            except Exception as exc:
                st.error(f"Retrieval error: {exc}")
                logger.exception("Retrieval error")
                st.stop()

        if not results:
            no_answer = (
                "I could not find this information in the legal database. "
                "Please ensure relevant PDFs have been ingested."
            )
            st.markdown(no_answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": no_answer, "citations": []}
            )
            st.stop()

        # ── Build context & generate answer ──────────────────────────────────
        context_str = PromptTemplates.build_context(results)
        system_prompt = PromptTemplates.legal_system_prompt()
        user_prompt = PromptTemplates.legal_user_prompt(prompt, context_str)

        # Stream response
        response_placeholder = st.empty()
        full_response = ""

        with st.spinner("Generating legal analysis…"):
            try:
                for token in llm.stream(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    conversation_history=st.session_state.messages[:-1],
                ):
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
            except Exception as exc:
                st.error(
                    f"LLM error: {exc}. Ensure Ollama is running with `ollama serve`."
                )
                logger.exception("LLM error")
                st.stop()

        response_placeholder.markdown(full_response)

        # ── Show citations ────────────────────────────────────────────────────
        citations = [
            {
                "act_name": r["metadata"].get("act_name", r["metadata"].get("source", "Unknown")),
                "page_number": r["metadata"].get("page_number", "?"),
                "section": r["metadata"].get("section", ""),
                "chapter": r["metadata"].get("chapter", ""),
                "score": r["score"],
                "text": r["text"],
            }
            for r in results
        ]

        with st.expander("📎 Legal Citations & Retrieved Context", expanded=False):
            for i, ctx in enumerate(citations, 1):
                score_pct = int((1 - ctx.get("score", 0)) * 100)
                section_str = f" · §{ctx.get('section', '')}" if ctx.get("section") else ""
                chapter_str = f" · Ch.{ctx.get('chapter', '')}" if ctx.get("chapter") else ""
                st.markdown(
                    f"""
                    <div class='citation-box'>
                        <div style='display:flex;justify-content:space-between;
                                    align-items:center;margin-bottom:4px'>
                            <strong style='color:#c9a84c'>
                                [{i}] {ctx.get("act_name", "Unknown Act")}
                            </strong>
                            <span class='citation-score'>Relevance {score_pct}%</span>
                        </div>
                        <div style='font-size:0.75rem;color:#8b949e;margin-bottom:6px'>
                            📄 Page {ctx.get("page_number","?")}{section_str}{chapter_str}
                        </div>
                        <div style='font-size:0.82rem;color:#c9d1d9;
                                    border-top:1px solid #30363d;padding-top:6px;
                                    max-height:120px;overflow-y:auto'>
                            {ctx.get("text","")[:500]}…
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Export response PDF button
        try:
            from utils.export import export_response_to_pdf  # optional util
            pdf_bytes = export_response_to_pdf(prompt, full_response, citations)
            st.download_button(
                "📄 Export as PDF",
                data=pdf_bytes,
                file_name=f"lawai_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
            )
        except Exception:
            pass  # PDF export is optional; skip if fpdf not installed

    # Store assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response, "citations": citations}
    )
