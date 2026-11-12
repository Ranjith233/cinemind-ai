"""
CineMind AI - Advanced Cinema Knowledge Assistant
Production-grade RAG application with hybrid retrieval, cross-encoder reranking,
page citations, and cinema research intelligence.
"""

import sys
from pathlib import Path
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="CineMind AI – Cinema Knowledge Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Environment & Dependency Validation
# ---------------------------------------------------------
missing_deps = []

# Check PyMuPDF
try:
    import pymupdf
except ImportError:
    try:
        import fitz
    except ImportError:
        missing_deps.append("pymupdf")

# Check Sentence Transformers
try:
    import sentence_transformers
except ImportError:
    missing_deps.append("sentence-transformers")

# Check ChromaDB
try:
    import chromadb
except ImportError:
    missing_deps.append("chromadb")

# Check Google GenAI SDK
try:
    from google import genai
except ImportError:
    missing_deps.append("google-genai")

if missing_deps:
    st.error("### ⚠️ Virtual Environment Required")
    st.markdown(
        f"""
        CineMind AI was launched using the system Python interpreter:
        `{sys.executable}`

        The following required packages are not installed in this interpreter:
        `{", ".join(missing_deps)}`

        ---
        ### 🚀 How to Launch CineMind AI Correctly:

        All dependencies are pre-installed in the dedicated project virtual environment (`.venv`).

        **Choose any of the following options:**

        1. **One-Click Batch Launcher (Recommended for Windows):**
           Double-click **`run_app.bat`** in the project folder.

        2. **PowerShell Script:**
           ```powershell
           .\\run_app.ps1
           ```

        3. **Direct Virtual Environment Streamlit:**
           ```powershell
           .\\.venv\\Scripts\\streamlit.exe run app.py
           ```

        4. **Activate Virtual Environment first:**
           ```powershell
           .\\.venv\\Scripts\\Activate.ps1
           streamlit run app.py
           ```
        """
    )
    st.stop()

from config.settings import settings
from config.logger import get_logger
from core.rag_pipeline import RAGPipeline
from ui.components import apply_cinematic_styles, render_brand_header
from ui.sidebar import render_sidebar
from ui.chat import render_chat_interface
from ui.library import render_document_library

logger = get_logger("App")


@st.cache_resource(show_spinner="Initializing CineMind AI Studio...")
def load_rag_pipeline() -> RAGPipeline:
    """Initialize persistent RAG pipeline singleton."""
    return RAGPipeline()


def main():
    # Apply cinematic dark theme
    apply_cinematic_styles()

    # Load RAG pipeline
    pipeline = load_rag_pipeline()

    # Render Sidebar
    config = render_sidebar(pipeline)

    # Render Header Banner
    render_brand_header(
        title="CINEMIND AI",
        subtitle="Advanced Cinema Intelligence, Screenplay Analysis & Film Research Assistant",
    )

    # Tabs
    tab_chat, tab_library, tab_arch = st.tabs(
        [
            "💬 Cinema Assistant",
            "📚 Document Library",
            "🔬 Architecture & Diagnostics",
        ]
    )

    with tab_chat:
        render_chat_interface(pipeline, config)

    with tab_library:
        render_document_library(pipeline)

    with tab_arch:
        st.markdown("### 🔬 Advanced RAG Architecture Pipeline")
        st.caption("Visual breakdown of CineMind AI's multi-stage retrieval and generation system.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                <div class="cinema-card">
                    <h4 style="color: #fde68a; margin-top: 0;">1. Ingestion & Document Processing</h4>
                    <ul style="font-size: 13px; color: #cbd5e1; line-height: 1.8;">
                        <li><b>Format Parsing:</b> PyMuPDF for PDF with TOC and page metadata; DOCX, TXT, CSV loaders.</li>
                        <li><b>Text Sanitization:</b> Linebreak dehyphenation, whitespace normalization, control character stripping.</li>
                        <li><b>Chapter Detection:</b> Hierarchical table of contents mapping and regex heading detection.</li>
                        <li><b>Tokenizer Splitter:</b> Tokenizer-aware sliding window preserving paragraph and sentence boundaries.</li>
                        <li><b>Vector Storage:</b> Persistent ChromaDB collection with cosine distance indexing.</li>
                        <li><b>Duplicate Guard:</b> SHA-256 cryptographic document fingerprinting.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
                <div class="cinema-card">
                    <h4 style="color: #fde68a; margin-top: 0;">2. Hybrid Retrieval & Reranking</h4>
                    <ul style="font-size: 13px; color: #cbd5e1; line-height: 1.8;">
                        <li><b>Query Reformulation:</b> Disambiguates follow-up pronouns using recent dialogue turns.</li>
                        <li><b>Dense Semantic Search:</b> 384-dimensional SentenceTransformer embeddings.</li>
                        <li><b>Sparse Lexical Search:</b> In-memory BM25Okapi over full vocabulary.</li>
                        <li><b>Rank Fusion:</b> Reciprocal Rank Fusion (RRF) with constant k=60.</li>
                        <li><b>Deep Cross-Encoder:</b> ms-marco-MiniLM-L-6-v2 cross-attention score evaluation.</li>
                        <li><b>Grounded Synthesis:</b> Google Gemini API inference with page citations.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Diagnostics & Metrics
        st.markdown("#### 📈 Real-Time Engine Metrics")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Vector Database", "ChromaDB Persistent")
        with m2:
            st.metric("Total Stored Vectors", pipeline.vector_store.count())
        with m3:
            st.metric("BM25 Corpus Size", pipeline.keyword_search.count())
        with m4:
            st.metric("Active LLM Model", pipeline.llm.model_name)

        # Recent Logs Viewer
        st.markdown("#### 📜 Application Execution Logs")
        log_file = settings.LOGS_DIR / "cinemind.log"
        if log_file.exists():
            log_lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            tail_lines = "\n".join(log_lines[-25:])
            st.code(tail_lines, language="log")
        else:
            st.info("No logs generated yet.")


if __name__ == "__main__":
    main()
