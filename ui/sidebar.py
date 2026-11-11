"""
CineMind AI - Premium Sidebar Component
Houses branding, session management, file uploads, model parameters, and system status.
"""

from pathlib import Path
import streamlit as st

from config.settings import settings
from core.rag_pipeline import RAGPipeline


def render_sidebar(pipeline: RAGPipeline) -> dict:
    """Renders the cinematic sidebar and returns user configuration."""
    with st.sidebar:
        # Branding
        st.markdown(
            """
            <div style="text-align: center; padding: 12px 0 20px 0;">
                <div style="font-size: 38px; line-height: 1;">🎞️</div>
                <div style="font-family: 'Cinzel', serif; font-size: 20px; font-weight: 700; color: #fde68a; letter-spacing: 0.1em; margin-top: 6px;">
                    CINEMIND AI
                </div>
                <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.15em;">
                    Cinema RAG Studio
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # Session Controls
        col_new, col_clear = st.columns([1, 1])
        with col_new:
            if st.button("➕ New Chat", use_container_width=True, type="primary"):
                new_id = pipeline.conv_db.create_session("New Cinema Inquiry")
                st.session_state["active_session_id"] = new_id
                st.rerun()

        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                active_id = st.session_state.get("active_session_id")
                if active_id:
                    pipeline.conv_db.clear_session(active_id)
                    st.rerun()

        # Conversation History Selector
        sessions = pipeline.conv_db.list_sessions()
        if sessions:
            st.markdown("##### 💬 Recent Conversations")
            session_options = {s.session_id: f"{s.title[:24]} ({s.updated_at[11:16]})" for s in sessions}

            curr_id = st.session_state.get("active_session_id")
            if not curr_id or curr_id not in session_options:
                curr_id = sessions[0].session_id
                st.session_state["active_session_id"] = curr_id

            selected_id = st.selectbox(
                "Select Session",
                options=list(session_options.keys()),
                format_func=lambda x: session_options[x],
                index=list(session_options.keys()).index(curr_id),
                label_visibility="collapsed",
            )
            if selected_id != curr_id:
                st.session_state["active_session_id"] = selected_id
                st.rerun()

        st.divider()

        # Document Upload
        st.markdown("##### 📁 Document Ingestion")
        uploaded_file = st.file_uploader(
            "Upload Cinema Documents",
            type=["pdf", "docx", "txt", "csv"],
            help="Upload film analysis, scripts, research papers, or books",
        )

        if uploaded_file is not None:
            upload_key = f"uploaded_{uploaded_file.name}_{uploaded_file.size}"
            if upload_key not in st.session_state:
                save_dest = settings.UPLOADS_DIR / uploaded_file.name
                with open(save_dest, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner(f"Ingesting & indexing {uploaded_file.name}..."):
                    record = pipeline.ingest_document(save_dest, force_reindex=False)
                    st.session_state[upload_key] = True
                    st.success(f"Indexed {record.filename}: {record.total_chunks} chunks!")
                    st.rerun()

        st.divider()

        # Model & API Configuration
        st.markdown("##### ⚙️ LLM Configuration")
        configured_key = settings.GEMINI_API_KEY
        user_api_key = st.text_input(
            "Gemini API Key",
            value=st.session_state.get("user_gemini_key", configured_key),
            type="password",
            placeholder="AIzaSy...",
            help="Enter your Google Gemini API key or set GEMINI_API_KEY in .env",
        )
        if user_api_key:
            st.session_state["user_gemini_key"] = user_api_key
            pipeline.llm.set_api_key(user_api_key)

        gemini_model = st.selectbox(
            "Gemini Model",
            options=settings.SUPPORTED_GEMINI_MODELS,
            index=0,
            help="Select generative model for synthesis",
        )
        pipeline.llm.set_model(gemini_model)

        st.divider()

        # Retrieval Settings
        st.markdown("##### 🔍 Retrieval Hyperparameters")
        top_k = st.slider("Initial Candidates (RRF Top-K)", min_value=4, max_value=25, value=settings.TOP_K, step=1)
        final_top_k = st.slider("Cross-Encoder Final Chunks", min_value=2, max_value=8, value=settings.FINAL_TOP_K, step=1)

        st.divider()

        # System Status
        st.markdown("##### 📊 System Status")
        chroma_count = pipeline.vector_store.count()
        doc_count = len(pipeline.meta_db.list_documents())

        llm_ready = "🟢 Ready" if pipeline.llm.is_configured else "🟡 Key Required"

        st.markdown(
            f"""
            <div style="font-size: 12px; color: #94a3b8; background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px;">
                <div><b>LLM Status:</b> {llm_ready}</div>
                <div><b>Indexed Docs:</b> {doc_count} documents</div>
                <div><b>Total Chunks:</b> {chroma_count} vectors</div>
                <div><b>Embedding:</b> all-MiniLM-L6-v2</div>
                <div><b>Reranker:</b> ms-marco-MiniLM-L-6-v2</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        return {
            "top_k": top_k,
            "final_top_k": final_top_k,
            "gemini_model": gemini_model,
        }
