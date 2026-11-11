"""
CineMind AI - Document Library & Management View
Displays indexed cinema documents, pagination stats, chunk counts, and management controls.
"""

import pandas as pd
import streamlit as st
from config.settings import settings
from core.rag_pipeline import RAGPipeline


def render_document_library(pipeline: RAGPipeline) -> None:
    """Renders document library dashboard and deletion/re-indexing controls."""
    st.markdown("### 📚 Cinema Document Library")
    st.caption("Manage persistent film research literature, monographs, and scripts.")

    docs = pipeline.meta_db.list_documents()

    if not docs:
        st.info("No documents are currently indexed. Upload files in the sidebar to begin.")
        return

    # Metrics Summary Row
    total_pages = sum(d.total_pages for d in docs)
    total_chunks = pipeline.vector_store.count()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Indexed Documents", len(docs))
    with col2:
        st.metric("Total Pages Analyzed", total_pages)
    with col3:
        st.metric("ChromaDB Vectors", total_chunks)
    with col4:
        st.metric("Storage Status", "Persistent SQLite & Chroma")

    st.markdown("---")

    # Document Table / Cards
    for doc in docs:
        with st.container():
            st.markdown(
                f"""
                <div class="cinema-card">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div>
                            <h4 style="margin: 0 0 6px 0; color: #ffffff; font-family: 'Cinzel', serif;">
                                🎬 {doc.title or doc.filename}
                            </h4>
                            <div style="font-size: 12px; color: #94a3b8;">
                                <span class="citation-badge">📁 {doc.filename}</span>
                                <span class="citation-badge">FORMAT: {doc.file_type.upper()}</span>
                                <span class="chapter-badge">PAGES: {doc.total_pages}</span>
                                <span class="chapter-badge">CHUNKS: {doc.total_chunks}</span>
                                <span style="margin-left: 8px;">Indexed on {doc.indexed_at[:10]}</span>
                            </div>
                        </div>
                        <div>
                            <span class="confidence-high">STATUS: {doc.status.upper()}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns([1, 1, 4])
            with c1:
                if st.button("🔄 Re-index", key=f"reindex_{doc.doc_id}", use_container_width=True):
                    with st.spinner(f"Re-indexing {doc.filename}..."):
                        pipeline.reindex_document(doc.doc_id)
                        st.success(f"Re-indexed {doc.filename} successfully!")
                        st.rerun()

            with c2:
                if st.button("🗑️ Delete", key=f"delete_{doc.doc_id}", use_container_width=True):
                    with st.spinner(f"Deleting {doc.filename}..."):
                        pipeline.delete_document(doc.doc_id)
                        st.warning(f"Deleted {doc.filename} from vector store.")
                        st.rerun()
