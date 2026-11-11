"""
CineMind AI - Interactive Chat Interface
Renders conversation flow, suggested cinema prompts, streaming responses, and citations.
"""

import streamlit as st
from config.logger import get_logger
from core.rag_pipeline import RAGPipeline
from ui.sources import render_sources_expander

logger = get_logger(__name__)

SUGGESTED_QUESTIONS = [
    "How did Tarantino develop his interest in cinema?",
    "What role did Video Archives play in his filmmaking journey?",
    "Which filmmakers and genres influenced his directing style?",
    "How does Reservoir Dogs use nonlinear storytelling?",
    "What makes Pulp Fiction's narrative structure unconventional?",
    "What inspired Tarantino's approach to Western films and Django Unchained?",
]


def render_chat_interface(pipeline: RAGPipeline, config: dict) -> None:
    """Renders the main cinema chat workspace."""
    session_id = st.session_state.get("active_session_id")
    if not session_id:
        session_id = pipeline.conv_db.create_session("New Cinema Inquiry")
        st.session_state["active_session_id"] = session_id

    # Retrieve existing messages
    messages = pipeline.conv_db.get_session_messages(session_id)

    # Empty state: Welcome screen with curated questions
    if not messages:
        st.markdown(
            """
            <div class="cinema-card" style="text-align: center; padding: 32px 24px; margin-bottom: 28px;">
                <div style="font-size: 42px; margin-bottom: 12px;">📽️</div>
                <h2 style="font-family: 'Cinzel', serif; font-size: 24px; color: #fde68a; margin-bottom: 8px;">
                    Welcome to CineMind AI
                </h2>
                <p style="color: #94a3b8; font-size: 14px; max-width: 620px; margin: 0 auto 20px auto; line-height: 1.6;">
                    Explore deep cinema scholarship, narrative architecture, and directorial craft grounded
                    in the book <i>Quentin Tarantino</i> by Ian Nathan, complete with page citations and hybrid retrieval.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### 💡 Explore In-Depth Film Inquiries")
        cols = st.columns(2)
        for idx, q_text in enumerate(SUGGESTED_QUESTIONS):
            col = cols[idx % 2]
            with col:
                if st.button(f"🎬 {q_text}", key=f"sug_{idx}", use_container_width=True):
                    st.session_state["pending_prompt"] = q_text
                    st.rerun()

    # Render conversation history
    for msg in messages:
        if msg.role == "user":
            with st.chat_message("user", avatar="🎬"):
                st.markdown(msg.content)
        else:
            with st.chat_message("assistant", avatar="🎞️"):
                st.markdown(msg.content)
                if msg.sources:
                    render_sources_expander(
                        citations=msg.sources,
                        confidence=msg.confidence,
                        latency_ms=msg.latency_ms,
                    )

    # Check for pending prompt from suggestion buttons or chat input
    pending = st.session_state.pop("pending_prompt", None)
    user_input = st.chat_input("Ask about Tarantino's movies, screenwriting, influences, or craft...")

    prompt_to_process = pending or user_input

    if prompt_to_process:
        # 1. Render User Message
        with st.chat_message("user", avatar="🎬"):
            st.markdown(prompt_to_process)

        # 2. Render Assistant Streaming Response
        with st.chat_message("assistant", avatar="🎞️"):
            response_placeholder = st.empty()

            with st.spinner("Analyzing cinema literature & reranking source passages..."):
                stream_gen, citations, confidence, latency_ms, rewritten_query = (
                    pipeline.query_stream(
                        user_question=prompt_to_process,
                        session_id=session_id,
                    )
                )

            # Stream response chunks
            full_response = ""
            for chunk in stream_gen:
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

            # Render expandable citations
            if citations:
                render_sources_expander(
                    citations=citations,
                    confidence=confidence,
                    latency_ms=latency_ms,
                )

            # Persist to database
            pipeline.conv_db.add_message(
                session_id=session_id,
                role="user",
                content=prompt_to_process,
            )
            pipeline.conv_db.add_message(
                session_id=session_id,
                role="assistant",
                content=full_response,
                sources=[
                    {
                        "source": c.source,
                        "page": c.page,
                        "chapter": c.chapter,
                        "excerpt": c.excerpt,
                        "score": round(c.score, 3),
                    }
                    for c in citations
                ],
                confidence=confidence,
                latency_ms=latency_ms,
            )

        st.rerun()
