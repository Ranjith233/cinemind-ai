"""
CineMind AI - Sources & Citations UI Component
Renders expandable, interactive source citations with page numbers, chapters, and excerpts.
"""

from typing import Any
import streamlit as st
from core.rag_pipeline import Citation


def render_sources_expander(
    citations: list[Citation] | list[dict[str, Any]],
    confidence: str = "High",
    latency_ms: float = 0.0,
) -> None:
    """Renders a collapsible cinema-style source verification drawer."""
    if not citations:
        return

    confidence_class = {
        "High": "confidence-high",
        "Medium": "confidence-medium",
        "Low": "confidence-low",
    }.get(confidence, "confidence-medium")

    header_label = f"Verified Source Citations ({len(citations)} references | Confidence: {confidence} | {latency_ms:.0f}ms)"

    with st.expander(header_label, expanded=False):
        for idx, cite in enumerate(citations, start=1):
            if isinstance(cite, dict):
                src = cite.get("source", "Document")
                page = cite.get("page", 1)
                chapter = cite.get("chapter", "General")
                excerpt = cite.get("excerpt", "")
                score = cite.get("score", 0.0)
            else:
                src = cite.source
                page = cite.page
                chapter = cite.chapter
                excerpt = cite.excerpt
                score = cite.score

            score_str = f"Score: {score:.2f}" if score is not None else ""

            st.markdown(
                f"""
                <div class="cinema-card" style="margin-bottom: 12px; padding: 14px 18px;">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                        <div>
                            <span class="citation-badge">📖 {src}</span>
                            <span class="citation-badge">📄 Page {page}</span>
                            <span class="chapter-badge">🎬 {chapter}</span>
                        </div>
                        <span style="font-size: 11px; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">
                            {score_str}
                        </span>
                    </div>
                    <div style="font-size: 13px; line-height: 1.6; color: #cbd5e1; font-style: italic; background: rgba(0,0,0,0.25); padding: 10px 14px; border-radius: 8px; border-left: 3px solid #d4af37;">
                        "{excerpt}"
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
