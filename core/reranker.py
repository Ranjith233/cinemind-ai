"""
CineMind AI - Cross-Encoder Reranker
Reranks retrieved candidate chunks using deep cross-attention for high precision relevance.
"""

from typing import Optional
from sentence_transformers import CrossEncoder

from config.logger import get_logger
from config.settings import settings
from core.retriever import RetrievedChunk

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Reranks candidate chunks using a trained Cross-Encoder model."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        logger.info(f"Loading CrossEncoder reranker model: {self.model_name}")
        self.model = CrossEncoder(self.model_name)
        logger.info("CrossEncoder reranker model loaded successfully.")

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        final_top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """
        Reranks a list of candidate chunks against the query.
        Returns top `final_top_k` chunks ordered by cross-attention relevance score.
        """
        if not candidates:
            return []

        top_k = final_top_k or settings.FINAL_TOP_K

        # Prepare (query, text) pairs
        pairs = [(query, chunk.text) for chunk in candidates]

        # Compute scores
        scores = self.model.predict(pairs)

        # Attach scores to chunks
        for idx, chunk in enumerate(candidates):
            score_val = float(scores[idx])
            chunk.rerank_score = score_val
            # Final ranking score is dominated by cross-encoder
            chunk.score = score_val

        # Sort descending by rerank score
        reranked = sorted(candidates, key=lambda c: c.rerank_score or -999.0, reverse=True)

        selected = reranked[:top_k]

        if selected:
            top_score = selected[0].rerank_score or 0.0
            logger.info(
                f"Reranked {len(candidates)} candidates -> selected top {len(selected)} "
                f"(highest score={top_score:.3f})"
            )

        return selected

    @staticmethod
    def evaluate_confidence(chunks: list[RetrievedChunk]) -> tuple[str, float]:
        """
        Evaluate retrieval confidence based on top rerank score.
        Returns (confidence_level: 'High'|'Medium'|'Low', max_score).
        """
        if not chunks or chunks[0].rerank_score is None:
            return "Low", -999.0

        top_score = chunks[0].rerank_score
        if top_score >= 3.0:
            return "High", top_score
        elif top_score >= -1.0:
            return "Medium", top_score
        else:
            return "Low", top_score
