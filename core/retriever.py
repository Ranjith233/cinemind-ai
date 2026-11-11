"""
CineMind AI - Hybrid Retriever with Reciprocal Rank Fusion (RRF)
Combines dense semantic vector retrieval and sparse lexical BM25 retrieval.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from config.logger import get_logger
from config.settings import settings
from core.embeddings import BaseEmbeddings
from core.keyword_search import BM25KeywordSearch
from core.vector_store import ChromaVectorStore

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """Represents a chunk retrieved by the search system."""
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float  # Final fused score or rerank score
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None
    source_rank: int = 0


class HybridRetriever:
    """
    Executes hybrid retrieval by running vector search and BM25 in parallel,
    then fusing ranks via Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embeddings: BaseEmbeddings,
        keyword_search: Optional[BM25KeywordSearch] = None,
        rrf_k: Optional[int] = None,
    ):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.keyword_search = keyword_search or BM25KeywordSearch()
        self.rrf_k = rrf_k or settings.RRF_K

    def initialize_keyword_index(self) -> None:
        """Load all chunks from vector store into the BM25 index."""
        chunks = self.vector_store.get_all_chunks()
        self.keyword_search.build_index(chunks)
        logger.info(f"Initialized HybridRetriever BM25 index with {len(chunks)} chunks.")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[RetrievedChunk]:
        """
        Execute Hybrid Retrieval:
        1. Dense semantic search
        2. Sparse BM25 search
        3. Reciprocal Rank Fusion (RRF)
        """
        k = top_k or settings.TOP_K

        # If BM25 index is not initialized or empty, try refreshing
        if self.keyword_search.count() == 0 and self.vector_store.count() > 0:
            self.initialize_keyword_index()

        # 1. Semantic Search
        query_emb = self.embeddings.embed_query(query)
        vector_results = self.vector_store.query(
            query_embedding=query_emb,
            top_k=k * 2,
            filters=filters,
        )

        # 2. BM25 Keyword Search
        bm25_results = self.keyword_search.search(
            query=query,
            top_k=k * 2,
            filters=filters,
        )

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, dict[str, Any]] = {}
        vector_scores_map: dict[str, float] = {}
        bm25_scores_map: dict[str, float] = {}

        # Process vector ranks
        for rank, res in enumerate(vector_results, start=1):
            cid = res["chunk_id"]
            chunk_map[cid] = res
            vector_scores_map[cid] = res["score"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        # Process BM25 ranks
        for rank, res in enumerate(bm25_results, start=1):
            cid = res["chunk_id"]
            chunk_map[cid] = res
            bm25_scores_map[cid] = res["score"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        # Sort all candidates by RRF score descending
        sorted_chunk_ids = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )

        final_chunks: list[RetrievedChunk] = []
        for rank_idx, cid in enumerate(sorted_chunk_ids[:k], start=1):
            raw = chunk_map[cid]
            final_chunks.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=raw["text"],
                    metadata=raw.get("metadata", {}),
                    score=rrf_scores[cid],
                    vector_score=vector_scores_map.get(cid, 0.0),
                    bm25_score=bm25_scores_map.get(cid, 0.0),
                    rrf_score=rrf_scores[cid],
                    source_rank=rank_idx,
                )
            )

        logger.info(
            f"Hybrid retrieval for '{query}' returned {len(final_chunks)} candidates "
            f"(vector={len(vector_results)}, bm25={len(bm25_results)})"
        )
        return final_chunks
