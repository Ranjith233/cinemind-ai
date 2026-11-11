"""
Tests for BM25 Keyword Search, Hybrid Retrieval (RRF), and Cross-Encoder Reranking.
"""

from pathlib import Path
import pytest
from core.embeddings import LocalSentenceTransformerEmbeddings
from core.vector_store import ChromaVectorStore
from core.keyword_search import BM25KeywordSearch
from core.retriever import HybridRetriever
from core.reranker import CrossEncoderReranker
from core.text_splitter import DocumentChunk


@pytest.fixture
def populated_components(tmp_path):
    chroma_path = tmp_path / "test_chroma"
    store = ChromaVectorStore(persist_path=chroma_path)
    embeddings = LocalSentenceTransformerEmbeddings()
    bm25 = BM25KeywordSearch()

    chunks = [
        DocumentChunk(
            chunk_id="tarantino_p12",
            doc_id="qt1",
            source_filename="Quentin-Tarantino.pdf",
            chapter="Video Archives",
            page_start=12,
            page_end=12,
            text="Quentin worked at Video Archives in Manhattan Beach with Roger Avary, watching countless exploitation and arthouse films.",
            token_count=20,
            metadata={"source": "Quentin-Tarantino.pdf", "page": 12, "chapter": "Video Archives", "doc_id": "qt1"},
        ),
        DocumentChunk(
            chunk_id="tarantino_p43",
            doc_id="qt1",
            source_filename="Quentin-Tarantino.pdf",
            chapter="Reservoir Dogs",
            page_start=43,
            page_end=43,
            text="Lawrence Bender produced Reservoir Dogs, which established Quentin's reputation at the Sundance Film Festival.",
            token_count=19,
            metadata={"source": "Quentin-Tarantino.pdf", "page": 43, "chapter": "Reservoir Dogs", "doc_id": "qt1"},
        ),
        DocumentChunk(
            chunk_id="tarantino_p101",
            doc_id="qt1",
            source_filename="Quentin-Tarantino.pdf",
            chapter="Pulp Fiction",
            page_start=101,
            page_end=101,
            text="Pulp Fiction features a circular, episodic narrative linking Vincent Vega, Jules Winnfield, and boxer Butch Coolidge.",
            token_count=21,
            metadata={"source": "Quentin-Tarantino.pdf", "page": 101, "chapter": "Pulp Fiction", "doc_id": "qt1"},
        ),
    ]

    embs = embeddings.embed_texts([c.text for c in chunks])
    store.add_chunks(chunks, embs)

    retriever = HybridRetriever(vector_store=store, embeddings=embeddings, keyword_search=bm25)
    retriever.initialize_keyword_index()

    reranker = CrossEncoderReranker()

    return retriever, reranker


def test_bm25_search(populated_components):
    retriever, _ = populated_components
    bm25 = retriever.keyword_search

    results = bm25.search("Lawrence Bender Sundance", top_k=2)
    assert len(results) > 0
    assert results[0]["chunk_id"] == "tarantino_p43"
    assert "Lawrence Bender" in results[0]["text"]


def test_hybrid_retrieval(populated_components):
    retriever, _ = populated_components
    results = retriever.retrieve("Where did Quentin work with Roger Avary?", top_k=3)

    assert len(results) == 3
    # Top result should be the Video Archives chunk
    assert results[0].chunk_id == "tarantino_p12"
    assert results[0].rrf_score > 0
    assert results[0].metadata["chapter"] == "Video Archives"
    assert results[0].metadata["page"] == 12


def test_cross_encoder_reranking(populated_components):
    retriever, reranker = populated_components
    query = "Tell me about Vincent Vega and Jules Winnfield in Pulp Fiction."

    candidates = retriever.retrieve(query, top_k=3)
    reranked = reranker.rerank(query, candidates, final_top_k=2)

    assert len(reranked) == 2
    # The Pulp Fiction chunk should be ranked #1 with high score
    assert reranked[0].chunk_id == "tarantino_p101"
    assert reranked[0].rerank_score is not None
    assert reranked[0].rerank_score > 0

    confidence, score = reranker.evaluate_confidence(reranked)
    assert confidence in ["High", "Medium"]
