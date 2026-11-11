"""
Tests for SentenceTransformer Embeddings and ChromaVectorStore.
"""

import shutil
from pathlib import Path
import pytest
from core.embeddings import LocalSentenceTransformerEmbeddings
from core.vector_store import ChromaVectorStore
from core.text_splitter import DocumentChunk


@pytest.fixture
def temp_chroma_path(tmp_path):
    return tmp_path / "test_chroma"


def test_embeddings_generation():
    emb_model = LocalSentenceTransformerEmbeddings()
    assert emb_model.dimension == 384

    texts = [
        "Quentin Tarantino worked at Video Archives in Manhattan Beach.",
        "Pulp Fiction won the Palme d'Or at Cannes Film Festival in 1994.",
    ]
    embs = emb_model.embed_texts(texts)
    assert len(embs) == 2
    assert len(embs[0]) == 384
    assert len(embs[1]) == 384

    query_emb = emb_model.embed_query("Where did Tarantino work?")
    assert len(query_emb) == 384


def test_chroma_vector_store(temp_chroma_path):
    store = ChromaVectorStore(persist_path=temp_chroma_path)
    emb_model = LocalSentenceTransformerEmbeddings()

    chunks = [
        DocumentChunk(
            chunk_id="doc1_c0",
            doc_id="doc1",
            source_filename="Tarantino.pdf",
            chapter="Video Archives",
            page_start=12,
            page_end=12,
            text="Quentin Tarantino spent formative years working as a clerk at Video Archives in Manhattan Beach.",
            token_count=18,
            metadata={"source": "Tarantino.pdf", "page": 12, "chapter": "Video Archives", "doc_id": "doc1"},
        ),
        DocumentChunk(
            chunk_id="doc1_c1",
            doc_id="doc1",
            source_filename="Tarantino.pdf",
            chapter="Reservoir Dogs",
            page_start=43,
            page_end=43,
            text="Reservoir Dogs introduced Tarantino's signature nonlinear crime narrative with Harvey Keitel.",
            token_count=16,
            metadata={"source": "Tarantino.pdf", "page": 43, "chapter": "Reservoir Dogs", "doc_id": "doc1"},
        ),
    ]

    embs = emb_model.embed_texts([c.text for c in chunks])
    store.add_chunks(chunks, embs)
    assert store.count() == 2

    # Query for Video Archives
    q_emb = emb_model.embed_query("Video Archives clerk Manhattan Beach")
    results = store.query(q_emb, top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "doc1_c0"
    assert "Video Archives" in results[0]["text"]
    assert results[0]["score"] > 0.4

    # Test delete
    store.delete_by_doc_id("doc1")
    assert store.count() == 0
