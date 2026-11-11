"""
Tests for End-to-End RAG Pipeline, Ingestion, Citations, and Conversation Database.
"""

from pathlib import Path
import pytest
from core.rag_pipeline import RAGPipeline
from core.vector_store import ChromaVectorStore
from database.metadata import MetadataDatabase
from database.conversation import ConversationDatabase
from config.settings import settings


@pytest.fixture
def test_pipeline(tmp_path):
    chroma_path = tmp_path / "chroma"
    db_path = tmp_path / "test.db"

    meta_db = MetadataDatabase(db_path=db_path)
    conv_db = ConversationDatabase(db_path=db_path)
    vector_store = ChromaVectorStore(persist_path=chroma_path)

    pipeline = RAGPipeline(
        vector_store=vector_store,
        meta_db=meta_db,
        conv_db=conv_db,
    )
    return pipeline


def test_txt_ingestion_and_query(test_pipeline, tmp_path):
    # Create sample cinema doc
    sample_txt = tmp_path / "tarantino_notes.txt"
    sample_txt.write_text(
        "Quentin Tarantino wrote True Romance and Natural Born Killers before Reservoir Dogs was filmed.\n\n"
        "He drew heavy inspiration from French New Wave director Jean-Luc Godard, naming his production "
        "company A Band Apart after Godard's film Bande a Part.\n\n"
        "In Reservoir Dogs, the iconic ear-slicing scene is accompanied by Stealers Wheel's 'Stuck in the Middle with You'.",
        encoding="utf-8",
    )

    # Ingest
    record = test_pipeline.ingest_document(sample_txt)
    assert record.status == "indexed"
    assert record.total_chunks > 0

    # Query
    session_id = test_pipeline.conv_db.create_session("Godard Influence")
    response = test_pipeline.query(
        "Which French New Wave director influenced Tarantino's production company name?",
        session_id=session_id,
    )

    assert response.confidence in ["High", "Medium"]
    assert len(response.citations) > 0
    assert response.citations[0].source == "tarantino_notes.txt"
    assert "Godard" in response.citations[0].excerpt

    # Check conversation database
    messages = test_pipeline.conv_db.get_session_messages(session_id)
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"
    assert len(messages[1].sources) > 0


def test_duplicate_prevention(test_pipeline, tmp_path):
    sample_txt = tmp_path / "duplicate_check.txt"
    sample_txt.write_text("Unique text content for duplicate detection test.", encoding="utf-8")

    rec1 = test_pipeline.ingest_document(sample_txt)
    # Re-ingest without force
    rec2 = test_pipeline.ingest_document(sample_txt, force_reindex=False)

    assert rec1.doc_id == rec2.doc_id
    assert rec1.file_hash == rec2.file_hash
