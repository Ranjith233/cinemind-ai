"""
Tests for PDF Loading, Text Cleaning, Chapter Detection, and Text Splitting.
"""

from pathlib import Path
import pytest
from core.pdf_loader import PDFLoader
from core.document_loader import DocumentLoader
from core.text_splitter import TokenizerAwareTextSplitter
from config.settings import settings


@pytest.fixture
def sample_pdf_path():
    path = settings.SOURCE_DIR / "Quentin-Tarantino.pdf"
    if not path.exists():
        pytest.skip(f"Quentin-Tarantino.pdf not found at {path}")
    return path


def test_pdf_loader_metadata(sample_pdf_path):
    loader = PDFLoader()
    doc = loader.load(sample_pdf_path)

    assert doc.total_pages == 277
    assert doc.filename == "Quentin-Tarantino.pdf"
    assert doc.doc_id is not None
    assert len(doc.doc_id) == 16
    assert len(doc.pages) == 277
    assert doc.file_type == "pdf"


def test_chapter_mapping(sample_pdf_path):
    loader = PDFLoader()
    doc = loader.load(sample_pdf_path)

    # Check known page chapter mappings from TOC
    # Page 12 is Video Archives
    page_12 = doc.pages[11]
    assert "Video Archives" in page_12.chapter

    # Page 43 is Reservoir Dogs
    page_43 = doc.pages[42]
    assert "Reservoir Dogs" in page_43.chapter

    # Page 101 is Pulp Fiction
    page_101 = doc.pages[100]
    assert "Pulp Fiction" in page_101.chapter


def test_text_cleaning():
    raw_text = "This is a demon-\nstration of dehyphen-\nation and   multiple    spaces.\n\n\n\nParagraph 2."
    cleaned = PDFLoader.clean_text(raw_text)

    assert "demonstration" in cleaned
    assert "dehyphenation" in cleaned
    assert "multiple spaces" in cleaned
    assert "\n\n\n" not in cleaned


def test_text_splitter(sample_pdf_path):
    loader = PDFLoader()
    doc = loader.load(sample_pdf_path)

    # Test splitting first 10 pages
    sample_doc = doc
    sample_doc.pages = doc.pages[:10]

    splitter = TokenizerAwareTextSplitter(chunk_size=400, chunk_overlap=80)
    chunks = splitter.split_document(sample_doc)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.chunk_id != ""
        assert chunk.token_count > 0
        assert chunk.token_count <= 500  # allowing minor variance for sentence preservation
        assert "page" in chunk.metadata
        assert "chapter" in chunk.metadata
        assert chunk.source_filename == "Quentin-Tarantino.pdf"
