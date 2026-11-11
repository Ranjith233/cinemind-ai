"""
CineMind AI - Ingest Primary Knowledge Source
Extracts, chunks, embeds, and indexes Quentin-Tarantino.pdf into ChromaDB & SQLite.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import time
from config.settings import settings
from config.logger import get_logger
from core.rag_pipeline import RAGPipeline

logger = get_logger("IngestBook")

def main():
    pdf_path = settings.SOURCE_DIR / "Quentin-Tarantino.pdf"
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found!")
        return

    print(f"=== INGESTING {pdf_path.name} ===")
    start_time = time.time()

    pipeline = RAGPipeline()
    record = pipeline.ingest_document(pdf_path, force_reindex=False)

    elapsed = time.time() - start_time
    print(f"=== INGESTION COMPLETE IN {elapsed:.2f}s ===")
    print(f"Document ID: {record.doc_id}")
    print(f"Title: {record.title}")
    print(f"Total Pages: {record.total_pages}")
    print(f"Total Chunks: {record.total_chunks}")
    print(f"Status: {record.status}")
    print(f"ChromaDB Total Chunks: {pipeline.vector_store.count()}")

if __name__ == "__main__":
    main()
