"""
CineMind AI - Document Metadata Database (SQLite)
Tracks uploaded documents, file hashes for duplicate detection, indexing status, and chunk counts.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Optional

from config.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


@dataclass
class DocumentRecord:
    doc_id: str
    filename: str
    file_path: str
    file_hash: str
    file_type: str
    file_size_bytes: int
    total_pages: int
    total_chunks: int
    title: str
    author: str
    status: str  # 'indexed', 'processing', 'error', 'pending'
    indexed_at: str


class MetadataDatabase:
    """Manages SQLite storage for document records and indexing metadata."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.DATABASE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create documents table if it doesn't exist."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL UNIQUE,
                    file_type TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    total_pages INTEGER NOT NULL,
                    total_chunks INTEGER NOT NULL,
                    title TEXT,
                    author TEXT,
                    status TEXT NOT NULL,
                    indexed_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def get_document_by_hash(self, file_hash: str) -> Optional[DocumentRecord]:
        """Check for existing duplicate document by SHA-256 hash."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE file_hash = ?", (file_hash,))
            row = cursor.fetchone()
            if row:
                return DocumentRecord(**dict(row))
        return None

    def get_document_by_id(self, doc_id: str) -> Optional[DocumentRecord]:
        """Fetch document record by doc_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
            row = cursor.fetchone()
            if row:
                return DocumentRecord(**dict(row))
        return None

    def list_documents(self) -> list[DocumentRecord]:
        """List all tracked documents ordered by indexing time."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents ORDER BY indexed_at DESC")
            rows = cursor.fetchall()
            return [DocumentRecord(**dict(r)) for r in rows]

    def add_or_update_document(self, doc: DocumentRecord) -> None:
        """Insert or update a document record."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents (
                    doc_id, filename, file_path, file_hash, file_type,
                    file_size_bytes, total_pages, total_chunks,
                    title, author, status, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.doc_id,
                    doc.filename,
                    doc.file_path,
                    doc.file_hash,
                    doc.file_type,
                    doc.file_size_bytes,
                    doc.total_pages,
                    doc.total_chunks,
                    doc.title,
                    doc.author,
                    doc.status,
                    doc.indexed_at,
                ),
            )
            conn.commit()
            logger.info(f"Metadata recorded for document: {doc.filename} (ID: {doc.doc_id})")

    def update_status(self, doc_id: str, status: str, total_chunks: Optional[int] = None) -> None:
        """Update the status and optionally chunk count of a document."""
        with self._get_connection() as conn:
            if total_chunks is not None:
                conn.execute(
                    "UPDATE documents SET status = ?, total_chunks = ? WHERE doc_id = ?",
                    (status, total_chunks, doc_id),
                )
            else:
                conn.execute(
                    "UPDATE documents SET status = ? WHERE doc_id = ?",
                    (status, doc_id),
                )
            conn.commit()

    def delete_document(self, doc_id: str) -> bool:
        """Delete document record from database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            if deleted:
                logger.info(f"Deleted metadata for doc_id: {doc_id}")
            return deleted
