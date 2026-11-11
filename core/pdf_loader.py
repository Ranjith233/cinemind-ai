"""
CineMind AI - PDF Document Loader
Extracts clean text, page numbers, chapters, and metadata using PyMuPDF.
"""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
from typing import Any, Optional
import pymupdf  # PyMuPDF

from config.logger import get_logger
from core.chapter_detector import ChapterDetector

logger = get_logger(__name__)


@dataclass
class DocumentPage:
    """Represents a single extracted page of a document."""
    doc_id: str
    source_filename: str
    page_number: int
    chapter: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedDocument:
    """Represents a fully extracted document."""
    doc_id: str
    filename: str
    file_path: str
    file_hash: str
    file_type: str
    total_pages: int
    pages: list[DocumentPage]
    title: Optional[str] = None
    author: Optional[str] = None
    file_size_bytes: int = 0


class PDFLoader:
    """Loads and cleans PDF documents with chapter mapping and metadata preservation."""

    def __init__(self):
        self.chapter_detector = ChapterDetector()

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """Compute SHA-256 hash of a file for duplicate detection."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans extracted text:
        - Reconnects hyphenated words across line breaks (e.g. 'struc-\nture' -> 'structure')
        - Normalizes multiple spaces and tabs to a single space
        - Preserves paragraph breaks (double newlines) while collapsing 3+ newlines
        - Strips control characters
        """
        if not text:
            return ""

        # Remove control characters except newline and tab
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

        # Dehyphenate across linebreaks
        text = re.sub(r"(\b\w+)-\n(\w+\b)", r"\1\2", text)

        # Replace carriage returns
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse excessive newlines (3 or more -> 2)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Replace excessive horizontal whitespace within a line
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text.strip()

    def load(self, file_path: str | Path) -> ExtractedDocument:
        """Extracts text and metadata from a PDF file."""
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_hash = self.compute_sha256(path)
        doc_id = file_hash[:16]
        file_size = path.stat().st_size

        logger.info(f"Loading PDF: {path.name} ({file_size / (1024*1024):.2f} MB, hash={doc_id})")

        doc = pymupdf.open(str(path))
        total_pages = len(doc)
        toc = doc.get_toc()

        # Build page-to-chapter mapping
        page_chapter_map = self.chapter_detector.build_page_to_chapter_map(toc, total_pages)

        # Extract PDF metadata
        pdf_meta = doc.metadata or {}
        doc_title = pdf_meta.get("title") or path.stem.replace("-", " ").replace("_", " ").title()
        doc_author = pdf_meta.get("author") or "Unknown"

        pages: list[DocumentPage] = []

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = doc[page_idx]

            raw_text = page.get_text("text")
            cleaned_text = self.clean_text(raw_text)

            # Determine chapter
            chapter = page_chapter_map.get(page_num, "")
            if not chapter:
                # Fallback to text heading detection
                detected = self.chapter_detector.detect_heading_in_text(cleaned_text)
                chapter = detected if detected else "General"

            page_meta = {
                "source": path.name,
                "file_path": str(path),
                "page": page_num,
                "total_pages": total_pages,
                "chapter": chapter,
                "doc_title": doc_title,
                "doc_author": doc_author,
                "doc_id": doc_id,
            }

            pages.append(
                DocumentPage(
                    doc_id=doc_id,
                    source_filename=path.name,
                    page_number=page_num,
                    chapter=chapter,
                    text=cleaned_text,
                    metadata=page_meta,
                )
            )

        doc.close()

        extracted = ExtractedDocument(
            doc_id=doc_id,
            filename=path.name,
            file_path=str(path),
            file_hash=file_hash,
            file_type="pdf",
            total_pages=total_pages,
            pages=pages,
            title=doc_title,
            author=doc_author,
            file_size_bytes=file_size,
        )

        logger.info(f"Successfully extracted {total_pages} pages from {path.name}")
        return extracted
