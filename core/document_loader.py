"""
CineMind AI - Unified Document Loader
Supports PDF, DOCX, TXT, and CSV formats with standardized metadata extraction.
"""

from pathlib import Path
from typing import Union
import pandas as pd
import docx2txt

from config.logger import get_logger
from core.pdf_loader import PDFLoader, ExtractedDocument, DocumentPage

logger = get_logger(__name__)


class DocumentLoader:
    """Unified document loader handling multiple document formats."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv"}

    def __init__(self):
        self.pdf_loader = PDFLoader()

    def load(self, file_path: Union[str, Path]) -> ExtractedDocument:
        """Route file loading to appropriate parser based on extension."""
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{ext}'. CineMind AI supports: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        if ext == ".pdf":
            return self.pdf_loader.load(path)
        elif ext == ".docx":
            return self._load_docx(path)
        elif ext == ".txt":
            return self._load_txt(path)
        elif ext == ".csv":
            return self._load_csv(path)
        else:
            raise ValueError(f"No handler configured for {ext}")

    def _load_docx(self, path: Path) -> ExtractedDocument:
        """Extract text from a DOCX document."""
        file_hash = PDFLoader.compute_sha256(path)
        doc_id = file_hash[:16]
        raw_text = docx2txt.process(str(path))
        cleaned = PDFLoader.clean_text(raw_text)

        # DOCX doesn't have native pagination; split into logical ~1500 char sections or paragraphs
        paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
        pages: list[DocumentPage] = []

        # Synthetic page grouping (~3 paragraphs per page)
        curr_page = 1
        buffer: list[str] = []
        for p in paragraphs:
            buffer.append(p)
            if len("\n\n".join(buffer)) >= 2000:
                page_text = "\n\n".join(buffer)
                pages.append(
                    DocumentPage(
                        doc_id=doc_id,
                        source_filename=path.name,
                        page_number=curr_page,
                        chapter="General",
                        text=page_text,
                        metadata={
                            "source": path.name,
                            "file_path": str(path),
                            "page": curr_page,
                            "chapter": "General",
                            "doc_title": path.stem.replace("-", " ").title(),
                            "doc_id": doc_id,
                        },
                    )
                )
                curr_page += 1
                buffer = []

        if buffer:
            pages.append(
                DocumentPage(
                    doc_id=doc_id,
                    source_filename=path.name,
                    page_number=curr_page,
                    chapter="General",
                    text="\n\n".join(buffer),
                    metadata={
                        "source": path.name,
                        "file_path": str(path),
                        "page": curr_page,
                        "chapter": "General",
                        "doc_title": path.stem.replace("-", " ").title(),
                        "doc_id": doc_id,
                    },
                )
            )

        return ExtractedDocument(
            doc_id=doc_id,
            filename=path.name,
            file_path=str(path),
            file_hash=file_hash,
            file_type="docx",
            total_pages=len(pages) or 1,
            pages=pages,
            title=path.stem.replace("-", " ").title(),
            author="Unknown",
            file_size_bytes=path.stat().st_size,
        )

    def _load_txt(self, path: Path) -> ExtractedDocument:
        """Extract text from a plain TXT file."""
        file_hash = PDFLoader.compute_sha256(path)
        doc_id = file_hash[:16]

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="latin-1")

        cleaned = PDFLoader.clean_text(content)
        page = DocumentPage(
            doc_id=doc_id,
            source_filename=path.name,
            page_number=1,
            chapter="General",
            text=cleaned,
            metadata={
                "source": path.name,
                "file_path": str(path),
                "page": 1,
                "chapter": "General",
                "doc_title": path.stem.replace("-", " ").title(),
                "doc_id": doc_id,
            },
        )

        return ExtractedDocument(
            doc_id=doc_id,
            filename=path.name,
            file_path=str(path),
            file_hash=file_hash,
            file_type="txt",
            total_pages=1,
            pages=[page],
            title=path.stem.replace("-", " ").title(),
            author="Unknown",
            file_size_bytes=path.stat().st_size,
        )

    def _load_csv(self, path: Path) -> ExtractedDocument:
        """Extract structured tabular data from a CSV file."""
        file_hash = PDFLoader.compute_sha256(path)
        doc_id = file_hash[:16]

        df = pd.read_csv(path)
        # Convert each row into a readable narrative format
        rows_text = []
        for idx, row in df.iterrows():
            row_items = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
            rows_text.append(f"Row {idx+1}: " + " | ".join(row_items))

        full_text = "\n\n".join(rows_text)
        page = DocumentPage(
            doc_id=doc_id,
            source_filename=path.name,
            page_number=1,
            chapter="Tabular Data",
            text=full_text,
            metadata={
                "source": path.name,
                "file_path": str(path),
                "page": 1,
                "chapter": "Tabular Data",
                "doc_title": path.stem.replace("-", " ").title(),
                "doc_id": doc_id,
                "row_count": len(df),
            },
        )

        return ExtractedDocument(
            doc_id=doc_id,
            filename=path.name,
            file_path=str(path),
            file_hash=file_hash,
            file_type="csv",
            total_pages=1,
            pages=[page],
            title=path.stem.replace("-", " ").title(),
            author="Unknown",
            file_size_bytes=path.stat().st_size,
        )
