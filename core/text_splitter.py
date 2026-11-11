"""
CineMind AI - Tokenizer-Aware Text Splitter
Splits document pages into semantic chunks while preserving paragraph boundaries,
tracking page numbers, and attaching chapter metadata.
"""

from dataclasses import dataclass, field
import os
from typing import Any, Optional
from transformers import AutoTokenizer

from config.logger import get_logger
from config.settings import settings
from core.pdf_loader import DocumentPage, ExtractedDocument

# Suppress HuggingFace symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logger = get_logger(__name__)


@dataclass
class DocumentChunk:
    """Represents a text chunk with complete provenance metadata."""
    chunk_id: str
    doc_id: str
    source_filename: str
    chapter: str
    page_start: int
    page_end: int
    text: str
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


class TokenizerAwareTextSplitter:
    """
    Splits document pages into chunks aware of token lengths and paragraph boundaries.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        model_to_use = model_name or settings.EMBEDDING_MODEL

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_to_use)
        except Exception as e:
            logger.warning(f"Could not load HuggingFace tokenizer '{model_to_use}': {e}. Using fallback estimator.")
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Count tokens accurately using tokenizer or estimation fallback."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        # Estimation fallback: 1 token ~ 0.75 words
        return max(1, int(len(text.split()) * 1.3))

    def split_document(self, document: ExtractedDocument) -> list[DocumentChunk]:
        """Splits an entire ExtractedDocument into a list of DocumentChunks."""
        chunks: list[DocumentChunk] = []
        chunk_idx = 0

        # We process page by page, retaining cross-page context when appropriate
        for page in document.pages:
            if not page.text.strip():
                continue

            page_chunks = self._split_page(page, document.filename, document.doc_id)
            for ch in page_chunks:
                chunk_id = f"{document.doc_id}_p{ch.page_start}_c{chunk_idx}"
                ch.chunk_id = chunk_id
                ch.metadata["chunk_id"] = chunk_id
                ch.metadata["chunk_index"] = chunk_idx
                chunks.append(ch)
                chunk_idx += 1

        # Annotate total chunks
        for ch in chunks:
            ch.metadata["total_chunks"] = len(chunks)

        logger.info(
            f"Split '{document.filename}' into {len(chunks)} chunks "
            f"(target chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks

    def _split_page(
        self, page: DocumentPage, filename: str, doc_id: str
    ) -> list[DocumentChunk]:
        """Splits a single page into chunks with paragraph awareness."""
        raw_paragraphs = page.text.split("\n\n")
        paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

        if not paragraphs:
            return []

        # If entire page fits comfortably within chunk_size, keep it intact
        page_tokens = self.count_tokens(page.text)
        if page_tokens <= self.chunk_size:
            chunk = DocumentChunk(
                chunk_id="",
                doc_id=doc_id,
                source_filename=filename,
                chapter=page.chapter,
                page_start=page.page_number,
                page_end=page.page_number,
                text=page.text,
                token_count=page_tokens,
                metadata={
                    **page.metadata,
                    "source": filename,
                    "page": page.page_number,
                    "chapter": page.chapter,
                    "doc_id": doc_id,
                    "token_count": page_tokens,
                },
            )
            return [chunk]

        # Multi-paragraph chunking with sliding window and paragraph boundary preservation
        chunks: list[DocumentChunk] = []
        current_paragraphs: list[str] = []
        current_token_count = 0

        for p in paragraphs:
            p_tokens = self.count_tokens(p)

            # If a single paragraph is larger than chunk_size, split by sentences
            if p_tokens > self.chunk_size:
                # Flush existing buffer
                if current_paragraphs:
                    chunk_text = "\n\n".join(current_paragraphs)
                    chunks.append(
                        self._create_chunk(
                            chunk_text, page, filename, doc_id, current_token_count
                        )
                    )
                    current_paragraphs = []
                    current_token_count = 0

                # Sentence-level splitting
                sub_chunks = self._split_large_paragraph(p, page, filename, doc_id)
                chunks.extend(sub_chunks)
                continue

            if current_token_count + p_tokens <= self.chunk_size:
                current_paragraphs.append(p)
                current_token_count += p_tokens
            else:
                # Chunk boundary reached: flush current chunk
                chunk_text = "\n\n".join(current_paragraphs)
                chunks.append(
                    self._create_chunk(
                        chunk_text, page, filename, doc_id, current_token_count
                    )
                )

                # Overlap retention: keep trailing paragraphs that fit within chunk_overlap
                overlap_paras: list[str] = []
                overlap_tokens = 0
                for op in reversed(current_paragraphs):
                    op_tok = self.count_tokens(op)
                    if overlap_tokens + op_tok <= self.chunk_overlap:
                        overlap_paras.insert(0, op)
                        overlap_tokens += op_tok
                    else:
                        break

                current_paragraphs = overlap_paras + [p]
                current_token_count = overlap_tokens + p_tokens

        # Flush remaining buffer
        if current_paragraphs:
            chunk_text = "\n\n".join(current_paragraphs)
            chunks.append(
                self._create_chunk(
                    chunk_text, page, filename, doc_id, current_token_count
                )
            )

        return chunks

    def _split_large_paragraph(
        self, text: str, page: DocumentPage, filename: str, doc_id: str
    ) -> list[DocumentChunk]:
        """Splits an oversized paragraph into sentence-bounded chunks."""
        import re

        # Split on sentence boundaries (. ! ?)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[DocumentChunk] = []
        curr_sentences: list[str] = []
        curr_tokens = 0

        for s in sentences:
            s_tok = self.count_tokens(s)
            if curr_tokens + s_tok <= self.chunk_size:
                curr_sentences.append(s)
                curr_tokens += s_tok
            else:
                if curr_sentences:
                    chunk_text = " ".join(curr_sentences)
                    chunks.append(
                        self._create_chunk(
                            chunk_text, page, filename, doc_id, curr_tokens
                        )
                    )
                curr_sentences = [s]
                curr_tokens = s_tok

        if curr_sentences:
            chunk_text = " ".join(curr_sentences)
            chunks.append(
                self._create_chunk(chunk_text, page, filename, doc_id, curr_tokens)
            )

        return chunks

    def _create_chunk(
        self,
        text: str,
        page: DocumentPage,
        filename: str,
        doc_id: str,
        token_count: int,
    ) -> DocumentChunk:
        return DocumentChunk(
            chunk_id="",
            doc_id=doc_id,
            source_filename=filename,
            chapter=page.chapter,
            page_start=page.page_number,
            page_end=page.page_number,
            text=text,
            token_count=token_count,
            metadata={
                **page.metadata,
                "source": filename,
                "page": page.page_number,
                "chapter": page.chapter,
                "doc_id": doc_id,
                "token_count": token_count,
            },
        )
