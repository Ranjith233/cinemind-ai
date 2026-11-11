"""
CineMind AI - End-to-End Advanced RAG Pipeline
Orchestrates ingestion, hybrid retrieval, cross-encoder reranking, context rewriting,
and grounded generation with page-level citations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Generator, Optional, Union

from config.logger import get_logger
from config.settings import settings
from core.document_loader import DocumentLoader
from core.embeddings import BaseEmbeddings, LocalSentenceTransformerEmbeddings
from core.keyword_search import BM25KeywordSearch
from core.llm import GeminiLLM
from core.prompts import RAG_GENERATION_TEMPLATE
from core.reranker import CrossEncoderReranker
from core.retriever import HybridRetriever, RetrievedChunk
from core.text_splitter import TokenizerAwareTextSplitter
from core.vector_store import ChromaVectorStore
from database.conversation import ConversationDatabase
from database.metadata import DocumentRecord, MetadataDatabase

logger = get_logger(__name__)


@dataclass
class Citation:
    source: str
    page: int
    chapter: str
    excerpt: str
    score: float
    chunk_id: str


@dataclass
class RAGResponse:
    answer: str
    citations: list[Citation]
    confidence: str
    latency_ms: float
    rewritten_query: str
    original_query: str
    retrieved_chunks: list[RetrievedChunk]


class RAGPipeline:
    """Production-grade RAG pipeline orchestrating retrieval, reranking, and generation."""

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        embeddings: Optional[BaseEmbeddings] = None,
        keyword_search: Optional[BM25KeywordSearch] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        llm: Optional[GeminiLLM] = None,
        meta_db: Optional[MetadataDatabase] = None,
        conv_db: Optional[ConversationDatabase] = None,
    ):
        self.meta_db = meta_db or MetadataDatabase()
        self.conv_db = conv_db or ConversationDatabase()
        self.document_loader = DocumentLoader()
        self.text_splitter = TokenizerAwareTextSplitter()
        self.embeddings = embeddings or LocalSentenceTransformerEmbeddings()
        self.vector_store = vector_store or ChromaVectorStore()
        self.keyword_search = keyword_search or BM25KeywordSearch()
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embeddings=self.embeddings,
            keyword_search=self.keyword_search,
        )
        self.reranker = reranker or CrossEncoderReranker()
        self.llm = llm or GeminiLLM()

        # Initialize keyword index from stored documents if Chroma has chunks
        if self.vector_store.count() > 0:
            self.retriever.initialize_keyword_index()

    def ingest_document(
        self, file_path: Union[str, Path], force_reindex: bool = False
    ) -> DocumentRecord:
        """
        Ingest a document:
        1. Extract text, page numbers, chapters, metadata
        2. Check for duplicates via SHA-256
        3. Split into token-aware chunks
        4. Compute vector embeddings
        5. Store in persistent ChromaDB
        6. Store metadata in SQLite
        7. Refresh BM25 index
        """
        path = Path(file_path).resolve()
        file_hash = DocumentLoader().pdf_loader.compute_sha256(path)

        # Check existing
        existing = self.meta_db.get_document_by_hash(file_hash)
        if existing and not force_reindex and existing.status == "indexed":
            logger.info(f"Document already indexed: {path.name} (ID: {existing.doc_id})")
            return existing

        doc_id = file_hash[:16]
        logger.info(f"Starting ingestion for: {path.name} (ID: {doc_id})")

        # 1. Load document
        extracted = self.document_loader.load(path)

        # 2. Split into chunks
        chunks = self.text_splitter.split_document(extracted)

        # 3. Compute embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embeddings.embed_texts(texts)

        # 4. Remove previous chunks if re-indexing
        if existing:
            self.vector_store.delete_by_doc_id(doc_id)

        # 5. Store in Chroma
        self.vector_store.add_chunks(chunks, embeddings)

        # 6. Save metadata
        now = datetime.now().isoformat()
        record = DocumentRecord(
            doc_id=doc_id,
            filename=path.name,
            file_path=str(path),
            file_hash=file_hash,
            file_type=extracted.file_type,
            file_size_bytes=extracted.file_size_bytes,
            total_pages=extracted.total_pages,
            total_chunks=len(chunks),
            title=extracted.title or path.stem,
            author=extracted.author or "Unknown",
            status="indexed",
            indexed_at=now,
        )
        self.meta_db.add_or_update_document(record)

        # 7. Refresh BM25 keyword index
        self.retriever.initialize_keyword_index()
        logger.info(f"Successfully completed ingestion for {path.name}: {len(chunks)} chunks indexed.")
        return record

    def delete_document(self, doc_id: str) -> bool:
        """Delete document from vector store and metadata database."""
        self.vector_store.delete_by_doc_id(doc_id)
        deleted = self.meta_db.delete_document(doc_id)
        self.retriever.initialize_keyword_index()
        return deleted

    def reindex_document(self, doc_id: str) -> Optional[DocumentRecord]:
        """Re-index an existing document from its file path."""
        record = self.meta_db.get_document_by_id(doc_id)
        if not record:
            return None
        return self.ingest_document(record.file_path, force_reindex=True)

    def retrieve_context(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        final_top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> tuple[str, list[RetrievedChunk], str, float]:
        """
        Retrieval stage:
        - Query rewriting
        - Hybrid search (Vector + BM25 + RRF)
        - Cross-Encoder reranking
        """
        # Context-aware query rewriting
        search_query = query
        if session_id:
            chat_history = self.conv_db.get_formatted_history(session_id, max_turns=3)
            if chat_history:
                search_query = self.llm.rewrite_query(chat_history, query)

        # Hybrid retrieval
        candidates = self.retriever.retrieve(
            query=search_query,
            top_k=top_k or settings.TOP_K,
            filters=filters,
        )

        # Cross-Encoder reranking
        reranked = self.reranker.rerank(
            query=search_query,
            candidates=candidates,
            final_top_k=final_top_k or settings.FINAL_TOP_K,
        )

        confidence, _ = self.reranker.evaluate_confidence(reranked)

        return search_query, reranked, confidence

    def build_context_string(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into clean numbered context blocks with metadata."""
        if not chunks:
            return "No relevant source documents found in knowledge base."

        formatted_blocks = []
        for idx, chunk in enumerate(chunks, start=1):
            source = chunk.metadata.get("source", "Unknown")
            page = chunk.metadata.get("page", 1)
            chapter = chunk.metadata.get("chapter", "General")
            block = (
                f"[Source Block {idx}]\n"
                f"Document: {source}\n"
                f"Page: {page}\n"
                f"Chapter/Section: {chapter}\n"
                f"Content:\n{chunk.text.strip()}\n"
            )
            formatted_blocks.append(block)

        return "\n----------------------------------------\n".join(formatted_blocks)

    def build_citations(self, chunks: list[RetrievedChunk]) -> list[Citation]:
        """Extract structured citations from top chunks."""
        citations = []
        for chunk in chunks:
            # Form clean excerpt (~200 chars)
            clean_text = chunk.text.strip().replace("\n", " ")
            excerpt = clean_text[:220] + ("..." if len(clean_text) > 220 else "")

            citations.append(
                Citation(
                    source=chunk.metadata.get("source", "Unknown"),
                    page=chunk.metadata.get("page", 1),
                    chapter=chunk.metadata.get("chapter", "General"),
                    excerpt=excerpt,
                    score=chunk.rerank_score or chunk.score,
                    chunk_id=chunk.chunk_id,
                )
            )
        return citations

    def query(
        self,
        user_question: str,
        session_id: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> RAGResponse:
        """
        Execute full RAG generation pipeline synchronously.
        """
        start_time = time.perf_counter()

        search_query, reranked_chunks, confidence = self.retrieve_context(
            query=user_question,
            session_id=session_id,
            filters=filters,
        )

        context_str = self.build_context_string(reranked_chunks)
        prompt = RAG_GENERATION_TEMPLATE.format(
            context_str=context_str,
            question=user_question,
        )

        answer = self.llm.generate(prompt)
        citations = self.build_citations(reranked_chunks)

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Persist to conversation database if session_id provided
        if session_id:
            self.conv_db.add_message(
                session_id=session_id,
                role="user",
                content=user_question,
            )
            self.conv_db.add_message(
                session_id=session_id,
                role="assistant",
                content=answer,
                sources=[
                    {
                        "source": c.source,
                        "page": c.page,
                        "chapter": c.chapter,
                        "excerpt": c.excerpt,
                        "score": round(c.score, 3),
                    }
                    for c in citations
                ],
                confidence=confidence,
                latency_ms=latency_ms,
            )

        return RAGResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            latency_ms=latency_ms,
            rewritten_query=search_query,
            original_query=user_question,
            retrieved_chunks=reranked_chunks,
        )

    def query_stream(
        self,
        user_question: str,
        session_id: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> tuple[Generator[str, None, None], list[Citation], str, float, str]:
        """
        Execute RAG generation pipeline with streaming response generator.
        Yields tokens in real-time.
        """
        start_time = time.perf_counter()

        search_query, reranked_chunks, confidence = self.retrieve_context(
            query=user_question,
            session_id=session_id,
            filters=filters,
        )

        context_str = self.build_context_string(reranked_chunks)
        prompt = RAG_GENERATION_TEMPLATE.format(
            context_str=context_str,
            question=user_question,
        )

        citations = self.build_citations(reranked_chunks)
        latency_ms = (time.perf_counter() - start_time) * 1000
        stream_gen = self.llm.generate_stream(prompt)

        return stream_gen, citations, confidence, latency_ms, search_query
