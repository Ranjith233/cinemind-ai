"""
CineMind AI - Vector Store (ChromaDB)
Persistent vector storage, chunk metadata index, and vector similarity retrieval.
"""

from pathlib import Path
from typing import Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from config.logger import get_logger
from config.settings import settings
from core.text_splitter import DocumentChunk

logger = get_logger(__name__)


class ChromaVectorStore:
    """Manages persistent ChromaDB vector collections and metadata."""

    COLLECTION_NAME = "cinemind_knowledge_base"

    def __init__(self, persist_path: Optional[Path] = None):
        self.persist_path = persist_path or settings.CHROMA_PATH
        self.persist_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing persistent ChromaDB at {self.persist_path}")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_path),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )

        # Cosine distance collection
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add document chunks and their precomputed embeddings to ChromaDB."""
        if not chunks:
            return

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]

        # Chroma metadata requires scalar types (str, int, float, bool)
        metadatas: list[dict[str, Any]] = []
        for chunk in chunks:
            meta = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "source": chunk.source_filename,
                "chapter": str(chunk.chapter),
                "page": int(chunk.page_start),
                "page_end": int(chunk.page_end),
                "token_count": int(chunk.token_count),
            }
            metadatas.append(meta)

        # Batch upsert in sizes of 250 to avoid any parameter count limitations
        batch_size = 250
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_docs = documents[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]
            batch_embs = embeddings[i : i + batch_size]

            self.collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
                embeddings=batch_embs,
            )

        logger.info(f"Persisted {len(chunks)} chunks to ChromaDB collection '{self.COLLECTION_NAME}'")

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Query vector store using an embedding vector.
        Returns list of matched chunks with cosine similarity scores and metadata.
        """
        total_count = self.collection.count()
        if total_count == 0:
            return []

        actual_k = min(top_k, total_count)
        where_clause = filters if filters else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        matched: list[dict[str, Any]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return matched

        ids = results["ids"][0]
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        for i in range(len(ids)):
            # Convert cosine distance to cosine similarity: sim = 1.0 - distance
            dist = distances[i] if i < len(distances) else 1.0
            similarity = max(0.0, 1.0 - dist)

            item = {
                "chunk_id": ids[i],
                "text": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "score": float(similarity),
                "distance": float(dist),
            }
            matched.append(item)

        return matched

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Deletes all chunks belonging to a document ID."""
        existing = self.collection.get(where={"doc_id": doc_id})
        ids_to_delete = existing.get("ids", [])
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} chunks from ChromaDB for doc_id '{doc_id}'")
        return len(ids_to_delete)

    def get_all_chunks(self) -> list[dict[str, Any]]:
        """Retrieve all stored chunks for building or refreshing BM25 indices."""
        total_count = self.collection.count()
        if total_count == 0:
            return []

        data = self.collection.get(include=["documents", "metadatas"])
        chunks = []
        ids = data.get("ids", [])
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])

        for i in range(len(ids)):
            chunks.append(
                {
                    "chunk_id": ids[i],
                    "text": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                }
            )
        return chunks

    def count(self) -> int:
        """Return total number of chunks stored."""
        return self.collection.count()

    def reset(self) -> None:
        """Reset and wipe the vector collection."""
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning(f"Reset vector collection '{self.COLLECTION_NAME}'")
