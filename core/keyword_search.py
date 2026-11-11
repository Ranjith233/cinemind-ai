"""
CineMind AI - BM25 Keyword Search Engine
Implements lexical search using BM25Okapi for precise keyword and entity matching.
"""

import re
from typing import Any, Optional
from rank_bm25 import BM25Okapi

from config.logger import get_logger

logger = get_logger(__name__)


class BM25KeywordSearch:
    """In-memory BM25 index built from knowledge base chunks."""

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_chunks: list[dict[str, Any]] = []
        self.tokenized_corpus: list[list[str]] = []

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Tokenize text into lowercase terms, preserving alphanumeric and quotes/dashes."""
        # Simple effective regex tokenization
        tokens = re.findall(r"\b[a-zA-Z0-9'-]+\b", text.lower())
        return tokens

    def build_index(self, chunks: list[dict[str, Any]]) -> None:
        """Build or refresh BM25 index from a list of chunk dicts."""
        self.corpus_chunks = chunks
        if not chunks:
            self.bm25 = None
            self.tokenized_corpus = []
            return

        self.tokenized_corpus = [self.tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"Built BM25 index over {len(chunks)} document chunks.")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Execute BM25 keyword search.
        Applies metadata filtering and normalizes scores between 0 and 1.
        """
        if not self.bm25 or not self.corpus_chunks:
            return []

        tokens = self.tokenize(query)
        if not tokens:
            return []

        raw_scores = self.bm25.get_scores(tokens)
        max_score = max(raw_scores) if len(raw_scores) > 0 and max(raw_scores) > 0 else 1.0

        results: list[dict[str, Any]] = []
        for idx, score in enumerate(raw_scores):
            if score <= 0:
                continue

            chunk = self.corpus_chunks[idx]
            meta = chunk.get("metadata", {})

            # Filter verification
            if filters:
                match = True
                for k, v in filters.items():
                    if meta.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            norm_score = float(score / max_score)
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": meta,
                    "score": norm_score,
                    "bm25_raw_score": float(score),
                }
            )

        # Sort descending by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def count(self) -> int:
        return len(self.corpus_chunks)
