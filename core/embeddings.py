"""
CineMind AI - Modular Embeddings Provider
Implements BaseEmbeddings interface with SentenceTransformers as default local provider.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from config.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


class BaseEmbeddings(ABC):
    """Abstract Base Class for embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate vector embeddings for a list of texts."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Generate vector embedding for a single search query."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding vector dimension."""
        pass


class LocalSentenceTransformerEmbeddings(BaseEmbeddings):
    """Local SentenceTransformer embedding provider."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        if hasattr(self.model, "get_embedding_dimension"):
            self._dim = self.model.get_embedding_dimension()
        else:
            self._dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded. Dimension: {self._dim}")

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Encode multiple texts in batches."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Encode a single query."""
        embedding = self.model.encode(
            query,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.tolist()
