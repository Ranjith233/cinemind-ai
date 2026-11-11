"""
CineMind AI - Global Configuration & Settings
Manages environment variables, directory paths, model hyperparameters, and defaults.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Base directory is the project root (cinemind-ai)
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env if present
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)


class Settings:
    """Application configuration settings."""

    # Project directories
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    SOURCE_DIR: Path = BASE_DIR / "data" / "source"
    UPLOADS_DIR: Path = BASE_DIR / "data" / "uploads"
    PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    STORAGE_DIR: Path = BASE_DIR / "storage"
    CHROMA_PATH: Path = BASE_DIR / os.getenv("CHROMA_PATH", "storage/chroma")
    DATABASE_PATH: Path = BASE_DIR / "storage" / "cinemind.db"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # API Keys & LLM Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    SUPPORTED_GEMINI_MODELS: list[str] = [
        "gemini-2.5-flash",
        "gemini-1.5-flash",
        "gemini-2.5-pro",
        "gemini-1.5-pro",
    ]

    # Retrieval & Embedding Configuration
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    RERANKER_MODEL: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # Chunking Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    # Retrieval Hyperparameters
    TOP_K: int = int(os.getenv("TOP_K", "10"))
    FINAL_TOP_K: int = int(os.getenv("FINAL_TOP_K", "4"))
    RRF_K: int = int(os.getenv("RRF_K", "60"))

    # Confidence Threshold for Reranking / Grounding
    MIN_RERANK_SCORE: float = float(os.getenv("MIN_RERANK_SCORE", "-10.0"))

    def ensure_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        for directory in [
            self.DATA_DIR,
            self.SOURCE_DIR,
            self.UPLOADS_DIR,
            self.PROCESSED_DIR,
            self.STORAGE_DIR,
            self.CHROMA_PATH,
            self.LOGS_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
