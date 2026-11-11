"""
CineMind AI - Conversation History Database (SQLite)
Maintains conversation sessions, user questions, AI responses, citations, and performance metrics.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any, Optional
import uuid

from config.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


@dataclass
class ChatMessageRecord:
    message_id: str
    session_id: str
    role: str  # 'user' or 'assistant'
    content: str
    sources_json: str
    confidence: str
    latency_ms: float
    created_at: str

    @property
    def sources(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self.sources_json) if self.sources_json else []
        except Exception:
            return []


@dataclass
class ChatSessionRecord:
    session_id: str
    title: str
    created_at: str
    updated_at: str


class ConversationDatabase:
    """Manages SQLite storage for conversation sessions and messages."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.DATABASE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create sessions and messages tables."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources_json TEXT,
                    confidence TEXT,
                    latency_ms REAL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
                )
                """
            )
            conn.commit()

    def create_session(self, title: str = "New Cinema Conversation") -> str:
        """Create a new chat session."""
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO chat_sessions (session_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, title, now, now),
            )
            conn.commit()
            logger.info(f"Created chat session: {session_id} ('{title}')")
        return session_id

    def list_sessions(self) -> list[ChatSessionRecord]:
        """List all chat sessions ordered by latest update."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chat_sessions ORDER BY updated_at DESC")
            rows = cursor.fetchall()
            return [ChatSessionRecord(**dict(r)) for r in rows]

    def get_session_messages(self, session_id: str) -> list[ChatMessageRecord]:
        """Fetch all messages in chronological order for a session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            )
            rows = cursor.fetchall()
            return [ChatMessageRecord(**dict(r)) for r in rows]

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[list[dict[str, Any]]] = None,
        confidence: str = "High",
        latency_ms: float = 0.0,
    ) -> str:
        """Add a message to a session and update session's updated_at timestamp."""
        msg_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        sources_json = json.dumps(sources or [])

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (
                    message_id, session_id, role, content, sources_json, confidence, latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (msg_id, session_id, role, content, sources_json, confidence, latency_ms, now),
            )
            # Update session timestamp and optionally update title if first user message
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chat_messages WHERE session_id = ?", (session_id,))
            msg_count = cursor.fetchone()[0]

            if msg_count <= 2 and role == "user":
                # Generate clean title from first query
                title = content[:35] + ("..." if len(content) > 35 else "")
                conn.execute(
                    "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                    (title, now, session_id),
                )
            else:
                conn.execute(
                    "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
                    (now, session_id),
                )
            conn.commit()

        return msg_id

    def delete_session(self, session_id: str) -> bool:
        """Delete an entire session and its messages."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted

    def clear_session(self, session_id: str) -> bool:
        """Clear all messages in a session while keeping the session entry."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_formatted_history(self, session_id: str, max_turns: int = 3) -> str:
        """Return formatted recent dialog turns for query rewriting."""
        messages = self.get_session_messages(session_id)
        if not messages:
            return ""

        # Take last 2 * max_turns messages
        recent = messages[-(max_turns * 2) :]
        turns = []
        for m in recent:
            prefix = "User" if m.role == "user" else "Assistant"
            turns.append(f"{prefix}: {m.content}")

        return "\n".join(turns)
