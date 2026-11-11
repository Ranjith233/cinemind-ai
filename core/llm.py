"""
CineMind AI - Gemini LLM Integration
Uses the official google-genai SDK for grounded cinema answer generation and streaming.
"""

from collections.abc import Generator
from typing import Optional
from google import genai
from google.genai import types

from config.logger import get_logger
from config.settings import settings
from core.prompts import (
    CINEMA_SPECIALIST_SYSTEM_PROMPT,
    RAG_GENERATION_TEMPLATE,
    QUERY_REWRITE_PROMPT,
)

logger = get_logger(__name__)


class GeminiLLM:
    """Manages LLM inference using Google Gemini via google-genai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name or settings.GEMINI_MODEL
        self._client: Optional[genai.Client] = None

        if self.api_key:
            self._init_client()

    def set_api_key(self, api_key: str) -> None:
        """Update API key dynamically at runtime."""
        self.api_key = api_key.strip()
        if self.api_key:
            self._init_client()
        else:
            self._client = None

    def set_model(self, model_name: str) -> None:
        """Update active Gemini model dynamically."""
        self.model_name = model_name.strip()
        logger.info(f"Active Gemini model set to: {self.model_name}")

    def _init_client(self) -> None:
        """Initialize official google-genai Client with v1beta API options."""
        try:
            self._client = genai.Client(
                api_key=self.api_key,
                http_options={"api_version": "v1beta"},
            )
            logger.info("google-genai Client initialized successfully with v1beta.")
        except Exception as e:
            logger.error(f"Failed to initialize google-genai client: {e}")
            self._client = None

    @property
    def is_configured(self) -> bool:
        """Return True if API key is provided and client is ready."""
        return bool(self._client and self.api_key)

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.3,
    ) -> str:
        """Generate response synchronously with fallback retry."""
        if not self.is_configured:
            return self._offline_fallback_response(prompt)

        sys_inst = system_instruction or CINEMA_SPECIALIST_SYSTEM_PROMPT
        config = types.GenerateContentConfig(
            system_instruction=sys_inst,
            temperature=temperature,
        )

        models_to_try = [self.model_name]
        if self.model_name != "gemini-1.5-flash":
            models_to_try.append("gemini-1.5-flash")

        last_error = None
        for m in models_to_try:
            try:
                response = self._client.models.generate_content(
                    model=m,
                    contents=prompt,
                    config=config,
                )
                return response.text or ""
            except Exception as e:
                logger.warning(f"Generation attempt with {m} failed: {e}. Trying next fallback...")
                last_error = e

        logger.error(f"Gemini API error during generate: {last_error}")
        raise RuntimeError(f"Gemini API generation failed: {last_error}")

    def generate_stream(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.3,
    ) -> Generator[str, None, None]:
        """Stream response chunks in real-time with fallback retry."""
        if not self.is_configured:
            fallback = self._offline_fallback_response(prompt)
            # Yield in word chunks to simulate smooth streaming
            words = fallback.split(" ")
            for i in range(0, len(words), 3):
                chunk = " ".join(words[i : i + 3]) + " "
                yield chunk
            return

        sys_inst = system_instruction or CINEMA_SPECIALIST_SYSTEM_PROMPT
        config = types.GenerateContentConfig(
            system_instruction=sys_inst,
            temperature=temperature,
        )

        models_to_try = [self.model_name]
        if self.model_name != "gemini-1.5-flash":
            models_to_try.append("gemini-1.5-flash")

        success = False
        for m in models_to_try:
            try:
                response_stream = self._client.models.generate_content_stream(
                    model=m,
                    contents=prompt,
                    config=config,
                )
                for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text
                success = True
                break
            except Exception as e:
                logger.warning(f"Streaming attempt with {m} failed: {e}")

        if not success:
            yield "\n\n*[Gemini API service momentarily unavailable. Please retry.]*"

    def rewrite_query(self, chat_history: str, latest_question: str) -> str:
        """
        Rewrite conversational follow-up questions into standalone search queries.
        """
        if not chat_history or not chat_history.strip():
            return latest_question

        if not self.is_configured:
            return latest_question

        prompt = QUERY_REWRITE_PROMPT.format(
            chat_history=chat_history,
            latest_question=latest_question,
        )

        try:
            config = types.GenerateContentConfig(
                temperature=0.0,
            )
            resp = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            rewritten = resp.text.strip() if resp.text else latest_question
            logger.info(f"Query rewritten: '{latest_question}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}. Falling back to original question.")
            return latest_question

    def _offline_fallback_response(self, prompt: str) -> str:
        """
        Graceful offline response when GEMINI_API_KEY is not yet supplied.
        Guides the user to configure their key in .env or the UI sidebar.
        """
        return (
            "⚠️ **Gemini API Key Required**\n\n"
            "CineMind AI requires a Google Gemini API key to generate grounded LLM responses. "
            "Please provide your `GEMINI_API_KEY` either:\n"
            "1. In the sidebar input field under **Configuration**\n"
            "2. In your `.env` file (`GEMINI_API_KEY=your_key_here`)\n\n"
            "Once configured, CineMind AI will synthesize grounded responses citing source pages."
        )
