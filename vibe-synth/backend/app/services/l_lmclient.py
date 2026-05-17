# =============================================================================
# File: backend/app/services/llm_client.py
# Purpose: Unified LLM client that abstracts over multiple providers
#          (OpenAI, Anthropic, local). All generation requests in the
#          codebase go through this single interface — routes and generators
#          never call provider SDKs directly.
# =============================================================================

import time
import logging
from typing import AsyncIterator

from app.config import Settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the LLM provider returns an error or times out."""
    pass


class LLMClient:
    """
    Thin async wrapper around LLM provider APIs.

    Supports three backends controlled by settings.llm_provider:
      - "openai"    — OpenAI Chat Completions API (gpt-4o, etc.)
      - "anthropic" — Anthropic Messages API (claude-* models)
      - "local"     — HTTP endpoint compatible with OpenAI API schema
                      (Ollama, LM Studio, vLLM, etc.)

    All methods are async and follow the same signature so callers are
    provider-agnostic. Streaming is supported via generate_stream().
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client  = self._build_client()

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _build_client(self):
        """
        Instantiate the correct provider SDK based on settings.llm_provider.
        Returns a raw SDK client object; higher-level methods wrap it.
        """
        provider = self.settings.llm_provider.lower()

        if provider == "openai":
            try:
                from openai import AsyncOpenAI
                return AsyncOpenAI(api_key=self.settings.llm_api_key)
            except ImportError:
                raise LLMError(
                    "openai package not installed. Run: pip install openai"
                )

        if provider == "anthropic":
            try:
                import anthropic
                return anthropic.AsyncAnthropic(api_key=self.settings.llm_api_key)
            except ImportError:
                raise LLMError(
                    "anthropic package not installed. Run: pip install anthropic"
                )

        if provider == "local":
            try:
                from openai import AsyncOpenAI
                return AsyncOpenAI(
                    api_key="local",
                    base_url="http://localhost:11434/v1",  # Ollama default
                )
            except ImportError:
                raise LLMError(
                    "openai package not installed. Run: pip install openai"
                )

        raise LLMError(
            f"Unknown llm_provider '{provider}'. "
            "Valid options: 'openai', 'anthropic', 'local'."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        """
        Send a single-turn generation request and return the full response
        text. Blocks until the complete response is available.

        Args:
            system_prompt: Instructions that set the model's role and output
                           format (loaded from generation/prompts/*.txt).
            user_prompt:   The user-facing Vibe description or task string.
            max_tokens:    Override settings.llm_max_tokens for this call.

        Returns:
            Raw text content of the model's response.

        Raises:
            LLMError: On API errors, timeouts, or unexpected response shapes.
        """
        tokens   = max_tokens or self.settings.llm_max_tokens
        provider = self.settings.llm_provider.lower()
        t0       = time.perf_counter()

        logger.debug(
            "LLM generate | provider=%s model=%s tokens=%d",
            provider, self.settings.llm_model, tokens,
        )

        try:
            if provider in ("openai", "local"):
                response = await self._client.chat.completions.create(
                    model=self.settings.llm_model,
                    temperature=self.settings.llm_temperature,
                    max_tokens=tokens,
                    messages=[
                        {"role": "system",  "content": system_prompt},
                        {"role": "user",    "content": user_prompt},
                    ],
                )
                content = response.choices[0].message.content or ""

            elif provider == "anthropic":
                response = await self._client.messages.create(
                    model=self.settings.llm_model,
                    max_tokens=tokens,
                    temperature=self.settings.llm_temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                content = response.content[0].text

            else:
                raise LLMError(f"Unsupported provider: {provider}")

        except Exception as exc:
            raise LLMError(f"LLM call failed: {exc}") from exc

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.debug("LLM generate completed in %d ms", elapsed_ms)
        return content

    async def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """
        Streaming variant of generate(). Yields text chunks as they arrive
        from the provider. Useful for long code-generation responses where
        the caller wants to display tokens progressively.

        Usage:
            async for chunk in client.generate_stream(sys, usr):
                print(chunk, end="", flush=True)
        """
        tokens   = max_tokens or self.settings.llm_max_tokens
        provider = self.settings.llm_provider.lower()

        try:
            if provider in ("openai", "local"):
                stream = await self._client.chat.completions.create(
                    model=self.settings.llm_model,
                    temperature=self.settings.llm_temperature,
                    max_tokens=tokens,
                    stream=True,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta

            elif provider == "anthropic":
                async with self._client.messages.stream(
                    model=self.settings.llm_model,
                    max_tokens=tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                ) as stream:
                    async for text in stream.text_stream:
                        yield text

            else:
                raise LLMError(f"Unsupported provider: {provider}")

        except Exception as exc:
            raise LLMError(f"LLM stream failed: {exc}") from exc

    async def self_correct(
        self,
        system_prompt: str,
        original_user_prompt: str,
        failed_code: str,
        error_message: str,
        max_attempts: int = 2,
    ) -> tuple[str, int]:
        """
        Ask the LLM to fix its own previously generated code after a
        compilation or validation failure.

        Returns:
            (corrected_code, attempts_used) — where attempts_used is the
            number of correction rounds needed (1-indexed).

        Raises:
            LLMError: If all attempts fail or the provider returns an error.
        """
        correction_prompt = (
            f"The following code failed to compile.\n\n"
            f"ORIGINAL REQUEST:\n{original_user_prompt}\n\n"
            f"FAILED CODE:\n```\n{failed_code}\n```\n\n"
            f"COMPILER ERROR:\n{error_message}\n\n"
            "Please provide a corrected version of the code only. "
            "Do not include explanations — output the fixed code block directly."
        )

        for attempt in range(1, max_attempts + 1):
            logger.warning(
                "Self-correction attempt %d/%d for failed code", attempt, max_attempts
            )
            corrected = await self.generate(
                system_prompt=system_prompt,
                user_prompt=correction_prompt,
            )
            if corrected.strip():
                return corrected, attempt

        raise LLMError(
            f"Self-correction failed after {max_attempts} attempts. "
            "Compiler error persisted."
        )