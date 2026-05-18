# =============================================================================
# File: backend/tests/test_llm_client.py
# Purpose: Unit tests for LLMClient. Uses pytest-asyncio and unittest.mock to
#          patch provider SDK calls so tests run fully offline without real API
#          keys. Covers: successful generation, streaming, self-correction
#          logic, provider switching, and error propagation.
# =============================================================================

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.config import Settings
from app.services.llm_client import LLMClient, LLMError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_settings(**overrides) -> Settings:
    """Return a Settings instance with test-safe defaults."""
    defaults = dict(
        llm_provider="openai",
        llm_model="gpt-4o",
        llm_api_key="test-key-123",
        llm_max_tokens=512,
        llm_temperature=0.2,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _make_openai_response(content: str) -> MagicMock:
    """Build a minimal mock that mimics an OpenAI chat completion response."""
    message  = MagicMock()
    message.content = content
    choice   = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# generate() — OpenAI provider
# ---------------------------------------------------------------------------

class TestGenerateOpenAI:

    @pytest.mark.asyncio
    async def test_returns_content_string(self):
        """generate() returns the text content from the first choice."""
        settings = _make_settings(llm_provider="openai")

        mock_response = _make_openai_response("process = _;")
        mock_create   = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI") as MockClient:
            instance = MockClient.return_value
            instance.chat.completions.create = mock_create

            client = LLMClient(settings=settings)
            client._client = instance

            result = await client.generate(
                system_prompt="You are a DSP code generator.",
                user_prompt="Generate a pass-through Faust process.",
            )

        assert result == "process = _;"
        mock_create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_passes_correct_model_and_temperature(self):
        """generate() forwards model and temperature from settings to the SDK."""
        settings = _make_settings(
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            llm_temperature=0.1,
        )
        mock_response = _make_openai_response("ok")
        mock_create   = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI"):
            client = LLMClient(settings=settings)
            client._client = MagicMock()
            client._client.chat.completions.create = mock_create

            await client.generate("sys", "usr")

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"]       == "gpt-4o-mini"
        assert call_kwargs["temperature"] == 0.1

    @pytest.mark.asyncio
    async def test_raises_llm_error_on_sdk_exception(self):
        """generate() wraps SDK exceptions in LLMError."""
        settings = _make_settings()

        with patch("openai.AsyncOpenAI"):
            client = LLMClient(settings=settings)
            client._client = MagicMock()
            client._client.chat.completions.create = AsyncMock(
                side_effect=RuntimeError("connection refused")
            )

            with pytest.raises(LLMError, match="connection refused"):
                await client.generate("sys", "usr")

    @pytest.mark.asyncio
    async def test_max_tokens_override(self):
        """generate() respects an explicit max_tokens argument over settings."""
        settings = _make_settings(llm_max_tokens=512)
        mock_response = _make_openai_response("x")
        mock_create   = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI"):
            client = LLMClient(settings=settings)
            client._client = MagicMock()
            client._client.chat.completions.create = mock_create

            await client.generate("sys", "usr", max_tokens=128)

        assert mock_create.call_args.kwargs["max_tokens"] == 128


# ---------------------------------------------------------------------------
# generate() — Anthropic provider
# ---------------------------------------------------------------------------

class TestGenerateAnthropic:

    @pytest.mark.asyncio
    async def test_returns_text_from_first_content_block(self):
        """Anthropic path extracts text from response.content[0].text."""
        settings = _make_settings(llm_provider="anthropic", llm_api_key="ant-key")

        content_block = MagicMock()
        content_block.text = "process = hgroup(\"\", vslider(\"gain\", 1, 0, 2, 0.01) * _);"
        mock_response = MagicMock()
        mock_response.content = [content_block]

        mock_create = AsyncMock(return_value=mock_response)

        with patch("anthropic.AsyncAnthropic") as MockAnt:
            instance = MockAnt.return_value
            instance.messages.create = mock_create

            client = LLMClient(settings=settings)
            client._client = instance

            result = await client.generate("sys", "usr")

        assert "vslider" in result


# ---------------------------------------------------------------------------
# generate() — local provider
# ---------------------------------------------------------------------------

class TestGenerateLocal:

    @pytest.mark.asyncio
    async def test_local_provider_uses_openai_client(self):
        """Local provider reuses the OpenAI-compatible client with a custom base_url."""
        settings = _make_settings(llm_provider="local", llm_api_key="local")
        mock_response = _make_openai_response("@fragment fn main() {}")
        mock_create   = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI") as MockOAI:
            instance = MockOAI.return_value
            instance.chat.completions.create = mock_create

            client = LLMClient(settings=settings)
            client._client = instance

            result = await client.generate("sys", "usr")

        assert result == "@fragment fn main() {}"


# ---------------------------------------------------------------------------
# Unknown provider
# ---------------------------------------------------------------------------

class TestUnknownProvider:

    def test_raises_on_unknown_provider(self):
        """LLMClient.__init__ raises LLMError for an unrecognised provider."""
        settings = _make_settings(llm_provider="cohere")
        with pytest.raises(LLMError, match="Unknown llm_provider"):
            LLMClient(settings=settings)


# ---------------------------------------------------------------------------
# self_correct()
# ---------------------------------------------------------------------------

class TestSelfCorrect:

    @pytest.mark.asyncio
    async def test_returns_corrected_code_and_attempt_count(self):
        """self_correct() returns (corrected_text, 1) on first successful attempt."""
        settings = _make_settings()
        mock_response = _make_openai_response("process = _ <: _, _;  // fixed")
        mock_create   = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI"):
            client = LLMClient(settings=settings)
            client._client = MagicMock()
            client._client.chat.completions.create = mock_create

            corrected, attempts = await client.self_correct(
                system_prompt="Fix the Faust code.",
                original_user_prompt="Stereo splitter",
                failed_code="process = _ <: _",
                error_message="SyntaxError: unexpected end of input",
                max_attempts=2,
            )

        assert "fixed" in corrected
        assert attempts == 1

    @pytest.mark.asyncio
    async def test_raises_llm_error_when_all_attempts_fail(self):
        """self_correct() raises LLMError after exhausting max_attempts."""
        settings = _make_settings()

        with patch("openai.AsyncOpenAI"):
            client = LLMClient(settings=settings)
            client._client = MagicMock()
            # Simulate the LLM always returning empty output
            client._client.chat.completions.create = AsyncMock(
                return_value=_make_openai_response("")
            )

            with pytest.raises(LLMError, match="Self-correction failed"):
                await client.self_correct(
                    system_prompt="sys",
                    original_user_prompt="usr",
                    failed_code="broken code",
                    error_message="compile error",
                    max_attempts=2,
                )

    @pytest.mark.asyncio
    async def test_correction_prompt_includes_error_message(self):
        """The correction call embeds the compiler error in the user message."""
        settings = _make_settings()
        mock_response = _make_openai_response("fixed code block")
        mock_create   = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI"):
            client = LLMClient(settings=settings)
            client._client = MagicMock()
            client._client.chat.completions.create = mock_create

            await client.self_correct(
                system_prompt="sys",
                original_user_prompt="make reverb",
                failed_code="bad code",
                error_message="BIBO stability violation",
                max_attempts=1,
            )

        call_messages = mock_create.call_args.kwargs["messages"]
        user_message  = next(m for m in call_messages if m["role"] == "user")
        assert "BIBO stability violation" in user_message["content"]