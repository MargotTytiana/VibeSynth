# =============================================================================
# File: backend/tests/test_prompt_builder.py
# Purpose: Unit tests for PromptBuilder. Patches filesystem reads so tests
#          run without requiring the actual prompt template files to exist.
#          Covers: DSP prompt assembly, shader prompt assembly, constraint
#          injection, few-shot formatting, and correction context retrieval.
# =============================================================================

import json
import pytest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path

from app.config import Settings
from app.services.prompt_builder import PromptBuilder, _load_text, _load_few_shots
from app.models.request_models import (
    GenerateDSPRequest,
    GenerateShaderRequest,
    InputType,
    TargetPlatform,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_settings() -> Settings:
    return Settings(
        llm_provider="openai",
        llm_api_key="test-key",
        llm_model="gpt-4o",
    )


def _dsp_request(**overrides) -> GenerateDSPRequest:
    defaults = dict(
        prompt="Make it sound like a spring reverb in a metal box",
        input_type=InputType.audio_stream,
        sample_rate=44100,
        max_parameters=6,
        force_refresh=False,
    )
    defaults.update(overrides)
    return GenerateDSPRequest(**defaults)


def _shader_request(**overrides) -> GenerateShaderRequest:
    defaults = dict(
        prompt="Glitchy pixel sort effect with warm colour bleed",
        input_type=InputType.video_stream,
        target_platform=TargetPlatform.webgpu,
        max_parameters=4,
        force_refresh=False,
    )
    defaults.update(overrides)
    return GenerateShaderRequest(**defaults)


# Minimal realistic template content used as mock file returns
_DSP_TEMPLATE  = "You are an expert Faust DSP programmer. Output only valid Faust code."
_SHADER_TEMPLATE = "You are an expert WGSL/GLSL shader programmer. Output only valid shader code."
_FEW_SHOTS = {
    "dsp": [
        {
            "vibe":   "warm tape saturation",
            "output": "process = ef.cubicnl(0.3, 0.0);",
        }
    ],
    "shader": [
        {
            "vibe":   "soft bloom glow",
            "output": "@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1.0); }",
        }
    ],
}


def _patch_templates():
    """
    Returns a context-manager stack that patches all three template file reads.
    Clears the lru_cache on _load_text and _load_few_shots before each use
    so stale cached values from other tests do not leak in.
    """
    _load_text.cache_clear()
    _load_few_shots.cache_clear()

    text_side_effects = {
        "system_dsp.txt":    _DSP_TEMPLATE,
        "system_shader.txt": _SHADER_TEMPLATE,
    }

    def fake_read_text(path: Path, encoding="utf-8"):  # noqa: ARG001
        return text_side_effects.get(path.name, "")

    def fake_exists(self):
        return True

    return (
        patch.object(Path, "read_text",  fake_read_text),
        patch.object(Path, "exists",     fake_exists),
        patch(
            "builtins.open",
            mock_open(read_data=json.dumps(_FEW_SHOTS)),
        ),
    )


# ---------------------------------------------------------------------------
# DSP prompt tests
# ---------------------------------------------------------------------------

class TestBuildDSPPrompts:

    def _build(self, request=None):
        _load_text.cache_clear()
        _load_few_shots.cache_clear()
        request = request or _dsp_request()
        patches = _patch_templates()
        with patches[0], patches[1], patches[2]:
            builder = PromptBuilder(settings=_make_settings())
            return builder.build_dsp_prompts(request)

    def test_returns_two_non_empty_strings(self):
        """build_dsp_prompts() returns a (system, user) tuple of non-empty strings."""
        system, user = self._build()
        assert isinstance(system, str) and len(system) > 0
        assert isinstance(user, str)   and len(user) > 0

    def test_system_contains_base_template(self):
        """System prompt includes the base DSP template text."""
        system, _ = self._build()
        assert _DSP_TEMPLATE in system

    def test_system_contains_faust_constraint(self):
        """System prompt declares Faust as the required output language."""
        system, _ = self._build()
        assert "Faust" in system

    def test_system_contains_max_parameters_constraint(self):
        """System prompt injects the max_parameters value from the request."""
        request = _dsp_request(max_parameters=3)
        system, _ = self._build(request)
        assert "3" in system

    def test_system_contains_sample_rate(self):
        """System prompt injects the target sample rate from the request."""
        request = _dsp_request(sample_rate=48000)
        system, _ = self._build(request)
        assert "48000" in system

    def test_system_contains_bibo_constraint(self):
        """System prompt includes the BIBO stability requirement."""
        system, _ = self._build()
        assert "BIBO" in system

    def test_user_prompt_contains_vibe_description(self):
        """User prompt wraps the caller's Vibe description in quotes."""
        request = _dsp_request(prompt="icy reverb tail")
        _, user = self._build(request)
        assert "icy reverb tail" in user

    def test_user_prompt_contains_input_type(self):
        """User prompt includes the input_type value."""
        _, user = self._build()
        assert "audio_stream" in user

    def test_system_contains_few_shot_example(self):
        """System prompt includes the few-shot example vibe text."""
        system, _ = self._build()
        assert "warm tape saturation" in system


# ---------------------------------------------------------------------------
# Shader prompt tests
# ---------------------------------------------------------------------------

class TestBuildShaderPrompts:

    def _build(self, request=None):
        _load_text.cache_clear()
        _load_few_shots.cache_clear()
        request = request or _shader_request()
        patches = _patch_templates()
        with patches[0], patches[1], patches[2]:
            builder = PromptBuilder(settings=_make_settings())
            return builder.build_shader_prompts(request)

    def test_returns_two_non_empty_strings(self):
        system, user = self._build()
        assert len(system) > 0 and len(user) > 0

    def test_system_contains_base_shader_template(self):
        """System prompt includes the base shader template text."""
        system, _ = self._build()
        assert _SHADER_TEMPLATE in system

    def test_wgsl_language_injected_for_webgpu(self):
        """WGSL is specified as output language when target_platform is webgpu."""
        request = _shader_request(target_platform=TargetPlatform.webgpu)
        system, _ = self._build(request)
        assert "WGSL" in system

    def test_glsl_language_injected_for_webgl(self):
        """GLSL ES 3.0 is specified as output language when target_platform is webgl."""
        _load_text.cache_clear()
        _load_few_shots.cache_clear()
        request = _shader_request(target_platform=TargetPlatform.webgl)
        patches = _patch_templates()
        with patches[0], patches[1], patches[2]:
            builder = PromptBuilder(settings=_make_settings())
            system, _ = builder.build_shader_prompts(request)
        assert "GLSL" in system

    def test_system_contains_loop_unroll_constraint(self):
        """System prompt enforces the loop-unrolling rule for shaders."""
        system, _ = self._build()
        assert "loop" in system.lower() and "constant" in system.lower()

    def test_user_prompt_contains_vibe_description(self):
        """User prompt includes the caller's Vibe text."""
        request = _shader_request(prompt="neon rain streaks")
        _, user = self._build(request)
        assert "neon rain streaks" in user

    def test_user_prompt_contains_platform(self):
        """User prompt mentions the target platform."""
        request = _shader_request(target_platform=TargetPlatform.webgpu)
        _, user = self._build(request)
        assert "webgpu" in user.lower()

    def test_few_shot_shader_example_included(self):
        """System prompt includes the shader few-shot example vibe text."""
        system, _ = self._build()
        assert "soft bloom glow" in system


# ---------------------------------------------------------------------------
# build_correction_context()
# ---------------------------------------------------------------------------

class TestBuildCorrectionContext:

    def _context(self, gen_type: str) -> str:
        _load_text.cache_clear()
        _load_few_shots.cache_clear()
        patches = _patch_templates()
        with patches[0], patches[1], patches[2]:
            builder = PromptBuilder(settings=_make_settings())
            return builder.build_correction_context(gen_type)

    def test_dsp_correction_returns_dsp_template(self):
        """Correction context for 'dsp' returns the DSP system prompt."""
        ctx = self._context("dsp")
        assert _DSP_TEMPLATE in ctx

    def test_shader_correction_returns_shader_template(self):
        """Correction context for 'shader' returns the shader system prompt."""
        ctx = self._context("shader")
        assert _SHADER_TEMPLATE in ctx

    def test_unknown_type_raises_value_error(self):
        """An unrecognised generation_type raises ValueError."""
        _load_text.cache_clear()
        patches = _patch_templates()
        with patches[0], patches[1], patches[2]:
            builder = PromptBuilder(settings=_make_settings())
            with pytest.raises(ValueError, match="Unknown generation_type"):
                builder.build_correction_context("audio")