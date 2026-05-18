# =============================================================================
# File: generation/shader/glsl_generator.py
# Purpose: Orchestrates end-to-end GLSL ES 3.0 (WebGL) fragment shader
#          generation. Mirrors wgsl_generator.py in structure but targets
#          WebGL/GLSL output. Selects the GLSL skeleton from shader_templates,
#          assembles prompts with the GLSL constraint set, validates for
#          GLSL-specific safety rules, and returns clean shader source.
# =============================================================================

import json
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

from app.config import Settings
from app.models.request_models import GenerateShaderRequest
from app.services.llm_client import LLMClient, LLMError
from app.services.prompt_builder import PromptBuilder
from app.services.validator import Validator, ValidationResult

logger = logging.getLogger(__name__)

SHADER_TEMPLATES_PATH = Path("generation/shader/shader_templates.json")


# ---------------------------------------------------------------------------
# Template loader  (separate cache from wgsl_generator to avoid collision)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_glsl_templates() -> list[dict]:
    """Load and cache the shader template list for the GLSL generator."""
    if not SHADER_TEMPLATES_PATH.exists():
        logger.warning(
            "Shader templates not found at %s — GLSL template hints disabled.",
            SHADER_TEMPLATES_PATH,
        )
        return []
    with SHADER_TEMPLATES_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("templates", [])


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class GLSLGenerationResult:
    """
    Complete output of one GLSL shader generation run.

    Attributes:
        shader_code:              Clean, validated GLSL ES 3.0 source.
        parameters:               Parsed parameter manifest (list of dicts).
        matched_template_id:      ID of the closest template archetype matched.
        compile_time_ms:          Wall-clock time for the full pipeline.
        self_correction_attempts: Number of LLM self-correction rounds used.
        validation:               Final ValidationResult (always passed=True).
        template_confidence:      Tag-match score for the selected template (0–1).
    """
    shader_code:              str
    parameters:               list[dict]
    matched_template_id:      str
    compile_time_ms:          int
    self_correction_attempts: int
    validation:               ValidationResult
    template_confidence:      float


# ---------------------------------------------------------------------------
# Template matcher  (GLSL variant — extracts glsl_skeleton instead of wgsl)
# ---------------------------------------------------------------------------

def _match_glsl_template(vibe: str) -> tuple[str, float, dict | None]:
    """
    Score all shader templates against the Vibe and return the best match.
    Identical scoring logic to the WGSL variant; the only difference is
    that callers use glsl_skeleton from the returned dict.

    Returns:
        (template_id, confidence, template_dict | None)
    """
    templates = _load_glsl_templates()
    if not templates:
        return "passthrough", 0.0, None

    stop = {"a", "an", "the", "and", "or", "of", "in", "with", "like",
            "make", "feel", "look", "looking", "is", "it", "to", "be"}
    tokens = [
        "".join(c for c in w.lower() if c.isalpha() or c == "-")
        for w in vibe.lower().split()
        if w not in stop and len(w) > 2
    ]

    best_score    = 0.0
    best_template = None

    for tmpl in templates:
        if tmpl["id"] == "passthrough":
            continue
        tags  = [t.lower() for t in tmpl.get("tags", [])]
        score = 0.0
        for token in tokens:
            for tag in tags:
                if token == tag:
                    score += 2.0
                    break
                elif token in tag or tag in token:
                    score += 0.5
        normalised = score / max(len(tokens), 1)
        if normalised > best_score:
            best_score    = normalised
            best_template = tmpl

    if best_template is None or best_score < 0.15:
        passthrough = next((t for t in templates if t["id"] == "passthrough"), None)
        return "passthrough", 0.0, passthrough

    return best_template["id"], round(best_score, 4), best_template


# ---------------------------------------------------------------------------
# GLSL-specific hint builder
# ---------------------------------------------------------------------------

def _build_glsl_hint(template: dict | None) -> str:
    """
    Build a GLSL-specific template hint injected into the user prompt.
    Includes the GLSL skeleton as a starting-point reference.
    """
    if template is None or template.get("id") == "passthrough":
        return ""

    skeleton = template.get("glsl_skeleton", "")
    name     = template.get("name", template["id"])
    desc     = template.get("description", "")

    hint_parts = [
        f"Suggested base effect: {name} — {desc}",
        "Extend and refine the following GLSL ES 3.0 skeleton to match the Vibe:",
        f"```glsl\n{skeleton}\n```",
        "Add, remove, or modify parameters as needed. "
        "Ensure all for-loop bounds are compile-time constants.",
    ]
    return "\n\n".join(hint_parts)


# ---------------------------------------------------------------------------
# GLSL preamble validator helper
# ---------------------------------------------------------------------------

def _has_glsl_preamble(code: str) -> bool:
    """
    Check that the generated GLSL code begins with the required ES 3.0 preamble.
    The Validator catches the missing void main() but does not check the version
    directive — this supplements it.
    """
    first_lines = code.strip()[:200]
    return "#version 300 es" in first_lines


def _has_precision_declaration(code: str) -> bool:
    """Check for 'precision highp float' or 'precision mediump float'."""
    return bool(
        "precision highp float" in code or "precision mediump float" in code
    )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class GLSLGenerator:
    """
    End-to-end GLSL ES 3.0 fragment shader generation pipeline.

    Pipeline stages:
      1. Template matching  — find the closest archetype from shader_templates.json
      2. Prompt assembly    — build (system, user) prompts with GLSL skeleton hint
      3. LLM generation     — call the LLM for raw GLSL source
      4. GLSL preamble fix  — auto-inject missing #version / precision if absent
      5. Validation         — static safety and structure checks
      6. Self-correction    — fix validation errors via LLM
      7. Return GLSLGenerationResult

    Usage:
        gen    = GLSLGenerator(settings, llm_client, prompt_builder, validator)
        result = await gen.generate(request)
    """

    MAX_CORRECTION_ATTEMPTS = 2

    def __init__(
        self,
        settings:       Settings,
        llm_client:     LLMClient,
        prompt_builder: PromptBuilder,
        validator:      Validator,
    ) -> None:
        self.settings       = settings
        self.llm_client     = llm_client
        self.prompt_builder = prompt_builder
        self.validator      = validator

    async def generate(self, request: GenerateShaderRequest) -> GLSLGenerationResult:
        """
        Run the full GLSL generation pipeline.

        Args:
            request: Validated GenerateShaderRequest with target_platform=webgl.

        Returns:
            GLSLGenerationResult with clean shader source and parameter manifest.

        Raises:
            ValueError: If validation fails after all self-correction attempts.
            LLMError:   If the LLM call fails unrecoverably.
        """
        t0 = time.perf_counter()

        # ------------------------------------------------------------------
        # Stage 1 — template matching
        # ------------------------------------------------------------------
        tmpl_id, confidence, template = _match_glsl_template(request.prompt)
        logger.info(
            "GLSLGenerator | template matched: %s (confidence=%.3f)",
            tmpl_id, confidence,
        )

        # ------------------------------------------------------------------
        # Stage 2 — prompt assembly with GLSL skeleton hint
        # ------------------------------------------------------------------
        system_prompt, user_prompt = self.prompt_builder.build_shader_prompts(request)

        if confidence >= 0.2 and template:
            hint = _build_glsl_hint(template)
            if hint:
                user_prompt = f"{user_prompt}\n\n{hint}"
                logger.debug("GLSLGenerator | GLSL skeleton hint injected.")

        # ------------------------------------------------------------------
        # Stage 3 — LLM generation
        # ------------------------------------------------------------------
        logger.info(
            "GLSLGenerator | calling LLM for vibe='%s'", request.prompt[:60]
        )
        raw_output = await self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # ------------------------------------------------------------------
        # Stage 4 — auto-fix missing GLSL preamble
        # The LLM occasionally forgets the version directive or precision
        # declaration. We patch these silently before validation so they do
        # not waste a self-correction round on a trivial omission.
        # ------------------------------------------------------------------
        raw_output = self._patch_preamble(raw_output)

        # ------------------------------------------------------------------
        # Stage 5 — validation
        # ------------------------------------------------------------------
        validation = self.validator.validate_shader(
            raw_output=raw_output,
            target_platform="webgl",
        )
        correction_attempts = 0

        # ------------------------------------------------------------------
        # Stage 6 — self-correction
        # ------------------------------------------------------------------
        if not validation.passed:
            logger.warning(
                "GLSLGenerator | validation failed — self-correcting. Errors: %s",
                validation.errors,
            )
            correction_system = self.prompt_builder.build_correction_context("shader")
            corrected, correction_attempts = await self.llm_client.self_correct(
                system_prompt=correction_system,
                original_user_prompt=user_prompt,
                failed_code=validation.clean_code,
                error_message="\n".join(validation.errors),
                max_attempts=self.MAX_CORRECTION_ATTEMPTS,
            )
            # Apply preamble patch to corrected output too
            corrected = self._patch_preamble(corrected)

            validation = self.validator.validate_shader(corrected, target_platform="webgl")
            if not validation.passed:
                raise ValueError(
                    f"GLSL generation failed after {correction_attempts} "
                    f"self-correction attempt(s). Errors: {validation.errors}"
                )
            logger.info(
                "GLSLGenerator | self-correction succeeded after %d attempt(s)",
                correction_attempts,
            )

        compile_time_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "GLSLGenerator | complete | params=%d corrections=%d time_ms=%d",
            len(validation.parameters), correction_attempts, compile_time_ms,
        )

        return GLSLGenerationResult(
            shader_code=validation.clean_code,
            parameters=validation.parameters,
            matched_template_id=tmpl_id,
            compile_time_ms=compile_time_ms,
            self_correction_attempts=correction_attempts,
            validation=validation,
            template_confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _patch_preamble(self, raw: str) -> str:
        """
        Silently inject the GLSL ES 3.0 preamble lines if the LLM omitted them.
        Only patches if the directives are genuinely absent — does not duplicate.

        Patches applied (in order at the top of the extracted code block):
          1. #version 300 es
          2. precision highp float;
        """
        import re

        # Extract just the code block content for patching
        code_block_re = re.compile(
            r"(```glsl\s*)([\s\S]*?)(```)", re.IGNORECASE
        )
        match = code_block_re.search(raw)
        if not match:
            return raw  # No fenced block found — leave for validator to flag

        open_fence = match.group(1)
        code       = match.group(2)
        close_fence = match.group(3)

        prefix_lines: list[str] = []

        if not _has_glsl_preamble(code):
            prefix_lines.append("#version 300 es")
            logger.debug("GLSLGenerator | auto-injected '#version 300 es'")

        if not _has_precision_declaration(code):
            prefix_lines.append("precision highp float;")
            logger.debug("GLSLGenerator | auto-injected 'precision highp float;'")

        if prefix_lines:
            patched_code = "\n".join(prefix_lines) + "\n" + code
            return raw[: match.start()] + open_fence + patched_code + close_fence + raw[match.end():]

        return raw

    def preview_template(self, vibe: str) -> dict:
        """
        Return the matched GLSL template skeleton for a Vibe without calling
        the LLM. Useful for notebook exploration.

        Returns:
            Dict with: template_id, confidence, glsl_skeleton,
            default_parameters, parameter_hints.
        """
        tmpl_id, confidence, template = _match_glsl_template(vibe)
        if template is None:
            return {"template_id": "passthrough", "confidence": 0.0}
        return {
            "template_id":        tmpl_id,
            "confidence":         confidence,
            "glsl_skeleton":      template.get("glsl_skeleton", ""),
            "default_parameters": template.get("default_parameters", {}),
            "parameter_hints":    template.get("parameter_hints", {}),
        }