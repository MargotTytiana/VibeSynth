# =============================================================================
# File: generation/shader/wgsl_generator.py
# Purpose: Orchestrates end-to-end WGSL (WebGPU) fragment shader generation.
#          Combines shader template selection from shader_templates.json,
#          prompt assembly via PromptBuilder, LLM generation via LLMClient,
#          and static validation via Validator into a single async pipeline.
#          Mirrors faust_generator.py in structure but targets WebGPU shaders.
# =============================================================================

import json
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

from app.config import Settings
from app.models.request_models import GenerateShaderRequest, TargetPlatform
from app.services.llm_client import LLMClient, LLMError
from app.services.prompt_builder import PromptBuilder
from app.services.validator import Validator, ValidationResult

logger = logging.getLogger(__name__)

SHADER_TEMPLATES_PATH = Path("generation/shader/shader_templates.json")


# ---------------------------------------------------------------------------
# Template loader
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_shader_templates() -> list[dict]:
    """Load and cache the shader template list from shader_templates.json."""
    if not SHADER_TEMPLATES_PATH.exists():
        logger.warning(
            "Shader templates not found at %s — template hints disabled.",
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
class WGSLGenerationResult:
    """
    Complete output of one WGSL shader generation run.

    Attributes:
        shader_code:              Clean, validated WGSL source.
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
# Template matcher
# ---------------------------------------------------------------------------

def _match_template(vibe: str) -> tuple[str, float, dict | None]:
    """
    Score all shader templates against the Vibe and return the best match.

    Scoring mirrors ASTBuilder: exact token match +2.0, substring +0.5,
    normalised by token count.

    Returns:
        (template_id, confidence, template_dict | None)
    """
    templates = _load_shader_templates()
    if not templates:
        return "passthrough", 0.0, None

    # Tokenise
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
        # Fall back to passthrough
        passthrough = next((t for t in templates if t["id"] == "passthrough"), None)
        return "passthrough", 0.0, passthrough

    return best_template["id"], round(best_score, 4), best_template


# ---------------------------------------------------------------------------
# Hint builder
# ---------------------------------------------------------------------------

def _build_template_hint(template: dict | None) -> str:
    """
    Render a short natural-language hint describing the matched template
    to be injected into the LLM user prompt.
    """
    if template is None or template.get("id") == "passthrough":
        return ""
    return (
        f"Suggested base effect: {template.get('name', template['id'])} — "
        f"{template.get('description', '')} "
        f"(refine and extend to match the Vibe description more precisely)."
    )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class WGSLGenerator:
    """
    End-to-end WGSL fragment shader generation pipeline.

    Pipeline stages:
      1. Template matching — find the closest archetype from shader_templates.json
      2. Prompt assembly  — build (system, user) prompts with template hint
      3. LLM generation   — call the LLM for raw WGSL source
      4. Validation       — static safety and structure checks
      5. Self-correction  — fix validation errors via LLM
      6. Return WGSLGenerationResult

    Usage:
        gen    = WGSLGenerator(settings, llm_client, prompt_builder, validator)
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

    async def generate(self, request: GenerateShaderRequest) -> WGSLGenerationResult:
        """
        Run the full WGSL generation pipeline.

        Args:
            request: Validated GenerateShaderRequest with target_platform=webgpu.

        Returns:
            WGSLGenerationResult with clean shader code and parameter manifest.

        Raises:
            ValueError:  If validation fails after all self-correction attempts.
            LLMError:    If the LLM call fails unrecoverably.
        """
        t0 = time.perf_counter()

        # ------------------------------------------------------------------
        # Stage 1 — template matching
        # ------------------------------------------------------------------
        tmpl_id, confidence, template = _match_template(request.prompt)
        logger.info(
            "WGSLGenerator | template matched: %s (confidence=%.3f)",
            tmpl_id, confidence,
        )

        # ------------------------------------------------------------------
        # Stage 2 — prompt assembly
        # ------------------------------------------------------------------
        system_prompt, user_prompt = self.prompt_builder.build_shader_prompts(request)

        if confidence >= 0.2 and template:
            hint = _build_template_hint(template)
            if hint:
                user_prompt = f"{user_prompt}\n\n{hint}"
                logger.debug("WGSLGenerator | template hint injected: %s", hint[:120])

        # ------------------------------------------------------------------
        # Stage 3 — LLM generation
        # ------------------------------------------------------------------
        logger.info(
            "WGSLGenerator | calling LLM for vibe='%s'", request.prompt[:60]
        )
        raw_output = await self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # ------------------------------------------------------------------
        # Stage 4 — validation
        # ------------------------------------------------------------------
        validation = self.validator.validate_shader(
            raw_output=raw_output,
            target_platform="webgpu",
        )
        correction_attempts = 0

        # ------------------------------------------------------------------
        # Stage 5 — self-correction
        # ------------------------------------------------------------------
        if not validation.passed:
            logger.warning(
                "WGSLGenerator | validation failed — self-correcting. Errors: %s",
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
            validation = self.validator.validate_shader(corrected, target_platform="webgpu")
            if not validation.passed:
                raise ValueError(
                    f"WGSL generation failed after {correction_attempts} "
                    f"self-correction attempt(s). Errors: {validation.errors}"
                )
            logger.info(
                "WGSLGenerator | self-correction succeeded after %d attempt(s)",
                correction_attempts,
            )

        compile_time_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "WGSLGenerator | complete | params=%d corrections=%d time_ms=%d",
            len(validation.parameters), correction_attempts, compile_time_ms,
        )

        return WGSLGenerationResult(
            shader_code=validation.clean_code,
            parameters=validation.parameters,
            matched_template_id=tmpl_id,
            compile_time_ms=compile_time_ms,
            self_correction_attempts=correction_attempts,
            validation=validation,
            template_confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Utility — preview template skeleton without calling LLM
    # ------------------------------------------------------------------

    def preview_template(self, vibe: str) -> dict:
        """
        Return the matched template skeleton for a Vibe without calling
        the LLM. Useful for notebook exploration and UI previews.

        Returns:
            Dict with keys: template_id, confidence, wgsl_skeleton,
            default_parameters, parameter_hints.
        """
        tmpl_id, confidence, template = _match_template(vibe)
        if template is None:
            return {"template_id": "passthrough", "confidence": 0.0}
        return {
            "template_id":        tmpl_id,
            "confidence":         confidence,
            "wgsl_skeleton":      template.get("wgsl_skeleton", ""),
            "default_parameters": template.get("default_parameters", {}),
            "parameter_hints":    template.get("parameter_hints", {}),
        }