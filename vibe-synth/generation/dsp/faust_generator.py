# =============================================================================
# File: generation/dsp/faust_generator.py
# Purpose: Orchestrates end-to-end Faust DSP code generation for a single
#          request. Combines the ASTBuilder (template selection), PromptBuilder
#          (prompt assembly), LLMClient (code generation), and Validator
#          (safety checks) into a single callable pipeline. This module is the
#          primary entry point used by routes_dsp.py when serving generation
#          requests.
# =============================================================================

import time
import logging
from dataclasses import dataclass

from app.config import Settings
from app.models.request_models import GenerateDSPRequest
from app.services.llm_client import LLMClient, LLMError
from app.services.prompt_builder import PromptBuilder
from app.services.validator import Validator, ValidationResult
from generation.dsp.ast_builder import ASTBuilder, SignalAST

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class FaustGenerationResult:
    """
    Complete output of one Faust generation run.

    Attributes:
        faust_code:            Clean, validated Faust source code.
        parameters:            Parsed parameter manifest (list of dicts).
        ast:                   The SignalAST that guided generation.
        compile_time_ms:       Wall-clock time for the full pipeline.
        self_correction_attempts: Number of LLM self-correction rounds used.
        validation:            The final ValidationResult (always passed=True).
        ast_confidence:        Confidence score from the AST builder (0–1).
    """
    faust_code:               str
    parameters:               list[dict]
    ast:                      SignalAST
    compile_time_ms:          int
    self_correction_attempts: int
    validation:               ValidationResult
    ast_confidence:           float


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class FaustGenerator:
    """
    End-to-end Faust DSP generation pipeline.

    Pipeline stages:
      1. ASTBuilder  — select and compose DSP templates from the Vibe description
      2. PromptBuilder — assemble (system_prompt, user_prompt) with AST hint injected
      3. LLMClient.generate() — call the LLM for raw Faust code output
      4. Validator.validate_dsp() — static safety and structural checks
      5. LLMClient.self_correct() — fix compilation errors if validation fails
      6. Return FaustGenerationResult

    Usage:
        generator = FaustGenerator(settings, llm_client, prompt_builder, validator)
        result    = await generator.generate(request)
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
        self.ast_builder    = ASTBuilder()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def generate(self, request: GenerateDSPRequest) -> FaustGenerationResult:
        """
        Run the full generation pipeline for a DSP request.

        Args:
            request: Validated GenerateDSPRequest from the API layer.

        Returns:
            FaustGenerationResult with clean code, parameters, and metadata.

        Raises:
            LLMError:   If the LLM call fails and cannot be recovered.
            ValueError: If validation fails after all self-correction attempts.
        """
        t0 = time.perf_counter()

        # ------------------------------------------------------------------
        # Stage 1 — AST construction
        # ------------------------------------------------------------------
        ast = self.ast_builder.build(request.prompt)
        logger.info(
            "FaustGenerator | AST built | nodes=%d confidence=%.3f",
            len(ast.nodes), ast.confidence,
        )

        # ------------------------------------------------------------------
        # Stage 2 — Prompt assembly with AST hint
        # ------------------------------------------------------------------
        system_prompt, user_prompt = self.prompt_builder.build_dsp_prompts(request)

        # Inject the AST chain hint into the user prompt when confidence is
        # high enough to be useful (avoids misleading hints on poor matches)
        if ast.confidence >= 0.2 and ast.nodes:
            ast_hint = ast.to_faust_hint()
            user_prompt = f"{user_prompt}\n\n{ast_hint}"
            logger.debug("FaustGenerator | AST hint injected: %s", ast_hint)

        # ------------------------------------------------------------------
        # Stage 3 — LLM generation
        # ------------------------------------------------------------------
        logger.info("FaustGenerator | calling LLM for vibe='%s'", request.prompt[:60])
        raw_output = await self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        logger.debug("FaustGenerator | raw LLM output length: %d chars", len(raw_output))

        # ------------------------------------------------------------------
        # Stage 4 — Validation
        # ------------------------------------------------------------------
        validation = self.validator.validate_dsp(raw_output)
        correction_attempts = 0

        # ------------------------------------------------------------------
        # Stage 5 — Self-correction loop
        # ------------------------------------------------------------------
        if not validation.passed:
            logger.warning(
                "FaustGenerator | validation failed, entering self-correction | errors: %s",
                validation.errors,
            )
            correction_system = self.prompt_builder.build_correction_context("dsp")

            corrected_output, correction_attempts = await self.llm_client.self_correct(
                system_prompt=correction_system,
                original_user_prompt=user_prompt,
                failed_code=validation.clean_code,
                error_message="\n".join(validation.errors),
                max_attempts=self.MAX_CORRECTION_ATTEMPTS,
            )

            # Re-validate the corrected output
            validation = self.validator.validate_dsp(corrected_output)
            if not validation.passed:
                raise ValueError(
                    f"Faust generation failed after {correction_attempts} "
                    f"self-correction attempt(s). Final errors: {validation.errors}"
                )

            logger.info(
                "FaustGenerator | self-correction succeeded after %d attempt(s)",
                correction_attempts,
            )

        compile_time_ms = int((time.perf_counter() - t0) * 1000)

        logger.info(
            "FaustGenerator | generation complete | "
            "params=%d corrections=%d time_ms=%d",
            len(validation.parameters),
            correction_attempts,
            compile_time_ms,
        )

        return FaustGenerationResult(
            faust_code=validation.clean_code,
            parameters=validation.parameters,
            ast=ast,
            compile_time_ms=compile_time_ms,
            self_correction_attempts=correction_attempts,
            validation=validation,
            ast_confidence=ast.confidence,
        )

    # ------------------------------------------------------------------
    # Utility — generate from a raw vibe string (convenience wrapper)
    # ------------------------------------------------------------------

    async def generate_from_vibe(
        self,
        vibe:          str,
        sample_rate:   int = 44100,
        max_parameters: int = 8,
    ) -> FaustGenerationResult:
        """
        Convenience method that constructs a GenerateDSPRequest internally
        and calls generate(). Useful for scripting and batch experiments.

        Args:
            vibe:           Natural language Vibe description.
            sample_rate:    Target sample rate in Hz.
            max_parameters: Maximum number of exposed UI parameters.

        Returns:
            FaustGenerationResult — same as generate().
        """
        from app.models.request_models import InputType

        request = GenerateDSPRequest(
            prompt=vibe,
            input_type=InputType.audio_stream,
            sample_rate=sample_rate,
            max_parameters=max_parameters,
            force_refresh=True,
        )
        return await self.generate(request)

    # ------------------------------------------------------------------
    # Utility — inspect AST without calling the LLM
    # ------------------------------------------------------------------

    def inspect_ast(self, vibe: str) -> dict:
        """
        Return the AST for a Vibe description without calling the LLM.
        Useful for debugging template selection and modifier behaviour
        from the experiments notebooks.

        Args:
            vibe: Natural language Vibe description.

        Returns:
            Dict representation of the SignalAST (see SignalAST.to_dict()).
        """
        ast = self.ast_builder.build(vibe)
        return ast.to_dict()