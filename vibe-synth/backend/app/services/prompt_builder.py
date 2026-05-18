# =============================================================================
# File: backend/app/services/prompt_builder.py
# Purpose: Assembles the final system and user prompts sent to the LLM for
#          each generation request. Loads base prompt templates from disk
#          (generation/prompts/), injects runtime context (target platform,
#          parameter constraints, few-shot examples), and returns a clean
#          (system_prompt, user_prompt) pair ready for LLMClient.generate().
# =============================================================================

import json
import logging
from pathlib import Path
from functools import lru_cache

from app.config import Settings
from app.models.request_models import GenerateDSPRequest, GenerateShaderRequest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths to prompt template files (relative to project root)
# ---------------------------------------------------------------------------
PROMPTS_DIR      = Path("generation/prompts")
DSP_SYSTEM_FILE  = PROMPTS_DIR / "system_dsp.txt"
SHADER_SYSTEM_FILE = PROMPTS_DIR / "system_shader.txt"
FEW_SHOT_FILE    = PROMPTS_DIR / "few_shot_examples.json"


@lru_cache(maxsize=1)
def _load_text(path: Path) -> str:
    """
    Load and cache a plain-text prompt template from disk.
    Cached so repeated requests do not hit the filesystem each time.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt template not found: {path}. "
            "Ensure generation/prompts/ files are present."
        )
    content = path.read_text(encoding="utf-8").strip()
    logger.debug("Loaded prompt template: %s (%d chars)", path.name, len(content))
    return content


@lru_cache(maxsize=1)
def _load_few_shots() -> dict:
    """
    Load and cache the few-shot examples JSON file.
    Returns the parsed dict; keys are 'dsp' and 'shader'.
    """
    if not FEW_SHOT_FILE.exists():
        logger.warning("Few-shot examples file not found at %s — skipping.", FEW_SHOT_FILE)
        return {"dsp": [], "shader": []}
    with FEW_SHOT_FILE.open(encoding="utf-8") as f:
        data = json.load(f)
    logger.debug("Loaded %d DSP and %d shader few-shot examples.",
                 len(data.get("dsp", [])), len(data.get("shader", [])))
    return data


def _format_few_shots(examples: list[dict], max_examples: int = 2) -> str:
    """
    Render a list of few-shot example dicts into a plain-text block that can
    be appended to a system prompt.

    Each example dict must have keys: 'vibe' (str) and 'output' (str).
    Only the first max_examples entries are included to stay within token budget.
    """
    if not examples:
        return ""

    lines = ["--- FEW-SHOT EXAMPLES (do not reproduce verbatim) ---"]
    for i, ex in enumerate(examples[:max_examples], start=1):
        lines.append(f"\nExample {i}:")
        lines.append(f"  Vibe: {ex.get('vibe', '')}")
        lines.append(f"  Output:\n{ex.get('output', '')}")
    lines.append("--- END OF EXAMPLES ---")
    return "\n".join(lines)


class PromptBuilder:
    """
    Constructs (system_prompt, user_prompt) pairs for DSP and shader
    generation requests.

    The system prompt carries:
      - The base instruction template loaded from disk
      - Relevant few-shot examples
      - Hard constraints derived from the request (max parameters, platform)

    The user prompt carries:
      - The caller's natural language Vibe description
      - Any additional runtime context (sample rate, input type)
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # ------------------------------------------------------------------
    # DSP prompt
    # ------------------------------------------------------------------

    def build_dsp_prompts(self, request: GenerateDSPRequest) -> tuple[str, str]:
        """
        Build and return (system_prompt, user_prompt) for a DSP generation
        request.

        The system prompt instructs the LLM to output valid Faust DSP code
        only, along with a JSON parameter manifest block.
        """
        base_system = _load_text(DSP_SYSTEM_FILE)
        few_shots   = _load_few_shots().get("dsp", [])
        few_shot_block = _format_few_shots(few_shots)

        constraints = (
            f"HARD CONSTRAINTS:\n"
            f"- Output language: Faust (functional audio programming language)\n"
            f"- Maximum exposed parameters: {request.max_parameters}\n"
            f"- Target sample rate: {request.sample_rate} Hz\n"
            f"- The algorithm MUST be BIBO-stable (Bounded-Input Bounded-Output)\n"
            f"- No infinite loops or unbounded recursion\n"
            f"- After the Faust code block, output a JSON block tagged "
            f"  ```json-params``` containing the parameter manifest\n"
        )

        system_prompt = "\n\n".join(
            part for part in [base_system, few_shot_block, constraints] if part
        )

        user_prompt = (
            f"Generate a Faust DSP algorithm for the following Vibe description:\n\n"
            f"\"{request.prompt}\"\n\n"
            f"Input type: {request.input_type.value}\n"
            f"Sample rate: {request.sample_rate} Hz\n"
        )

        logger.debug(
            "Built DSP prompts | system=%d chars user=%d chars",
            len(system_prompt), len(user_prompt),
        )
        return system_prompt, user_prompt

    # ------------------------------------------------------------------
    # Shader prompt
    # ------------------------------------------------------------------

    def build_shader_prompts(self, request: GenerateShaderRequest) -> tuple[str, str]:
        """
        Build and return (system_prompt, user_prompt) for a shader generation
        request.

        The system prompt instructs the LLM to output a valid WGSL or GLSL
        fragment shader, depending on request.target_platform.
        """
        base_system = _load_text(SHADER_SYSTEM_FILE)
        few_shots   = _load_few_shots().get("shader", [])
        few_shot_block = _format_few_shots(few_shots)

        language = "WGSL" if request.target_platform.value == "webgpu" else "GLSL ES 3.0"

        constraints = (
            f"HARD CONSTRAINTS:\n"
            f"- Output language: {language}\n"
            f"- Target platform: {request.target_platform.value}\n"
            f"- Maximum exposed parameters: {request.max_parameters}\n"
            f"- All loops MUST have a compile-time constant iteration count "
            f"  (loop unrolling enforced — no dynamic loop bounds)\n"
            f"- No recursion of any kind\n"
            f"- After the shader code block, output a JSON block tagged "
            f"  ```json-params``` containing the parameter manifest\n"
        )

        system_prompt = "\n\n".join(
            part for part in [base_system, few_shot_block, constraints] if part
        )

        user_prompt = (
            f"Generate a {language} fragment shader for the following Vibe description:\n\n"
            f"\"{request.prompt}\"\n\n"
            f"Input type: {request.input_type.value}\n"
            f"Target platform: {request.target_platform.value}\n"
        )

        logger.debug(
            "Built shader prompts | system=%d chars user=%d chars",
            len(system_prompt), len(user_prompt),
        )
        return system_prompt, user_prompt

    # ------------------------------------------------------------------
    # Self-correction prompt (delegates to LLMClient.self_correct)
    # ------------------------------------------------------------------

    def build_correction_context(self, generation_type: str) -> str:
        """
        Return the appropriate system prompt to use during a self-correction
        loop — same base template as the original generation, so the model
        stays in the right output mode.
        """
        if generation_type == "dsp":
            return _load_text(DSP_SYSTEM_FILE)
        if generation_type == "shader":
            return _load_text(SHADER_SYSTEM_FILE)
        raise ValueError(
            f"Unknown generation_type '{generation_type}'. Expected 'dsp' or 'shader'."
        )