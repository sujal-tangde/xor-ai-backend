"""Background LLM analysis of uploaded hardware/teardown images."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
import threading
import time
from typing import Any

import litellm
from PIL import Image

from src.core.config import AWS_REGION, LLM_API_BASE, LLM_API_KEY, LLM_MODEL
from src.services.file_storage import get_supabase

logger = logging.getLogger(__name__)

LLM_MAX_SIDE = 1024
LLM_MAX_RETRIES = 4
LLM_RETRY_BASE_DELAY = 2.0

# Delimiter lines the analysis/merge prompts must emit between the two outputs.
THEORY_DELIMITER = "=== OUTPUT 1: THEORY ANALYSIS SUMMARY ==="
STRUCTURED_DELIMITER = "=== OUTPUT 2: STRUCTURED ANALYSIS SUMMARY ==="

PRODUCT_ANALYSIS_PROMPT = """You are an expert electronics reverse-engineering and should-cost analyst. You are given images of a physical electronic product (PCB top/bottom/close-ups, enclosure, product/assembly, retail box, labels) and any provided files (manuals, datasheets, box text). You ONLY have the physical product — never assume access to Gerbers, CAD, schematics, or firmware source. Never invent values, markings, or prices; if something is not visible or determinable, mark it unknown. State estimates with a confidence (0-1) and cite the evidence (which image/file and what you saw).

Analyze ALL provided images and files exhaustively and produce TWO outputs, in this exact order, separated by the exact delimiter lines shown.

=== OUTPUT 1: THEORY ANALYSIS SUMMARY ===
Write prose (GitHub-flavored Markdown). Explain:
- What the product most likely is and its primary function/use case.
- The system architecture: major functional blocks (power, MCU/processor, memory, RF/wireless, sensors, connectors, I/O, analog) and how they interconnect, inferred from the visible parts.
- The role of each major component (ICs, connectors, crystals, electromechanical parts) and why it's there.
- The enclosure: likely material, finish, manufacturing process, and how the assembly goes together.
- Notable design observations, target market, and rough product class/positioning.
Keep claims tied to evidence; flag inferences vs. things actually read off the board.

=== OUTPUT 2: STRUCTURED ANALYSIS SUMMARY ===
Output a SINGLE valid JSON object only — no prose, no markdown fences, nothing before or after.

Design the schema yourself to best fit what this specific product reveals. There is no fixed template — choose the keys, nesting, and arrays that most faithfully and completely capture the structured facts (product identity, enclosure, PCB, exhaustive component list, connectors/IO, architecture blocks, dimensions, and anything else relevant). Be exhaustive on components: every IC, connector, crystal/oscillator, electromechanical part, LED, inductor, and electrolytic/tantalum cap is its own entry; group identical small passives (0402/0603 R/C) into one entry with total qty. Never lump the whole board into one entry. Transcribe top-marks EXACTLY as seen.

JSON rules:
- Must be strictly valid and parseable. Numbers as numbers (not strings); use null for unknown values, never guess.
- Every measurement, identification, and estimate should carry a confidence and evidence reference where it makes sense.
- Include unreadable items, requested follow-up evidence (close-up/microscope photos, calipers, multimeter, weight), and any assumptions made.
- Do not include MPN/pricing guesses you can't support from markings — leave the part unidentified if the top-mark isn't legible enough."""

MERGE_CONTEXT_PROMPT = """You are an expert electronics reverse-engineering and should-cost analyst maintaining a single cumulative understanding of ONE physical electronic product as new images of it are analyzed over time.

You are given the project's EXISTING accumulated analysis (theory prose + structured JSON) and the NEW analysis just produced from an additional image of the SAME product. Merge them into one consolidated, up-to-date analysis that reflects everything known so far.

Merge rules:
- Treat all inputs as views of the same product. Combine evidence; do not duplicate facts that appear in both.
- Add anything new the latest image reveals (newly visible components, markings, connectors, enclosure details).
- When the new image gives a clearer reading than before (e.g. a legible top-mark that was previously unknown), prefer the higher-confidence reading and update it.
- Never invent values, markings, or prices. Keep anything still unknown as unknown/null. Preserve confidences (0-1) and evidence references.
- Keep the component list exhaustive and de-duplicated: one entry per distinct IC, connector, crystal/oscillator, electromechanical part, LED, inductor, electrolytic/tantalum cap; group identical small passives into one entry with total qty.

Produce TWO outputs, in this exact order, separated by the exact delimiter lines shown.

=== OUTPUT 1: THEORY ANALYSIS SUMMARY ===
The consolidated theory analysis as GitHub-flavored Markdown prose (same scope and style as before), reflecting the combined knowledge.

=== OUTPUT 2: STRUCTURED ANALYSIS SUMMARY ===
A SINGLE valid JSON object only — no prose, no markdown fences, nothing before or after. The consolidated structured analysis. Must be strictly valid and parseable; numbers as numbers; null for unknowns; carry confidence and evidence where it makes sense.

EXISTING THEORY ANALYSIS:
{existing_theory}

EXISTING STRUCTURED ANALYSIS (JSON):
{existing_structured}

NEW IMAGE THEORY ANALYSIS:
{new_theory}

NEW IMAGE STRUCTURED ANALYSIS (JSON):
{new_structured}
"""


def _llm_completion_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": LLM_MODEL,
        "api_key": LLM_API_KEY or None,
        "aws_region_name": AWS_REGION,
    }
    if LLM_API_BASE:
        kwargs["api_base"] = LLM_API_BASE
    return kwargs


def _is_retryable_llm_error(exc: Exception) -> bool:
    name = type(exc).__name__
    if name in {"ServiceUnavailableError", "InternalServerError", "APIConnectionError", "Timeout"}:
        return True
    message = str(exc).lower()
    return any(
        phrase in message
        for phrase in (
            "unexpected error",
            "unable to process your request",
            "try your request again",
            "throttling",
            "rate limit",
            "timeout",
        )
    )


def _invoke_llm(messages: list[dict[str, Any]], *, max_tokens: int) -> str:
    last_error: Exception | None = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            response = litellm.completion(
                **_llm_completion_kwargs(),
                messages=messages,
                max_tokens=max_tokens,
                stream=False,
            )
            content = response.choices[0].message.content
            return _extract_text(content)
        except Exception as exc:
            last_error = exc
            if attempt >= LLM_MAX_RETRIES - 1 or not _is_retryable_llm_error(exc):
                raise
            delay = LLM_RETRY_BASE_DELAY * (2**attempt)
            logger.warning(
                "LLM call failed (attempt %s/%s), retrying in %.1fs: %s",
                attempt + 1,
                LLM_MAX_RETRIES,
                delay,
                exc,
            )
            time.sleep(delay)
    raise last_error or RuntimeError("LLM call failed")


def _prepare_image_for_llm(compressed_jpeg: bytes) -> bytes:
    """Downscale for Bedrock vision — storage compression keeps full resolution."""
    img = Image.open(io.BytesIO(compressed_jpeg))
    if img.mode != "RGB":
        img = img.convert("RGB")

    width, height = img.size
    scale = min(1.0, LLM_MAX_SIDE / max(width, height))
    if scale < 1.0:
        img = img.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.LANCZOS,
        )

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80, optimize=True)
    return buf.getvalue()


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content) if content else ""


def _split_dual_output(text: str) -> tuple[str, str]:
    """Split a dual-output LLM response into (theory_markdown, structured_json_text).

    The prompts ask the model to separate the two outputs with the OUTPUT 1 /
    OUTPUT 2 delimiter lines. We split on the OUTPUT 2 marker and strip the
    OUTPUT 1 header off the theory half; both halves degrade gracefully if the
    model omits or reformats a delimiter.
    """
    structured_marker = re.search(
        r"^={2,}\s*OUTPUT\s*2\b.*$", text, flags=re.MULTILINE | re.IGNORECASE
    )
    if structured_marker:
        theory_part = text[: structured_marker.start()]
        structured_part = text[structured_marker.end() :]
    else:
        # No OUTPUT 2 delimiter — fall back to splitting at the first JSON object.
        brace = text.find("{")
        if brace == -1:
            return text.strip(), ""
        theory_part = text[:brace]
        structured_part = text[brace:]

    theory = re.sub(
        r"^={2,}\s*OUTPUT\s*1\b.*$",
        "",
        theory_part,
        flags=re.MULTILINE | re.IGNORECASE,
    ).strip()
    return theory, structured_part.strip()


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort parse of a JSON object from model output (tolerates fences)."""
    if not text:
        return None
    cleaned = text.strip()
    # Strip ```json ... ``` fences if the model added them despite instructions.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def analyze_image_dual(compressed_jpeg: bytes) -> tuple[str, dict[str, Any] | None]:
    """Analyze an image and return (theory_markdown, structured_dict).

    Sends the vision-safe JPEG with the reverse-engineering prompt that produces
    both a prose theory summary and a structured JSON summary, then splits and
    parses the response. ``structured_dict`` is ``None`` if the JSON half could
    not be parsed.
    """
    llm_jpeg = _prepare_image_for_llm(compressed_jpeg)
    b64 = base64.b64encode(llm_jpeg).decode("ascii")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PRODUCT_ANALYSIS_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
            ],
        }
    ]
    raw = _invoke_llm(messages, max_tokens=8192)
    theory, structured_text = _split_dual_output(raw)
    structured = _parse_json_object(structured_text)
    if structured is None and structured_text:
        logger.warning("Could not parse structured JSON from image analysis output")
    return theory, structured


def _merge_contexts(
    existing_theory: str,
    existing_structured: str,
    new_theory: str,
    new_structured: str,
) -> tuple[str, dict[str, Any] | None]:
    """LLM-merge the existing project context with a new image's analysis."""
    prompt = MERGE_CONTEXT_PROMPT.format(
        existing_theory=existing_theory or "(none yet)",
        existing_structured=existing_structured or "{}",
        new_theory=new_theory or "(none)",
        new_structured=new_structured or "{}",
    )
    raw = _invoke_llm([{"role": "user", "content": prompt}], max_tokens=8192)
    theory, structured_text = _split_dual_output(raw)
    structured = _parse_json_object(structured_text)
    return theory, structured


def _update_analysis_status(
    file_id: str,
    status: str,
    analysis: str | None = None,
) -> None:
    client = get_supabase()
    payload: dict[str, Any] = {"image_analysis_status": status}
    if analysis is not None:
        payload["image_analysis"] = analysis
    client.table("uploaded_files").update(payload).eq("id", file_id).execute()


def _get_project_contexts(project_id: str) -> tuple[str, str]:
    """Return the project's current (context, structured_context) as strings."""
    client = get_supabase()
    result = (
        client.table("projects")
        .select("context, structured_context")
        .eq("id", project_id)
        .execute()
    )
    row = result.data[0] if result.data else {}
    return (row.get("context") or "", row.get("structured_context") or "")


def _update_project_contexts(
    project_id: str, context: str, structured_context: str
) -> None:
    client = get_supabase()
    client.table("projects").update(
        {
            "context": context,
            "structured_context": structured_context,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        }
    ).eq("id", project_id).execute()


# Serialize read-modify-write of a project's context so concurrent image uploads
# to the same project don't clobber each other's merge (each analysis runs in a
# background executor thread).
_project_locks: dict[str, threading.Lock] = {}
_project_locks_guard = threading.Lock()


def _project_lock(project_id: str) -> threading.Lock:
    with _project_locks_guard:
        lock = _project_locks.get(project_id)
        if lock is None:
            lock = threading.Lock()
            _project_locks[project_id] = lock
        return lock


def merge_project_context(
    project_id: str, new_theory: str, new_structured: dict[str, Any] | None
) -> None:
    """Fold a new image's analysis into the project's accumulated context.

    For the first image the new analysis simply becomes the context; afterwards
    it is LLM-merged with the existing context into a single consolidated theory
    summary and structured JSON object, then written back.
    """
    new_structured_text = (
        json.dumps(new_structured, ensure_ascii=False, indent=2)
        if new_structured is not None
        else ""
    )

    with _project_lock(project_id):
        existing_theory, existing_structured = _get_project_contexts(project_id)
        existing_theory = existing_theory.strip()
        existing_structured = existing_structured.strip()

        if not existing_theory and not existing_structured:
            merged_theory = new_theory.strip()
            merged_structured_text = new_structured_text
        else:
            merged_theory, merged_structured = _merge_contexts(
                existing_theory,
                existing_structured,
                new_theory,
                new_structured_text,
            )
            merged_theory = merged_theory.strip() or existing_theory
            if merged_structured is not None:
                merged_structured_text = json.dumps(
                    merged_structured, ensure_ascii=False, indent=2
                )
            else:
                # Merge produced unparseable JSON — keep the existing structured
                # context rather than corrupting it.
                logger.warning(
                    "Context merge returned unparseable JSON for project %s; "
                    "keeping previous structured_context",
                    project_id,
                )
                merged_structured_text = existing_structured or new_structured_text

        _update_project_contexts(project_id, merged_theory, merged_structured_text)


def _run_image_analysis(
    file_id: str, compressed_data: bytes, project_id: str | None
) -> None:
    try:
        logger.info("Starting background image analysis for file %s", file_id)
        theory, structured = analyze_image_dual(compressed_data)
        # The theory prose doubles as the per-file analysis surfaced by the
        # get_image_analysis tool.
        _update_analysis_status(file_id, "processed", theory)
        logger.info("Completed background image analysis for file %s", file_id)
    except Exception:
        logger.exception("Background image analysis failed for file %s", file_id)
        try:
            _update_analysis_status(file_id, "failed")
        except Exception:
            logger.exception("Failed to update analysis status for file %s", file_id)
        return

    if not project_id:
        return
    try:
        merge_project_context(project_id, theory, structured)
        logger.info("Merged image %s into project %s context", file_id, project_id)
    except Exception:
        logger.exception(
            "Failed to merge image %s into project %s context", file_id, project_id
        )


def schedule_image_analysis(
    file_id: str, compressed_data: bytes, project_id: str | None = None
) -> None:
    """Fire-and-forget: run LLM analysis in a thread pool without blocking the caller."""

    async def _task() -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, _run_image_analysis, file_id, compressed_data, project_id
        )

    asyncio.create_task(_task())
