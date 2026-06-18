"""Shared LLM analysis primitives for the insight pipeline.

This module centralises everything the image, document, and knowledge-base
pipelines have in common: the reverse-engineering prompts, the dual-output
(theory prose + structured JSON) parsing, retryable LLM invocation, and image
preparation for vision models. Images, document map/reduce, and KB merge all
build on these helpers so the prompt/format contract lives in one place.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from typing import Any

import litellm
from PIL import Image

from src.core.config import AWS_REGION, LLM_API_BASE, LLM_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

LLM_MAX_SIDE = 1024
LLM_MAX_RETRIES = 4
LLM_RETRY_BASE_DELAY = 2.0

# Delimiter lines the analysis/merge prompts must emit between the two outputs.
THEORY_DELIMITER = "=== OUTPUT 1: THEORY ANALYSIS SUMMARY ==="
STRUCTURED_DELIMITER = "=== OUTPUT 2: STRUCTURED ANALYSIS SUMMARY ==="

# Reference shape for the structured analysis. The model may add/remove keys to
# best fit the product, but this anchors the expected fields and nesting.
STRUCTURED_REFERENCE = """{
  "product": {"name": null, "type": null, "primary_function": null, "product_class": null, "target_market": null, "confidence": 0.0},
  "enclosure": {"material": null, "finish": null, "process": null, "dimensions_mm": {"length": null, "width": null, "height": null}, "wall_thickness_mm": null, "weight_g": null, "confidence": 0.0},
  "pcb": {"dimensions_mm": {"length": null, "width": null, "thickness": null}, "layer_count_estimate": null, "sided": null, "soldermask_color": null, "surface_finish": null, "mounting_holes": null, "confidence": 0.0},
  "components": [{"ref_des": null, "type": null, "package": null, "top_mark": null, "value": null, "qty_per_unit": 1, "location": null, "function": null, "mpn": null, "manufacturer": null, "confidence": 0.0}],
  "connectors_io": [{"ref_des": null, "type": null, "pin_count": null, "pitch_mm": null, "function": null, "confidence": 0.0}],
  "architecture_blocks": [{"block": null, "components": [], "description": null}],
  "design_observations": [],
  "assumptions": [],
  "extra_insights": {}
}"""

PRODUCT_ANALYSIS_PROMPT = f"""You are an expert electronics reverse-engineering and should-cost analyst. You are given images of a physical electronic product (PCB top/bottom/close-ups, enclosure, product/assembly, retail box, labels). You ONLY have the physical product — never assume access to Gerbers, CAD, schematics, or firmware source. Never invent values, markings, or prices; if something is not visible or determinable, mark it unknown. State estimates with a confidence (0-1) and cite the evidence (which image and what you saw).

Analyze the image exhaustively and produce TWO outputs, in this exact order, separated by the exact delimiter lines shown.

{THEORY_DELIMITER}
Write prose (GitHub-flavored Markdown). Explain:
- What the product most likely is and its primary function/use case.
- The system architecture: major functional blocks (power, MCU/processor, memory, RF/wireless, sensors, connectors, I/O, analog) and how they interconnect, inferred from the visible parts.
- The role of each major component (ICs, connectors, crystals, electromechanical parts) and why it's there.
- The enclosure: likely material, finish, manufacturing process, and how the assembly goes together.
- Notable design observations, target market, and rough product class/positioning.
Keep claims tied to evidence; flag inferences vs. things actually read off the board.

{STRUCTURED_DELIMITER}
Output a SINGLE valid JSON object only — no prose, no markdown fences, nothing before or after.

Use the following shape as a reference (add or drop keys to best fit this specific product; keep the overall structure):
{STRUCTURED_REFERENCE}

Be exhaustive on components: every IC, connector, crystal/oscillator, electromechanical part, LED, inductor, and electrolytic/tantalum cap is its own entry; group identical small passives (0402/0603 R/C) into one entry with total qty. Never lump the whole board into one entry. Transcribe top-marks EXACTLY as seen.

JSON rules:
- Must be strictly valid and parseable. Numbers as numbers (not strings); use null for unknown values, never guess.
- Every measurement, identification, and estimate should carry a confidence and evidence reference where it makes sense.
- Do not include MPN/pricing guesses you can't support from markings — leave the part unidentified if the top-mark isn't legible enough."""

# Document map step: extract structured facts from ONE chunk of document text.
DOC_MAP_PROMPT = f"""You are an expert electronics reverse-engineering and should-cost analyst. You are given ONE excerpt from a document about a physical electronic product (a datasheet, manual, spec sheet, retail box text, or certification document). Extract only what THIS excerpt states. Never invent values; if the excerpt does not state something, omit it. Cite confidence (0-1) where it makes sense.

Produce TWO outputs, in this exact order, separated by the exact delimiter lines shown.

{THEORY_DELIMITER}
A short Markdown summary of what this excerpt reveals about the product (function, electrical specs, components, ratings, certifications, mechanical details). Keep it to what the text actually says.

{STRUCTURED_DELIMITER}
A SINGLE valid JSON object only — no prose, no fences. Use this reference shape (drop keys not addressed by this excerpt):
{STRUCTURED_REFERENCE}

Only include fields this excerpt supports. Numbers as numbers; null/omit for unknowns.

DOCUMENT EXCERPT:
{{chunk}}"""

# Document reduce step: consolidate the per-chunk partials into one analysis.
DOC_REDUCE_PROMPT = f"""You are an expert electronics reverse-engineering and should-cost analyst. You are given several PARTIAL analyses, each extracted from a different excerpt of the SAME document about ONE physical electronic product. Consolidate them into a single, de-duplicated, coherent analysis of the whole document.

Merge rules:
- Combine evidence; do not duplicate facts that appear in multiple partials.
- Prefer higher-confidence / more specific readings when partials disagree.
- Never invent values; keep unknowns as null. Keep component/spec lists exhaustive and de-duplicated.

Produce TWO outputs, in this exact order, separated by the exact delimiter lines shown.

{THEORY_DELIMITER}
The consolidated theory analysis as GitHub-flavored Markdown prose.

{STRUCTURED_DELIMITER}
A SINGLE valid JSON object only — no prose, no fences. Same reference shape as the partials; numbers as numbers; null for unknowns.

PARTIAL ANALYSES (JSON-encoded list of {{theory, structured}}):
{{partials}}"""

MERGE_CONTEXT_PROMPT = f"""You are an expert electronics reverse-engineering and should-cost analyst maintaining a single cumulative understanding of ONE physical electronic product as new uploads (images and documents) of it are analyzed over time.

You are given the project's EXISTING accumulated analysis (theory prose + structured JSON) and the NEW analysis just produced from an additional upload of the SAME product. Merge them into one consolidated, up-to-date analysis that reflects everything known so far.

Merge rules:
- The merged result MUST be a SUPERSET of the EXISTING analysis. This is the most important rule: never drop, shorten, or summarize away anything already recorded. Every component, connector, architecture block, design observation, assumption, spec, marking, and confidence/evidence reference already present MUST survive into the output unchanged unless the new upload gives a strictly better reading of that exact item.
- Treat all inputs as views of the same product. Combine evidence; do not duplicate facts that appear in both.
- Add anything new the latest upload reveals (newly visible components, markings, connectors, enclosure details, electrical specs) ON TOP OF everything already known.
- When the new upload gives a clearer reading than before (e.g. a legible top-mark or a datasheet spec that was previously unknown), prefer the higher-confidence reading and update that one field — but do not delete the surrounding context.
- Never invent values, markings, or prices. Keep anything still unknown as unknown/null. Preserve confidences (0-1) and evidence references.
- Keep the component list exhaustive and de-duplicated: one entry per distinct IC, connector, crystal/oscillator, electromechanical part, LED, inductor, electrolytic/tantalum cap; group identical small passives into one entry with total qty. The merged component list must be at least as long as the existing one.
- If the new upload adds nothing to a particular field or section, copy the existing value through verbatim. The theory prose must stay at least as detailed as the existing theory.

Produce TWO outputs, in this exact order, separated by the exact delimiter lines shown.

{THEORY_DELIMITER}
The consolidated theory analysis as GitHub-flavored Markdown prose (same scope and style as before), reflecting the combined knowledge.

{STRUCTURED_DELIMITER}
A SINGLE valid JSON object only — no prose, no markdown fences, nothing before or after. The consolidated structured analysis. Must be strictly valid and parseable; numbers as numbers; null for unknowns; carry confidence and evidence where it makes sense.

EXISTING THEORY ANALYSIS:
{{existing_theory}}

EXISTING STRUCTURED ANALYSIS (JSON):
{{existing_structured}}

NEW THEORY ANALYSIS:
{{new_theory}}

NEW STRUCTURED ANALYSIS (JSON):
{{new_structured}}
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


def is_retryable_llm_error(exc: Exception) -> bool:
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


def invoke_llm(messages: list[dict[str, Any]], *, max_tokens: int) -> str:
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
            return extract_text(content)
        except Exception as exc:
            last_error = exc
            if attempt >= LLM_MAX_RETRIES - 1 or not is_retryable_llm_error(exc):
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


def prepare_image_for_llm(compressed_jpeg: bytes) -> bytes:
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


def extract_text(content: Any) -> str:
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


def split_dual_output(text: str) -> tuple[str, str]:
    """Split a dual-output response into (theory_markdown, structured_json_text)."""
    structured_marker = re.search(
        r"^={2,}\s*OUTPUT\s*2\b.*$", text, flags=re.MULTILINE | re.IGNORECASE
    )
    if structured_marker:
        theory_part = text[: structured_marker.start()]
        structured_part = text[structured_marker.end() :]
    else:
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


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort parse of a JSON object from model output (tolerates fences)."""
    if not text:
        return None
    cleaned = text.strip()
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
    """Analyze an image and return (theory_markdown, structured_dict)."""
    llm_jpeg = prepare_image_for_llm(compressed_jpeg)
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
    raw = invoke_llm(messages, max_tokens=8192)
    theory, structured_text = split_dual_output(raw)
    structured = parse_json_object(structured_text)
    if structured is None and structured_text:
        logger.warning("Could not parse structured JSON from image analysis output")
    return theory, structured


def analyze_document_chunk(chunk: str) -> tuple[str, dict[str, Any] | None]:
    """Map step: extract a partial (theory, structured) from one document chunk."""
    prompt = DOC_MAP_PROMPT.format(chunk=chunk)
    raw = invoke_llm([{"role": "user", "content": prompt}], max_tokens=4096)
    theory, structured_text = split_dual_output(raw)
    return theory, parse_json_object(structured_text)


def reduce_document_partials(
    partials: list[dict[str, Any]],
) -> tuple[str, dict[str, Any] | None]:
    """Reduce step: consolidate per-chunk partials into one (theory, structured)."""
    payload = json.dumps(partials, ensure_ascii=False, indent=2)
    prompt = DOC_REDUCE_PROMPT.format(partials=payload)
    raw = invoke_llm([{"role": "user", "content": prompt}], max_tokens=8192)
    theory, structured_text = split_dual_output(raw)
    return theory, parse_json_object(structured_text)


def merge_dual(
    existing_theory: str,
    existing_structured: str,
    new_theory: str,
    new_structured: str,
) -> tuple[str, dict[str, Any] | None]:
    """Merge an existing (theory, structured) with a new one into a consolidated pair."""
    prompt = MERGE_CONTEXT_PROMPT.format(
        existing_theory=existing_theory or "(none yet)",
        existing_structured=existing_structured or "{}",
        new_theory=new_theory or "(none)",
        new_structured=new_structured or "{}",
    )
    raw = invoke_llm([{"role": "user", "content": prompt}], max_tokens=8192)
    theory, structured_text = split_dual_output(raw)
    return theory, parse_json_object(structured_text)
