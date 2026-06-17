"""Background LLM analysis of uploaded hardware/teardown images."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import re
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

IMAGE_ANALYSIS_PROMPT = """You are an expert electronics hardware engineer and product teardown specialist with deep knowledge of PCB design, electronic components, mechanical assemblies, enclosures, and manufacturing.

Carefully analyze the provided image and extract every detail you can see:

1. **Image Type**
   - Is this a PCB, assembled product, enclosure, teardown, component close-up, or assembly photo?

2. **Enclosure & Mechanical Parts**
   - Enclosure material (plastic, metal, aluminum, etc.)
   - Color, finish, and form factor
   - Screws, standoffs, clips, hinges, gaskets, or mounting hardware
   - Vents, ports, cutouts, labels, or stickers on the enclosure

3. **PCB Details** (if visible)
   - PCB color, size estimation, layer type
   - All visible components:
     - Type (resistor, capacitor, IC, transistor, diode, inductor, fuse, relay, connector, etc.)
     - Reference designator (R1, C1, U1, etc.)
     - Value or part number if printed
     - Package type (SMD, through-hole, QFP, BGA, SOP, DIP, etc.)
   - Silkscreen labels, version numbers, date codes

4. **Connectors & Interfaces**
   - Internal and external connectors (USB, HDMI, JST, barrel jack, headers, antenna, etc.)
   - Cable harnesses, ribbon cables, or wire bundles

5. **ICs & Chips**
   - All chip markings, part numbers, manufacturer logos
   - Any identifiable microcontroller, processor, memory, or communication chip

6. **Power Components**
   - Battery, power supply section, transformers, heat sinks, thermal pads

7. **Labels & Markings**
   - Any text, barcodes, QR codes, serial numbers, certifications (CE, FCC, UL, RoHS), or branding visible anywhere

8. **Assembly & Build Quality**
   - Solder quality, any rework or modifications visible
   - Overall assembly method (screwed, snapped, glued)

List every single visible detail. If something is partially visible or uncertain, include it with a confidence level (high/medium/low). Do not skip anything."""

SUMMARIZE_ANALYSIS_PROMPT = """You are an expert electronics hardware engineer. Summarize the following image analysis for storage in a database.

Requirements:
- Output plain text only. Do NOT use markdown: no asterisks, no hash headers, no bullet dashes, no backticks, no numbered markdown lists.
- Write in clear, compact prose using short paragraphs or simple line breaks between topics.
- Preserve ALL hardware-critical facts: image type, enclosure material and color, PCB details, every component reference designator and value/part number mentioned, IC/chip markings, connector types, power components, labels, certifications, serial numbers, and assembly notes.
- Include confidence levels (high/medium/low) when the source text mentions uncertainty.
- Remove repetition and filler, but never drop a specific part number, designator, marking, or measurable detail.
- Target roughly 150-400 words unless the source has unusually many distinct components; in that case use up to 600 words rather than omitting parts.

Image analysis to summarize:

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


def _strip_markdown(text: str) -> str:
    """Best-effort cleanup if the model still returns markdown formatting."""
    cleaned = text.strip()
    cleaned = re.sub(r"^#{1,6}\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


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


def analyze_image(compressed_jpeg: bytes) -> str:
    """Send a vision-safe JPEG to the LLM and return the analysis text."""
    llm_jpeg = _prepare_image_for_llm(compressed_jpeg)
    b64 = base64.b64encode(llm_jpeg).decode("ascii")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": IMAGE_ANALYSIS_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
            ],
        }
    ]
    return _invoke_llm(messages, max_tokens=4096)


def summarize_analysis(raw_analysis: str) -> str:
    """Condense the full analysis to plain text while keeping critical hardware details."""
    messages = [
        {
            "role": "user",
            "content": f"{SUMMARIZE_ANALYSIS_PROMPT}{raw_analysis}",
        }
    ]
    summary = _invoke_llm(messages, max_tokens=2048)
    return _strip_markdown(summary)


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


def _run_image_analysis(file_id: str, compressed_data: bytes) -> None:
    try:
        logger.info("Starting background image analysis for file %s", file_id)
        raw_analysis = analyze_image(compressed_data)
        summary = summarize_analysis(raw_analysis)
        _update_analysis_status(file_id, "processed", summary)
        logger.info("Completed background image analysis for file %s", file_id)
    except Exception:
        logger.exception("Background image analysis failed for file %s", file_id)
        try:
            _update_analysis_status(file_id, "failed")
        except Exception:
            logger.exception("Failed to update analysis status for file %s", file_id)


def schedule_image_analysis(file_id: str, compressed_data: bytes) -> None:
    """Fire-and-forget: run LLM analysis in a thread pool without blocking the caller."""

    async def _task() -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _run_image_analysis, file_id, compressed_data)

    asyncio.create_task(_task())
