"""LLM intent router for chat turns.

A single cheap/fast LLM call classifies the latest user message into one of four
intents. This REPLACES the old brittle regex/keyword intent detection in
``chat_agent`` (``_is_report_edit_command`` / ``_is_report_generate_command`` /
the ``[SYSTEM DIRECTIVE]`` forcing). There are no keyword lists here — the model
reads the message (in the context of the last few turns) and decides, so the
"phrased it differently so it didn't match" class of bugs disappears.

Intents:
  - ``generate`` — build a NEW report from scratch.
  - ``edit``     — change ANYTHING about the existing report (title, prose,
                   layout, alignment, structure, images, cost numbers, …).
  - ``fetch``    — view / re-open the existing report unchanged.
  - ``chat``     — greetings, questions, discussion, or any product question
                   that does not create/modify/open a report.

The router does NOT decide whether a report exists; the ``edit``/``fetch``
handlers deal with absence gracefully (an "edit" with no report becomes a
friendly "generate one first" reply).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from src.core.config import LLM_INTENT_MODEL
from src.services.llm_analysis import invoke_llm, parse_json_object

logger = logging.getLogger(__name__)

Intent = Literal["generate", "edit", "fetch", "chat"]
_VALID: set[str] = {"generate", "edit", "fetch", "chat"}

_SYSTEM = """You classify the user's latest message in an electronics should-cost \
chat assistant. The assistant can generate a should-cost PDF report for a product, \
edit that report, re-display it, or just talk. Decide what the LATEST user message \
wants, using the prior turns only for context (e.g. to resolve "do it", "yes", \
"that one").

Return STRICT JSON only, no prose:
{"intent": "generate" | "edit" | "fetch" | "chat", "reasoning": "one short line"}

Definitions:
- "generate": the user wants a NEW report built from scratch — e.g. "generate a \
should-cost report", "make a BOM cost PDF", "create the cost breakdown", or \
confirming a prior offer to generate ("yes, go ahead and build it").
- "edit": the user wants to CHANGE anything about the report that already exists — \
its title, any heading or wording, a section, the layout/alignment/styling, an \
image (add/remove/replace), or any cost number/quantity/price. ANY modification, \
however phrased ("rename it", "make the heading say X", "update the name real \
quick", "center the title", "bold the summary", "attach this image", "set U1 to \
$3.10", "remove the second image"). If the user attached an image and wants it in \
the report, that is "edit".
- "fetch": the user wants to SEE/open/pull up the existing report with NO changes \
("show me the report", "open the cost report", "can I see it again?").
- "chat": greetings, thanks, small talk, or any question/discussion that is not \
asking to create, change, or open a report — including product questions \
("what's the BOM?", "is there a crystal near the MCU?", "what does this IC do?").

Edges:
- Distinguish "generate" (new) from "edit" (change existing). "regenerate the \
title" or "redo the summary" is an EDIT (a tweak), not a from-scratch generate. \
Only choose "generate" when they clearly want a brand-new/rebuilt report.
- A question ABOUT the report's contents ("what's the total cost?") is "chat", \
not "fetch" (fetch is only for re-displaying the document)."""


def _format_context(messages: list[dict[str, Any]]) -> str:
    """Render the last few turns as a compact transcript for the classifier."""
    recent = [m for m in messages if m.get("role") in ("user", "assistant")][-5:]
    lines: list[str] = []
    for m in recent:
        role = "User" if m.get("role") == "user" else "Assistant"
        content = str(m.get("content", "")).strip().replace("\n", " ")
        if len(content) > 600:
            content = content[:600] + "…"
        if m.get("role") == "user" and m.get("file_ids"):
            content += f"  [attached {len(m['file_ids'])} file(s)]"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def classify_sync(messages: list[dict[str, Any]]) -> Intent | None:
    """Classify the latest user turn. Returns the intent, or None on failure.

    A None return is the caller's signal to fall back to the general tool-using
    path (so a transient router failure never wedges report generation/Q&A).
    """
    if not messages:
        return None
    transcript = _format_context(messages)
    try:
        raw = invoke_llm(
            [
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Conversation so far:\n{transcript}\n\n"
                        "Classify the LATEST user message. Return only the JSON."
                    ),
                },
            ],
            max_tokens=200,
            model=LLM_INTENT_MODEL,
        )
    except Exception:
        logger.warning("Intent router LLM call failed", exc_info=True)
        return None

    data = parse_json_object(raw)
    if not isinstance(data, dict):
        logger.warning("Intent router returned unparseable output: %r", raw[:200])
        return None
    intent = str(data.get("intent", "")).strip().lower()
    if intent not in _VALID:
        logger.warning("Intent router returned invalid intent: %r", intent)
        return None
    logger.info("Intent router: %s (%s)", intent, str(data.get("reasoning", ""))[:120])
    return intent  # type: ignore[return-value]


async def classify(messages: list[dict[str, Any]]) -> Intent | None:
    """Async wrapper — runs the blocking LLM call off the event loop."""
    return await asyncio.to_thread(classify_sync, messages)
