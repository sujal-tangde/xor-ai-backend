"""Background ingestion of report Q&A answers into the insight pipeline.

When the report tool asks the user questions and gets answers, the useful facts
in those answers should not be lost — they are extracted into a project insight
(``project_insights``) which then folds into the project knowledge base via the
normal recompute path. Runs as an RQ job so the report flow never blocks on it.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from src.services.insights import record_insight
from src.services.llm_analysis import invoke_llm, parse_json_object

logger = logging.getLogger(__name__)

_QA_INSIGHT_PROMPT = """You are an expert electronics reverse-engineering and should-cost analyst. \
A user answered some clarifying questions about a physical electronic product they are having cost-analyzed. \
Extract only the product-relevant facts their answers actually state. Never invent values; ignore non-answers \
(e.g. "I don't know" or skipped questions).

Return ONLY a JSON object of this exact shape:
{{"theory": "<short markdown prose of what these answers reveal about the product>",
  "structured": {{"product": {{}}, "pcb": {{}}, "enclosure": {{}}, "components": [], "assumptions": [], "extra_insights": {{}}}}}}

Drop any structured keys the answers don't address. Numbers as numbers; omit unknowns.

QUESTIONS AND ANSWERS (JSON):
{qa}
"""


def answered_pairs(qa_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter to questions the user actually answered (text or file)."""
    return [
        qa
        for qa in (qa_pairs or [])
        if isinstance(qa, dict) and (qa.get("answer") or qa.get("file_ids"))
    ]


def extract_qa_facts(
    qa_pairs: list[dict[str, Any]],
) -> tuple[str, dict[str, Any] | None]:
    """Extract (theory, structured) facts from answered questions. ``(\"\", None)`` if none.

    Shared by the async KB-ingest job and the synchronous in-report merge so the
    answers shape the report being generated, not just future ones.
    """
    answered = answered_pairs(qa_pairs)
    if not answered:
        return "", None
    prompt = _QA_INSIGHT_PROMPT.format(qa=json.dumps(answered, ensure_ascii=False)[:8000])
    try:
        raw = invoke_llm([{"role": "user", "content": prompt}], max_tokens=2048)
        data = parse_json_object(raw) or {}
    except Exception:
        logger.exception("Q&A fact extraction failed")
        return "", None
    theory = str(data.get("theory") or "").strip()
    structured = data.get("structured")
    if not isinstance(structured, dict):
        structured = None
    return theory, structured


def ingest_qa_insight(
    project_id: str,
    user_id: str | None,
    qa_pairs: list[dict[str, Any]],
) -> None:
    """Extract facts from answered questions and record them as an insight."""
    if not answered_pairs(qa_pairs):
        return

    theory, structured = extract_qa_facts(qa_pairs)
    if not theory and not structured:
        logger.info("Q&A produced no extractable facts for project %s", project_id)
        return

    try:
        record_insight(
            project_id=project_id,
            user_id=user_id,
            file_id=str(uuid.uuid4()),  # synthetic: Q&A is not tied to an upload
            media_kind="qa",
            theory=theory,
            structured=structured,
        )
    except Exception:
        logger.exception("Failed to record Q&A insight for project %s", project_id)
