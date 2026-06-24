"""The ``report_edit`` agent tool.

Edits an EXISTING should-cost report that was already generated and saved for this
conversation — without re-running the full report-generation pipeline (no KB read,
no HILT questions, no Mouser/PCBWay teardown). It operates on the report's saved
structured JSON (``report_json``), applies only the requested changes, re-renders
the markdown + PDF, persists them, and streams the revised report to the screen.

Use this instead of ``report_generation`` whenever the user wants to tweak a report
that already exists: change the title, rewrite or reword a section, add/remove a
section, reorder/realign sections, attach or remove an image, change the target
volume, edit/remove/add a BOM line, or any other adjustment to the current report.

Why a separate tool: re-running ``report_generation`` for a small edit is slow and
wasteful (it re-reads the KB, may re-ask clarifying questions, and re-prices the
whole BOM). ``report_edit`` is specialized for in-place edits and is fast because it
mutates the already-computed report and only recomputes the numbers an edit touches.

The governing principle is unchanged: **the LLM narrates, the code computes.** A
user-supplied price is re-tagged ``Est`` / user-provided; only the requested fields
and their direct roll-ups move — unrelated numbers never silently change.
"""

from __future__ import annotations

import logging

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.agent.tools.report_tool import _cfg, _generate_modification
from src.agent.tools.validation import invalid_project_id_message, is_uuid

logger = logging.getLogger(__name__)

TOOL_NAME = "report_edit"
TOOL_LABEL = "report editing"


@tool(TOOL_NAME)
def report_edit_tool(
    edit_request: str,
    config: RunnableConfig = None,  # type: ignore[assignment]
    file_ids: list[str] | None = None,
) -> str:
    """Edit the existing should-cost report for this conversation, in place.

    Call this WHENEVER the user wants to change a report that has ALREADY been
    generated in this conversation — never rebuild it with ``report_generation``
    for an edit. This is fast: it revises the saved report and re-renders the PDF
    without re-reading the knowledge base, re-asking questions, or re-pricing the
    whole BOM. The revised markdown is shown on the right and a fresh PDF is made
    ready to download.

    Typical requests that should call this tool:
      - Report title: "rename the report", "change the title to X".
      - A section's heading: "change the '08 · Methodology & Confidence' heading
        to 'Myth & Confi'", "rename the Market Context section to 'Pricing'".
        (This renames just that section's heading, NOT the report title.)
      - A section's text: "shorten the executive summary", "reword the market
        section", "add more detail to the assembly analysis".
      - Structure: "remove the architecture section", "drop the methodology
        section", "take out the market context".
      - Images: "attach this photo below the executive summary", "add this image
        to the report", "remove the image".
      - Cost inputs: "use 5,000 units", "remove the LED line", "set the MCU price
        to ₹120", "change the per-joint assembly rate", "re-price U1 live",
        "add a 10µF capacitor line".
      - Any other adjustment to the report that already exists.

    If there is NO report yet in this conversation, the tool says so — in that case
    call ``report_generation`` to create one first.

    Args:
        edit_request: The user's change request, verbatim (e.g. "rename it to
            'BOM Cost — Rev B' and remove the market section"). Describe WHAT to
            change in plain language; the tool figures out the concrete edits and
            recomputes any affected numbers deterministically.
        file_ids: The IDs of any images the user ATTACHED to this message that they
            want embedded in (or used by) the report. Pass them whenever the user
            asks to attach/add/embed/insert an image or photo. Attached IDs also
            reach the tool automatically via the request context, so passing them
            here is belt-and-suspenders.
    """
    cfg = _cfg(config)
    project_id = cfg.get("project_id")
    user_id = cfg.get("user_id")
    conversation_id = cfg.get("conversation_id")

    # Attached image IDs can arrive two ways: threaded through the request config
    # (reliable, set by chat_stream) or passed by the model as a tool arg. Merge
    # both so an attach works regardless of which path delivered them.
    cfg_file_ids = [str(x) for x in (cfg.get("file_ids") or [])]
    arg_file_ids = [str(x) for x in (file_ids or [])]
    file_ids = list(dict.fromkeys(cfg_file_ids + arg_file_ids))

    if not edit_request or not edit_request.strip():
        return (
            "Tell me what you'd like to change about the report (e.g. the title, a "
            "section, the volume, or a BOM line) and I'll edit it."
        )
    if not project_id:
        return "No project is associated with this conversation, so I can't edit a report."
    if not is_uuid(str(project_id)):
        return invalid_project_id_message(str(project_id))

    request = edit_request.strip()
    return _generate_modification(
        str(project_id),
        conversation_id,
        user_id,
        request,
        request,
        file_ids=file_ids,
    )
