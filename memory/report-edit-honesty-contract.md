---
name: report-edit-honesty-contract
description: report edits are free-form HTML verified by a computed diff; honesty is structural (no diff = no success), not prompt-dependent
metadata:
  type: project
---

## How report editing works now (free-form HTML + diff verification)

Should-cost reports are stored as free-form HTML (the `reports.html` column). After
generation the user can change ANYTHING — title, prose, layout, alignment, section
structure, images, and cost numbers (numbers are NO LONGER code-protected for
edits). There is no fixed schema and no closed set of operations. The old
`classify_edit()` + operation-applier system in `src/services/report_edit.py` and
the `report_edit` agent tool were DELETED. (`report_edit.py` now contains only
`_add_images`, still used by the generation pipeline to embed attached images.)

### Routing — LLM intent router, no regex
`src/agent/intent_router.py` makes one cheap-model call (`LLM_INTENT_MODEL`,
falls back to `LLM_MODEL`) classifying the latest message into
`generate | edit | fetch | chat`. This REPLACES all the old regex report-intent
detection (`_is_report_edit_command`, `_is_report_generate_command`,
`_detect_title_rename`, `_force_tool_directive`, `_claims_report_action`, and the
`[SYSTEM DIRECTIVE]` injection) — so "phrased it differently so it didn't match"
bugs are gone. `chat_agent.chat_stream` routes:
- `edit` → `_edit_flow` (deterministic HTML editor; bypasses the deep agent).
- `generate` / `fetch` → the deep agent (`report_generation` / `get_report`),
  HILT intact. A light `_tool_nudge` keeps the model calling the right tool.
- `chat` → plain no-tools reply for small talk (`_is_small_talk`), else the deep
  agent for substantive product questions.

### The HTML edit engine — `src/agent/html_editor.py`
`edit(html, request, image_urls)` asks the capable model (`LLM_MODEL`) to return
the COMPLETE modified HTML (delimiter protocol `===CHANGES===/===UNABLE===/===HTML===`,
which survives full-document output far better than JSON-escaping). Then it
VERIFIES deterministically (plain Python, no LLM):
- `compute_diff` / `diff_is_empty` — token/normalized comparison. **Empty diff =>
  the change didn't happen.** No `report_ready`, honest failure message. This is
  why fabrication is now structurally impossible: the success summary is built by
  `build_summary_from_diff` from the ACTUAL diff, never from the model's claims.
- `existing_images_preserved` — every original `<img src>` must survive unless the
  request explicitly asked to remove/replace (`removal_requested`). Stops
  "add an image" from dropping existing images. Retries once, else rejects.
- `change_scope_ratio` — if an edit changed far more than expected (> 0.6), retry
  stricter; if still huge, proceed with a warning appended to the summary.

### Persistence — edited HTML is authoritative
`report_tool.persist_edited_html` renders the PDF from the edited HTML, uploads it,
and updates `html` + `markdown` (+ title re-read from the cover `<h1>`/`<title>`).
It deliberately does NOT write `report_json` — after a free-form edit that JSON is
a stale generation artifact and must never clobber the edited HTML. The preview
panel (`ReportPanel.jsx`) renders the HTML in a sandboxed `<iframe>` so
layout/style edits are actually visible; markdown (via `markdownify`) is a
fallback. `report_ready` now carries `html`.

**Why:** for EDITS the principle is no longer "LLM narrates, code computes" — the
user may change numbers freely. The new contract is: **the assistant can only
report a change the computed diff confirms.** (Generation still computes numbers
in code and fills the locked template — only edits are free-form.)
**How to apply:** never build an edit success message from the model's
self-description; always derive it from the diff, and never emit `report_ready`
when the diff is empty or an existing image vanished unrequested.
