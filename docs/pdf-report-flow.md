# PDF & Should-Cost Report — End-to-End Flow

This document describes how should-cost PDF reports are **generated**, **edited**, **stored**, **streamed to the UI**, and **downloaded** in the should-cost application.

---

## Table of contents

1. [Architecture overview](#architecture-overview)
2. [Tools and skill](#tools-and-skill)
3. [What gets stored](#what-gets-stored)
4. [Entry point: chat WebSocket](#entry-point-chat-websocket)
5. [Agent routing (`chat_agent.py`)](#agent-routing-chat_agentpy)
6. [Flow A: Generate a new report](#flow-a-generate-a-new-report)
7. [Flow B: Edit an existing report](#flow-b-edit-an-existing-report)
8. [Flow C: Fetch / re-display a report](#flow-c-fetch--re-display-a-report)
9. [Edit engine details (`report_edit.py`)](#edit-engine-details-report_editpy)
10. [Rendering and PDF creation](#rendering-and-pdf-persistence)
11. [WebSocket stream events](#websocket-stream-events)
12. [Frontend (UI)](#frontend-ui)
13. [HTTP report API](#http-report-api)
14. [Anti-fabrication backstop](#anti-fabrication-backstop)
15. [Common failure points](#common-failure-points)
16. [File index](#file-index)

---

## Architecture overview

```
User message (Chat UI)
        │
        ▼
WebSocket  /ws/chat  (chat.py)
        │
        ▼
chat_stream()  (chat_agent.py)
        │
        ├── greeting path ──► plain LLM (no tools, no PDF)
        │
        └── tool path ──► Deep Agent (LangGraph + Bedrock LLM)
                    │
                    ├── Skills: pdf-report/SKILL.md (guidance only)
                    ├── System prompt (tool usage rules)
                    └── Tools:
                          report_generation  → full pipeline → new PDF
                          report_edit        → mutate saved JSON → re-render PDF
                          get_report         → load from DB → re-display only
        │
        ▼
Stream events  (report_progress, report_ready, stream_delta, …)
        │
        ▼
ChatApp.jsx  →  ReportPanel.jsx  (preview + download)
        │
        ▼
Supabase  reports  table +  reports  storage bucket
```

**Governing principle:** the LLM narrates; the code computes. All cost numbers are computed in Python with `Live` / `Est` source tags. The LLM classifies edit intent and rewrites prose — it does not invent prices.

---

## Tools and skill

| Component | Name | Purpose |
|-----------|------|---------|
| **Skill** | `pdf-report` | Loaded from `src/agent/skills/pdf-report/SKILL.md`. Tells the agent which tool to call for generate vs edit vs fetch. Does not execute anything itself. |
| **Tool** | `report_generation` | Build a **new** report from the project knowledge base (KB). Full pipeline: KB → HILT questions → pricing → PDF. |
| **Tool** | `report_edit` | Change an **existing** saved report in place. Fast: no KB re-read, no HILT, no full re-pricing unless a line is re-priced. |
| **Tool** | `get_report` | Load a saved report from the database and re-display it. No computation, no changes. |

### Skill loading

On every agent turn with tools enabled, `chat_agent.py` reads all `*/SKILL.md` files under `src/agent/skills/` and injects them into the deep-agent `files` channel:

- Virtual path: `/skills/pdf-report/SKILL.md`
- Consumed by deepagents `SkillsMiddleware` alongside the system prompt

### Tool registration

All three report tools are registered in `src/agent/tools/__init__.py` via `get_agent_tools()`.

---

## What gets stored

Each report row in the Supabase `reports` table holds:

| Field | Role |
|-------|------|
| `report_json` | **Single source of truth** — structured JSON with `_compute`, BOM lines, prose sections, images, hidden sections, etc. All edits mutate this. |
| `html` | Rendered locked-template HTML (used for PDF generation). |
| `markdown` | Rendered markdown for the in-app preview panel. |
| `pdf_path` / `pdf_url` | PDF uploaded to the `reports` storage bucket; URL is public. |
| `conversation_id` | Chat where the report was created (used to find “latest report for this chat”). |
| `project_id` | Product/project the report belongs to (fallback lookup across chats). |
| `title`, `volume`, `user_id`, timestamps | Metadata. |

Related table: `report_questions` — HILT question/answer pairs from report **generation** (not used during edit).

---

## Entry point: chat WebSocket

**File:** `src/routers/chat.py`  
**Endpoint:** `WS /ws/chat`

1. User sends `{ "message": "...", "chat_id": "<uuid>", "file_ids": [...] }` (optional attached images).
2. Handler resolves `project_id` from the conversation (or from `project_id` on first message in a new chat).
3. Builds `agent_messages = history + user_message`.
4. Calls `chat_stream(agent_messages, project_id, user_id, conversation_id)`.
5. Forwards every streamed event to the client (`stream_delta`, `tool_start`, `report_progress`, `report_ready`, etc.).
6. On completion, persists user + assistant messages (including slim report metadata on the assistant bubble).

**HILT resume:** If generation pauses for clarifying questions, the client sends `{ "type": "report_answers", "answers": {...} }` and `resume_stream(thread_id, answers, ...)` continues the same tool run.

---

## Agent routing (`chat_agent.py`)

Before the deep agent runs, several deterministic checks run:

### 1. Greeting / small-talk bypass

`_needs_tools()` returns `false` for pure greetings (“hi”, “thanks”) with no product keywords. Those messages go to `_direct_chat_stream()` — **no tools at all**. That path explicitly cannot generate or edit reports.

Keywords that keep the message on the tool path include: `report`, `pdf`, `title`, `bom`, `cost`, `change`, etc.

### 2. Report intent detection (regex)

| Function | Detects |
|----------|---------|
| `_is_report_edit_command()` | Imperative edit requests: verbs (`change`, `rename`, `remove`, …) + cues (`report`, `title`, `section`, `bom`, `pdf`, …). Also title renames via `_detect_title_rename()`. |
| `_is_report_generate_command()` | New report requests: “generate/create/make … report / cost breakdown / PDF”. |

If edit or generate intent is detected, a **`[SYSTEM DIRECTIVE]`** block is appended to the user message forcing the model to call the correct tool before replying.

### 3. Context threaded into tools

LangGraph config (`configurable`) carries:

- `thread_id` — fresh UUID per turn (for HILT checkpointing)
- `project_id`, `user_id`, `conversation_id`
- `file_ids` — attached image IDs from the latest user message (reliable path for image embed)

### 4. Deep agent execution

`create_deep_agent()` with:

- Bedrock LLM (`ChatLiteLLM`)
- All agent tools
- System prompt (includes report tool rules)
- Skills path: `/skills/`
- `MemorySaver` checkpointer (for HILT `interrupt()` during generation)

The agent streams three LangGraph channels:

- `messages` — token deltas, tool call lifecycle
- `custom` — `report_progress` and `report_ready` from tools
- `updates` — `__interrupt__` for HILT questions

---

## Flow A: Generate a new report

**Trigger:** User asks to generate a should-cost report, BOM cost PDF, cost breakdown, etc.  
**Tool:** `report_generation` (`src/agent/tools/report_tool.py`)

### Sequence

```
report_generation(request)
    │
    ├─ 1. reading_kb
    │      get_project_knowledge_base(project_id)
    │      → theory_context + structured_context
    │      (fail if nothing analyzed yet)
    │
    ├─ 2. hilt  (Human-in-the-loop)
    │      report_builder.assess_missing() → list of questions
    │      if questions: interrupt() → UI shows form → user answers
    │      → resume_stream() continues
    │      Answers merged into structured data + persisted to report_questions
    │      → enqueue_qa_insight() for async KB update
    │
    ├─ 3–9. report_pipeline.run_pipeline()
    │      resolving_mpns → pricing (parts DB) → fab_quote (PCBWay)
    │      → non_quotable estimates → market_context (Tavily)
    │      → fx → duty → assembly → volume_curve → aggregate
    │      Each stage emits report_progress
    │
    ├─ Optional: embed attached images if user asked + file_ids present
    │
    └─ 10–11. _render_store_stream()
           render_html + render_markdown
           → render_pdf_from_html (Playwright → xhtml2pdf fallback)
           → create_report() in DB
           → upload_report_pdf() to Supabase bucket
           → emit report_ready
```

### Pipeline stages (`report_pipeline.py`)

Deterministic compute after KB + HILT:

- MPN resolution
- Parallel BOM pricing (JLCPCB / parts DB), PCB fab quote, non-quotable blocks, market context
- FX (USD→INR), customs duty, SMT assembly model, volume curve
- Aggregate into one `report_json` document

**Volume:** New reports are generated for a **single unit** (recurring per-unit cost; NRE listed separately).

### HILT (questions during generation only)

- Implemented with LangGraph `interrupt({"type": "report_questions", "questions": [...]})`
- UI receives `report_questions` WebSocket event with `thread_id`
- User submits answers → `report_answers` message → `resume_stream(thread_id, answers)`
- Question list is cached per `thread_id` so resume does not re-ask different questions

**Edits do not use HILT.**

---

## Flow B: Edit an existing report

**Trigger:** User asks to change title, section, BOM line, volume, attach image, etc.  
**Tool:** `report_edit` (`src/agent/tools/report_edit_tool.py`)

### Sequence

```
report_edit(edit_request, file_ids?)
    │
    ├─ Merge file_ids from config + tool args
    │
    └─ _generate_modification()  (report_tool.py)
           │
           ├─ Load base report:
           │     latest_report_for_conversation(conversation_id)
           │     else latest_report_for_project(project_id, user_id)
           │     (fail if no report_json)
           │
           ├─ report_edit.apply_edit(report_json, request, image_refs)
           │     ├─ Title fast-path: regex _detect_title_rename() → set_title (skip LLM)
           │     ├─ Else: classify_edit() → LLM returns operations[]
           │     ├─ Apply each operation to report_json / _compute
           │     └─ recompute_numeric() if BOM/pricing/volume changed
           │
           ├─ Image fallback: if user attached image + intent detected but classifier
           │     didn't add_image → _add_images() directly
           │
           ├─ If nothing changed → honest failure message, NO report_ready
           │
           └─ If changed → _render_store_stream(existing_report_id)
                  update_report() + re-upload PDF
                  emit report_ready
```

### Report lookup order

1. Latest report in **this conversation**
2. Else latest report for the **project** (any prior chat for the same product)

This allows editing a report created in an earlier conversation on the same project.

### Deprecated edit path on `report_generation`

`report_generation(modification_request=...)` still delegates to `_generate_modification()` for backward compatibility. New code and the skill should always use `report_edit`.

---

## Flow C: Fetch / re-display a report

**Trigger:** “Show me the report”, “open my cost report”, etc.  
**Tool:** `get_report` (`src/agent/tools/get_report_tool.py`)

### Sequence

```
get_report(report_id?)
    │
    ├─ Resolve record:
    │     report_id (if given, user-scoped)
    │     else latest_report_for_conversation()
    │     else latest_report_for_project()
    │
    ├─ Load markdown (stored or render from report_json)
    │
    └─ Emit report_ready (same event as generate/edit)
           No DB write, no PDF re-render
```

---

## Edit engine details (`report_edit.py`)

### Classification

`classify_edit(request, report_json)` sends a prompt to the LLM with:

- A summary of current report state (volume, stage costs, BOM lines, prose fields)
- The user's edit request

Returns JSON:

```json
{
  "operations": [ { "op": "...", ... } ],
  "needs_sourcing": false,
  "summary": "..."
}
```

If classification fails, falls back to `rewrite_prose` on the executive summary.

### Supported operations

| Operation | Effect |
|-----------|--------|
| `set_title` | Change whole report title (`meta.title`) |
| `set_section_title` | Rename one section heading (`section_titles`) |
| `remove_section` | Hide section (`hidden_sections`) |
| `rewrite_prose` | LLM rewrites narrative field (executive, architecture, market, data confidence) |
| `set_volume` | Change production volume → recompute |
| `set_qty` | Change line quantity → recompute |
| `remove_line` | Remove BOM line → recompute |
| `set_unit_price` | User-supplied price on a line (tagged `Est`) → recompute |
| `set_stage_cost` | Override per-unit cost for a manufacturing stage → recompute |
| `set_rate` | Change assembly rate card (per-joint / setup / stencil) → recompute |
| `reprice_line` | Live parts DB lookup for one line → recompute |
| `add_line` | Add BOM line (optional live pricing) → recompute |
| `add_image` | Embed image (from attachments or URL in message) |
| `remove_image` | Remove embedded image(s) |

### Title rename fast-path

Regex `_detect_title_rename()` handles unambiguous phrasings (“change the title to X”, “call it X”) **without** calling the classifier — avoids confusing report title vs section heading.

### Honesty contract

- Summary reflects **only operations that actually applied**
- If zero changes: tool returns explicit failure; **no** `report_ready` event
- Tool return text includes `NOTE:` when image embed failed so the assistant does not claim success

See also: `memory/report-edit-honesty-contract.md`

---

## Rendering and PDF persistence

**Files:** `src/services/report_template.py`, `src/services/reports.py`

### `_render_store_stream()` (shared by generate and edit)

1. **`report_template.render_html(report_json)`** — locked HTML template
2. **`report_template.render_markdown(report_json)`** — in-app preview
3. **`reports.render_pdf_from_html(html)`**
   - Primary: Playwright headless Chromium
   - Fallback: xhtml2pdf
4. **Persist**
   - New report: `create_report()` then `upload_report_pdf()`
   - Edit: `update_report(existing_id, ...)` then re-upload PDF (overwrite `{report_id}.pdf`)
5. **Stream** `report_ready` with `report_id`, `title`, `markdown`, `pdf_url`, `volume`, `fx_rate`

---

## WebSocket stream events

| Event type | When | UI effect |
|------------|------|-----------|
| `stream_start` | Turn begins | Show streaming state |
| `stream_delta` | LLM token | Append assistant text |
| `stream_reset` | New message after tool | Clear partial text |
| `tool_start` / `tool_query` / `tool_end` | Tool lifecycle | Tool status in message bubble |
| `tools_used` | End of turn | Final tool summary |
| `report_progress` | Pipeline / edit stage | Progress bar + stage checklist |
| `report_questions` | HILT during **generation** | Question form in chat |
| `report_ready` | PDF saved | Open ReportPanel, set markdown + download URL |
| `stream_end` | Turn complete | Finalize message |

---

## Frontend (UI)

### Chat orchestration

**File:** `should-cost-ui/src/pages/ChatApp.jsx`

- Opens WebSocket to `/ws/chat`
- Handles all stream events above
- On `report_ready`: sets `activeReport`, opens panel, attaches report metadata to assistant message
- On history load: report metadata on messages is slim (`report_id`, `title`, `volume`); full markdown fetched on demand

### Report panel

**File:** `should-cost-ui/src/components/chat/ReportPanel.jsx`

- Shows markdown preview (`MarkdownRenderer`)
- Download button → `downloadReportPdf(reportId)` (authenticated fetch, not plain `<a href>`)
- If only `report_id` present (history): fetches full report via REST

### Progress UI

**File:** `should-cost-ui/src/components/chat/ReportProgress.jsx`

- Driven by `report_progress` events and `reportStages` on the assistant message

---

## HTTP report API

**File:** `src/routers/reports.py`  
**Prefix:** `/api/reports` (auth required)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/{report_id}` | Metadata + markdown + html + report_json for preview |
| `GET` | `/{report_id}/download` | Stream PDF bytes (from bucket, or re-render from stored HTML) |

**Frontend client:** `should-cost-ui/src/services/reportsApi.js`

---

## Anti-fabrication backstop

**File:** `src/agent/chat_agent.py`

Problem: the LLM sometimes **claims** a report was updated without calling any tool.

Mitigation after each tool-enabled turn:

1. If `_is_report_edit_command()` or `_is_report_generate_command()` matched
2. AND no report tool ran (`report_generation`, `report_edit`)
3. AND not paused for HILT
4. AND the reply looks like a success claim (`_claims_report_action()`)

Then:

- **`stream_reset`** clears fabricated text
- For **edit** intent with `project_id`: **`_run_direct_edit()`** runs `report_edit` logic directly (no LLM), streaming real progress + `report_ready`
- For **generate** intent: honest message that nothing was created

---

## Common failure points

| Symptom | Likely cause |
|---------|----------------|
| Bot says “updated” but PDF unchanged | LLM did not call `report_edit`; classifier mapped request to no ops; or greeting path (no tools) |
| “Generate a report first” | No `report_json` in this conversation or project |
| “Couldn’t apply that change” | `classify_edit()` ops did not match any line/section; empty `changes` list |
| Image not embedded | No `file_ids` on message; or image resolve failed; tool returns `NOTE:` — assistant must not claim success |
| PDF download 404 | Playwright/render failed at save time; row exists but no PDF in bucket |
| Full rebuild instead of quick edit | Model called `report_generation` instead of `report_edit` |
| Questions during edit | Should not happen — HILT is generation-only; if seen, wrong tool was invoked |
| Skill “not working” | Skill is guidance only; failure is usually tool selection or `apply_edit` classification |

---

## File index

### Agent layer

| File | Role |
|------|------|
| `src/agent/skills/pdf-report/SKILL.md` | Agent skill: when to use each report tool |
| `src/agent/chat_agent.py` | Routing, streaming, intent detection, backstop |
| `src/agent/tools/report_tool.py` | `report_generation`, `_generate_modification`, `_render_store_stream` |
| `src/agent/tools/report_edit_tool.py` | `report_edit` tool wrapper |
| `src/agent/tools/get_report_tool.py` | `get_report` tool |
| `src/agent/tools/__init__.py` | Tool registration |

### Services

| File | Role |
|------|------|
| `src/services/report_edit.py` | Edit classification + operation appliers |
| `src/services/report_pipeline.py` | Full pricing/compute pipeline + `recompute_numeric()` |
| `src/services/report_builder.py` | HILT `assess_missing()` |
| `src/services/report_template.py` | HTML + markdown rendering |
| `src/services/reports.py` | DB CRUD, PDF render/upload, latest-report lookups |
| `src/services/parts_pricing.py` | Live re-pricing during edits |
| `src/services/file_storage.py` | Resolve attached image IDs to URLs |

### API & UI

| File | Role |
|------|------|
| `src/routers/chat.py` | WebSocket chat + stream forwarding |
| `src/routers/reports.py` | REST preview + PDF download |
| `should-cost-ui/src/pages/ChatApp.jsx` | WebSocket client, event handling |
| `should-cost-ui/src/components/chat/ReportPanel.jsx` | Report preview panel |
| `should-cost-ui/src/components/chat/ReportProgress.jsx` | Generation progress UI |
| `should-cost-ui/src/services/reportsApi.js` | REST client for report fetch/download |

### Related docs

| File | Role |
|------|------|
| `memory/report-edit-honesty-contract.md` | Edit success/failure messaging rules |

---

## Quick decision guide

| User intent | Tool to call |
|-------------|--------------|
| “Generate / create / make a should-cost report” | `report_generation` |
| “Change title / remove section / re-price U1 / attach this photo” | `report_edit` |
| “Show me the report again” | `get_report` |
| “Regenerate from scratch” (explicit full rebuild) | `report_generation` |
| “Regenerate it” (but only a small tweak) | `report_edit` |
