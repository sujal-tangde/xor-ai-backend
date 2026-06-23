---
name: report-edit-honesty-contract
description: report tools must never claim success without running; covers tool-skip fabrication, classifier vocab, and honest reporting
metadata:
  type: project
---

## Anti-fabrication harness (the root cause behind "it says done but nothing changed")

The deep agent intermittently REPLIES that it generated/edited the report **without calling any tool** — for both generation and edits. The prompt alone can't stop this. `src/agent/chat_agent.py` now enforces it deterministically:
- `_is_report_edit_command` / `_is_report_generate_command` detect report intent (verb + report/field cue; generate = verb + "report"). Both exclude trailing-"?" info questions.
- On report intent, `_force_tool_directive` is appended to the last user turn so the model must call the tool first.
- **Backstop:** `chat_stream` tracks whether a `report_generation`/`report_edit` tool actually ran (via the `tools_used` event / HILT `questions`). If the user asked for a report action, no report tool ran, and the reply is empty or `_claims_report_action` matches → it yields `{"type":"reset"}` (UI clears the fabricated bubble) and then does the real thing: edits run directly via `_run_direct_edit` (bridges the sync `_generate_modification` to async through a thread + `loop.call_soon_threadsafe` queue); generation can't run inline (HILT `interrupt()` needs LangGraph) so it emits an honest "I didn't actually generate it" message.
- `report_tool._generate_modification` accepts injected `emit`/`writer` (split out of `_make_emit` via `_emit_for_writer`) so the direct path streams identical `report_progress`/`report_ready` events.
- Known gap: requests phrased as a question ("can you rename it to X?") aren't force/backstopped (precision tradeoff) — they rely on the agent calling the tool.

## Edit application bugs (fixed earlier, same file family)

The should-cost `report_edit` flow (`src/services/report_edit.py` + `src/agent/tools/report_tool.py::_generate_modification`) edits a saved `report_json._compute` then `recompute_numeric` rolls up the numbers. An LLM classifier (`classify_edit`) maps free-text into a fixed operation vocabulary.

Two failure modes caused a real "I updated it / no it's unchanged" gaslighting loop with a user:
1. **Missing vocabulary** — there was no op to set a manufacturing-stage's per-unit cost directly (only `set_rate` per-joint/setup/stencil). "Make PCB assembly cost ₹90" was inexpressible, so the edit no-op'd. Fixed by adding `set_stage_cost` (targets: assembly / pcb_fabrication / firmware / enclosure / final_assembly) and exposing current `stage_costs_per_unit_inr` in `_state_summary`.
2. **Dishonest success** — summary was taken from the classifier's optimistic `summary` field, and `_generate_modification` always returned "Updated the report". Fixed: summary now derives from the actual `changes` list; empty `changes` + no image added → return an explicit "couldn't apply that, report unchanged, do NOT tell the user it was updated" and skip the re-render/`report_ready` stream.

**Why:** the contract is "the LLM narrates, the code computes" — but the tool must never claim a change the code didn't make. **How to apply:** when adding a new editable field, add BOTH a classifier op AND make sure `changes.append(...)` only fires when something really changed; never report success off the classifier's self-description.
