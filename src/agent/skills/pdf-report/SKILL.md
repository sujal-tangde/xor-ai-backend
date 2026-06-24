---
name: pdf-report
description: Generate a professional electronics should-cost PDF report for a project, and re-display existing reports. Use whenever the user asks to "generate a report", "make a should-cost report", "BOM cost report", "cost breakdown PDF", or to view an existing report ("show me the report"). Editing an existing report is handled automatically by the application's HTML edit engine before it reaches you — you never edit reports yourself.
license: MIT
---

# Should-Cost PDF Report

## When to use

Use this skill when the user wants a generated should-cost report about their
product — "generate a report", "give me a should-cost report", "make a BOM cost
PDF", "cost breakdown" — or to view a report that already exists.

There are two tools you call directly:

- **`report_generation`** — create a NEW report from scratch.
- **`get_report`** — fetch an existing report from the database and re-display it
  WITHOUT changing anything ("show me the report", "open my cost report", "pull
  the report up again").

**Editing an existing report is NOT a tool you call.** When the user asks to
change a report (title, wording, a section, layout/alignment, an image, a cost
number — anything), the application detects that and routes it to a dedicated
free-form **HTML edit engine** before the request ever reaches you. So you never
edit a report yourself, and you should not try to with the filesystem tools.

## How to generate a new report

Call the **`report_generation`** tool. Do not try to build the PDF yourself.
`report_generation` performs the entire pipeline and streams the result to the
user's screen.

Pass:
- `request`: the user's message verbatim.
- `file_ids`: when the user attached image(s) they want embedded in the new
  report, pass the attached file IDs (they also reach the tool via context).

## How to fetch / re-display an existing report

Call the **`get_report`** tool when the user just wants to SEE a report that was
already produced — "show me the report", "open the cost report", "can I see my
report again?". It loads the saved report and re-renders it in the preview panel
with its download button; it does not recompute anything.

Pass:
- `report_id` (optional): only when the user references a specific report. Omit it
  to fetch the most recent report for this conversation, falling back to the
  latest report for the project.

If no report exists for the product yet, `get_report` says so — then call
`report_generation` to create one.

## What `report_generation` does (so you can set expectations)

1. Reads the project knowledge base (theory + structured context). If nothing has
   been analyzed yet, it tells the user to upload photos/datasheets first.
2. **Human-in-the-loop:** if material information is missing, it asks the user a
   few short questions (production volume, an illegible IC marking, PCB layer
   count, enclosure material, etc.). You do not need to ask these yourself — the
   tool handles the question round and pauses for the user's answers.
3. Resolves and prices the BOM via the parts database.
4. Optionally pulls extra market/product context from web search.
5. Composes the report, renders a PDF, stores it, and shows it on the right side
   of the screen with a download button.

## How edits work (for your awareness — you do not run these)

When the user asks to change the current report, the HTML edit engine:
- takes the report's stored HTML and the user's request,
- produces the complete modified HTML with ONLY the requested change applied,
- VERIFIES the change with a computed diff (an unchanged document is reported as
  an honest failure — never as success), preserves existing images unless the
  user asked to remove one, and re-renders the PDF.

Because the user-facing summary is built from the verified diff, the system never
claims a change that did not actually happen.

## Rules

- Never invent component prices or cost figures — generation uses only real
  component pricing and clearly-labelled heuristic estimates.
- Keep the question round minimal; the tool already limits how many questions are
  asked. Do not pile on additional questions of your own.
- After a tool returns, relay its result naturally and invite the user to request
  changes or view the report.
