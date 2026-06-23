---
name: pdf-report
description: Generate AND edit a professional electronics should-cost PDF report for a project. Use whenever the user asks to "generate a report", "make a should-cost report", "BOM cost report", "cost breakdown PDF", or to EDIT an existing report — "change the title", "remove a section", "reorder sections", "attach an image", "use 5000 units", "re-price a line". Covers gathering context, asking the user clarifying questions (human-in-the-loop), pricing the BOM, rendering a downloadable PDF, and editing it in place.
license: MIT
---

# Should-Cost PDF Report Generation

## When to use

Use this skill whenever the user wants a generated report about their product —
e.g. "generate a report", "give me a should-cost report", "make a BOM cost PDF",
"cost breakdown", "report this product's cost". Also use it when the user asks to
**change/edit** a report you already produced in this conversation.

There are three tools, and choosing the right one matters:

- **`report_generation`** — create a NEW report from scratch.
- **`report_edit`** — edit a report that ALREADY exists in this conversation,
  in place. Use this for every change to an existing report; it is much faster
  because it does not re-read the knowledge base, ask new questions, or re-price
  the whole BOM.
- **`get_report`** — fetch an existing report from the database and re-display it
  WITHOUT generating or changing anything. Use it when the user just wants to see
  or re-open a report ("show me the report", "open my cost report", "pull the
  report up again").

## How to generate a new report

Call the **`report_generation`** tool. Do not try to build the PDF yourself with
the filesystem tools — `report_generation` performs the entire pipeline and
streams the result to the user's screen.

Pass:
- `request`: the user's message verbatim (it is used to infer intent and any
  stated production volume, e.g. "10k units").

## How to edit an existing report

Call the **`report_edit`** tool (NOT `report_generation`) whenever the user wants
to change the report that was already generated in this conversation. It applies
only the requested changes to the saved report, re-renders the markdown + PDF, and
streams the revised version to the screen.

Pass:
- `edit_request`: the user's change request, verbatim. Examples of what it covers:
  - **Report title:** "change the title to X", "rename the report".
  - **A section's heading:** "rename the '08 · Methodology & Confidence' section
    to 'Myth & Confi'", "change the Market Context heading to 'Pricing'". This
    renames only that section's heading — not the whole report title.
  - **A section's wording:** "shorten the executive summary", "reword the market
    section", "add more detail to the assembly analysis".
  - **Structure:** "remove the architecture section", "reorder the sections",
    "drop the methodology section".
  - **Images:** "attach this photo below the executive summary", "add this image
    to the report", "remove the image".
  - **Cost inputs:** "use 5,000 units", "remove the LED line", "set the MCU price
    to ₹120", "change the per-joint assembly rate", "re-price U1", "add a 10µF
    capacitor line".
- `file_ids`: when the user attached an image to embed, the attached file IDs
  reach the tool automatically — just convey the attach intent in `edit_request`
  (passing the IDs explicitly is also fine).

If there is no report yet in this conversation, `report_edit` will say so — in
that case call `report_generation` first to create one.

## How to fetch / re-display an existing report

Call the **`get_report`** tool when the user just wants to SEE a report that was
already produced, not create or change one — e.g. "show me the report", "open the
cost report", "can I see my report again?". It loads the saved report from the
database and re-renders it in the preview panel with its download button; it does
not recompute anything.

Pass:
- `report_id` (optional): only when the user references a specific report. Omit it
  to fetch the most recent report for this conversation, falling back to the
  latest report for the project.

If no report exists for the product yet, `get_report` says so — then call
`report_generation` to create one.

When the user says "regenerate it" but only wants a tweak, prefer `report_edit`.
Only use `report_generation` when they explicitly want the report rebuilt from
scratch or when no report exists yet.

## What the tool does (so you can set expectations)

1. Reads the project knowledge base (theory + structured context). If nothing has
   been analyzed yet, it will tell the user to upload photos/datasheets first.
2. **Human-in-the-loop:** if material information is missing, it asks the user a
   few short questions (production volume, an illegible IC marking, PCB layer
   count, enclosure material, etc.). The user may answer, skip a question, or
   upload a file. You do not need to ask these yourself — the tool handles the
   question round and pauses for the user's answers.
3. Resolves and prices the BOM via the DigiKey API.
4. Optionally pulls extra market/product context from web search.
5. Composes the report, renders a PDF, stores it, and shows it on the right side
   of the screen with a download button.

## After it returns

The tool returns a short confirmation. Relay it to the user naturally and invite
them to request changes. When they ask for a change, call `report_edit` with
`edit_request` set — do not regenerate.

## Rules

- Never invent component prices or cost figures — the tools use only real
  component pricing and clearly-labelled heuristic estimates.
- Keep the question round minimal; the tool already limits how many questions are
  asked. Do not pile on additional questions of your own.
- For changes to an existing report, always use `report_edit` rather than
  rebuilding with `report_generation` — it is faster and preserves the report's
  data. Only rebuild from scratch when the user explicitly asks for it.
