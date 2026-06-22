---
name: pdf-report
description: Generate a professional electronics should-cost PDF report for a project. Use whenever the user asks to "generate a report", "make a should-cost report", "BOM cost report", "cost breakdown PDF", or anything similar. Covers gathering context, asking the user clarifying questions (human-in-the-loop), pricing the BOM via DigiKey, rendering a downloadable PDF, and revising it.
license: MIT
---

# Should-Cost PDF Report Generation

## When to use

Use this skill whenever the user wants a generated report about their product —
e.g. "generate a report", "give me a should-cost report", "make a BOM cost PDF",
"cost breakdown", "report this product's cost". Also use it when the user asks to
**change/modify** a report you already produced in this conversation.

## How to run it

Call the **`report_generation`** tool. Do not try to build the PDF yourself with
the filesystem tools — `report_generation` performs the entire pipeline and
streams the result to the user's screen.

Pass:
- `request`: the user's message verbatim (it is used to infer intent and any
  stated production volume, e.g. "10k units").
- `modification_request`: ONLY when the user is asking to change a report that
  was already generated in this conversation (e.g. "change the title to X",
  "rename the report", "use 5,000 units instead", "remove the LED line", "add
  more detail to the assembly section", "attach this photo to the PDF and
  regenerate"). Put their change request here so the existing report is revised
  in place instead of rebuilt from scratch. If the user attached an image to
  embed, the attached file IDs reach the tool automatically — just convey the
  attach intent in the change request.

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
them to request modifications. When they ask for a change, call
`report_generation` again with `modification_request` set.

## Rules

- Never invent component prices or cost figures — the tool uses only real
  DigiKey pricing and clearly-labelled heuristic estimates.
- Keep the question round minimal; the tool already limits how many questions are
  asked. Do not pile on additional questions of your own.
