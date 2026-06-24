"""Deterministic HTML template-fill for the should-cost report.

The structure here is derived once from ``dummy_report_xor_cost.html`` (the
locked format): the same 9 sections, table columns, metric cards, source/
confidence tags and stage bars, with the dummy's CSS embedded verbatim. Only the
*structure* of the dummy is authoritative — its numbers are placeholders and are
never copied. All real values come from the aggregated pipeline JSON.

The LLM never touches this file's output: prose fields arrive pre-written in the
JSON; everything numeric/tabular is formatted here in code.
"""

from __future__ import annotations

import html
from typing import Any

# CSS lifted verbatim from dummy_report_xor_cost.html (the locked styling).
_CSS = """
:root{
  --ink:#1a2230; --sub:#4a5568; --muted:#718096; --line:#e4e8ee; --hair:#eef1f5;
  --brand:#143a5a; --brand-2:#1d4e77; --rule:#0f2e49;
  --pos:#1c6b45; --neg:#9a3324; --warn:#8a5a00; --chipbg:#eef2f6;
  --tablehdr:#f2f5f8;
}
*{box-sizing:border-box}
html{-webkit-print-color-adjust:exact;print-color-adjust:exact}
body{font-family:"Segoe UI",-apple-system,BlinkMacSystemFont,Roboto,Helvetica,Arial,sans-serif;
  color:var(--ink);margin:0;background:#eceff3;font-size:13px;line-height:1.55}
.doc{max-width:920px;margin:24px auto;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.08)}
.cover{padding:40px 48px 28px;border-bottom:3px solid var(--rule)}
.cover .org{display:flex;justify-content:space-between;align-items:flex-start;font-size:11px;color:var(--muted)}
.cover .org .logo{font-weight:800;letter-spacing:.04em;color:var(--brand);font-size:15px}
.cover .conf{text-transform:uppercase;letter-spacing:.14em;font-size:10px;color:var(--neg);font-weight:700}
.cover h1{font-size:25px;font-weight:700;margin:26px 0 4px;letter-spacing:-.01em}
.cover .subtitle{font-size:14px;color:var(--sub);font-weight:400;margin:0}
.docmeta{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);margin-top:26px}
.docmeta>div{padding:10px 14px;border-right:1px solid var(--line)}
.docmeta>div:last-child{border-right:none}
.docmeta .k{font-size:9.5px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted)}
.docmeta .v{font-size:13px;font-weight:600;margin-top:3px}
section{padding:26px 48px;border-bottom:1px solid var(--hair)}
.sec-no{font-size:11px;font-weight:700;color:var(--brand);letter-spacing:.04em}
.sec-h{font-size:17px;font-weight:700;margin:2px 0 14px;padding-bottom:8px;border-bottom:2px solid var(--rule)}
p{margin:0 0 11px}
.lead{color:var(--sub);max-width:74ch}
.twocol{display:grid;grid-template-columns:1fr 1fr;gap:28px}
.threecol{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
table{width:100%;border-collapse:collapse;font-size:12px;margin:6px 0 4px}
th{background:var(--tablehdr);text-align:left;padding:7px 9px;font-size:10px;text-transform:uppercase;
  letter-spacing:.05em;color:var(--sub);font-weight:700;border-bottom:1.5px solid var(--rule)}
td{padding:6px 9px;border-bottom:1px solid var(--line);vertical-align:top}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tbody tr:nth-child(even) td{background:#fafbfc}
tfoot td{font-weight:700;background:#eef2f6;border-top:2px solid var(--rule);border-bottom:none}
.kv{width:100%;font-size:12.5px}
.kv td{padding:5px 0;border-bottom:1px solid var(--hair)}
.kv td:first-child{color:var(--sub);width:52%}
.kv td:last-child{text-align:right;font-weight:600;font-variant-numeric:tabular-nums}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:0;border:1px solid var(--line)}
.metrics>div{padding:16px 18px;border-right:1px solid var(--line)}
.metrics>div:last-child{border-right:none}
.metrics .lbl{font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}
.metrics .val{font-size:23px;font-weight:700;margin-top:6px;font-variant-numeric:tabular-nums}
.metrics .meta{font-size:11px;color:var(--muted);margin-top:3px}
.tag{display:inline-block;font-size:9.5px;font-weight:700;padding:1px 6px;border-radius:3px;
  text-transform:uppercase;letter-spacing:.03em}
.tag.live{background:#e7f1ea;color:var(--pos)}
.tag.est{background:#f6efdf;color:var(--warn)}
.src{font-size:10px;color:var(--muted);display:block;margin-top:2px}
.callout{border-left:3px solid var(--brand);background:#f6f9fc;padding:11px 14px;font-size:12px;color:var(--sub);margin:12px 0}
.callout.warn{border-left-color:var(--warn);background:#fbf7ec}
.note{font-size:10.5px;color:var(--muted);margin-top:8px;font-style:italic}
ul.tight{margin:6px 0 11px;padding-left:18px}
ul.tight li{margin-bottom:4px;color:var(--sub)}
.stagerow{display:grid;grid-template-columns:1fr 150px 86px;gap:14px;align-items:center;
  padding:9px 0;border-bottom:1px solid var(--hair)}
.stagerow .nm{font-weight:600;font-size:12.5px}
.stagerow .ds{font-size:11px;color:var(--muted)}
.track{height:7px;background:#e9edf2;border-radius:4px;overflow:hidden}
.fill{height:100%;background:var(--brand)}
.stagerow .amt{text-align:right;font-weight:700;font-variant-numeric:tabular-nums}
.arch{display:grid;grid-template-columns:1fr;gap:0;border:1px solid var(--line);font-size:12px}
.arch .blk{display:grid;grid-template-columns:130px 1fr;border-bottom:1px solid var(--line)}
.arch .blk:last-child{border-bottom:none}
.arch .blk .h{background:var(--tablehdr);padding:8px 11px;font-weight:700;color:var(--brand-2);border-right:1px solid var(--line)}
.arch .blk .b{padding:8px 11px;color:var(--sub)}
footer{padding:18px 48px 30px;color:var(--muted);font-size:10.5px;line-height:1.5}
@media screen{
  html{overflow-x:auto;overflow-y:auto;scrollbar-width:thin;scrollbar-color:#c4cad4 transparent}
  html::-webkit-scrollbar{width:5px;height:5px}
  html::-webkit-scrollbar-thumb{background:#c4cad4;border-radius:9999px}
  html::-webkit-scrollbar-track{background:transparent}
  body{margin:0;background:#fff;overflow-x:auto}
  .doc{max-width:100%;width:100%;margin:0;box-shadow:none;overflow-x:hidden}
  section{overflow-x:auto;max-width:100%;scrollbar-width:thin;scrollbar-color:#c4cad4 transparent}
  section::-webkit-scrollbar{width:5px;height:5px}
  section::-webkit-scrollbar-thumb{background:#c4cad4;border-radius:9999px}
  section::-webkit-scrollbar-track{background:transparent}
}
@media print{.doc{box-shadow:none;margin:0}}
"""


def esc(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def fmt_inr(amount: Any, decimals: int = 2) -> str:
    try:
        val = float(amount)
    except (TypeError, ValueError):
        return "—"
    return f"₹{val:,.{decimals}f}"


def _bom_price_label(tag: str | None, note: str | None = None) -> str:
    """Plain-language BOM line pricing status (shown in the Source column)."""
    if str(tag or "").lower() == "live":
        return "Found in parts DB"
    note_l = (note or "").lower()
    if "generic passive" in note_l:
        return "Estimated (generic part)"
    if "not resolved" in note_l:
        return "Part not identified — estimated"
    if "live price break unavailable" in note_l:
        return "Found but no price — estimated"
    return "Not found — estimated"


def _tag_html(tag: str, source: str | None) -> str:
    """Generic source tag for fab/assembly/market rows."""
    cls = "live" if str(tag).lower() == "live" else "est"
    label = "Live quote" if cls == "live" else "Estimated"
    src = f'<span class="src">{esc(source)}</span>' if source else ""
    return f'<span class="tag {cls}">{esc(label)}</span>{src}'


def _bom_price_html(tag: str | None, note: str | None = None) -> str:
    cls = "live" if str(tag or "").lower() == "live" else "est"
    return f'<span class="tag {cls}">{esc(_bom_price_label(tag, note))}</span>'


def _confidence_tag(level: str | None) -> str:
    text = (level or "").strip() or "Med"
    cls = "live" if text.lower() in {"high", "live"} else "est"
    return f'<span class="tag {cls}">{esc(text)}</span>'


def _sec_title(rj: dict[str, Any], key: str, default: str) -> str:
    """The heading text for a section.

    Returns the user's per-section override from ``rj["section_titles"]`` (set via
    an edit's ``set_section_title`` op) when present, else the locked default. Only
    the heading TEXT is overridable — the section number and layout never change.
    """
    override = (rj.get("section_titles") or {}).get(key)
    if isinstance(override, str) and override.strip():
        return override.strip()
    return default


# --------------------------------------------------------------------------- #
# Section builders
# --------------------------------------------------------------------------- #
def _cover(rj: dict[str, Any]) -> str:
    meta = rj.get("meta") or {}
    product = rj.get("product") or {}
    title = meta.get("title") or "Reverse Engineering Cost Report"
    return f"""
  <div class="cover">
    <div class="org">
      <span class="logo">ELECBITS · TEARDOWN COST ENGINE</span>
      <span class="conf">Confidential — Internal</span>
    </div>
    <h1>{esc(title)}</h1>
    <p class="subtitle">{esc(product.get('name'))}{(' — ' + esc(product.get('subtitle'))) if product.get('subtitle') else ''}</p>
    <div class="docmeta">
      <div><div class="k">Product</div><div class="v">{esc(product.get('name'))}</div></div>
      <div><div class="k">Cost basis</div><div class="v">{_basis_label(meta.get('volume'))}</div></div>
      <div><div class="k">Inputs</div><div class="v">{esc(meta.get('inputs'))}</div></div>
      <div><div class="k">Reporting currency</div><div class="v">INR (₹)</div></div>
    </div>
  </div>"""


def _basis_label(volume: Any) -> str:
    """Cost basis shown on the cover/header — per-unit by default, qty if given."""
    try:
        v = int(volume or 1)
    except (TypeError, ValueError):
        v = 1
    return "Per unit" if v <= 1 else f"{v:,} units"


def _executive(rj: dict[str, Any]) -> str:
    product = rj.get("product") or {}
    metrics = rj.get("metrics") or {}
    dc = rj.get("dataConfidence") or {}
    findings = "".join(
        f"<li>{esc(item)}</li>" for item in (product.get("key_findings") or [])
    )
    callout_cls = "callout" if dc.get("all_live") else "callout warn"
    bom_per_unit = (rj.get("bom") or {}).get("subtotal_inr")
    return f"""
  <section>
    <div class="sec-no">01</div>
    <div class="sec-h">{esc(_sec_title(rj, "executive", "Executive Summary"))}</div>
    <p class="lead">{esc(product.get('summary_prose'))}</p>
    <div class="metrics">
      <div>
        <div class="lbl">Unit cost (1 unit)</div>
        <div class="val">{fmt_inr(metrics.get('ex_works_selected'))}</div>
        <div class="meta">Recurring per-unit · excl. one-time NRE</div>
      </div>
      <div>
        <div class="lbl">BOM cost / unit</div>
        <div class="val">{fmt_inr(bom_per_unit)}</div>
        <div class="meta">Components, landed (incl. duty)</div>
      </div>
      <div>
        <div class="lbl">One-time NRE (separate)</div>
        <div class="val">{fmt_inr(metrics.get('one_time_nre_inr'), 0)}</div>
        <div class="meta">Tooling, firmware, line setup — one-time</div>
      </div>
    </div>
    <div class="{callout_cls}"><strong>Data Confidence &amp; Notes.</strong> {esc(dc.get('prose'))}</div>
    <p style="margin-top:16px"><strong>Key findings.</strong></p>
    <ul class="tight">{findings}</ul>
  </section>"""


def _product_overview(rj: dict[str, Any]) -> str:
    product = rj.get("product") or {}
    kv_rows = "".join(
        f"<tr><td>{esc(row.get('k'))}</td><td>{esc(row.get('v'))}</td></tr>"
        for row in (product.get("overview") or [])
    )
    subs = product.get("subsystems") or []
    sub_rows = "".join(
        f"<tr><td>{esc(s.get('subsystem'))}</td><td>{esc(s.get('basis'))}</td>"
        f"<td>{_confidence_tag(s.get('confidence'))}</td></tr>"
        for s in subs
    ) or '<tr><td colspan="3" style="color:var(--muted)">No subsystem breakdown available.</td></tr>'
    return f"""
  <section>
    <div class="sec-no">02</div>
    <div class="sec-h">{esc(_sec_title(rj, "product_overview", "Product Overview"))}</div>
    <div class="twocol">
      <div>
        <table class="kv">{kv_rows}</table>
      </div>
      <div>
        <p style="font-weight:600;margin-bottom:6px">Identified subsystems &amp; evidence quality</p>
        <table>
          <thead><tr><th>Subsystem</th><th>Identification basis</th><th>Conf.</th></tr></thead>
          <tbody>{sub_rows}</tbody>
        </table>
      </div>
    </div>
  </section>"""


def _architecture(rj: dict[str, Any]) -> str:
    arch = rj.get("architecture") or {}
    blocks = "".join(
        f'<div class="blk"><div class="h">{esc(b.get("block"))}</div>'
        f'<div class="b">{esc(b.get("description"))}</div></div>'
        for b in (arch.get("blocks") or [])
    ) or '<div class="blk"><div class="h">Architecture</div><div class="b">Not enough detail to break out functional blocks.</div></div>'
    insight = arch.get("insight")
    insight_html = (
        f'<div class="callout"><strong>Cost-driver insight:</strong> {esc(insight)}</div>'
        if insight else ""
    )
    return f"""
  <section>
    <div class="sec-no">03</div>
    <div class="sec-h">{esc(_sec_title(rj, "architecture", "Architecture Analysis"))}</div>
    <p class="lead">{esc(arch.get('prose'))}</p>
    <div class="arch">{blocks}</div>
    {insight_html}
  </section>"""


def _cost_by_stage(rj: dict[str, Any]) -> str:
    stages = rj.get("stages") or {}
    rows = stages.get("rows") or []
    vol = int(stages.get("volume") or rj.get("metrics", {}).get("selected_volume") or 0)
    bars = ""
    for r in rows:
        pct = max(0.0, min(100.0, float(r.get("pct") or 0)))
        bars += (
            f'<div class="stagerow"><div><div class="nm">{esc(r.get("stage"))}</div>'
            f'<div class="ds">{pct:.0f}% of unit cost</div></div>'
            f'<div class="track"><div class="fill" style="width:{pct:.0f}%"></div></div>'
            f'<div class="amt">{fmt_inr(r.get("amount"))}</div></div>'
        )
    total = stages.get("total")
    bars += (
        f'<div class="stagerow" style="border-bottom:none;margin-top:4px">'
        f'<div><div class="nm" style="color:var(--brand)">Total unit cost (1 unit)</div>'
        f'<div class="ds">Recurring per-unit cost · one-time NRE shown separately</div></div>'
        f'<div></div><div class="amt" style="font-size:15px;color:var(--brand)">{fmt_inr(total)}</div></div>'
    )
    return f"""
  <section>
    <div class="sec-no">04</div>
    <div class="sec-h">{esc(_sec_title(rj, "cost_by_stage", "Cost by Manufacturing Stage"))}</div>
    <p class="lead">Recurring per-unit cost for a single unit, decomposed by standard manufacturing stage.
    One-time NRE (tooling, firmware, line setup) is reported separately and is not included here.
    Bars are proportional to contribution.</p>
    {bars}
  </section>"""


def _bom(rj: dict[str, Any]) -> str:
    bom = rj.get("bom") or {}
    rows_html = ""
    for r in bom.get("rows") or []:
        rows_html += (
            "<tr>"
            f"<td>{esc(r.get('sno'))}</td>"
            f"<td>{esc(r.get('mpn'))}</td>"
            f"<td>{esc(r.get('make'))}</td>"
            f"<td>{esc(r.get('description'))}</td>"
            f"<td>{esc(r.get('designator'))}</td>"
            f"<td class='n'>{esc(r.get('qty'))}</td>"
            f"<td>{esc(r.get('pkg'))}</td>"
            f"<td class='n'>{fmt_inr(r.get('unit_inr'))}</td>"
            f"<td class='n'>{esc(r.get('bcd_igst'))}</td>"
            f"<td class='n'>{fmt_inr(r.get('ext_inr'))}</td>"
            f"<td>{_bom_price_html(r.get('tag'), r.get('note'))}</td>"
            "</tr>"
        )
    if not rows_html:
        rows_html = '<tr><td colspan="11" style="color:var(--muted)">No components identified.</td></tr>'
    subtotal = bom.get("subtotal_inr")
    return f"""
  <section>
    <div class="sec-no">05</div>
    <div class="sec-h">{esc(_sec_title(rj, "bom", "Bill of Materials"))}</div>
    <p class="lead">Reconstructed BOM with per-line landed cost. Unit pricing reflects the qty break at the
    selected volume, duty-loaded per HSN classification. Each line shows whether the price was
    <strong>found in the parts database</strong> or <strong>estimated</strong> because the MPN was not found.</p>
    <table>
      <thead>
        <tr>
          <th>S.No</th><th>MPN</th><th>Make</th><th>Description</th><th>Desig.</th>
          <th class="n">Qty</th><th>Pkg</th><th class="n">Unit ₹</th><th class="n">BCD/IGST</th>
          <th class="n">Ext. ₹</th><th>Price status</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
      <tfoot>
        <tr><td colspan="9">BOM subtotal — landed (incl. BCD, SWS, IGST)</td>
        <td class="n">{fmt_inr(subtotal)}</td><td></td></tr>
      </tfoot>
    </table>
    <p class="note">Duty is modeled per-MPN from the inferred HSN code. Lines marked
    &ldquo;Not found — estimated&rdquo; use a predicted price by component type, not a real catalog quote.</p>
  </section>"""


def _fab_assembly(rj: dict[str, Any]) -> str:
    fab = rj.get("fab") or {}
    params = fab.get("params") or {}
    assembly = rj.get("assembly") or {}
    fab_kv = (
        f"<tr><td>Base material</td><td>{esc(params.get('Material') or 'FR-4')}</td></tr>"
        f"<tr><td>Layers</td><td>{esc(params.get('layers'))}</td></tr>"
        f"<tr><td>Dimensions</td><td>{esc(params.get('Length'))} × {esc(params.get('width'))} mm</td></tr>"
        f"<tr><td>Thickness</td><td>{esc(params.get('Thickness'))} mm</td></tr>"
        f"<tr><td>Solder mask</td><td>{esc(params.get('SolderMask'))}</td></tr>"
        f"<tr><td>Surface finish</td><td>{esc(params.get('surface'))}</td></tr>"
        f"<tr><td>Source</td><td>{_tag_html(fab.get('tag'), fab.get('source'))}</td></tr>"
        f"<tr style='font-weight:700'><td>Fab cost / board</td><td>{fmt_inr(fab.get('selected_inr'))}</td></tr>"
    )
    asm_kv = (
        f"<tr><td>Total solder joints</td><td>{esc(assembly.get('total_joints'))}</td></tr>"
        f"<tr><td>Setup fee (NRE)</td><td>{fmt_inr(assembly.get('setup_fee_inr'), 0)}</td></tr>"
        f"<tr><td>Stencil (NRE)</td><td>{fmt_inr(assembly.get('stencil_fee_inr'), 0)}</td></tr>"
        f"<tr><td>Rate per joint</td><td>{fmt_inr(assembly.get('rate_per_joint_inr'), 2)}</td></tr>"
        f"<tr><td>Source</td><td>{_tag_html(assembly.get('tag'), assembly.get('source'))}</td></tr>"
        f"<tr style='font-weight:700'><td>Assembly / board (per-unit)</td><td>{fmt_inr(assembly.get('per_unit_inr'))}</td></tr>"
    )
    return f"""
  <section>
    <div class="sec-no">06</div>
    <div class="sec-h">{esc(_sec_title(rj, "fab_assembly", "PCB Fabrication & Assembly Detail"))}</div>
    <div class="twocol">
      <div>
        <p style="font-weight:700;color:var(--brand-2);margin-bottom:4px">A · PCB Fabrication — PCBWay</p>
        <table class="kv">{fab_kv}</table>
      </div>
      <div>
        <p style="font-weight:700;color:var(--brand-2);margin-bottom:4px">B · PCB Assembly — SMT + THT</p>
        <table class="kv">{asm_kv}</table>
      </div>
    </div>
    <p class="note">Placement count is derived from the BOM. The per-joint rate and setup NRE come from a
    configurable Indian EMS rate card. PCBWay USD quotes are converted to INR at the rate shown on the cover.</p>
  </section>"""


def _market(rj: dict[str, Any]) -> str:
    mc = rj.get("marketContext") or {}
    comparables = mc.get("comparables") or []
    comp_rows = "".join(
        f"<tr><td>{esc(c.get('name'))}</td><td class='n'>{fmt_inr(c.get('retail_mrp_inr'), 0)}</td>"
        f"<td>{esc(c.get('note'))}</td></tr>"
        for c in comparables
    )
    comp_block = (
        f"""<table>
          <thead><tr><th>Comparable</th><th class="n">Retail MRP</th><th>Note</th></tr></thead>
          <tbody>{comp_rows}</tbody>
        </table>"""
        if comp_rows
        else '<p class="note">Retail comparables were not available for this run.</p>'
    )
    obs = "".join(f"<li>{esc(o)}</li>" for o in (mc.get("observations") or []))
    obs_block = f'<ul class="tight">{obs}</ul>' if obs else '<p class="note">No sourcing observations gathered.</p>'
    margin = mc.get("margin_band")
    margin_line = f'<p class="note">Implied gross margin band: {esc(margin)} (approximate).</p>' if margin else ""
    return f"""
  <section>
    <div class="sec-no">07</div>
    <div class="sec-h">{esc(_sec_title(rj, "market", "Market Context"))}</div>
    <p class="lead">{esc(mc.get('prose'))}</p>
    <div class="twocol">
      <div>
        <p style="font-weight:600;margin-bottom:4px">Comparable units &amp; margin band</p>
        {comp_block}
        {margin_line}
        <p class="note">Web-sourced context (low confidence) — tagged Est. Not a vendor quotation.</p>
      </div>
      <div>
        <p style="font-weight:600;margin-bottom:4px">Sourcing &amp; supply observations</p>
        {obs_block}
      </div>
    </div>
  </section>"""


def _methodology(rj: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{esc(m.get('stage'))}</td><td>{esc(m.get('source'))}</td>"
        f"<td>{esc(m.get('method'))}</td><td>{_confidence_tag(m.get('confidence'))}</td></tr>"
        for m in (rj.get("methodology") or [])
    )
    dq = "".join(f"<li>{esc(note)}</li>" for note in (rj.get("dataQuality") or []))
    dq_block = (
        f'<p style="font-weight:600;margin:14px 0 4px">Data quality notes</p><ul class="tight">{dq}</ul>'
        if dq else ""
    )
    return f"""
  <section style="border-bottom:none">
    <div class="sec-no">08</div>
    <div class="sec-h">{esc(_sec_title(rj, "methodology", "Methodology & Confidence"))}</div>
    <table>
      <thead><tr><th>Stage</th><th>Data source</th><th>Method</th><th>Confidence</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    {dq_block}
  </section>"""


def _images(rj: dict[str, Any]) -> str:
    images = [img for img in (rj.get("images") or []) if isinstance(img, dict) and img.get("url")]
    if not images:
        return ""
    cards = ""
    for img in images:
        cards += (
            f'<figure style="margin:0">'
            f'<img src="{esc(img.get("url"))}" alt="{esc(img.get("caption") or "attached image")}" '
            f'style="width:100%;border:1px solid var(--line);border-radius:4px"/>'
            f"</figure>"
        )
    # Images render as a bare block — no "Reference Images" heading and no
    # boilerplate lead text — so attached photos appear cleanly on their own.
    return f"""
  <section>
    <div class="threecol" style="gap:18px">{cards}</div>
  </section>"""


def _image_position(rj: dict[str, Any]) -> str:
    """Where the attached-images block goes: 'after_executive' or 'end'."""
    for img in rj.get("images") or []:
        if isinstance(img, dict) and (img.get("position") or "").lower() == "after_executive":
            return "after_executive"
    return "end"


# --------------------------------------------------------------------------- #
# Markdown rendering (for the in-app preview panel — avoids the PDF's CSS).
# The PDF still renders from render_html(); this is display-only and carries the
# same numbers/structure as GitHub-flavored Markdown.
# --------------------------------------------------------------------------- #
def _mc(value: Any) -> str:
    """Markdown table cell: stringify and escape pipes/newlines."""
    if value is None:
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ").strip() or "—"


def _tag_text(tag: str | None, source: str | None, *, note: str | None = None) -> str:
    if str(source or "").lower() in {"jlcpcb", "rate-card"} or note is not None:
        return _bom_price_label(tag, note)
    if str(tag or "").lower() == "live":
        return "Live quote"
    return "Estimated"


def _md_table(header: list[str], aligns: list[str], rows: list[list[str]]) -> str:
    """Build ONE contiguous GFM table (no blank lines between header and rows)."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(aligns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _md_images(rj: dict[str, Any]) -> list[str]:
    images = [img for img in (rj.get("images") or []) if isinstance(img, dict) and img.get("url")]
    if not images:
        return []
    # Image-only: no "## Reference Images" heading and no caption line — just the
    # image(s), so attached photos appear cleanly on their own.
    return [f"![{str(img.get('caption') or 'attached image')}]({img['url']})" for img in images]


def render_markdown(report_json: dict[str, Any]) -> str:
    rj = report_json or {}
    meta = rj.get("meta") or {}
    product = rj.get("product") or {}
    metrics = rj.get("metrics") or {}
    hidden = {str(s).lower() for s in (rj.get("hidden_sections") or [])}
    out: list[str] = []

    # Header
    out.append(f"# {meta.get('title') or 'Reverse Engineering Cost Report'}")
    sub_bits = []
    if product.get("subtitle"):
        sub_bits.append(str(product["subtitle"]))
    sub_bits.append("All costs in INR (₹)")
    out.append("*" + " · ".join(sub_bits) + "*")

    image_blocks = _md_images(rj)
    img_after_exec = _image_position(rj) == "after_executive"

    # 01 Executive Summary
    if "executive" not in hidden:
        out.append(f"## 01 · {_sec_title(rj, 'executive', 'Executive Summary')}")
        if product.get("summary_prose"):
            out.append(str(product["summary_prose"]).strip())
        bom_per_unit = (rj.get("bom") or {}).get("subtotal_inr")
        out.append(_md_table(
            ["Metric", "Value"], ["---", "---:"],
            [
                ["Unit cost (1 unit)", fmt_inr(metrics.get("ex_works_selected"))],
                ["BOM cost / unit (landed)", fmt_inr(bom_per_unit)],
                ["One-time NRE (separate)", fmt_inr(metrics.get("one_time_nre_inr"), 0)],
            ],
        ))
        dc = rj.get("dataConfidence") or {}
        if dc.get("prose"):
            out.append(f"> **Data Confidence & Notes.** {dc['prose']}")
        findings = product.get("key_findings") or []
        if findings:
            out.append("**Key findings.**\n" + "\n".join(f"- {f}" for f in findings))

    # Reference images placed directly below the Executive Summary, if requested.
    if image_blocks and img_after_exec:
        out.extend(image_blocks)

    # 02 Product Overview
    if "product_overview" not in hidden:
        out.append(f"## 02 · {_sec_title(rj, 'product_overview', 'Product Overview')}")
        overview = product.get("overview") or []
        if overview:
            out.append(_md_table(
                ["Field", "Value"], ["---", "---"],
                [[_mc(r.get("k")), _mc(r.get("v"))] for r in overview],
            ))
        subs = product.get("subsystems") or []
        if subs:
            out.append("**Identified subsystems & evidence quality**")
            out.append(_md_table(
                ["Subsystem", "Identification basis", "Conf."], ["---", "---", "---"],
                [[_mc(s.get("subsystem")), _mc(s.get("basis")), _mc(s.get("confidence"))] for s in subs],
            ))

    # 03 Architecture Analysis
    if "architecture" not in hidden:
        arch = rj.get("architecture") or {}
        out.append(f"## 03 · {_sec_title(rj, 'architecture', 'Architecture Analysis')}")
        if arch.get("prose"):
            out.append(str(arch["prose"]).strip())
        blocks = arch.get("blocks") or []
        if blocks:
            out.append("\n".join(
                f"- **{_mc(b.get('block'))}** — {_mc(b.get('description'))}" for b in blocks
            ))
        if arch.get("insight"):
            out.append(f"> **Cost-driver insight:** {arch['insight']}")

    # 04 Cost by Manufacturing Stage
    if "cost_by_stage" not in hidden:
        stages = rj.get("stages") or {}
        out.append(f"## 04 · {_sec_title(rj, 'cost_by_stage', 'Cost by Manufacturing Stage')} (1 unit, recurring)")
        stage_rows = [
            [_mc(r.get("stage")), f"{r.get('pct', 0):.0f}%", fmt_inr(r.get("amount"))]
            for r in stages.get("rows") or []
        ]
        stage_rows.append(["**Total unit cost (1 unit)**", "", f"**{fmt_inr(stages.get('total'))}**"])
        out.append(_md_table(["Stage", "Share", "Per-unit"], ["---", "---:", "---:"], stage_rows))

    # 05 Bill of Materials
    if "bom" not in hidden:
        bom = rj.get("bom") or {}
        bom_rows = [
            [
                _mc(r.get("sno")), _mc(r.get("mpn")), _mc(r.get("make")), _mc(r.get("description")),
                _mc(r.get("designator")), _mc(r.get("qty")), _mc(r.get("pkg")),
                fmt_inr(r.get("unit_inr")), _mc(r.get("bcd_igst")), fmt_inr(r.get("ext_inr")),
                _mc(_tag_text(r.get("tag"), r.get("source"), note=r.get("note"))),
            ]
            for r in bom.get("rows") or []
        ]
        bom_rows.append(["", "", "", "", "", "", "", "", "**Subtotal**", f"**{fmt_inr(bom.get('subtotal_inr'))}**", ""])
        out.append(f"## 05 · {_sec_title(rj, 'bom', 'Bill of Materials')}")
        out.append(_md_table(
            ["S.No", "MPN", "Make", "Description", "Desig.", "Qty", "Pkg", "Unit ₹", "BCD/IGST", "Ext ₹", "Price status"],
            ["---:", "---", "---", "---", "---", "---:", "---", "---:", "---", "---:", "---"],
            bom_rows,
        ))
        out.append("*Duty is modeled per-MPN from the inferred HSN code. Lines marked \"Not found — estimated\" use a predicted price by component type.*")

    # 06 PCB Fab & Assembly
    if "fab_assembly" not in hidden:
        fab = rj.get("fab") or {}
        params = fab.get("params") or {}
        asm = rj.get("assembly") or {}
        out.append(f"## 06 · {_sec_title(rj, 'fab_assembly', 'PCB Fabrication & Assembly Detail')}")
        out.append("**A · PCB Fabrication — PCBWay**")
        out.append(_md_table(["Parameter", "Value"], ["---", "---"], [
            ["Base material", _mc(params.get("Material") or "FR-4")],
            ["Layers", _mc(params.get("layers"))],
            ["Dimensions", f"{_mc(params.get('Length'))} × {_mc(params.get('width'))} mm"],
            ["Thickness", f"{_mc(params.get('Thickness'))} mm"],
            ["Surface finish", _mc(params.get("surface"))],
            ["Source", _mc(_tag_text(fab.get("tag"), fab.get("source")))],
            ["**Fab cost / board**", f"**{fmt_inr(fab.get('selected_inr'))}**"],
        ]))
        out.append("**B · PCB Assembly — SMT + THT**")
        out.append(_md_table(["Parameter", "Value"], ["---", "---"], [
            ["Total solder joints", _mc(asm.get("total_joints"))],
            ["Setup fee (NRE)", fmt_inr(asm.get("setup_fee_inr"), 0)],
            ["Stencil (NRE)", fmt_inr(asm.get("stencil_fee_inr"), 0)],
            ["Rate per joint", fmt_inr(asm.get("rate_per_joint_inr"), 2)],
            ["Source", _mc(_tag_text(asm.get("tag"), asm.get("source")))],
            ["**Assembly / board (per-unit)**", f"**{fmt_inr(asm.get('per_unit_inr'))}**"],
        ]))

    # 07 Market Context
    if "market" not in hidden:
        mc = rj.get("marketContext") or {}
        out.append(f"## 07 · {_sec_title(rj, 'market', 'Market Context')}")
        if mc.get("prose"):
            out.append(str(mc["prose"]).strip())
        comparables = mc.get("comparables") or []
        if comparables:
            out.append(_md_table(
                ["Comparable", "Retail MRP", "Note"], ["---", "---:", "---"],
                [[_mc(c.get("name")), fmt_inr(c.get("retail_mrp_inr"), 0), _mc(c.get("note"))] for c in comparables],
            ))
        if mc.get("margin_band"):
            out.append(f"*Implied gross margin band: {mc['margin_band']} (approximate).*")
        obs = mc.get("observations") or []
        if obs:
            out.append("**Sourcing & supply observations**\n" + "\n".join(f"- {o}" for o in obs))

    # 08 Methodology
    if "methodology" not in hidden:
        out.append(f"## 08 · {_sec_title(rj, 'methodology', 'Methodology & Confidence')}")
        out.append(_md_table(
            ["Stage", "Data source", "Method", "Confidence"], ["---", "---", "---", "---"],
            [[_mc(m.get("stage")), _mc(m.get("source")), _mc(m.get("method")), _mc(m.get("confidence"))]
             for m in rj.get("methodology") or []],
        ))
        dq = rj.get("dataQuality") or []
        if dq:
            out.append("**Data quality notes**\n" + "\n".join(f"- {n}" for n in dq))

    # Reference Images at the end (default placement).
    if image_blocks and not img_after_exec:
        out.extend(image_blocks)

    return "\n\n".join(out)


def render_html(report_json: dict[str, Any]) -> str:
    """Fill the locked template skeleton from the aggregated JSON."""
    rj = report_json or {}
    meta = rj.get("meta") or {}
    hidden = {str(s).lower() for s in (rj.get("hidden_sections") or [])}
    images_html = _images(rj)
    img_after_exec = _image_position(rj) == "after_executive"
    sections = [
        ("executive", _executive(rj)),
        ("product_overview", _product_overview(rj)),
        ("architecture", _architecture(rj)),
        ("cost_by_stage", _cost_by_stage(rj)),
        ("bom", _bom(rj)),
        ("fab_assembly", _fab_assembly(rj)),
        ("market", _market(rj)),
        ("methodology", _methodology(rj)),
    ]
    parts = [_cover(rj)]
    for key, section_html in sections:
        if key in hidden:
            continue
        parts.append(section_html)
        if key == "executive" and img_after_exec and images_html:
            parts.append(images_html)
    if images_html and not img_after_exec:
        parts.append(images_html)
    parts.append(
        """
  <footer>
    Generated by the Elecbits Teardown Cost Engine. Figures are estimates for internal planning and carry the
    confidence bands stated in Section 08. Live distributor pricing and FX are timestamped at generation and will
    drift over time. All amounts are single-unit landed costs in Indian Rupees (₹) unless otherwise noted; one-time
    NRE is shown separately. This document is not a vendor quotation and does not constitute a commercial offer.
  </footer>"""
    )
    body = "".join(parts)
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>"
        f"<title>{esc(meta.get('title') or 'Reverse Engineering Cost Report')}</title>"
        f"<style>{_CSS}</style></head><body><div class='doc'>{body}</div></body></html>"
    )
