"""
src/rag/report_render.py
--------------------------
Presentation-only renderers for the evidence report (run report + per-unit
reports). Pure functions of dicts: no pipeline logic, no I/O. Text output is
ASCII-safe for Windows consoles; all log-derived text goes through
`display_safe` so RTLO / control characters are shown as visible escapes.
"""

from typing import Any, Dict, List

from src.mapping.text_prep import display_safe

SECTION_TITLES = [
    ("observed_facts", "Observed facts (from input logs)"),
    ("computed_findings", "Computed findings (clustering / novelty / graph)"),
    ("technique_candidates", "ATT&CK technique candidates (semantic similarity leads)"),
    ("reference_information", "Retrieved ATT&CK reference passages"),
    ("graph_evidence", "Graph evidence (traversal paths)"),
    ("ontology_facts", "Ontology facts"),
    ("inferences", "Inferences (hedged)"),
    ("uncertainty_and_missing_evidence", "Uncertainty and missing evidence"),
    ("recommended_investigation_steps", "Recommended investigation steps"),
]


def _ascii(s: str) -> str:
    return display_safe(s).encode("ascii", "replace").decode("ascii")


def _item_line(key: str, it: Any, max_len: int) -> str:
    if isinstance(it, str):
        return _ascii(it)[:max_len]
    if key == "technique_candidates":
        return (f"#{it.get('rank')} {it.get('technique_id')} {_ascii(it.get('technique_name'))} "
                f"[similarity {it.get('similarity_score')}, {it.get('status')}] "
                f"unit context events (not technique-specific): {', '.join(u[:8] for u in it.get('supporting_event_uids', [])[:4])}")
    if key == "reference_information":
        txt = _ascii(it.get("text", ""))
        return f"{it.get('ref_id')}: {_ascii(it.get('title'))} -- {txt[:max_len//2]}{'...' if len(txt) > max_len//2 else ''}"
    tag = f"[{it['id']}] " if it.get("id") else ""
    line = tag + _ascii(it.get("text", ""))
    if it.get("certainty"):
        line += f" (certainty: {it['certainty']})"
    return line[:max_len + 40]


def render_unit_text(unit_report: Dict[str, Any], max_line: int = 300, sections: bool = True) -> List[str]:
    out = [f"{unit_report['unit_id']} [{unit_report['scope']}] {_ascii(unit_report['title'])}",
           f"  Summary ({unit_report.get('generated_by')}{', MOCK' if unit_report.get('is_mock') else ''}): "
           f"{_ascii(unit_report.get('analyst_summary'))}"]
    if not sections:
        return out
    for key, title in SECTION_TITLES:
        items = unit_report.get("sections", {}).get(key) or []
        out.append(f"  {title}:")
        if not items:
            out.append("    (none)")
        for it in items:
            out.append("    - " + _item_line(key, it, max_line))
    return out


def render_report_text(report: Dict[str, Any], unit_reports: List[Dict[str, Any]], detail_units: int = 5) -> str:
    L: List[str] = ["TECHNICAL APPENDIX - per-unit detail. The prioritized report is analyst_report.md.", "",
                    _ascii(report.get("title", "report")), "=" * 72,
                    f"run_id {report.get('run_id')}   generated {report.get('generated_at')}   "
                    f"produced by: {report.get('generated_by')}"]
    if report.get("status_banner"):
        L.append("STATUS: " + _ascii(report["status_banner"]))
    ov = report.get("overview", {})
    L += ["", "OVERVIEW"]
    for k in ("events", "clusters", "noise_events", "noise_units_analysed", "units_analysed", "novelty_label_counts",
              "technique_outcomes", "corpus_version", "retrieval_mode", "graphrag_claim"):
        if k in ov:
            L.append(f"  {k}: {_ascii(ov[k])}")
    L += ["", "TRIAGE (" + _ascii(report.get("triage_note", "")) + ")"]
    for kf in report.get("key_findings", []):
        L.append(f"  {kf['unit_id']:<4} novelty {kf.get('triage_priority_novelty')}  {kf.get('outcome')}")
        L.append("       " + _ascii(kf.get("headline")))
    L += ["", "CONCLUSIONS"] + ["  - " + _ascii(c) for c in report.get("conclusions", [])]
    v = report.get("validation", {})
    L += ["", f"EVIDENCE VALIDATION: {'passed' if v.get('passed') else 'FAILED'} ({v.get('issue_count', 0)} issue(s))"]
    L += ["  - " + _ascii(i) for i in v.get("issues", [])[:10]]
    L += ["", "DETAIL (top units by novelty)", "-" * 72]
    by_id = {u["unit_id"]: u for u in unit_reports}
    for kf in report.get("key_findings", [])[:detail_units]:
        ur = by_id.get(kf["unit_id"])
        if ur:
            L += render_unit_text(ur) + [""]
    L += ["DISCLAIMER: " + _ascii(report.get("disclaimer", ""))]
    return "\n".join(L) + "\n"


def render_report_markdown(report: Dict[str, Any], unit_reports: List[Dict[str, Any]]) -> str:
    L = ["> **Technical appendix** - per-unit detail for every unit. The prioritized, deduplicated report is `analyst_report.md`.", "",
         f"# {report.get('title')}", "", f"*run `{report.get('run_id')}` - {report.get('generated_at')} - "
         f"produced by `{report.get('generated_by')}`*", ""]
    if report.get("status_banner"):
        L += [f"> **{report['status_banner']}**", ""]
    L += ["## Triage", report.get("triage_note", ""), ""]
    for kf in report.get("key_findings", []):
        L.append(f"- **{kf['unit_id']}** (novelty {kf.get('triage_priority_novelty')}, {kf.get('outcome')}): {kf.get('headline')}")
    L += ["", "## Conclusions"] + [f"- {c}" for c in report.get("conclusions", [])] + [""]
    for ur in unit_reports:
        L += [f"## {ur['unit_id']} - {ur['title']}", f"*{ur.get('generated_by')}*", "", ur.get("analyst_summary", ""), ""]
        for key, title in SECTION_TITLES:
            items = ur.get("sections", {}).get(key) or []
            L.append(f"### {title}")
            L += [f"- {_item_line(key, it, 400)}" for it in items] or ["- (none)"]
            L.append("")
    L += ["---", report.get("disclaimer", "")]
    return "\n".join(L) + "\n"
