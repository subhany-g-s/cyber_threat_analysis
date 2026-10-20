"""
src/rag/analyst_render.py
---------------------------
Presentation-only renderer for the analyst report built by findings.py (and
optionally enriched by finding_llm.py). Pure function of a dict: no pipeline
logic, no I/O. One layout is produced as Markdown; the terminal version is the
same text with Markdown emphasis removed and non-ASCII characters replaced
(Windows consoles). Log-derived text was already passed through `display_safe`
when the report was built, so control characters appear as visible escapes.

Layout: A executive summary | B prioritized findings | C overall assessment |
D investigation plan | E pointers to the technical appendix.
"""

import re
from typing import Any, Dict, List

SHOW = {"observed": 4, "why": 3, "evidence": 5, "attack": 3, "missing": 4, "benign": 3, "steps": 4}


def _src_mark(item: Dict[str, Any]) -> str:
    return " [LLM]" if str(item.get("source", "")).startswith("llm:") else ""


def common_missing(a: Dict[str, Any]) -> List[str]:
    """Missing-evidence items that apply to every shown finding: stated once (section C), not repeated per finding."""
    fs = a["findings"]
    if len(fs) < 2:
        return []
    sets = [{m["text"] for m in f["missing_evidence"]} for f in fs]
    return sorted(set.intersection(*sets))


def _units(f: Dict[str, Any]) -> str:
    ids = f["unit_ids"]
    return ", ".join(ids[:5]) + (f" (+{len(ids) - 5} more)" if len(ids) > 5 else "")


def _finding_md(f: Dict[str, Any], skip_missing: List[str]) -> List[str]:
    a = f["analysis"]
    llm = a["source"].startswith("llm:")
    L = [f"### {f['id']} - [{f['priority'].upper()}] {f['title']}", ""]
    merged = (f" Combined units ({_units(f)}) because " + " and ".join(f["merge_basis"]) + ".") if f["merged"] else f" Unit: {f['unit_ids'][0]}."
    L.append(f"*Status: **{f['status']}** ({f['status_reason']}). Priority basis: {f['priority_reason']}.{merged} "
             f"Text source: {'LLM (' + a['source'][4:] + '), validated' if llm else 'deterministic'}.*")
    L += ["", "**What was observed**"] + [f"- {t}" for t in f["observed"][:SHOW["observed"]]]
    L += ["", "**Why it matters**"] + [f"- {t}" for t in f["why_it_matters"][:SHOW["why"]]]
    if llm:
        L += ["", "**Analyst assessment (LLM, validated against the evidence)**",
              f"- What happened: {a['what_happened']['text']} ({', '.join(a['what_happened']['evidence'])})",
              f"- Why unusual: {a['why_unusual']['text']}"]
        if a["security_relevance"]["text"]:
            L.append(f"- Security relevance (hypothesis): {a['security_relevance']['text']}")
    L += ["", "**Supporting evidence**"]
    for e in f["evidence"][:SHOW["evidence"]]:
        ind = f" Indicators: {', '.join(e['indicators'])}." if e["indicators"] else ""
        nov = f" Novelty {e['novelty_score']}." if e["novelty_score"] is not None else ""
        L.append(f"- `{e['ref']}` {e['timestamp'][:19].replace('T', ' ') + ' UTC' if e['timestamp'] else 'time n/a'}, {e['host'] or 'host n/a'}: "
                 f"{e['summary']}. Source: {e['source']}.{nov}{ind}")
    L += ["", "**ATT&CK assessment** (text-similarity leads, not confirmed techniques)"]
    if not f["attack"]:
        L.append(f"- {f['attack_note']}")
    for t in f["attack"][:SHOW["attack"]]:
        tac = f" ({', '.join(t['tactics'][:2])})" if t["tactics"] else ""
        L.append(f"- **{t['technique_id']} {t['name']}**{tac}: unit-level similarity {t['unit_similarity']} (uncalibrated). Fit: **{t['fit']}**.")
        L.append(f"  - Why a candidate: {t['llm_rationale'] if t.get('llm_rationale') else t['why_candidate']}")
        if t["event_support"]:
            L.append("  - Event-level support: " + ", ".join(f"`{s['ref']}` ({s['event_level_similarity']})" for s in t["event_support"]) + " - ranking signal only.")
        else:
            L.append(f"  - Specific support: none. {t['support_basis'][0].upper() + t['support_basis'][1:]}.")
        miss = t.get("llm_missing") or (f"ATT&CK detection guidance suggests: {t['detection_guidance']}" if t["detection_guidance"] else "")
        if miss:
            L.append(f"  - Missing: {miss}")
    if f["attack"]:
        L.append("  - Limits: " + f["attack"][0]["limitations"].split(" Unit-level")[0])
    own_missing = [m for m in f["missing_evidence"] if m["text"] not in skip_missing][:SHOW["missing"]]
    L += ["", "**Missing evidence**" + (" (besides the gaps common to all findings, see C)" if skip_missing else "")]
    L += [f"- {m['text']}{_src_mark(m)}" for m in own_missing] or ["- Nothing beyond the gaps common to all findings."]
    L += ["", "**Benign explanations to rule out**"] + [f"- {m['text']}{_src_mark(m)}" for m in f["benign_explanations"][:SHOW["benign"]]]
    L += ["", "**Recommended next steps**"]
    for i, s in enumerate(f["next_steps"][:SHOW["steps"]], start=1):
        L.append(f"{i}. {s['action']}{_src_mark(s)}")
    c = a["conclusion"]
    L += ["", f"**Can be concluded:** {c['supported']}", f"**Remains uncertain:** {c['uncertain']}"]
    if f["related"]:
        L.append("**Related, not merged:** " + "; ".join(
            f"{r['unit_id']} ({r['finding'] if r['finding'] else 'lower-priority candidate, in appendix' if r['candidate'] else 'not a finding'}): {r['relation']}"
            for r in f["related"]))
    L += ["", "**Evidence references:** " + ", ".join(f"`{r}`" for r in f["evidence_refs"]), ""]
    return L


def render_analyst_markdown(a: Dict[str, Any], header: Dict[str, Any]) -> str:
    ex, mode = a["executive_summary"], a["analysis_mode"]
    L = ["# SUBHANY threat analysis - suspicious-behaviour report", "",
         f"*Run `{header.get('run_id')}` - generated {header.get('generated_at')} - analysis mode: **{mode['mode'].upper()}***", ""]
    L += ["## A. Executive summary", "", ex["assessment"], ""]
    L.append(f"- **Scope:** {ex['events_analysed']} events analysed -> {ex['clusters']} behaviour clusters and {ex['noise_events']} unclustered (noise) events; "
             f"{ex['units_reviewed']} units reviewed.")
    pr = ", ".join(f"{k} {v}" for k, v in ex["findings_by_priority"].items()) or "none"
    L.append(f"- **Findings worth investigating:** {ex['findings']} ({pr})"
             + (f"; {ex['further_candidates_not_shown']} further lower-priority candidate(s) kept in the technical appendix." if ex["further_candidates_not_shown"] else "."))
    if ex["highest_priority"]:
        h = ex["highest_priority"]
        L.append(f"- **Highest priority:** {h['finding']} [{h['priority']}] {h['title']} - {h['status']}.")
    if ex["main_reasons"]:
        L.append("- **Main reasons for concern:**")
        L += [f"  - {r}" for r in ex["main_reasons"]]
    L.append("- **Evidence limitations:**")
    L += [f"  - {r}" for r in ex["evidence_limitations"]]
    L.append(f"- **Analysis mode:** {ex['analysis_mode']}")
    L += ["", "## B. Prioritized suspicious-behaviour findings", ""]
    if not a["findings"]:
        L += ["No finding met the reporting criteria (see the prioritization rules in `analyst_report.json`).", ""]
    skip = common_missing(a)
    for f in a["findings"]:
        L += _finding_md(f, skip)
    o = a["overall_assessment"]
    L += ["## C. Overall assessment", ""]
    if o["significant_patterns"]:
        L += ["**Most significant patterns**"] + [f"- {s}" for s in o["significant_patterns"]] + [""]
    L += [f"**Common explanation?** {o['common_explanation']}", ""]
    L += ["**What can be concluded**"] + [f"- {s}" for s in o["can_conclude"]] + ["", "**What cannot be concluded**"] + [f"- {s}" for s in o["cannot_conclude"]] + [""]
    if o["investigate_first"]:
        L.append(f"**Investigate first:** {o['investigate_first']['finding']} - {o['investigate_first']['why']}.")
    gaps = skip + [s for s in o["most_useful_additional_evidence"] if s not in skip]
    if gaps:
        L += ["", "**Additional evidence that would improve the assessment most** (gaps common to the findings first)"] + [f"- {s}" for s in gaps[:4]]
    L += ["", "## D. Prioritized investigation plan", ""]
    for s in a["investigation_plan"]:
        L.append(f"- [ ] **{s['order']}. ({s['finding']}, {s['priority']})** {s['action']}{_src_mark(s)}")
        if s.get("why"):
            L.append(f"  - Why: {s['why']}")
        L.append(f"  - Strengthens the hypothesis if: {s.get('strengthens_if', '-')}  |  weakens it if: {s.get('weakens_if', '-')}")
    if not a["investigation_plan"]:
        L.append("- No actions: nothing met the reporting criteria.")
    L += ["", "## E. Technical appendix", "",
          f"Full detail (all {header.get('units_total', '?')} units, every ATT&CK candidate with scores, reference passages, graph paths, "
          "uncertainties, validation) is kept in the technical appendix, not here:",
          "- `evidence_report.json` (machine-readable: run report, all unit reports, evidence registries) with `evidence_report.md` / `evidence_report.txt` as readable renderings",
          "- `analyst_report.json` (this report, structured, with the prioritization rules and evidence index)",
          "- `run_result.json`, `noise_review.json` (every noise event with its measured signals)"
          if header.get("exported") else
          "- re-run with `--export-dir outputs/<name>` to write `evidence_report.json` / `.md` / `.txt` (all units, candidates, scores, passages, graph paths) and `analyst_report.json`.",
          "", "---", a["disclaimer"], ""]
    return "\n".join(L)


def _plain(md: str) -> str:
    out = []
    for line in md.splitlines():
        line = re.sub(r"^#{1,4}\s*", "", line)
        line = line.replace("**", "").replace("`", "")
        line = re.sub(r"(?<![\w*])\*(.+?)\*(?![\w*])", r"\1", line)
        out.append(line)
    return "\n".join(out)


def render_analyst_text(a: Dict[str, Any], header: Dict[str, Any]) -> str:
    return _plain(render_analyst_markdown(a, header)).encode("ascii", "replace").decode("ascii") + "\n"
