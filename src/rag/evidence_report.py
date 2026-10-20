"""
src/rag/evidence_report.py
----------------------------
The final, evidence-grounded report.

Design: FACTS ARE NEVER WRITTEN BY AN LLM. Every observed fact, computed
finding, technique candidate, reference passage, graph path and ontology fact
is assembled deterministically from the run's own data and carries evidence
ids. This module builds the FULL per-unit technical record (the "technical
appendix"). The short, prioritized analyst report - and the only LLM step, which
adds validated narrative per finding - live in findings.py / finding_llm.py.

Sections (kept separate on purpose):
    observed_facts           from the input logs
    computed_findings        GAE/HDBSCAN/novelty/graph computations
    technique_candidates     semantic ATT&CK leads (cosine similarity, not probability)
    reference_information    retrieved ATT&CK passages (trusted reference corpus)
    graph_evidence           retrieved relationship paths + related units
    ontology_facts           from the ontology module, when supplied
    inferences               hedged, with their basis
    uncertainty_and_missing_evidence
    recommended_investigation_steps

Evidence ids:  EV:<event_uid>   GP:<unit>:<n>   REF:<doc id>   ONT:<entity id>
               TC:<technique id>   U:<unit id>
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from src.mapping.semantic_mapper import SIMILARITY_DISCLAIMER
from src.mapping.text_prep import display_safe

REPORT_SCHEMA_VERSION = "3.0"   # 3.0: analyst report added (findings.py); unit/appendix structure unchanged; unit-level LLM prose removed
DETERMINISTIC_BACKEND = "deterministic-evidence-template"

DISCLAIMER = ("This report lists leads for analyst review. Technique candidates come from text-embedding "
              "similarity and are not confirmed findings; novelty scores measure statistical/structural "
              "unusualness, not maliciousness; no actor attribution is made or implied.")

SCOPE_LABEL = {"cluster": "Behaviour cluster", "noise_group": "Unclustered (noise) group",
               "noise_event": "Unclustered (noise) event"}


def _short(uid: str) -> str:
    return uid[:8]


class EvidenceReportBuilder:
    def __init__(self, run_info: Dict[str, Any], event_uids: Sequence[str],
                 provenance_by_uid: Optional[Dict[str, Dict[str, Any]]] = None):
        """run_info: run_id, corpus_version, embedder{model_id,is_standin}, thresholds, ontology_present, platform_filter."""
        self.run = run_info
        self.event_uids = set(event_uids)
        self.prov = provenance_by_uid or {}

    # ------------------------------------------------------------------ helpers
    def _ev(self, uid: str) -> str:
        return f"EV:{uid}"

    def _src(self, uid: str) -> str:
        p = self.prov.get(uid) or {}
        return p.get("source_ref") or "input log"

    # ------------------------------------------------------------------ unit report
    def build_unit(self, unit: Dict[str, Any]) -> Dict[str, Any]:
        prof, nov, mapping, ctx = unit["profile"], unit["novelty"], unit["mapping"], unit["retrieval"]
        uid_, scope = unit["unit_id"], unit["scope"]
        reps = [u for u in prof.get("representative_event_uids", []) if u in self.event_uids]
        size = unit["size"]
        reg: Dict[str, Dict[str, Any]] = {}

        def reg_add(eid: str, kind: str, **kw):
            reg.setdefault(eid, dict(kind=kind, **kw))
            return eid

        # ---------------- observed facts (input logs) ----------------
        facts: List[Dict[str, Any]] = []

        def fact(text, evs):
            facts.append({"id": f"F{len(facts) + 1}", "text": text, "source": "input logs",
                          "evidence": [reg_add(self._ev(u), "event", source_ref=self._src(u)) for u in evs if u in self.event_uids]})

        desc = prof.get("event_descriptions") or []
        if desc:
            fact(f"{size} event(s); log activity types: " +
                 "; ".join(f"{display_safe(d)} x{c}" for d, c in desc[:3]) + ".", reps)
        if prof.get("dominant_event_ids"):
            fact("EventID distribution: " + ", ".join(f"{e} x{c}" for e, c in prof["dominant_event_ids"][:4]) + ".", reps)
        if prof.get("hostname_distribution"):
            hosts = list(prof["hostname_distribution"].items())
            fact("Host(s): " + ", ".join(f"{h} ({c})" for h, c in hosts[:4]) + ".", reps)
        src = [display_safe(p) for p, _ in (prof.get("common_source_images") or [])[:3]]
        tgt = [display_safe(p) for p, _ in (prof.get("common_target_images") or [])[:3]]
        if src:
            fact("Source process image(s): " + "; ".join(src) + (". Target process image(s): " + "; ".join(tgt) if tgt else "") + ".", reps)
        if prof.get("common_destination_ips"):
            ports = ", ".join(p for p, _ in (prof.get("network_ports") or [])[:3])
            fact("Network destination(s): " + ", ".join(ip for ip, _ in prof["common_destination_ips"][:3]) +
                 (f" on port(s) {ports}" if ports else "") + ".", reps)
        if prof.get("registry_targets"):
            fact("Registry target(s): " + "; ".join(display_safe(k, 80) for k, _ in prof["registry_targets"][:2]) + ".", reps)
        for cmd in (prof.get("command_texts") or [])[:2]:
            fact(f'Command / script text recorded in the log: "{display_safe(cmd, 160)}"', reps)
        span = prof.get("temporal_span") or {}
        if span.get("earliest"):
            fact(f"Time span: {span['earliest']} to {span.get('latest')}.", reps)

        # ---------------- computed findings ----------------
        comp: List[Dict[str, Any]] = []

        def computed(text, source, evs=()):
            comp.append({"id": f"C{len(comp) + 1}", "text": text, "source": source,
                         "evidence": [reg_add(f"U:{uid_}", "unit")] +
                                     [reg_add(self._ev(u), "event", source_ref=self._src(u)) for u in evs if u in self.event_uids]})

        if scope == "cluster":
            computed(f"{size} events were grouped into cluster {unit['cluster_id']} by HDBSCAN on pretrained-GAE embeddings.",
                     "GAE + HDBSCAN")
        else:
            computed(f"{size} event(s) were NOT assigned to any cluster (HDBSCAN noise) and are analysed as a "
                     f"{'related group' if scope == 'noise_group' else 'single event'}; this is event-level evidence, "
                     f"not a behaviour cluster.", "HDBSCAN + noise review")
            ns = unit.get("noise_signals") or {}
            if ns:
                computed("Measured properties consistent with being unclustered: " +
                         ", ".join(f"{k} ({v})" for k, v in sorted(ns.items(), key=lambda kv: -kv[1])[:5]) +
                         ". HDBSCAN reports no per-point reason.", "noise review")
        if nov:
            top_ev = [e["event_uid"] for e in nov.get("top_events", [])[:3]]
            lc = nov.get("label_counts") or {}
            computed(f"Novelty (statistical/structural unusualness, not maliciousness): max {nov.get('max')}, mean {nov.get('mean')}; "
                     "labels " + ", ".join(f"{k}={v}" for k, v in sorted(lc.items())) + ".", "novelty detection", top_ev)
        gc = ctx.get("graph_contribution") or {}
        if gc:
            computed(f"Graph traversal from {len(ctx.get('seed_events') or [])} seed event(s) reached "
                     f"{gc.get('traversal_events', 0)} further event(s) ({gc.get('cross_unit_events', 0)} outside this unit) "
                     f"and {gc.get('entity_bridged_events', 0)} more via shared entities; retrieval backend: "
                     f"{ctx.get('retrieval_mode')}.", "GraphRAG retrieval")

        # ---------------- technique candidates ----------------
        techs: List[Dict[str, Any]] = []
        ref_ids_by_tid: Dict[str, List[str]] = {}
        for r in ctx.get("references") or []:
            ref_ids_by_tid.setdefault(r["technique_id"], []).append(r["ref_id"])
        for c in (mapping or {}).get("top_k_techniques", []):
            tc = reg_add(f"TC:{c['technique_id']}", "technique_candidate", similarity_score=c["similarity_score"],
                         status=c["mapping_status"])
            techs.append({
                "id": tc, "technique_id": c["technique_id"], "technique_name": c["technique_name"],
                "tactics": c.get("tactics") or [c.get("tactic")], "rank": c.get("rank"),
                "similarity_score": c["similarity_score"], "status": c["mapping_status"],
                "supporting_event_uids": [u for u in c.get("supporting_event_uids", []) if u in self.event_uids],
                "reference_ids": [reg_add(r, "reference") for r in ref_ids_by_tid.get(c["technique_id"], [])],
                "caveat": "semantic similarity lead - not a confirmed technique execution",
                "supporting_events_scope": "unit-level representative events (general context for the whole unit, NOT specific to this technique)",
                "scope": scope,
            })
            for u in techs[-1]["supporting_event_uids"]:
                reg_add(self._ev(u), "event", source_ref=self._src(u))
        outcome = (mapping or {}).get("overall_outcome", "NO_MATCH") if mapping else "UNAVAILABLE"

        # ---------------- reference information ----------------
        refs = []
        for r in ctx.get("references") or []:
            refs.append(dict(r, id=reg_add(r["ref_id"], "reference", source=r.get("source"),
                                           corpus_version=r.get("corpus_version"))))

        # ---------------- graph evidence ----------------
        gev: List[Dict[str, Any]] = []
        for p in ctx.get("paths") or []:
            chain = p.get("edges") or []
            nodes = ([chain[0]["from"]] if chain else []) + [e["to"] for e in chain]
            text = " ".join([f"EV[{_short(nodes[0])}]"] +
                            [f"-{e['rel']}-> EV[{_short(e['to'])}]" for e in chain]) if chain else ""
            gev.append({"id": reg_add(p["path_id"], "graph_path", status=p.get("status")),
                        "text": f"{text}  (ends in {p.get('end_unit') or 'unclustered event'})",
                        "events": [reg_add(self._ev(n), "event", source_ref=self._src(n)) for n in nodes if n in self.event_uids],
                        "status": p.get("status", "observed"), "provenance": p.get("provenance")})
        related = []
        for r in ctx.get("related_units") or []:
            related.append({"unit_id": r["unit_id"], "via": r["via"], "graph_links": r.get("graph_links", 0),
                            "narrative_similarity": r.get("narrative_similarity"),
                            "candidate_techniques": r.get("candidate_techniques", [])})
            reg_add(f"U:{r['unit_id']}", "unit")

        # ---------------- ontology facts ----------------
        onto = []
        for o in ctx.get("ontology_facts") or []:
            onto.append({"id": reg_add(o["ref_id"], "ontology", status=o.get("status")),
                         "text": f"Ontology module entity '{o.get('canonical_name')}' (type {o.get('entity_type')}, "
                                 f"{o.get('status')}) cites event {_short(o['evidence_event_uid'])}.",
                         "confidence": o.get("confidence"), "ontology_uri": o.get("ontology_uri"),
                         "provenance": f"ontology module: {o.get('provenance')}",
                         "evidence": [reg_add(self._ev(o["evidence_event_uid"]), "event")]})

        # ---------------- inferences (hedged, deterministic) ----------------
        inf: List[Dict[str, Any]] = []
        best = next((t for t in techs if t["status"] == "CANDIDATE"), None)
        if best:
            inf.append({"text": f"The behaviour narrative is semantically similar to ATT&CK {best['technique_id']} "
                                f"{best['technique_name']} (cosine {best['similarity_score']}). This is a lead to check "
                                f"against the raw events, not evidence that the technique was used.",
                        "basis": [best["id"]] + [r for r in best["reference_ids"][:1]], "certainty": "low",
                        "source": DETERMINISTIC_BACKEND})
        elif outcome == "UNAVAILABLE":
            inf.append({"text": "ATT&CK semantic mapping was not available in this run (see run warnings), so no technique "
                                "candidates are given for this unit.", "basis": [reg_add(f"U:{uid_}", "unit")],
                        "certainty": "low", "source": DETERMINISTIC_BACKEND})
        elif outcome == "WEAK_ONLY":
            inf.append({"text": "Only weak semantic similarity to ATT&CK techniques was found; no technique is proposed as a candidate.",
                        "basis": [t["id"] for t in techs[:2]], "certainty": "low", "source": DETERMINISTIC_BACKEND})
        else:
            inf.append({"text": "No ATT&CK technique was semantically similar enough to propose (no match). This does not "
                                "mean the activity is benign.", "basis": [reg_add(f"U:{uid_}", "unit")],
                        "certainty": "low", "source": DETERMINISTIC_BACKEND})
        cross_units = sorted({r["unit_id"] for r in related if "graph" in r["via"]})
        if cross_units:
            inf.append({"text": f"Events here are linked in the event graph (same process id or same host/time window) to "
                                f"{', '.join(cross_units[:4])}; the activity may be part of a longer sequence spanning those groups.",
                        "basis": [p["id"] for p in gev[:3]], "certainty": "low", "source": DETERMINISTIC_BACKEND})
        if scope != "cluster" and (nov.get("max") or 0) >= 0.5:
            inf.append({"text": "This unclustered activity is statistically unusual in this dataset; unusualness can come from rare "
                                "but legitimate administration as well as from malicious activity.",
                        "basis": [reg_add(f"U:{uid_}", "unit")], "certainty": "low", "source": DETERMINISTIC_BACKEND})

        # ---------------- uncertainty & missing evidence ----------------
        unc = [SIMILARITY_DISCLAIMER]
        if self.run.get("embedder", {}).get("is_standin"):
            unc.append("Semantic scores come from the offline LEXICAL stand-in embedder (word overlap, not meaning); "
                       "they are not comparable to SBERT scores and are for development only.")
        unc.append("Similarity thresholds are uncalibrated defaults: no labelled ground truth ships with this project.")
        if not prof.get("command_texts"):
            unc.append("No command-line / script text was recorded for this unit's events, so process behaviour is "
                       "inferred only from event types, process names and relationships.")
        if not self.run.get("ontology_present"):
            unc.append("No ontology-module data was supplied; there are no ontology-derived facts.")
        elif not onto:
            unc.append("Ontology data was supplied but none of it cites events in this unit's neighbourhood.")
        if ctx.get("truncated"):
            unc.append("Graph traversal hit the event cap; the neighbourhood shown is partial.")
        if ctx.get("hub_entities"):
            unc.append("Highly shared entities (" + ", ".join(h["name"] for h in ctx["hub_entities"][:3]) +
                       ") were not expanded to avoid flooding the context.")
        if gc.get("status") == "no_additional_evidence_from_graph":
            unc.append("Graph traversal found no additional evidence for this unit; retrieval here is effectively "
                       "seed-event lookup plus semantic references.")
        if self.run.get("platform_filter"):
            unc.append(f"ATT&CK matching was restricted to platform(s) {self.run['platform_filter']}; techniques for other "
                       f"platforms were not considered.")

        # ---------------- recommended steps ----------------
        steps: List[Dict[str, Any]] = []
        if reps:
            steps.append({"text": "Open the raw log records for the representative events and confirm the observed facts: " +
                                  ", ".join(f"{_short(u)} ({self._src(u)})" for u in reps[:3]) + ".",
                          "basis": [self._ev(u) for u in reps[:3]], "source": DETERMINISTIC_BACKEND})
        if best:
            det = next((r for r in refs if r["technique_id"] == best["technique_id"] and r["passage_type"] == "detection"), None)
            if det:
                steps.append({"text": f"Compare against ATT&CK's detection guidance for {best['technique_id']}: {det['text'][:240]}",
                              "basis": [det["id"], best["id"]], "source": DETERMINISTIC_BACKEND})
            else:
                steps.append({"text": f"Check whether the behaviour matches the ATT&CK description of {best['technique_id']} "
                                      f"(see reference passages) in the raw events.", "basis": [best["id"]],
                              "source": DETERMINISTIC_BACKEND})
        if gev:
            steps.append({"text": "Inspect the linked events on the retrieved relationship paths to see whether the sequence "
                                  "continues outside this unit.", "basis": [g["id"] for g in gev[:2]], "source": DETERMINISTIC_BACKEND})
        if scope != "cluster":
            steps.append({"text": "Decide whether this one-off activity is expected for the host/user (change windows, admin tooling) "
                                  "before escalating.", "basis": [reg_add(f"U:{uid_}", "unit")], "source": DETERMINISTIC_BACKEND})
        if not steps:
            steps.append({"text": "No specific follow-up is suggested by the retrieved evidence.", "basis": [], "source": DETERMINISTIC_BACKEND})

        top_desc = display_safe((desc[0][0] if desc else "mixed events"), 60)
        host = next(iter(prof.get("hostname_distribution") or {}), None)
        title = f"{SCOPE_LABEL[scope]} {uid_}: {top_desc}" + (f" on {host}" if host else "") + f" ({size} event{'s' if size != 1 else ''})"

        for u in reps:
            reg_add(self._ev(u), "event", source_ref=self._src(u))
        return {
            "unit_id": uid_, "scope": scope, "title": title,
            "generated_by": DETERMINISTIC_BACKEND, "is_mock": False, "llm_used": False,
            "analyst_summary": self._deterministic_summary(unit, outcome, techs, gc),
            "sections": {
                "observed_facts": facts, "computed_findings": comp, "technique_candidates": techs,
                "reference_information": refs, "graph_evidence": gev, "related_units": related,
                "ontology_facts": onto, "inferences": inf, "uncertainty_and_missing_evidence": unc,
                "recommended_investigation_steps": steps},
            "technique_outcome": outcome,
            "evidence_registry": reg,
            "disclaimer": DISCLAIMER,
        }

    @staticmethod
    def _deterministic_summary(unit, outcome, techs, gc) -> str:
        nov = unit["novelty"] or {}
        bits = [f"{SCOPE_LABEL[unit['scope']]} {unit['unit_id']} contains {unit['size']} event(s); "
                f"max novelty {nov.get('max')} ({', '.join(f'{k}={v}' for k, v in sorted((nov.get('label_counts') or {}).items()))})."]
        cands = [t for t in techs if t["status"] == "CANDIDATE"]
        if cands:
            bits.append("Semantic ATT&CK candidates (leads, not findings): " +
                        ", ".join(f"{t['technique_id']} ({t['similarity_score']})" for t in cands[:3]) + ".")
        elif outcome == "WEAK_ONLY":
            bits.append("Only weak semantic similarity to ATT&CK was found.")
        elif outcome == "UNAVAILABLE":
            bits.append("ATT&CK mapping was unavailable.")
        else:
            bits.append("No ATT&CK technique matched.")
        if gc.get("cross_unit_events"):
            bits.append(f"Graph traversal links it to {gc['cross_unit_events']} event(s) outside the unit.")
        return " ".join(bits)

    # ------------------------------------------------------------------ validation
    def validate_unit_report(self, report: Dict[str, Any]) -> List[str]:
        """Structural + grounding validation of a finished unit report."""
        issues: List[str] = []
        reg = report.get("evidence_registry", {})
        sec = report.get("sections", {})
        for name in ("observed_facts", "computed_findings", "technique_candidates", "reference_information",
                     "graph_evidence", "ontology_facts", "inferences", "uncertainty_and_missing_evidence",
                     "recommended_investigation_steps"):
            if name not in sec:
                issues.append(f"missing section {name}")
        for f in sec.get("observed_facts", []):
            for e in f.get("evidence", []):
                if e.startswith("EV:") and e[3:] not in self.event_uids:
                    issues.append(f"observed fact {f['id']} cites event not in this run: {e}")
            if not f.get("source"):
                issues.append(f"observed fact {f['id']} has no provenance")
        for t in sec.get("technique_candidates", []):
            s = t.get("similarity_score")
            if not isinstance(s, (int, float)) or not (0.0 <= s <= 1.0):
                issues.append(f"technique {t.get('technique_id')} has score outside [0,1]: {s}")
            if t.get("status") not in ("CANDIDATE", "WEAK_MATCH", "NO_CONFIDENT_MATCH"):
                issues.append(f"technique {t.get('technique_id')} has invalid status {t.get('status')}")
            for u in t.get("supporting_event_uids", []):
                if u not in self.event_uids:
                    issues.append(f"technique {t['technique_id']} cites event not in this run: {u}")
            for r in t.get("reference_ids", []):
                if r not in reg:
                    issues.append(f"technique {t['technique_id']} cites unknown reference {r}")
        for r in sec.get("reference_information", []):
            if not r.get("source") or not r.get("corpus_version"):
                issues.append(f"reference {r.get('ref_id')} lacks source/corpus_version provenance")
        for key in ("inferences", "recommended_investigation_steps", "ontology_facts"):
            for it in sec.get(key, []):
                for b in it.get("basis", []) + it.get("evidence", []):
                    if b not in reg:
                        issues.append(f"{key} cites unknown evidence id {b}")
        for eid in reg:
            if eid.startswith("EV:") and eid[3:] not in self.event_uids:
                issues.append(f"registry lists event not in this run: {eid}")
        return issues

    # ------------------------------------------------------------------ run-level report
    def build_run_report(self, unit_reports: List[Dict[str, Any]], units_meta: List[Dict[str, Any]],
                         overview: Dict[str, Any], llm_info: Dict[str, Any], top_n: int) -> Dict[str, Any]:
        prio = {m["unit_id"]: m for m in units_meta}
        ranked = sorted(unit_reports, key=lambda r: (-(prio[r["unit_id"]].get("priority") or 0.0), r["unit_id"]))
        key_findings = []
        for r in ranked[:top_n]:
            m = prio[r["unit_id"]]
            cands = [t for t in r["sections"]["technique_candidates"] if t["status"] == "CANDIDATE"]
            key_findings.append({
                "unit_id": r["unit_id"], "scope": r["scope"], "title": r["title"], "triage_priority_novelty": m.get("priority"),
                "headline": r["analyst_summary"],
                "technique_candidates": [{"technique_id": t["technique_id"], "similarity_score": t["similarity_score"],
                                          "status": t["status"]} for t in cands[:3]],
                "outcome": r["technique_outcome"],
            })
        issues = []
        for r in unit_reports:
            for i in self.validate_unit_report(r):
                issues.append(f"{r['unit_id']}: {i}")
        return {
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "title": "SUBHANY cyber threat analysis - evidence-grounded report",
            "run_id": self.run.get("run_id"),
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "generated_by": llm_info.get("generated_by", DETERMINISTIC_BACKEND),
            "llm": llm_info,
            "overview": overview,
            "triage_note": "Units are ordered by maximum novelty (statistical unusualness) - a triage order, not a severity rating.",
            "key_findings": key_findings,
            "conclusions": self._conclusions(overview, unit_reports),
            "unit_report_count": len(unit_reports),
            "validation": {"passed": not issues, "issue_count": len(issues), "issues": issues[:40]},
            "disclaimer": DISCLAIMER,
        }

    @staticmethod
    def _conclusions(overview: Dict[str, Any], unit_reports: List[Dict[str, Any]]) -> List[str]:
        out = []
        with_cand = [r for r in unit_reports if r["technique_outcome"] == "CANDIDATES"]
        out.append(f"{len(with_cand)} of {len(unit_reports)} analysed units have at least one ATT&CK technique candidate "
                   f"(semantic leads, not confirmed techniques); {sum(1 for r in unit_reports if r['technique_outcome'] == 'NO_MATCH')} "
                   f"have no match.")
        n_noise = overview.get("noise_events")
        if n_noise:
            out.append(f"{n_noise} events were not clustered; they were reviewed separately and "
                       f"{overview.get('noise_units_analysed', 0)} noise unit(s) received semantic analysis as event-level evidence.")
        hi = overview.get("high_novelty_events")
        if hi:
            out.append(f"{hi} event(s) are labelled HIGH_NOVELTY (statistically unusual, not necessarily malicious).")
        out.append("No actor attribution or confirmed attack is claimed; every candidate needs analyst verification against the raw events.")
        return out
