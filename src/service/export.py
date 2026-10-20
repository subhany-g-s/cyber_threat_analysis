"""
src/service/export.py
-----------------------
OPTIONAL file export of a finished RunResult. Nothing is written unless the
caller passes an export directory (CLI: --export-dir).

PRIMARY report (short, prioritized, for analysts):
    analyst_report.md      readable report: summary, findings, overall assessment, plan
    analyst_report.json    the same, structured (+ prioritization rules and evidence index)
TECHNICAL APPENDIX (everything, nothing hidden):
    evidence_report.json   run report + ALL unit reports + the analyst report
    evidence_report.md/.txt  readable renderings of every unit (long)
    run_result.json, noise_review.json, novelty_scores.csv, embedding_clusters.csv, ...

The legacy names/shapes (cluster_profiles_v2.json, semantic_mitre_mappings.json,
embedding_clusters.csv, novelty_scores.csv, rag_ready_clusters.json) are unchanged so the
older file-based tools keep working.
"""

import csv
import json
import os
from typing import Any, Dict, List

from src.rag.analyst_render import render_analyst_markdown
from src.rag.report_render import render_report_markdown, render_report_text
from src.service.result import RunResult


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str, ensure_ascii=True)


def export_run(result: RunResult, export_dir: str) -> Dict[str, Any]:
    os.makedirs(export_dir, exist_ok=True)
    written: List[str] = []

    def out(name: str) -> str:
        written.append(name)
        return os.path.join(export_dir, name)

    _write_json(out("run_result.json"), result.to_dict())

    unit_reports = [u.report for u in result.units if u.report]
    if result.analyst_report:
        header = {"run_id": result.run_id, "generated_at": (result.report or {}).get("generated_at"), "units_total": len(result.units),
                  "exported": True}
        _write_json(out("analyst_report.json"), result.analyst_report)
        with open(out("analyst_report.md"), "w", encoding="utf-8") as f:
            f.write(render_analyst_markdown(result.analyst_report, header))
    if result.report:
        _write_json(out("evidence_report.json"), {"report": result.report, "units": unit_reports,
                                                  "analyst_report": result.analyst_report})
        with open(out("evidence_report.txt"), "w", encoding="utf-8") as f:
            f.write(render_report_text(result.report, unit_reports, detail_units=len(unit_reports)))
        with open(out("evidence_report.md"), "w", encoding="utf-8") as f:
            f.write(render_report_markdown(result.report, unit_reports))

    art = result.artifacts
    cluster_units = [u for u in result.units if u.scope == "cluster"]
    _write_json(out("cluster_profiles_v2.json"), [u.profile for u in cluster_units])
    _write_json(out("semantic_mitre_mappings.json"), [u.mapping for u in cluster_units])
    _write_json(out("noise_review.json"), result.noise)
    _write_json(out("rag_ready_clusters.json"), [{
        "cluster_id": u.cluster_id, "cluster_size": u.size, "narrative": u.narrative,
        "representative_event_uids": u.profile.get("representative_event_uids", []),
        "mitre_candidates": u.mapping.get("top_k_techniques", []),
        "mapping_outcome": u.mapping.get("overall_outcome"),
    } for u in cluster_units])

    if art is not None and art.row_to_uid and art.labels is not None:
        with open(out("embedding_clusters.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["embedding_index", "event_uid", "cluster_id", "membership_probability", "is_noise"])
            for i, uid in enumerate(art.row_to_uid):
                w.writerow([i, uid, int(art.labels[i]), float(art.probabilities[i]), int(art.labels[i]) == -1])
        fields = ["event_uid", "cluster_id", "density_score", "embedding_distance_score", "reconstruction_error",
                  "graph_rarity_score", "novelty_score", "novelty_label"]
        with open(out("novelty_scores.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in art.novelty_rows:
                w.writerow({k: r.get(k) for k in fields})
        try:
            from src.ontology.export import export_findings
            _write_json(out("ontology_findings.json"), export_findings(result))
        except Exception as e:                              # export must not abort for the optional hand-off file
            result.warnings.append(f"ontology_findings.json not written: {type(e).__name__}: {e}")
            written.remove("ontology_findings.json") if "ontology_findings.json" in written else None
    return {"directory": os.path.abspath(export_dir), "files": written}
