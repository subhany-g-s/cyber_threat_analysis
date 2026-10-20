"""
src/cli/render.py
-------------------
Terminal presentation of a StageResult / RunResult. ASCII only (safe for
cp1252/cp437 Windows consoles); every value shown is read from the result -
nothing is typed in by hand. Log-derived text is passed through display_safe.
"""

import textwrap
from typing import Any, Dict, List

from src.mapping.text_prep import display_safe
from src.rag.analyst_render import render_analyst_text
from src.service.result import FAILED, OK, SKIPPED, WARN, RunResult, StageResult

WIDTH = 78
_BADGE = {OK: "[ OK ]", WARN: "[WARN]", FAILED: "[FAIL]", SKIPPED: "[SKIP]"}

# metrics shown per stage (in this order, only when present in the real result)
STAGE_KEYS: Dict[int, List[str]] = {
    1: ["path", "format", "source_id", "total_lines", "valid_records", "invalid_records", "blank_lines", "valid_share_pct"],
    2: ["normalized_count", "unique_events", "duplicate_records_collapsed", "skipped_count", "missing_timestamp",
        "missing_hostname", "event_id_distribution"],
    3: ["num_nodes", "num_edges", "same_pid_edges", "same_host_temporal_edges", "isolated_nodes",
        "num_connected_components", "avg_degree", "max_degree"],
    4: ["source", "checkpoint", "in_channels", "hidden_channels", "latent_channels", "feature_matrix_shape",
        "embedding_shape", "retrained", "schema_coverage_pct"],
    5: ["algorithm", "min_cluster_size", "min_samples", "total_events", "clusters", "noise_events", "noise_share_pct",
        "mean_membership_probability", "largest_clusters"],
    6: ["label_counts", "score_mean", "score_max", "mean_score_noise_events", "mean_score_clustered_events", "thresholds"],
    7: ["cluster_profiles", "noise_events_preserved", "noise_groups", "noise_single_events", "noise_review_units_total",
        "noise_review_units_analysed", "noise_events_with_clustered_neighbors", "noise_signal_counts"],
    8: ["corpus_version", "attack_version", "platform_filter", "model_id", "is_standin_embedder", "top_k",
        "candidate_threshold", "weak_threshold", "units_mapped", "outcomes", "outcomes_by_scope"],
    9: ["retrieval_mode", "run_id", "units_retrieved", "units_where_graph_contributed", "units_without_graph_evidence",
        "totals", "store_stats", "neo4j", "ontology"],
    10: ["units_reported", "findings", "findings_by_priority", "analysis_mode", "validation_passed", "validation_issue_count"],
}


def _fmt(v: Any, depth: int = 0) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, dict):
        items = [f"{k}={_fmt(x, depth + 1)}" for k, x in list(v.items())[:10]]
        more = f", +{len(v) - 10} more" if len(v) > 10 else ""
        return "{" + ", ".join(items) + more + "}"
    if isinstance(v, (list, tuple)):
        s = "[" + ", ".join(_fmt(x, depth + 1) for x in list(v)[:6]) + (", ..." if len(v) > 6 else "") + "]"
        return s
    return display_safe(v)


def _ascii(s: str) -> str:
    return s.encode("ascii", "replace").decode("ascii")


def _wrap(prefix: str, text: str, indent: str) -> List[str]:
    return textwrap.wrap(_ascii(text), WIDTH, initial_indent=prefix, subsequent_indent=indent) or [prefix]


def render_stage(st: StageResult, total: int = 10) -> str:
    L = ["", "-" * WIDTH, f"[{st.number}/{total}] {st.name}",
         f"      Status: {_BADGE.get(st.status, st.status)}   Elapsed: {st.elapsed_s:.2f}s", "-" * WIDTH]
    if st.status == SKIPPED and not st.metrics:
        L += _wrap("  ", st.explanation or "Skipped.", "  ")
        return "\n".join(L)
    for k in STAGE_KEYS.get(st.number, []):
        if k in st.metrics and st.metrics[k] not in (None, ""):
            L += _wrap(f"  {k}: ", _fmt(st.metrics[k]), "      ")
    if st.error:
        L += _wrap("  ERROR: ", st.error, "         ")
    for w in st.warnings:
        L += _wrap("  WARNING: ", w, "           ")
    if st.explanation:
        L += [""] + _wrap("  What this stage did: ", st.explanation, "  ")
    return "\n".join(L)


def render_header(cfg: Dict[str, Any]) -> str:
    return "\n".join(["=" * WIDTH, " SUBHANY cyber threat analysis pipeline", "=" * WIDTH,
                      f"  input: {display_safe(cfg.get('input_path'))}   format: {cfg.get('input_format')}",
                      f"  embedder: {cfg.get('embedder')} ({cfg.get('embedding_model')})   llm: {cfg.get('llm')}   "
                      f"neo4j: {'on' if cfg.get('neo4j_enabled') else 'off (local mode)'}"])


def render_summary(res: RunResult, show_report: bool = True, detail_units: int = 5) -> str:
    """`detail_units` is kept for API compatibility; the number of findings shown is set by --report-top-n at build time."""
    L = ["", "=" * WIDTH, " RUN SUMMARY", "=" * WIDTH,
         f"  status: {'COMPLETED' if res.ok else 'COMPLETED WITH ERRORS'}   run_id: {res.run_id or '-'}   "
         f"total: {res.metadata.get('total_seconds', 0)}s"]
    for s in res.stages:
        L.append(f"  {s.number:>2}. {_BADGE.get(s.status, s.status)} {s.name:<46} {s.elapsed_s:>7.2f}s")
    v = (res.report or {}).get("validation")
    if v:
        L.append(f"  EVIDENCE VALIDATION: {'passed' if v.get('passed') else 'FAILED (' + str(len(v.get('issues', []))) + ' issue(s))'}")
    if res.errors:
        L += ["", "  Errors:"] + [x for e in res.errors for x in _wrap("   - ", e, "     ")]
    if res.warnings:
        L += ["", f"  Warnings ({len(res.warnings)}):"] + [x for w in res.warnings[:15] for x in _wrap("   - ", w, "     ")]
        if len(res.warnings) > 15:
            L.append(f"   ... {len(res.warnings) - 15} more (see --json)")
    ex = res.metadata.get("exported")
    L += ["", "  Files written: " + (f"{len(ex['files'])} in {ex['directory']}" if ex else "none (default; use --export-dir)")]
    if show_report and res.analyst_report:
        header = {"run_id": res.run_id, "generated_at": (res.report or {}).get("generated_at"), "units_total": len(res.units),
                  "exported": bool(ex)}
        L += ["", render_analyst_text(res.analyst_report, header)]
    return "\n".join(L)
