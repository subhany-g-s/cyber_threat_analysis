"""
src/rag/context_builder.py
------------------------------
Formats a ClusterContext (retriever.py) into the exact text block the LLM
prompt embeds. Kept as its own module (not inlined into prompt_templates)
so the formatting logic is independently testable and reusable if a
different prompt style is needed later.

Design choices for conciseness / non-redundancy:
  - Techniques already below the mapper's confidence floor
    (NO_CONFIDENT_MATCH) are still shown, but clearly labeled, so the LLM
    can accurately say "no confident technique match" instead of omitting
    them and inventing something more confident-sounding.
  - evidence_snippets are already deduplicated upstream (cluster_profiler.py)
    -- this function does not re-fetch or expand them.
  - High-novelty events are capped (default top 5, set by the retriever)
    and shown as a compact table-style block, not full event dumps.
  - Structured profile fields are summarized (counts, top-N), never the
    full raw member list -- the narrative text already carries the
    human-readable summary of the same data.
"""

from src.rag.retriever import ClusterContext


def format_techniques_block(techniques: list) -> str:
    if not techniques:
        return "  (no technique candidates were returned by the semantic mapper for this cluster)"
    lines = []
    for t in techniques:
        lines.append(
            f"  - {t['technique_id']} {t['technique_name']} (tactic: {t['tactic']}) "
            f"similarity={t['similarity_score']:.4f} status={t['mapping_status']}"
        )
    return "\n".join(lines)


def format_novelty_block(ctx: ClusterContext) -> str:
    if not ctx.novelty_label_counts:
        return "  (no novelty data available for this cluster's member events)"
    label_line = ", ".join(f"{k}={v}" for k, v in sorted(ctx.novelty_label_counts.items()))
    lines = [
        f"  Cluster novelty: mean={ctx.mean_novelty_score}, max={ctx.max_novelty_score}",
        f"  Label distribution across {ctx.cluster_size} member events: {label_line}",
    ]
    if ctx.high_novelty_events:
        lines.append("  Highest-novelty member events:")
        for e in ctx.high_novelty_events:
            lines.append(f"    - {e['event_uid']}  score={e['novelty_score']}  label={e['novelty_label']}")
    return "\n".join(lines)


def build_context_block(ctx: ClusterContext) -> str:
    """The single formatted text block inserted into the user prompt.
    Every line traces to a specific retrieved field -- nothing here is
    free-generated."""
    evidence = "\n".join(f'  - "{s}"' for s in ctx.evidence_snippets) if ctx.evidence_snippets else "  (none captured)"

    return f"""Cluster ID: {ctx.cluster_id}
Cluster Size: {ctx.cluster_size} events

Summary:
{ctx.narrative}

Structured Profile:
  Dominant EventIDs: {ctx.structured_profile['dominant_event_ids']}
  Hosts involved: {list(ctx.structured_profile['hostname_distribution'].keys())}
  Common source processes: {[p for p, _ in ctx.structured_profile['common_source_images']]}
  Common target processes: {[p for p, _ in ctx.structured_profile['common_target_images']]}
  Network destinations: {[ip for ip, _ in ctx.structured_profile['common_destination_ips']]}
  Time span: {ctx.structured_profile['temporal_span']['earliest']} to {ctx.structured_profile['temporal_span']['latest']}

Evidence snippets:
{evidence}

Top MITRE ATT&CK Technique Candidates (semantic similarity, NOT rule-based; may include NO_CONFIDENT_MATCH):
{format_techniques_block(ctx.techniques)}

Novelty Insights:
{format_novelty_block(ctx)}
"""
