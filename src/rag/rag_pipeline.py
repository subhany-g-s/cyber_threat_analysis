"""
src/rag/rag_pipeline.py
---------------------------
Orchestrates: retrieve -> build context -> prompt -> LLM -> parse/validate
-> ThreatReport. This is the only module callers need to import for
"generate a report for cluster N".

    INPUT (cluster_id) -> RETRIEVER -> CONTEXT BUILDER -> LLM -> FINAL REPORT
"""

import json
from typing import List, Optional

from src.rag.context_builder import build_context_block
from src.rag.llm_clients import LLMClient
from src.rag.prompt_templates import SYSTEM_PROMPT, build_user_prompt
from src.rag.report_schema import ReportParseError, ThreatReport, parse_llm_json_response
from src.rag.retriever import RagIndex


def generate_cluster_report(cluster_id: int, index: RagIndex, llm: LLMClient) -> ThreatReport:
    """Raises KeyError if cluster_id doesn't exist, ReportParseError if the
    LLM's response is malformed or cites a technique it wasn't shown."""

    # 1. RETRIEVER
    ctx = index.get_cluster_context(cluster_id)

    # 2. CONTEXT BUILDER
    context_block = build_context_block(ctx)
    user_prompt = build_user_prompt(context_block)

    # 3. LLM
    raw_response = llm.generate(SYSTEM_PROMPT, user_prompt)

    # 4. PARSE + VALIDATE (anti-hallucination check happens here, see report_schema.py)
    try:
        raw_json = json.loads(raw_response)
    except json.JSONDecodeError as e:
        raise ReportParseError(f"LLM response was not valid JSON: {e}. Raw response: {raw_response!r}")

    allowed_ids = [t["technique_id"] for t in ctx.techniques]
    report = parse_llm_json_response(
        raw_json, cluster_id=cluster_id, generated_by=llm.backend_name,
        grounding_event_uids=ctx.representative_event_uids,
        cluster_size=ctx.cluster_size, max_novelty_score=ctx.max_novelty_score,
        allowed_technique_ids=allowed_ids,
    )
    return report


def generate_reports_for_clusters(cluster_ids: List[int], index: RagIndex, llm: LLMClient,
                                    on_error: str = "raise") -> List[ThreatReport]:
    """on_error: 'raise' (default) stops on the first failure; 'skip' logs
    and continues -- useful for a batch run where one bad cluster shouldn't
    block the rest, but failures are still printed, never silent."""
    reports = []
    for cid in cluster_ids:
        try:
            reports.append(generate_cluster_report(cid, index, llm))
        except (KeyError, ReportParseError) as e:
            if on_error == "raise":
                raise
            print(f"WARNING: report generation failed for cluster {cid}: {e}")
    return reports
