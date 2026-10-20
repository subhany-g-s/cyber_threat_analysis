"""
tests/test_rag_pipeline.py
------------------------------
Tests the retriever -> context builder -> LLM client -> schema validation
chain using TemplateLLMClient (no model server required) and real-shaped
synthetic data. Does NOT call a real LLM -- that path (OllamaLLMClient)
requires a running local model server and is documented, not tested, in
README.

Run with:
    python -m pytest tests/test_rag_pipeline.py -v
or:
    python tests/test_rag_pipeline.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.context_builder import build_context_block, format_techniques_block
from src.rag.llm_clients import TemplateLLMClient
from src.rag.rag_pipeline import generate_cluster_report
from src.rag.report_schema import ReportParseError, parse_llm_json_response
from src.rag.retriever import RagIndex


def _write_fixture_output_dir() -> str:
    """Builds a tiny, real-shaped output/ directory (the same files
    scripts/run_pipeline.py produces) so RagIndex can be tested without
    depending on a full pipeline run."""
    d = tempfile.mkdtemp()

    profiles = [{
        "cluster_id": 0, "cluster_size": 2,
        "dominant_event_ids": [[1, 2]],
        "hostname_distribution": {"HOST1": 2},
        "common_source_images": [["C:\\Windows\\System32\\powershell.exe", 1]],
        "common_target_images": [],
        "common_destination_ips": [],
        "temporal_span": {"earliest": "2020-01-01T00:00:00Z", "latest": "2020-01-01T00:00:05Z"},
        "evidence_snippets": ["powershell -enc ABC"],
        "representative_event_uids": ["a1", "a2"],
        "text": "Cluster 0 contains 2 events, 100.0% of which are EventID 1. "
                "Repeated execution of powershell.exe observed.",
    }]
    with open(os.path.join(d, "cluster_profiles_v2.json"), "w") as f:
        json.dump(profiles, f)

    mappings = [{
        "cluster_id": 0,
        "cluster_text": profiles[0]["text"],
        "top_k_techniques": [
            {"technique_id": "T1059.001", "technique_name": "PowerShell", "tactic": "Execution",
             "similarity_score": 0.81, "mapping_status": "CANDIDATE"},
            {"technique_id": "T1486", "technique_name": "Data Encrypted for Impact", "tactic": "Impact",
             "similarity_score": 0.12, "mapping_status": "NO_CONFIDENT_MATCH"},
        ],
        "representative_event_uids": ["a1", "a2"],
    }]
    with open(os.path.join(d, "semantic_mitre_mappings.json"), "w") as f:
        json.dump(mappings, f)

    with open(os.path.join(d, "embedding_clusters.csv"), "w") as f:
        f.write("embedding_index,event_uid,cluster_id,membership_probability,is_noise\n")
        f.write("0,a1,0,0.9,False\n")
        f.write("1,a2,0,0.95,False\n")

    with open(os.path.join(d, "novelty_scores.csv"), "w") as f:
        f.write("event_uid,cluster_id,density_score,embedding_distance_score,reconstruction_error,"
                 "graph_rarity_score,novelty_score,novelty_label\n")
        f.write("a1,0,0.1,0.1,0.1,0.1,0.1,NORMAL\n")
        f.write("a2,0,0.6,0.5,0.4,0.5,0.5,SUSPICIOUS\n")

    return d


def test_rag_index_loads_and_joins():
    d = _write_fixture_output_dir()
    index = RagIndex(output_dir=d)
    assert index.available_cluster_ids() == [0]

    ctx = index.get_cluster_context(0)
    assert ctx.cluster_id == 0
    assert ctx.cluster_size == 2
    assert len(ctx.techniques) == 2
    assert ctx.mean_novelty_score == 0.3  # mean(0.1, 0.5)
    assert ctx.max_novelty_score == 0.5
    assert ctx.novelty_label_counts == {"NORMAL": 1, "SUSPICIOUS": 1}


def test_rag_index_rejects_unknown_cluster():
    d = _write_fixture_output_dir()
    index = RagIndex(output_dir=d)
    try:
        index.get_cluster_context(999)
        assert False, "expected KeyError for unknown cluster_id"
    except KeyError:
        pass


def test_rag_index_requires_all_four_files():
    d = tempfile.mkdtemp()  # empty dir
    try:
        RagIndex(output_dir=d)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_context_block_contains_grounding_fields():
    d = _write_fixture_output_dir()
    index = RagIndex(output_dir=d)
    ctx = index.get_cluster_context(0)
    block = build_context_block(ctx)

    assert "Cluster ID: 0" in block
    assert "T1059.001" in block
    assert "NO_CONFIDENT_MATCH" in block
    assert "SUSPICIOUS" in block  # novelty label should surface in the block


def test_techniques_block_handles_empty_list():
    assert "no technique candidates" in format_techniques_block([])


def test_template_llm_produces_schema_valid_report():
    d = _write_fixture_output_dir()
    index = RagIndex(output_dir=d)
    llm = TemplateLLMClient()

    report = generate_cluster_report(0, index, llm)
    assert report.cluster_id == 0
    assert report.generated_by == "template"
    assert len(report.likely_techniques) == 2
    assert report.likely_techniques[0].technique_id == "T1059.001"
    assert report.grounding_event_uids == ["a1", "a2"]


def test_schema_rejects_hallucinated_technique_id():
    """The core anti-hallucination check: a technique_id not present in
    allowed_technique_ids must be rejected, not silently accepted."""
    raw_json = {
        "title": "t", "summary": "s", "observed_behavior": "b", "risk_assessment": "r",
        "likely_techniques": [
            {"technique_id": "T9999.999", "technique_name": "Made Up Technique", "tactic": "Fake",
             "similarity_score": 0.99, "confidence_label": "CANDIDATE", "reasoning": "invented"},
        ],
    }
    try:
        parse_llm_json_response(
            raw_json, cluster_id=0, generated_by="test", grounding_event_uids=[],
            cluster_size=1, max_novelty_score=None, allowed_technique_ids=["T1059.001"],
        )
        assert False, "expected ReportParseError for a technique_id not in allowed_technique_ids"
    except ReportParseError:
        pass


def test_schema_rejects_missing_required_field():
    raw_json = {"title": "t", "summary": "s"}  # missing observed_behavior, likely_techniques, risk_assessment
    try:
        parse_llm_json_response(raw_json, cluster_id=0, generated_by="test", grounding_event_uids=[],
                                 cluster_size=1, max_novelty_score=None)
        assert False, "expected ReportParseError for missing required fields"
    except ReportParseError:
        pass


def test_empty_technique_list_is_valid():
    """A cluster with zero CANDIDATE techniques must produce a valid report
    with an empty likely_techniques list, not an error and not an invented technique."""
    raw_json = {
        "title": "t", "summary": "s", "observed_behavior": "b",
        "risk_assessment": "No confident technique match was found for this cluster.",
        "likely_techniques": [],
    }
    report = parse_llm_json_response(raw_json, cluster_id=0, generated_by="test", grounding_event_uids=[],
                                      cluster_size=1, max_novelty_score=None, allowed_technique_ids=[])
    assert report.likely_techniques == []


if __name__ == "__main__":
    tests = [
        test_rag_index_loads_and_joins,
        test_rag_index_rejects_unknown_cluster,
        test_rag_index_requires_all_four_files,
        test_context_block_contains_grounding_fields,
        test_techniques_block_handles_empty_list,
        test_template_llm_produces_schema_valid_report,
        test_schema_rejects_hallucinated_technique_id,
        test_schema_rejects_missing_required_field,
        test_empty_technique_list_is_valid,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll RAG pipeline tests passed.")
