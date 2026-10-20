import json
import os

import numpy as np
import pytest

from src.service.pipeline import PipelineService
from src.service.result import FAILED, OK, SKIPPED, STAGE_NAMES, WARN
from tests.conftest import ROOT, base_config


def test_ten_stages_in_order_with_callbacks(small_run):
    assert [s.name for s in small_run.stages] == STAGE_NAMES and len(STAGE_NAMES) == 10
    assert [s.number for s in small_run._callback_events] == list(range(1, 11))
    assert all(s.status in (OK, WARN) for s in small_run.stages), [(s.number, s.status, s.error) for s in small_run.stages]
    assert small_run.ok and not small_run.errors
    assert all(s.elapsed_s >= 0 and s.explanation for s in small_run.stages)


def test_metrics_are_real_not_templated(small_run):
    assert small_run.parsing["valid_records"] == 400
    assert small_run.graph["num_nodes"] == small_run.normalization["unique_events"]
    assert small_run.gae["retrained"] is False and small_run.gae["in_channels"] == 78
    assert small_run.gae["embedding_shape"] == [small_run.graph["num_nodes"], 32]
    cl = small_run.clustering
    assert cl["total_events"] == cl["noise_events"] + sum(1 for l in small_run.artifacts.labels if l != -1)


def test_result_is_json_serialisable_and_in_memory(small_run, tmp_path):
    json.dumps(small_run.to_dict())
    assert small_run.artifacts.embeddings is not None and small_run.artifacts.knowledge_graph is not None
    assert "exported" not in small_run.metadata                       # default: nothing written


def test_checkpoint_and_schema_files_untouched_by_a_run(small_input, mitre_cache):
    import hashlib
    paths = [os.path.join(ROOT, "models", "gae_model.pt"), os.path.join(ROOT, "models", "gae_feature_schema.json")]
    h = lambda: [hashlib.sha256(open(p, "rb").read()).hexdigest() for p in paths]
    before = h()
    res = PipelineService(base_config(small_input, mitre_cache)).run()
    assert res.gae["retrained"] is False and h() == before


def test_missing_input_fails_stage1_and_skips_rest(mitre_cache, tmp_path):
    res = PipelineService(base_config(str(tmp_path / "nope.jsonl"), mitre_cache)).run()
    assert not res.ok and res.stages[0].status == FAILED and "not found" in res.stages[0].error
    assert all(s.status == SKIPPED for s in res.stages[1:])
    json.dumps(res.to_dict())


def test_empty_and_garbage_input(mitre_cache, tmp_path):
    e = tmp_path / "e.jsonl"
    e.write_text("")
    assert PipelineService(base_config(str(e), mitre_cache)).run().stages[0].status == FAILED
    g = tmp_path / "g.jsonl"
    g.write_text("garbage\n")
    assert PipelineService(base_config(str(g), mitre_cache)).run().stages[0].status == FAILED


def test_missing_checkpoint_fails_stage4_with_clear_message(small_input, mitre_cache):
    res = PipelineService(base_config(small_input, mitre_cache, checkpoint_path="models/does_not_exist.pt")).run()
    assert [s.status for s in res.stages[:3]] == [OK, OK, OK]
    assert res.stages[3].status == FAILED and "not found" in res.stages[3].error and "retrain" in res.stages[3].error
    assert all(s.status == SKIPPED for s in res.stages[4:])


def test_incompatible_schema_fails_stage4(small_input, mitre_cache, tmp_path):
    schema = json.load(open(os.path.join(ROOT, "models", "gae_feature_schema.json")))
    bad = tmp_path / "schema.json"
    schema['feature_names'] = schema['feature_names'][:-1]      # no longer matches the checkpoint's in_channels=78
    bad.write_text(json.dumps(schema))
    res = PipelineService(base_config(small_input, mitre_cache, feature_schema_path=str(bad))).run()
    assert res.stages[3].status == FAILED
    assert res.stages[3].error and ("Incompatible" in res.stages[3].error or "schema" in res.stages[3].error.lower())


def test_corrupt_checkpoint_fails_cleanly(small_input, mitre_cache, tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a checkpoint")
    res = PipelineService(base_config(small_input, mitre_cache, checkpoint_path=str(bad))).run()
    assert res.stages[3].status == FAILED and res.stages[3].error


def test_precomputed_embeddings_path_and_shape_check(small_run, small_input, mitre_cache, tmp_path):
    art = small_run.artifacts
    npy, mp = tmp_path / "z.npy", tmp_path / "m.json"
    np.save(npy, art.embeddings)
    mp.write_text(json.dumps({"row_to_event_uid": art.row_to_uid}))
    res = PipelineService(base_config(small_input, mitre_cache, precomputed_embeddings=str(npy),
                                      precomputed_mapping=str(mp))).run()
    assert res.stages[3].status in (OK, WARN) and res.gae["source"] != "pretrained_checkpoint"
    np.save(npy, art.embeddings[:-5])                                   # wrong row count
    res2 = PipelineService(base_config(small_input, mitre_cache, precomputed_embeddings=str(npy),
                                       precomputed_mapping=str(mp))).run()
    assert res2.stages[3].status == FAILED


def test_missing_mitre_corpus_degrades_without_fallback(small_input, tmp_path):
    res = PipelineService(base_config(small_input, str(tmp_path / "empty_cache"))).run()
    assert res.stages[7].status == FAILED and "update_mitre_corpus" in res.stages[7].error
    assert all(res.stages[i].status in (OK, WARN) for i in range(7))     # data stages unaffected
    assert res.stages[8].status in (OK, WARN, SKIPPED, FAILED)
    assert not res.mapping.get("legacy_curated_corpus")


def test_llm_unavailable_reports_deterministic_not_error(small_input, mitre_cache):
    res = PipelineService(base_config(small_input, mitre_cache, llm="ollama", ollama_host="http://127.0.0.1:9")).run()
    s10 = res.stages[9]
    assert s10.status in (OK, WARN) and res.ok
    assert any("LLM not used" in w and "deterministic fallback" in w for w in s10.warnings)
    assert res.analyst_report["analysis_mode"]["mode"] == "deterministic"
    assert "Deterministic fallback" in res.report["status_banner"]


def test_export_only_when_asked(small_input, mitre_cache, tmp_path):
    out = tmp_path / "out"
    res = PipelineService(base_config(small_input, mitre_cache, export_dir=str(out))).run()
    names = set(os.listdir(out))
    assert {"run_result.json", "evidence_report.txt", "evidence_report.md", "ontology_findings.json", "noise_review.json",
            "cluster_profiles_v2.json", "semantic_mitre_mappings.json", "embedding_clusters.csv",
            "novelty_scores.csv", "rag_ready_clusters.json"} <= names
    assert res.metadata["exported"]["files"]
