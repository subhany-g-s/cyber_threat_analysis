import copy
import json

import pytest

from src.graphrag import schema as S
from src.ontology.contract import (ContractError, load_ontology_document, validate_findings_document,
                                   validate_ontology_document)
from src.ontology.export import export_findings
from src.ontology.resolution import ingest_ontology, resolve_ontology
from src.service.pipeline import PipelineService
from tests.conftest import ROOT, base_config


def ent(eid, **kw):
    d = {"entity_id": eid, "entity_type": "AnyType", "canonical_name": "n-" + eid, "source_id": "doc:1", "status": "observed",
         "confidence": {"value": 0.7, "method": "m"}, "evidence": [{"text": "t"}]}
    d.update(kw)
    return d


def doc(entities, rels=()):
    return {"contract": "subhany.ontology-entities", "contract_version": "1.0.0", "producer": {"name": "p"},
            "entities": entities, "relationships": list(rels)}


def test_valid_example_file_loads():
    d, r = load_ontology_document(f"{ROOT}/docs/examples/ontology_entities.example.json")
    assert r.ok and r.entities == 2 and r.relationships == 1


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d.update(contract="other"), "contract must be"),
    (lambda d: d.update(contract_version="2.0.0"), "major version"),
    (lambda d: d.update(contract_version="one"), "semver"),
    (lambda d: d.pop("producer"), "producer.name"),
    (lambda d: d["entities"][0].pop("source_id"), "source_id"),
    (lambda d: d["entities"][0].update(status="maybe"), "status"),
    (lambda d: d["entities"][0].update(confidence={"value": 1.5, "method": "m"}), "confidence.value"),
    (lambda d: d["entities"][0].update(confidence={"value": 0.5}), "method"),
    (lambda d: d["entities"].append(ent("e1")), "duplicates"),
    (lambda d: d["relationships"].append({"source_entity_id": "e1", "target_entity_id": "ghost", "relationship_type": "r",
                                          "source_id": "s", "status": "observed"}), "not an entity_id"),
    (lambda d: d["entities"][0].update(status="inferred", evidence=[], confidence=None), "inferred"),
])
def test_contract_rejects_bad_documents(mutate, needle):
    d = doc([ent("e1")])
    mutate(d)
    r = validate_ontology_document(d)
    assert not r.ok and any(needle in e for e in r.errors), r.errors


def test_same_name_different_ids_is_warned_never_merged():
    d = doc([ent("e1", canonical_name="Same"), ent("e2", canonical_name="same")])
    r = validate_ontology_document(d)
    assert r.ok and any("NOT merged" in w for w in r.warnings)


def test_unreadable_or_non_json_file(tmp_path):
    with pytest.raises(ContractError):
        load_ontology_document(str(tmp_path / "missing.json"))
    p = tmp_path / "bad.json"
    p.write_text("{oops")
    with pytest.raises(ContractError):
        load_ontology_document(str(p))


def _events_with_ip(art):
    for uid, ev in art.events_by_uid.items():
        if ev.get("destination_ip"):
            return uid, ev["destination_ip"]
    pytest.fail("fixture slice has no event with a destination IP")


def test_resolution_requires_strong_identifiers(small_run):
    art = small_run.artifacts
    store = copy.deepcopy(art.knowledge_graph)
    uid, ip = _events_with_ip(art)
    some_host = next(e["hostname"] for e in art.events_by_uid.values() if e.get("hostname"))
    d = doc([
        ent("by-ip", identifiers=[{"scheme": "ip_address", "value": ip}]),
        ent("by-evidence", evidence=[{"event_uid": uid}]),
        ent("dangling", evidence=[{"event_uid": "00000000-0000-0000-0000-000000000000"}]),
        ent("name-only", canonical_name=some_host),
        ent("weak-scheme", identifiers=[{"scheme": "hostname", "value": some_host}]),
        ent("nothing", canonical_name="completely unknown thing"),
    ])
    assert validate_ontology_document(d).ok
    rep = resolve_ontology(d, art.events_by_uid, store)
    linked = {e["src_key"] for e in store.iter_edges() if e["rel"] == S.EVIDENCED_BY}
    assert {"by-ip", "by-evidence"} <= linked
    assert not ({"name-only", "weak-scheme", "nothing", "dangling"} & linked)          # names / weak schemes never link
    assert any(m["entity_id"] == "name-only" and m["identity_asserted"] is False for m in rep["possible_name_matches"])
    assert rep["dangling_event_references"] == 1 and rep["unresolved"] == 4
    assert any(e["rel"] == S.RESOLVED_TO and e["src_key"] == "by-ip" for e in store.iter_edges())
    assert "names alone never merge" in rep["policy"]


def test_ingest_stores_opaque_types_verbatim(small_run):
    store = copy.deepcopy(small_run.artifacts.knowledge_graph)
    d = doc([ent("a", entity_type="Weird Type/X", ontology_uri="urn:x"), ent("b")],
            [{"source_entity_id": "a", "target_entity_id": "b", "relationship_type": "their_custom_rel", "source_id": "s", "status": "inferred",
              "evidence": [{"text": "t"}]}])
    out = ingest_ontology(store, d)
    assert out == {"entities": 2, "relationships": 1}
    assert store.get_node(S.ONTOLOGY, "a")["entity_type"] == "Weird Type/X"
    rel = [e for e in store.iter_edges() if e["rel"] == S.ONTOLOGY_RELATION][0]
    assert rel["props"]["relationship_type"] == "their_custom_rel" and rel["props"]["status"] == "inferred"
    ingest_ontology(store, d)                                                           # idempotent
    assert sum(1 for e in store.iter_edges() if e["rel"] == S.ONTOLOGY_RELATION) == 1


def test_export_matches_contract_and_event_set(small_run):
    doc_ = export_findings(small_run)
    assert validate_findings_document(doc_).ok
    art = small_run.artifacts
    assert [r["event_uid"] for r in doc_["records"]] == list(art.row_to_uid)
    r0 = doc_["records"][0]
    for k in ("event_uid", "cluster_id", "source_id", "novelty", "behavior_summary", "technique_candidates", "supporting_event_uids"):
        assert k in r0
    assert doc_["mapping"]["corpus_version"] == small_run.mapping["corpus_version"] and doc_["mapping"]["is_standin_embedder"] is True
    noise_recs = [r for r in doc_["records"] if r["cluster_id"] == -1]
    assert len(noise_recs) == small_run.clustering["noise_events"]
    assert {r["unit_scope"] for r in noise_recs} <= {"noise_group", "noise_event", "unreviewed_noise"}
    assert "not a probability" in doc_["mapping"]["similarity_semantics"]
    json.dumps(doc_)


def test_findings_validation_rejects_bad_scores_and_duplicates():
    base = {"contract": "subhany.pipeline-findings", "contract_version": "1.0.0", "mapping": {"corpus_version": "v"},
            "records": [{"event_uid": "a", "source_id": "s", "unit_scope": "cluster", "novelty": {"score": 0.2},
                         "technique_candidates": [{"technique_id": "T1", "similarity_score": 0.5}]}]}
    assert validate_findings_document(base).ok
    bad = copy.deepcopy(base)
    bad["records"][0]["technique_candidates"][0]["similarity_score"] = 1.7
    assert not validate_findings_document(bad).ok
    dup = copy.deepcopy(base)
    dup["records"].append(dup["records"][0])
    assert not validate_findings_document(dup).ok


def test_pipeline_with_ontology_file_end_to_end(small_input, mitre_cache, small_run, tmp_path):
    uid, ip = _events_with_ip(small_run.artifacts)
    p = tmp_path / "onto.json"
    p.write_text(json.dumps(doc([ent("x1", evidence=[{"event_uid": uid}])])))
    res = PipelineService(base_config(small_input, mitre_cache, ontology_path=str(p))).run()
    assert res.ontology["valid"] and res.ontology["entities"] == 1 and res.ontology["resolution"]["linked_to_events"] == 1
    edges = [e for e in res.artifacts.knowledge_graph.iter_edges() if e["rel"] == S.EVIDENCED_BY]
    assert [(e["src_key"], e["dst_key"]) for e in edges] == [("x1", uid)]
    assert edges[0]["props"]["status"] == S.STATUS_OBSERVED


def test_pipeline_with_invalid_ontology_warns_but_continues(small_input, mitre_cache, tmp_path):
    p = tmp_path / "onto.json"
    p.write_text(json.dumps({"contract": "wrong"}))
    res = PipelineService(base_config(small_input, mitre_cache, ontology_path=str(p))).run()
    assert res.ok and any("rejected by contract validation" in w for w in res.stages[8].warnings)
