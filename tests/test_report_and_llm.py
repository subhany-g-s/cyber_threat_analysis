import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.rag.evidence_report import EvidenceReportBuilder
from src.rag.llm_clients import OllamaLLMClient
from src.service.pipeline import PipelineService
from tests.conftest import base_config


def builder_for(res):
    mp = res.mapping
    return EvidenceReportBuilder(
        {"run_id": res.run_id, "corpus_version": mp["corpus_version"],
         "embedder": {"model_id": mp["model_id"], "is_standin": mp["is_standin_embedder"]}, "ontology_present": False,
         "platform_filter": mp["platform_filter"], "thresholds": (0.35, 0.25)},
        res.artifacts.row_to_uid, res.artifacts.provenance_by_uid)


def a_unit_with_candidates(res):
    for u in res.units:
        if u.report["sections"]["technique_candidates"]:
            return u
    pytest.fail("no unit with technique candidates in fixture run")


def test_all_sections_present_and_distinguished(small_run):
    r = small_run.units[0].report
    assert set(r["sections"]) >= {"observed_facts", "computed_findings", "technique_candidates", "reference_information",
                                  "graph_evidence", "ontology_facts", "inferences", "uncertainty_and_missing_evidence",
                                  "recommended_investigation_steps"}
    assert small_run.report["validation"]["passed"] and r["generated_by"] == "deterministic-evidence-template"
    assert "not" in r["disclaimer"].lower() and "confirmed" in r["disclaimer"].lower()


def test_validation_catches_fabricated_evidence(small_run):
    b = builder_for(small_run)
    good = small_run.units[0].report
    assert b.validate_unit_report(good) == []
    bad = copy.deepcopy(good)
    bad["sections"]["observed_facts"][0]["evidence"].append("EV:11111111-2222-3333-4444-555555555555")
    assert any("not in this run" in i for i in b.validate_unit_report(bad))
    bad = copy.deepcopy(a_unit_with_candidates(small_run).report)
    t = bad["sections"]["technique_candidates"][0]
    t["similarity_score"] = 1.4
    t["status"] = "CONFIRMED"
    t["supporting_event_uids"].append("deadbeef")
    t["reference_ids"].append("REF:made-up")
    issues = " | ".join(b.validate_unit_report(bad))
    assert "outside [0,1]" in issues and "invalid status" in issues and "not in this run" in issues and "unknown reference" in issues
    bad = copy.deepcopy(good)
    bad["sections"]["inferences"].append({"text": "x", "basis": ["U:NOPE"], "source": "s"})
    assert any("unknown evidence id" in i for i in b.validate_unit_report(bad))
    bad = copy.deepcopy(good)
    del bad["sections"]["uncertainty_and_missing_evidence"]
    assert any("missing section" in i for i in b.validate_unit_report(bad))


def test_every_candidate_is_a_retrieved_candidate_with_valid_range(small_run):
    for u in small_run.units:
        cands = {c["technique_id"] for c in u.mapping.get("top_k_techniques", [])}
        for t in u.report["sections"]["technique_candidates"]:
            assert t["technique_id"] in cands and 0.0 <= t["similarity_score"] <= 1.0
            assert t["caveat"].startswith("semantic similarity lead")


def grounded_reply(pkg):
    refs = [e["ref"] for e in pkg["events"]][:2]
    return {"what_happened": {"text": "The listed events were recorded for this unit.", "evidence": refs},
            "why_unusual": {"text": "Novelty is high relative to the rest of the dataset.", "evidence": refs[:1]},
            "security_relevance": {"text": "Possibly relevant; this is a hypothesis."},
            "attack": [], "missing_evidence": ["The raw source records have not been reviewed."],
            "benign_explanations": ["Routine administration could explain it."],
            "next_steps": [{"action": "Review the raw source records.", "why": "Confirms the fields.", "evidence": refs[:1]}],
            "conclusion": {"supported": "The events occurred.", "uncertain": "Intent is unknown."}}


# ------------------------------------------------------------------ Ollama wire format (LOCAL MOCK HTTP SERVER, not real Ollama)
class _H(BaseHTTPRequestHandler):
    seen = []
    chat_reply = None                       # None -> build a valid, grounded analysis from the package in the prompt
    models = ["mockmodel:latest"]

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"models": [{"name": n} for n in self.models]}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n))
        _H.seen.append((self.path, req))
        reply = self.chat_reply
        if reply is None:
            pkg = json.loads(req["messages"][-1]["content"].split("\n\n", 1)[1])
            reply = grounded_reply(pkg)
        body = json.dumps({"message": {"role": "assistant", "content": reply if isinstance(reply, str) else json.dumps(reply)}}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def mock_ollama():
    _H.seen = []
    srv = HTTPServer(("127.0.0.1", 0), _H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_ollama_client_wire_format(mock_ollama):
    c = OllamaLLMClient(model="mockmodel", host=mock_ollama)
    assert c.available() == (True, "")
    _H.chat_reply = {"ok": 1}
    out = c.generate("SYS", "USER")
    _H.chat_reply = None
    path, payload = _H.seen[-1]
    assert path == "/api/chat" and payload["model"] == "mockmodel" and payload["stream"] is False
    assert payload["options"]["temperature"] == 0 and [m["role"] for m in payload["messages"]] == ["system", "user"]
    assert payload["format"] == "json" and payload["options"]["num_ctx"] > 0


def test_ollama_availability_failures(mock_ollama):
    ok, why = OllamaLLMClient(model="other", host=mock_ollama).available()
    assert not ok and "ollama pull other" in why
    ok, why = OllamaLLMClient(model="m", host="http://127.0.0.1:9").available(timeout=0.5)
    assert not ok and "not reachable" in why


def test_pipeline_with_ollama_mock_server_end_to_end(small_input, mitre_cache, mock_ollama):
    res = PipelineService(base_config(small_input, mitre_cache, llm="ollama", ollama_model="mockmodel",
                                      ollama_host=mock_ollama, report_top_n=3)).run()
    a = res.analyst_report
    assert res.ok and a["analysis_mode"]["mode"] == "llm-assisted" and a["analysis_mode"]["accepted"] == 3 == len(a["findings"])
    assert all(f["analysis"]["source"] == "llm:ollama:mockmodel" for f in a["findings"][:3])
    assert len([p for p in _H.seen if p[0] == "/api/chat"]) == 3          # one call per analysed finding, no retries
    assert res.report["validation"]["passed"] and "LLM" in res.report["status_banner"]
