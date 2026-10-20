"""Analyst layer: findings, prioritisation, technique evidence, LLM validation and fallback.
All LLM behaviour here is exercised with fakes / a local mock HTTP server - never a real model."""
import copy
import json
import socket

import pytest

from src.rag import finding_llm as fl
from src.rag.analyst_render import render_analyst_markdown, render_analyst_text
from src.rag.findings import EvidenceIndex, event_markers, rate_priority
from src.rag.llm_clients import LLMUnavailableError, OllamaLLMClient
from tests.test_report_and_llm import grounded_reply


class FakeLLM:
    def __init__(self, reply, name="fake"):
        self.reply, self.backend_name, self.calls = reply, name, 0

    def generate(self, system, user):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        pkg = json.loads(user.split("\n\n", 1)[1])
        r = self.reply(pkg) if callable(self.reply) else self.reply
        return r if isinstance(r, str) else json.dumps(r)


def with_technique(finding, context_only=True, support=()):
    """Inject a synthetic ATT&CK candidate (the lexical fixture run produces none)."""
    f = copy.deepcopy(finding)
    f["attack"] = [{"technique_id": "T9001", "name": "Command Interpreter Abuse", "tactics": ["execution"], "unit_similarity": 0.4,
                    "status": "CANDIDATE", "event_support": [{"ref": r, "event_level_similarity": 0.5} for r in support],
                    "context_only": context_only, "detection_guidance": "Monitor process creation.", "fit": "weak",
                    "why_candidate": "x", "limitations": "x", "support_basis": "x"}]
    return f


@pytest.fixture
def analyst(small_run):
    return copy.deepcopy(small_run.analyst_report)


# ------------------------------------------------------------------ structure / determinism
def test_default_report_structure_and_brevity(small_run, tmp_path):
    a = small_run.analyst_report
    md = render_analyst_markdown(a, {"run_id": small_run.run_id, "generated_at": "t", "units_total": 70})
    for h in ("## A. Executive summary", "## B. Prioritized", "## C. Overall assessment", "## D. Prioritized investigation plan", "## E. Technical appendix"):
        assert h in md
    assert len(md.splitlines()) < 450 and a["analysis_mode"]["mode"] == "deterministic"
    assert len(a["findings"]) <= 5
    render_analyst_text(a, {"run_id": "x"}).encode("ascii")
    for f in a["findings"]:
        assert f["priority"] in ("Critical", "High", "Medium", "Low", "Informational") and f["priority_reason"] and f["evidence_refs"]
        assert f["missing_evidence"] and f["benign_explanations"] and f["next_steps"]


def test_all_evidence_refs_resolve(small_run):
    a = small_run.analyst_report
    index = a["evidence_index"]
    for f in a["findings"]:
        for r in f["evidence_refs"]:
            assert r.startswith(("U:", "TC:")) or r in index


def test_findings_never_merge_on_host_alone(small_run):
    for f in small_run.analyst_report["findings"]:
        if f["merged"]:
            basis = " ".join(f["merge_basis"])
            assert "process id" in basis or "indicator artefact" in basis
    units = [u for f in small_run.analyst_report["findings"] for u in f["unit_ids"]]
    assert len(units) == len(set(units))                       # a unit belongs to exactly one finding (dedup)


def test_priority_ignores_similarity_scores():
    # The ladder takes novelty tier + markers + scope only - there is no similarity input.
    import inspect
    assert "similarity" not in " ".join(inspect.signature(rate_priority).parameters)


def test_markers_and_hidden_characters():
    ev = {"event_id": 1, "raw_event": {"Image": "C:\\ProgramData\\x\\\u202ecod.scr", "CommandLine": "x"}}
    m = event_markers(ev)
    assert "control_chars" in m and "script_like_extension" in m and "staging_path" in m
    moji = {"event_id": 11, "raw_event": {"TargetFilename": "C:\\a\\\u00e2\u20ac\u00aecod.scr"}}
    assert "control_chars" in event_markers(moji)
    assert not event_markers({"event_id": 3, "destination_ip": "10.0.0.5", "raw_event": {"Image": "C:\\Windows\\System32\\svchost.exe"}})


def test_evidence_index_is_stable_and_resolvable():
    i = EvidenceIndex(["abc12345-0000", "def45678-0000"])
    r1, r2 = i.ref("abc12345-0000"), i.ref("def45678-0000")
    assert r1 != r2 and i.ref("abc12345-0000") == r1 and i.resolve(r1) == "abc12345-0000" and r1.startswith("EV:")


# ------------------------------------------------------------------ LLM validation
def pkg_of(analyst, with_tech=False, support=()):
    f = analyst["findings"][0]
    if with_tech:
        f = with_technique(f, True, support)
    return fl.build_package(f)


def test_valid_grounded_analysis_is_accepted(analyst):
    pkg = pkg_of(analyst)
    clean, problems, _ = fl.validate_analysis(grounded_reply(pkg), pkg)
    assert clean is not None and not problems


@pytest.mark.parametrize("mutate,needle", [
    (lambda r, p: r["what_happened"]["evidence"].append("EV:deadbeef"), "unknown evidence id"),
    (lambda r, p: r["why_unusual"].update(text="Resembles T1059.001 behaviour."), "outside the provided candidates"),
    (lambda r, p: r["conclusion"].update(supported="This is a confirmed attack by APT29."), "over-claims"),
    (lambda r, p: r["why_unusual"].update(text="See C:\\Windows\\totally\\invented.exe"), "not in the package"),
    (lambda r, p: r["what_happened"].update(text="Contacted 203.0.113.99 at night."), "not in the package"),
    (lambda r, p: r["what_happened"].update(evidence=[]), "at least one event"),
    (lambda r, p: r.pop("conclusion"), "conclusion"),
    (lambda r, p: r.update(attack=[{"technique_id": "T9999", "fit": "weak", "rationale": "r", "evidence": []}]), "hallucination"),
    (lambda r, p: r.update(attack=[{"technique_id": "T9001", "fit": "certain", "rationale": "r", "evidence": []}]), "fit must be"),
])
def test_invalid_llm_output_is_rejected(analyst, mutate, needle):
    pkg = pkg_of(analyst, with_tech=True)
    r = grounded_reply(pkg)
    mutate(r, pkg)
    clean, problems, _ = fl.validate_analysis(r, pkg)
    assert clean is None and needle in " ".join(problems)


def test_negated_overclaim_is_allowed_but_plain_overclaim_is_not():
    assert fl.find_overclaim("There is no evidence that the host was compromised.") is None
    assert fl.find_overclaim("The host was compromised by an attacker.")



def test_plausible_fit_is_capped_without_event_level_support(analyst):
    # The technique is context-only, so "plausible" must be capped to "weak".
    pkg = pkg_of(analyst, with_tech=True)
    r = grounded_reply(pkg)

    r["attack"] = [{
        "technique_id": "T9001",
        "fit": "plausible",
        "rationale": (
            "The wording is similar, but the supplied events do not "
            "establish this technique's behavior."
        ),
        "evidence": [pkg["events"][0]["ref"]],
        "missing": (
            "Additional event-level evidence is needed to establish "
            "the technique's behavior."
        ),
    }]

    clean, problems, adjustments = fl.validate_analysis(r, pkg)

    assert clean is not None, f"Validation failed: {problems}"
    assert clean["attack"][0]["fit"] == "weak"
    assert any("capped" in adjustment for adjustment in adjustments)
    assert not problems



def test_plausible_fit_kept_with_event_level_support(analyst):
    finding = analyst["findings"][0]
    ref = finding["evidence"][0]["ref"]

    pkg = fl.build_package(
        with_technique(
            finding,
            context_only=False,
            support=[ref],
        )
    )

    r = grounded_reply(pkg)
    r["attack"] = [{
        "technique_id": "T9001",
        "fit": "plausible",
        "rationale": (
            "The supplied event-level evidence directly supports "
            "the behavior described by this technique."
        ),
        "evidence": [ref],
        "missing": (
            "Further corroborating events would help establish "
            "the full scope of this behavior."
        ),
    }]

    clean, problems, adjustments = fl.validate_analysis(r, pkg)

    assert clean is not None, f"Validation failed: {problems}"
    assert clean["attack"][0]["fit"] == "plausible"
    assert not problems




def test_duplicate_and_contradictory_assessments(analyst):
    pkg = pkg_of(analyst, with_tech=True)
    ref = pkg["events"][0]["ref"]

    def item(fit):
        return {
            "technique_id": "T9001",
            "fit": fit,
            "rationale": (
                "The supplied evidence provides context for assessing "
                "this technique."
            ),
            "evidence": [ref],
            "missing": (
                "Additional event details would help establish "
                "whether this behavior occurred."
            ),
        }

    # Identical duplicate assessments should be deduplicated.
    r = grounded_reply(pkg)
    r["attack"] = [item("weak"), item("weak")]

    clean, problems, adjustments = fl.validate_analysis(r, pkg)

    assert clean is not None, f"Validation failed: {problems}"
    assert len(clean["attack"]) == 1
    assert any("duplicate" in adjustment for adjustment in adjustments)
    assert not problems

    # Conflicting assessments for the same technique must be rejected.
    r["attack"] = [item("weak"), item("insufficient")]

    clean, problems, adjustments = fl.validate_analysis(r, pkg)

    assert clean is None
    assert problems


def test_parse_json_tolerates_fences_and_rejects_garbage():
    assert fl.parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert fl.parse_json_object("not json") is None and fl.parse_json_object("[1]") is None and fl.parse_json_object("") is None


# ------------------------------------------------------------------ applying the LLM / fallback
def test_llm_cannot_change_scores_or_priority(analyst):
    before = [(f["id"], f["priority"], f["priority_reason"], f["novelty"], f["unit_ids"], f["evidence_refs"]) for f in analyst["findings"]]
    fl.apply_llm_analysis(analyst, FakeLLM(grounded_reply, "ollama:t"))
    after = [(f["id"], f["priority"], f["priority_reason"], f["novelty"], f["unit_ids"], f["evidence_refs"]) for f in analyst["findings"]]
    assert before == after


def test_mocked_llm_success_is_labelled(analyst):
    llm = FakeLLM(grounded_reply, "ollama:t")
    info = fl.apply_llm_analysis(analyst, llm)
    n = len(analyst["findings"])
    assert info["accepted"] == n and analyst["analysis_mode"]["mode"] == "llm-assisted" and llm.calls == n
    f = analyst["findings"][0]
    assert f["analysis"]["source"] == "llm:ollama:t"
    md = render_analyst_markdown(analyst, {"run_id": "r"})
    assert "LLM (ollama:t), validated" in md and "[LLM]" in md
    assert "LLM-assisted" in fl.mode_note(analyst)


def test_malformed_llm_output_falls_back_without_retry_storm(analyst):
    llm = FakeLLM("this is not json at all")
    fl.apply_llm_analysis(analyst, llm)
    assert analyst["analysis_mode"]["mode"] == "deterministic" and llm.calls == len(analyst["findings"])    # one call each, no retries
    assert all(f["analysis"]["source"] == "deterministic" and f["llm"]["accepted"] is False for f in analyst["findings"])
    assert "Deterministic fallback" in fl.mode_note(analyst)


def test_partial_acceptance_is_reported_as_partial(analyst):
    state = {"n": 0}

    def reply(pkg):
        state["n"] += 1
        return grounded_reply(pkg) if state["n"] == 1 else "garbage"
    fl.apply_llm_analysis(analyst, FakeLLM(reply))
    assert analyst["analysis_mode"]["mode"] == "llm-partial" and analyst["analysis_mode"]["accepted"] == 1


def test_unreachable_llm_stops_after_first_failure(analyst):
    llm = FakeLLM(LLMUnavailableError("connection refused"))
    info = fl.apply_llm_analysis(analyst, llm)
    assert llm.calls == 1 and info["not_attempted"] == len(analyst["findings"]) - 1
    assert analyst["analysis_mode"]["mode"] == "deterministic" and "connection refused" in analyst["analysis_mode"]["reason"]


def test_consecutive_generic_failures_trip_the_breaker(analyst):
    llm = FakeLLM(RuntimeError("boom"))
    fl.apply_llm_analysis(analyst, llm)
    assert llm.calls == 2 and analyst["analysis_mode"]["mode"] == "deterministic"


def test_max_findings_limits_calls(analyst):
    llm = FakeLLM(grounded_reply)
    fl.apply_llm_analysis(analyst, llm, max_findings=2)
    assert llm.calls == 2 and analyst["findings"][-1]["llm"]["attempted"] is False


def test_real_client_against_closed_port_raises_typed_error():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    with pytest.raises(LLMUnavailableError):
        OllamaLLMClient(model="m", host=f"http://127.0.0.1:{port}", timeout=2).generate("s", "u")


def test_missing_attack_candidate_is_rejected(analyst):
    pkg = pkg_of(analyst, with_tech=True)
    reply = grounded_reply(pkg)

    # The package contains T9001, but the model omits its assessment.
    reply["attack"] = []

    clean, problems, _ = fl.validate_analysis(reply, pkg)

    assert clean is None
    assert any(
        "missing assessments" in problem
        for problem in problems
    )
