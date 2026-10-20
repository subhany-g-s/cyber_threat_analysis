"""Shared fixtures. The heavy fixture (`small_run`) runs the REAL pipeline once on a
400-line slice of the shipped sample with the REAL pretrained GAE checkpoint and
the REAL cached ATT&CK corpus; only the embedder is the offline lexical stand-in
(SBERT needs a model download that is unavailable here)."""
import json
import os
import shutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for sub in ("graph", "gae", "clustering", "novelty"):
    sys.path.insert(0, os.path.join(ROOT, "src", sub))

SAMPLE = os.path.join(ROOT, "data", "sample_2000_events.json")


def make_bundle():
    """Small synthetic STIX bundle exercising: technique, sub-technique, revoked, deprecated,
    detection strategy/analytic, missing-description, platforms and tactics."""
    def ap(i, ext, name, desc, phases=("execution",), platforms=("Windows",), sub=False, **extra):
        o = {"type": "attack-pattern", "id": f"attack-pattern--{i:08d}-0000-4000-8000-000000000000", "name": name,
             "description": desc, "x_mitre_platforms": list(platforms), "x_mitre_is_subtechnique": sub,
             "kill_chain_phases": [{"kill_chain_name": "mitre-attack", "phase_name": p} for p in phases],
             "external_references": [{"source_name": "mitre-attack", "external_id": ext,
                                      "url": f"https://attack.mitre.org/techniques/{ext.replace('.', '/')}"}],
             "x_mitre_detection": "Monitor process creation."}
        o.update(extra)
        return o
    objs = [
        {"type": "x-mitre-collection", "id": "x-mitre-collection--00000000-0000-4000-8000-000000000001",
         "name": "Enterprise ATT&CK", "x_mitre_version": "99.9", "x_mitre_attack_spec_version": "3.3.0",
         "modified": "2030-01-01T00:00:00.000Z"},
        {"type": "x-mitre-tactic", "id": "x-mitre-tactic--00000000-0000-4000-8000-000000000002", "name": "Execution",
         "x_mitre_shortname": "execution"},
        {"type": "x-mitre-tactic", "id": "x-mitre-tactic--00000000-0000-4000-8000-000000000003", "name": "Persistence",
         "x_mitre_shortname": "persistence"},
        ap(1, "T9001", "Command Interpreter Abuse", "Adversaries abuse command interpreters such as powershell to execute commands and scripts."),
        ap(2, "T9001.001", "Powershell Encoded Commands", "Adversaries run powershell with encoded command lines to hide script content.", sub=True),
        ap(3, "T9002", "Registry Run Key Persistence", "Adversaries add registry run keys so programs start at logon.", phases=("persistence",)),
        ap(4, "T9003", "Old Removed Technique", "Superseded text.", revoked=True),
        ap(5, "T9004", "Deprecated Technique", "Deprecated text.", x_mitre_deprecated=True),
        ap(6, "T9005", "Linux Only Thing", "Adversaries modify cron entries on linux hosts.", platforms=("Linux",), phases=("persistence",)),
        {"type": "relationship", "id": "relationship--00000000-0000-4000-8000-000000000010", "relationship_type": "subtechnique-of",
         "source_ref": "attack-pattern--00000002-0000-4000-8000-000000000000",
         "target_ref": "attack-pattern--00000001-0000-4000-8000-000000000000"},
        {"type": "relationship", "id": "relationship--00000000-0000-4000-8000-000000000011", "relationship_type": "revoked-by",
         "source_ref": "attack-pattern--00000004-0000-4000-8000-000000000000",
         "target_ref": "attack-pattern--00000001-0000-4000-8000-000000000000"},
    ]
    return {"type": "bundle", "id": "bundle--00000000-0000-4000-8000-0000000000ff", "objects": objs}


@pytest.fixture
def bundle():
    return make_bundle()


@pytest.fixture(scope="session")
def small_input(tmp_path_factory):
    d = tmp_path_factory.mktemp("input")
    p = d / "small.jsonl"
    with open(SAMPLE, "r", encoding="utf-8") as src, open(p, "w", encoding="utf-8") as dst:
        for i, line in enumerate(src):
            if i >= 400:
                break
            dst.write(line)
    return str(p)


@pytest.fixture(scope="session")
def mitre_cache(tmp_path_factory):
    """A private copy of the real cached ATT&CK corpus (so tests never touch cache/mitre/embeddings)."""
    d = tmp_path_factory.mktemp("mitre_cache")
    src = os.path.join(ROOT, "cache", "mitre")
    cur = open(os.path.join(src, "CURRENT")).read().strip()
    shutil.copy(os.path.join(src, cur), d / cur)
    shutil.copy(os.path.join(src, "CURRENT"), d / "CURRENT")
    return str(d)


def base_config(input_path, mitre_cache, **kw):
    from src.service.config import PipelineConfig
    d = dict(input_path=input_path, project_root=ROOT, embedder="lexical", llm="none", min_cluster_size=8,
             min_samples=4, mitre_cache_dir=mitre_cache, noise_review_top_n=6)
    d.update(kw)
    return PipelineConfig(**d)


@pytest.fixture(scope="session")
def small_run(small_input, mitre_cache):
    from src.service.pipeline import PipelineService
    events = []
    res = PipelineService(base_config(small_input, mitre_cache), on_stage=events.append).run()
    res._callback_events = events
    return res
