"""
tests/test_profiling_and_mapping.py
--------------------------------------
Unit tests for src/profiling/cluster_profiler.py and
src/mapping/semantic_mapper.py using tiny synthetic data and the
TF-IDF stand-in embedder (no network access required -- see
tests/tfidf_embedder.py for why SBERT itself isn't used here).

Run with:
    python -m pytest tests/test_profiling_and_mapping.py -v
or:
    python tests/test_profiling_and_mapping.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.profiling.cluster_profiler import build_cluster_profile, build_cluster_profiles, cluster_to_text
from src.mapping.semantic_mapper import SemanticMitreMapper, TechniqueCorpus
from tests.tfidf_embedder import TfidfEmbedder


def _sample_events():
    return [
        {"event_uid": "a1", "event_id": 1, "hostname": "HOST1", "timestamp": "2020-01-01T00:00:00Z",
         "source_image": "C:\\Windows\\explorer.exe", "target_image": None, "destination_ip": None,
         "protocol": None, "raw_event": {"CommandLine": "notepad.exe"}},
        {"event_uid": "a2", "event_id": 1, "hostname": "HOST1", "timestamp": "2020-01-01T00:00:05Z",
         "source_image": "C:\\Windows\\System32\\powershell.exe", "target_image": None, "destination_ip": None,
         "protocol": None, "raw_event": {"CommandLine": "powershell -enc ABC"}},
    ]


def test_profiling_rejects_noise_cluster():
    events = _sample_events()
    try:
        build_cluster_profile(-1, events, [])
        assert False, "expected ValueError for cluster_id == -1"
    except ValueError:
        pass


def test_profiling_rejects_empty_cluster():
    try:
        build_cluster_profile(0, [], [])
        assert False, "expected ValueError for empty member_events"
    except ValueError:
        pass


def test_cluster_to_text_is_deterministic():
    events = _sample_events()
    p1 = build_cluster_profile(0, events, ["a2"])
    p2 = build_cluster_profile(0, events, ["a2"])
    assert p1.text == p2.text
    assert "powershell.exe" in p1.text
    assert str(p1.cluster_id) in p1.text


def test_build_cluster_profiles_excludes_noise():
    events_by_uid = {e["event_uid"]: e for e in _sample_events()}
    assignments = [
        {"event_uid": "a1", "cluster_id": -1, "membership_probability": 0.0},
        {"event_uid": "a2", "cluster_id": 0, "membership_probability": 0.9},
    ]
    profiles = build_cluster_profiles(assignments, events_by_uid)
    assert len(profiles) == 1
    assert profiles[0].cluster_id == 0


def test_semantic_mapper_returns_top_k_and_never_crashes_on_no_match():
    techniques = [
        {"technique_id": "T1059.001", "technique_name": "PowerShell", "tactic": "Execution",
         "description": "Use of PowerShell to execute commands or scripts."},
        {"technique_id": "T1486", "technique_name": "Data Encrypted for Impact", "tactic": "Impact",
         "description": "Files are encrypted to disrupt availability, as in ransomware."},
    ]
    events_by_uid = {e["event_uid"]: e for e in _sample_events()}
    assignments = [
        {"event_uid": "a1", "cluster_id": 0, "membership_probability": 0.9},
        {"event_uid": "a2", "cluster_id": 0, "membership_probability": 0.95},
    ]
    profiles = build_cluster_profiles(assignments, events_by_uid)

    embedder = TfidfEmbedder()
    corpus_texts = [f"{t['technique_id']} {t['technique_name']}: {t['description']}" for t in techniques]
    embedder.fit(corpus_texts + [p.text for p in profiles])
    corpus = TechniqueCorpus.build(techniques, embedder)

    mapper = SemanticMitreMapper(corpus, embedder, top_k=2, min_similarity=0.9)  # deliberately high floor
    mappings = mapper.map_clusters(profiles)

    assert len(mappings) == 1
    # Zero-similarity techniques are no longer padded into the list (never force a
    # technique onto a cluster), so the list holds between 1 and top_k entries here.
    assert 1 <= len(mappings[0].top_k_techniques) <= 2
    # With an unreachable similarity floor, nothing should be CANDIDATE --
    # must degrade to NO_CONFIDENT_MATCH / NO_MATCH, never crash or force a match.
    assert all(t.mapping_status == "NO_CONFIDENT_MATCH" for t in mappings[0].top_k_techniques)
    assert mappings[0].overall_outcome == "NO_MATCH"


def test_semantic_mapper_rejects_noise_cluster_id():
    class _FakeProfile:
        cluster_id = -1
        text = "noise"
        representative_event_uids = []

    techniques = [{"technique_id": "T1059.001", "technique_name": "PowerShell", "tactic": "Execution",
                   "description": "PowerShell execution."}]
    embedder = TfidfEmbedder()
    embedder.fit(["PowerShell execution.", "noise"])
    corpus = TechniqueCorpus.build(techniques, embedder)
    mapper = SemanticMitreMapper(corpus, embedder)

    try:
        mapper.map_clusters([_FakeProfile()])
        assert False, "expected ValueError when a noise cluster (-1) reaches the mapper"
    except ValueError:
        pass


def test_technique_corpus_cache_mismatch_is_detected():
    import json
    import numpy as np
    import tempfile

    techniques = [{"technique_id": "T1059.001", "technique_name": "PowerShell", "tactic": "Execution",
                   "description": "x"}]
    with tempfile.TemporaryDirectory() as d:
        vec_path = os.path.join(d, "v.npy")
        ids_path = os.path.join(d, "i.json")
        np.save(vec_path, np.zeros((1, 4)))
        with open(ids_path, "w") as f:
            json.dump(["T9999.999"], f)  # mismatched id

        try:
            TechniqueCorpus.load_cached(techniques, vec_path, ids_path)
            assert False, "expected ValueError for stale/mismatched cache"
        except ValueError:
            pass


if __name__ == "__main__":
    tests = [
        test_profiling_rejects_noise_cluster,
        test_profiling_rejects_empty_cluster,
        test_cluster_to_text_is_deterministic,
        test_build_cluster_profiles_excludes_noise,
        test_semantic_mapper_returns_top_k_and_never_crashes_on_no_match,
        test_semantic_mapper_rejects_noise_cluster_id,
        test_technique_corpus_cache_mismatch_is_detected,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
