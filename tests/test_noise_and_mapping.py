import numpy as np
import pytest

from src.mapping.embedders import LexicalCorpusEmbedder
from src.mapping.semantic_mapper import SemanticMitreMapper, TechniqueCorpus
from src.mapping.text_prep import display_safe, prepare_text, technique_embedding_text
from src.mitre import updater
from src.mitre.embedding_cache import cache_path, load_or_build
from tests.conftest import make_bundle


# ------------------------------------------------------------------ noise preservation
def test_every_noise_event_is_preserved_with_attributes_and_neighbors(small_run):
    art = small_run.artifacts
    noise_uids = {u for i, u in enumerate(art.row_to_uid) if art.labels[i] == -1}
    assert noise_uids, "fixture slice must contain noise for this test to mean anything"
    reviewed = {e["event_uid"] for e in small_run.noise["events"]}
    assert reviewed == noise_uids                                          # nothing disappears
    sample = small_run.noise["events"][0]
    for k in ("event_uid", "event_id", "hostname", "novelty_score", "graph_degree", "neighbors", "signals", "review_rank"):
        assert k in sample
    ranks = [e["review_rank"] for e in small_run.noise["events"]]
    assert ranks == list(range(1, len(ranks) + 1))
    scores = [e["novelty_score"] for e in small_run.noise["events"]]
    assert small_run.noise["events"][0]["novelty_score"] == max(scores)


def test_noise_units_cover_all_noise_and_are_not_clusters(small_run):
    units = small_run.noise["units"]
    covered = [u for r in units for u in r["event_uids"]]
    assert len(covered) == len(set(covered)) == small_run.clustering["noise_events"]
    assert {r["scope"] for r in units} <= {"noise_group", "noise_event"}
    for u in small_run.units:
        if u.scope != "cluster":
            assert u.cluster_id == -1 and u.unit_id[0] in "NE"
    assert all(u.scope == "cluster" or u.profile.get("kind") for u in small_run.units)


def test_noise_is_not_called_malicious_and_signals_are_not_causes(small_run):
    text = small_run.noise["explanation"].lower()
    assert "not" in text and "malicious" in text
    noise_reports = [u.report for u in small_run.units if u.scope != "cluster"]
    assert noise_reports
    for r in noise_reports:
        blob = str(r["sections"]["computed_findings"]).lower()
        assert "consistent with being unclustered" in blob or "no per-point reason" in blob or "not assigned" in blob
        assert "is malicious" not in str(r).lower()


def test_noise_units_get_semantic_analysis_without_normal_cluster(small_run):
    nu = [u for u in small_run.units if u.scope != "cluster"]
    assert nu and all(u.mapping and u.mapping["overall_outcome"] in ("CANDIDATES", "WEAK_ONLY", "NO_MATCH") for u in nu)
    assert small_run.stages[6].metrics["noise_events_preserved"] == small_run.clustering["noise_events"]


def test_cluster_minus_one_cannot_be_mapped_as_a_cluster(small_run):
    from src.profiling.cluster_profiler import build_cluster_profile
    with pytest.raises(ValueError):
        build_cluster_profile(-1, [], {}, None)


# ------------------------------------------------------------------ semantic mapper
@pytest.fixture
def mitre():
    return updater.build_corpus(make_bundle(), None, "local-file")


@pytest.fixture
def mapper_parts(mitre, tmp_path):
    emb = LexicalCorpusEmbedder(technique_embedding_text(t) for t in mitre.active())
    tc = TechniqueCorpus.from_mitre(mitre, emb, str(tmp_path), platforms=["Windows"])
    return mitre, emb, tc


def test_topk_ranking_scores_and_supporting_events(mapper_parts):
    _, emb, tc = mapper_parts
    m = SemanticMitreMapper(tc, emb, top_k=2, min_similarity=0.2, weak_similarity=0.1)
    cm = m.map_cluster(3, "powershell with encoded command lines executing scripts", ["ev-1", "ev-2"])
    ids = [t.technique_id for t in cm.top_k_techniques]
    assert len(ids) <= 2 and ids[0] in ("T9001", "T9001.001")
    scores = [t.similarity_score for t in cm.top_k_techniques]
    assert scores == sorted(scores, reverse=True) and all(0.0 < s <= 1.0 for s in scores)
    top = cm.top_k_techniques[0]
    assert top.supporting_event_uids == ["ev-1", "ev-2"] and top.technique_name and top.description_excerpt
    assert top.mapping_status == "CANDIDATE" and cm.overall_outcome == "CANDIDATES"
    assert "probab" not in top.as_dict().get("score_meaning", "").replace("not a probability", "")


def test_no_forced_match_and_platform_filter(mapper_parts):
    _, emb, tc = mapper_parts
    m = SemanticMitreMapper(tc, emb, top_k=5, min_similarity=0.35, weak_similarity=0.25)
    cm = m.map_cluster(1, "zzz qqq xxx unrelated gibberish tokens", ["e"])
    assert cm.overall_outcome == "NO_MATCH" and all(t.mapping_status == "NO_CONFIDENT_MATCH" for t in cm.top_k_techniques)
    assert "T9005" not in {t["technique_id"] for t in tc.techniques}          # Linux-only technique filtered out
    assert not any(t.similarity_score == 0 for t in cm.top_k_techniques)      # zero-score "candidates" are never listed


def test_thresholds_are_configurable_and_validated(mapper_parts):
    _, emb, tc = mapper_parts
    text = "powershell encoded command"
    strict = SemanticMitreMapper(tc, emb, top_k=3, min_similarity=0.99, weak_similarity=0.9).map_cluster(1, text, ["e"])
    loose = SemanticMitreMapper(tc, emb, top_k=3, min_similarity=0.05, weak_similarity=0.01).map_cluster(1, text, ["e"])
    assert strict.overall_outcome == "NO_MATCH" and loose.overall_outcome == "CANDIDATES"
    with pytest.raises(ValueError):
        SemanticMitreMapper(tc, emb, top_k=3, min_similarity=0.2, weak_similarity=0.5)
    with pytest.raises(ValueError):
        SemanticMitreMapper(tc, emb, top_k=0)


def test_map_clusters_rejects_noise_label(mapper_parts):
    _, emb, tc = mapper_parts

    class P:
        cluster_id = -1
        text = "x"
        representative_event_uids = []
    with pytest.raises(ValueError):
        SemanticMitreMapper(tc, emb).map_clusters([P()])


def test_embedding_cache_hit_and_invalidation(mitre, tmp_path):
    cache = str(tmp_path)
    emb = LexicalCorpusEmbedder(technique_embedding_text(t) for t in mitre.active())
    _, _, s1 = load_or_build(mitre, emb, cache)
    _, _, s2 = load_or_build(mitre, emb, cache)
    assert s1["cache"] == "built" and s2["cache"] == "hit"
    _, _, s3 = load_or_build(mitre, emb, cache, rebuild=True)
    assert s3["cache"] == "rebuilt"

    class Other(LexicalCorpusEmbedder):                                       # different model -> different cache key
        @property
        def model_id(self):
            return "other-model"
    o = Other(technique_embedding_text(t) for t in mitre.active())
    _, _, s4 = load_or_build(mitre, o, cache)
    assert s4["cache"] == "built" and s4["path"] != s1["path"]

    b2 = make_bundle()
    b2["objects"][3]["description"] = "Entirely different wording about command interpreters and shells."
    mitre2 = updater.build_corpus(b2, None, "local-file")
    assert mitre2.content_hash != mitre.content_hash
    _, _, s5 = load_or_build(mitre2, emb, cache)
    assert s5["cache"] in ("built", "rebuilt") and s5["path"] != s1["path"]
    assert not __import__("os").path.exists(s1["path"])                       # stale vectors for the old corpus pruned


def test_corrupt_cache_is_rebuilt_not_trusted(mitre, tmp_path):
    emb = LexicalCorpusEmbedder(technique_embedding_text(t) for t in mitre.active())
    _, _, s = load_or_build(mitre, emb, str(tmp_path))
    open(s["path"], "wb").write(b"garbage")
    _, vecs, s2 = load_or_build(mitre, emb, str(tmp_path))
    assert s2["cache"] in ("built", "rebuilt") and vecs.shape[0] == len(mitre.active())


def test_same_preprocessing_on_both_sides():
    t = {"technique_id": "T1", "technique_name": "Name", "description": "Desc (Citation: x) here", "tactics": ["Execution"]}
    assert prepare_text("  Mixed   CASE\r\ntext ") == prepare_text("mixed case text") or prepare_text("a  b") == prepare_text("a b")
    assert isinstance(technique_embedding_text(t), str) and "Citation" not in technique_embedding_text(t)


def test_display_safe_escapes_rtlo_and_controls():
    s = display_safe("a‮cod.scr\x07\nb")
    assert "‮" not in s and "\\u202e" in s and "\x07" not in s and "\n" not in s


def test_stand_in_embedder_is_flagged(small_run):
    assert small_run.mapping["is_standin_embedder"] is True
    assert any("LEXICAL" in w for w in small_run.stages[7].warnings)
    assert small_run.mapping["thresholds_calibrated_against_labels"] is False
    assert small_run.mapping["legacy_curated_corpus"] is False
    assert small_run.mapping["corpus_stats"]["active"] > 500
