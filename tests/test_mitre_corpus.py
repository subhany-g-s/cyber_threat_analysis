import json
import os

import pytest

from src.mitre import updater
from src.mitre.corpus import CorpusError, MitreCorpus, compute_content_hash, validate_techniques
from src.mitre.stix_parser import clean_description, parse_bundle
from tests.conftest import make_bundle


def test_parse_bundle_extracts_fields_and_status(bundle):
    p = parse_bundle(bundle)
    by = {t["technique_id"]: t for t in p.techniques}
    assert p.collection["attack_version"] == "99.9"
    assert set(by) >= {"T9001", "T9001.001", "T9002", "T9003", "T9004", "T9005"}
    assert by["T9001.001"]["parent_id"] == "T9001"
    assert by["T9001"]["tactics"] == ["Execution"]
    assert by["T9001"]["platforms"] == ["Windows"]
    assert by["T9003"]["status"] == "revoked" and by["T9003"]["revoked_by"] == "T9001"
    assert by["T9004"]["status"] == "deprecated"
    assert by["T9001"]["status"] == "active"
    assert "Monitor process creation" in json.dumps(by["T9001"]["detection"])


def test_revoked_and_deprecated_are_status_only_not_deleted_and_not_matchable(bundle):
    c = updater.build_corpus(bundle, None, "local-file")
    assert c.by_id("T9003") is not None                       # kept as metadata
    active_ids = {t["technique_id"] for t in c.active()}
    assert "T9003" not in active_ids and "T9004" not in active_ids
    assert {t["technique_id"] for t in c.active(["Windows"])} == {"T9001", "T9001.001", "T9002"}
    assert c.stats()["revoked"] == 1 and c.stats()["deprecated"] == 1


def test_not_a_bundle_rejected():
    with pytest.raises(ValueError):
        parse_bundle({"type": "nope"})
    with pytest.raises(updater.CorpusUpdateError):
        updater.build_corpus([1, 2], None, "local-file")


def test_validation_flags_duplicates_missing_ids_malformed_and_empty():
    good = {"technique_id": "T1234", "technique_name": "Name", "description": "A sufficiently long description of the technique.", "status": "active", "tactics": ["Execution"], "platforms": ["Windows"]}
    assert validate_techniques([good]).ok
    assert not validate_techniques([]).ok                               # empty corpus
    assert not validate_techniques([good, dict(good)]).ok               # duplicate
    assert not validate_techniques([dict(good, technique_id="")]).ok    # missing id
    assert not validate_techniques([dict(good, technique_name="")]).ok            # malformed
    assert not validate_techniques(["not a dict"]).ok


def test_update_from_file_is_versioned_idempotent_and_sets_current(tmp_path, bundle):
    src = tmp_path / "b.json"
    src.write_text(json.dumps(bundle))
    cache = str(tmp_path / "cache")
    o1 = updater.update_corpus(cache, from_file=str(src))
    assert o1.changed and os.path.exists(o1.path)
    assert updater.load_current(cache).version_label == o1.corpus.version_label
    assert o1.corpus.version_label.startswith("enterprise-attack-v99.9-")
    o2 = updater.update_corpus(cache, from_file=str(src))
    assert not o2.changed and o2.corpus.content_hash == o1.corpus.content_hash
    # a changed bundle becomes a NEW version, the old one stays cached
    b2 = make_bundle()
    b2["objects"][3]["description"] = "Completely different words about interpreters."
    src2 = tmp_path / "b2.json"
    src2.write_text(json.dumps(b2))
    o3 = updater.update_corpus(cache, from_file=str(src2))
    assert o3.changed and o3.corpus.content_hash != o1.corpus.content_hash
    assert len(updater.list_cached(cache)) == 2


def test_invalid_update_does_not_touch_cache(tmp_path):
    cache = str(tmp_path / "cache")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(updater.CorpusUpdateError):
        updater.update_corpus(cache, from_file=str(bad))
    assert updater.read_current_pointer(cache) is None
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"type": "bundle", "objects": []}))
    with pytest.raises(updater.CorpusUpdateError):
        updater.update_corpus(cache, from_file=str(empty))
    assert updater.read_current_pointer(cache) is None


def test_no_silent_fallback_when_no_corpus_cached(tmp_path):
    with pytest.raises(CorpusError):
        updater.load_corpus_for_run(str(tmp_path / "nothing"))


def test_analysis_never_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(updater, "download_bundle", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network used")))
    with pytest.raises(CorpusError):
        updater.load_corpus_for_run(str(tmp_path / "nothing"))


def test_tampered_corpus_detected(tmp_path, bundle):
    c = updater.build_corpus(bundle, None, "local-file")
    p = tmp_path / "c.json"
    c.save(str(p))
    d = json.loads(p.read_text())
    d["techniques"][0]["description"] = "tampered"
    p.write_text(json.dumps(d))
    with pytest.raises(CorpusError):
        MitreCorpus.load(str(p))
    with pytest.raises(CorpusError):
        updater.load_corpus_for_run(str(tmp_path), str(p))     # not treated as a legacy file either


def test_legacy_corpus_only_when_explicit_and_flagged(tmp_path):
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"techniques": [{"technique_id": "T1059.001", "technique_name": "PowerShell",
                                                  "description": "Adversaries abuse powershell.", "tactic": "Execution"}]}))
    c = updater.load_corpus_for_run(str(tmp_path / "none"), str(legacy))
    assert c.meta.get("legacy_curated") is True


def test_shipped_real_corpus_is_full_attack(mitre_cache):
    c = updater.load_current(mitre_cache)
    assert c.stats()["active"] > 500 and c.stats()["revoked"] > 0
    assert c.meta["attack_version"] and c.meta["source_url"] and c.meta["content_hash"] == compute_content_hash(c.techniques)


def test_clean_description_strips_markup():
    assert "(Citation" not in clean_description("Text (Citation: Foo 2020) more <code>x</code>.")
