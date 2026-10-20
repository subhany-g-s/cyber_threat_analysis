import csv
import json

import pytest

from src.service.adapters import InputError, ingest, sniff_format

REC = {"EventID": 1, "Hostname": "h1", "EventTime": "2020-05-01 22:55:23", "Message": "Process Create:\r\nProcessId: 4"}


def test_jsonl_default_and_provenance(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text("\n".join(json.dumps(dict(REC, EventID=i)) for i in (1, 2, 3)) + "\n")
    r = ingest(str(p))
    assert r.format == "jsonl" and len(r.records) == 3 and r.record_lines == [1, 2, 3]
    assert r.source_id.startswith("src:a.jsonl:") and len(r.sha256) == 64


def test_jsonl_malformed_lines_counted_not_repaired(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text(json.dumps(REC) + "\n{broken\n\n" + json.dumps(REC) + "\n")
    r = ingest(str(p))
    assert len(r.records) == 2 and r.stats["invalid_records"] == 1 and r.stats["blank_lines"] == 1
    assert r.warnings and r.record_lines == [1, 4]


def test_json_array_adapter(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps([REC, REC, "junk", 7]))
    assert sniff_format(str(p)) == "json-array"
    r = ingest(str(p))
    assert r.format == "json-array" and len(r.records) == 2 and r.stats["invalid_records"] == 2


def test_csv_adapter(tmp_path):
    p = tmp_path / "a.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EventID", "Hostname", "Message"])
        w.writerow(["10", "h1", "x"])
        w.writerow(["abc", "h1", "bad id"])
        w.writerow(["11", "h2", "y"])
    r = ingest(str(p))
    assert r.format == "csv" and [x["EventID"] for x in r.records] == [10, 11]
    assert r.stats["invalid_records"] == 1 and r.record_lines == [2, 4]


def test_csv_without_eventid_column_is_refused(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(InputError, match="EventID"):
        ingest(str(p))


@pytest.mark.parametrize("make,match", [
    (lambda d: str(d / "missing.jsonl"), "not found"),
    (lambda d: str(d), "directory"),
])
def test_unusable_paths(tmp_path, make, match):
    with pytest.raises(InputError, match=match):
        ingest(make(tmp_path))


def test_empty_and_all_invalid_inputs(tmp_path):
    e = tmp_path / "empty.jsonl"
    e.write_text("")
    with pytest.raises(InputError, match="empty"):
        ingest(str(e))
    b = tmp_path / "bad.jsonl"
    b.write_text("nope\nstill nope\n")
    with pytest.raises(InputError, match="No valid"):
        ingest(str(b))


def test_unknown_format_name(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text(json.dumps(REC))
    with pytest.raises(InputError, match="Unknown input format"):
        ingest(str(p), "xml")


def test_normalizer_stable_ids_and_dedup(tmp_path):
    from src.graph.normalizer import normalize_records
    a = dict(REC)
    ev1, _ = normalize_records([a, dict(a), dict(a, EventID=2)])
    ev2, _ = normalize_records([dict(a), dict(a, EventID=2)])
    assert len({e.event_uid for e in ev1}) == 2            # identical raw lines map to ONE uid (pipeline dedupes before graph build)
    assert {e.event_uid for e in ev1} == {e.event_uid for e in ev2}   # deterministic across runs
