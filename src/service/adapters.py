"""
src/service/adapters.py
-------------------------
STAGE 1 input adapters. The project's native format is JSON Lines (one JSON
object per line, Winlogbeat/Mordor style) and stays the default. Two extra
formats are supported ONLY through these explicit adapters, selected by
`--input-format` or by sniffing:

    jsonl       one JSON object per line            (native, parser.py)
    json-array  a single JSON array of objects
    csv         header row + one event per row; column names must be the
                raw field names the normalizer reads (EventID, Hostname,
                UtcTime, SourceImage, ...). "EventID" is converted to int;
                other cells stay strings; empty cells are treated as missing.

Every adapter returns the same IngestResult, including per-record source
line/row numbers so each event can carry provenance
(source_id + "file:line N").

Failure policy: a missing / empty / unreadable file or one with zero usable
records raises InputError (reported cleanly by the CLI); individual
malformed records are counted and skipped, never silently repaired.
"""

import csv
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.graph.parser import parse_log_file


class InputError(Exception):
    """The input cannot be used at all (missing, empty, unreadable, no valid records)."""


@dataclass
class IngestResult:
    records: List[Dict[str, Any]]
    record_lines: List[int]
    format: str
    source_id: str
    path: str
    file_size: int
    sha256: str
    stats: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    invalid_samples: List[str] = field(default_factory=list)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sniff_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return "csv"
    with open(path, "rb") as f:
        head = f.read(4096)
    stripped = head.lstrip(b"\xef\xbb\xbf \t\r\n")
    if stripped.startswith(b"["):
        return "json-array"
    return "jsonl"


# ------------------------------------------------------------------ adapters
def _read_jsonl(path: str, max_invalid_samples: int):
    records, report = parse_log_file(path, max_invalid_samples=max_invalid_samples)
    stats = report.as_dict()
    return records, report.record_lines, stats, report.invalid_samples


def _read_json_array(path: str, max_invalid_samples: int):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise InputError(f"{path} is not valid JSON ({e.msg} at line {e.lineno}). "
                         f"If it is JSON Lines, use --input-format jsonl.")
    if not isinstance(data, list):
        raise InputError(f"{path}: expected a JSON array of event objects, found {type(data).__name__}.")
    records, lines, invalid, samples = [], [], 0, []
    for i, obj in enumerate(data, start=1):
        if isinstance(obj, dict):
            records.append(obj)
            lines.append(i)
        else:
            invalid += 1
            if len(samples) < max_invalid_samples:
                samples.append(repr(obj)[:200])
    stats = {"total_lines": len(data), "blank_lines": 0, "valid_records": len(records), "invalid_records": invalid}
    return records, lines, stats, samples


def _read_csv(path: str, max_invalid_samples: int):
    records, lines, invalid, blank, samples = [], [], 0, 0, []
    total = 0
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise InputError(f"{path}: CSV has no header row.")
        if "EventID" not in reader.fieldnames:
            raise InputError(f"{path}: CSV needs an 'EventID' column (found: {reader.fieldnames[:8]}...).")
        for row_no, row in enumerate(reader, start=2):             # header is row 1
            total += 1
            if None in row:                                         # more cells than header columns
                invalid += 1
                if len(samples) < max_invalid_samples:
                    samples.append(f"row {row_no}: {len(row.get(None) or [])} extra cell(s)")
                continue
            rec = {k: v for k, v in row.items() if k and v not in (None, "")}
            if not rec:
                blank += 1
                continue
            eid = rec.get("EventID")
            if eid is not None:
                try:
                    rec["EventID"] = int(str(eid).strip())
                except ValueError:
                    invalid += 1
                    if len(samples) < max_invalid_samples:
                        samples.append(f"row {row_no}: non-numeric EventID {eid!r}")
                    continue
            records.append(rec)
            lines.append(row_no)
    stats = {"total_lines": total, "blank_lines": blank, "valid_records": len(records), "invalid_records": invalid}
    return records, lines, stats, samples


_ADAPTERS = {"jsonl": _read_jsonl, "json-array": _read_json_array, "csv": _read_csv}


def ingest(path: str, fmt: str = "auto", max_invalid_samples: int = 5) -> IngestResult:
    if not path:
        raise InputError("No input file given.")
    if not os.path.exists(path):
        raise InputError(f"Input file not found: {path}")
    if os.path.isdir(path):
        raise InputError(f"Input path is a directory, expected a file: {path}")
    size = os.path.getsize(path)
    if size == 0:
        raise InputError(f"Input file is empty (0 bytes): {path}")
    if fmt not in ("auto",) + tuple(_ADAPTERS):
        raise InputError(f"Unknown input format '{fmt}' (choose auto, jsonl, json-array, csv).")

    used = sniff_format(path) if fmt == "auto" else fmt
    try:
        sha = _sha256(path)
        records, lines, stats, samples = _ADAPTERS[used](path, max_invalid_samples)
    except OSError as e:
        raise InputError(f"Cannot read input file {path}: {e}")

    warnings: List[str] = []
    if stats["invalid_records"]:
        warnings.append(f"{stats['invalid_records']} malformed record(s) were skipped (not repaired).")
    if not records:
        hint = ""
        if used == "jsonl" and stats["invalid_records"] and stats["invalid_records"] == stats["total_lines"] - stats["blank_lines"]:
            hint = (" Every line failed to parse as a JSON object; if the file is a JSON array use "
                    "--input-format json-array, if it is CSV use --input-format csv.")
        raise InputError(f"No valid event records found in {path}.{hint}")

    source_id = f"src:{os.path.basename(path)}:{sha[:12]}"
    return IngestResult(records=records, record_lines=lines, format=used, source_id=source_id, path=path,
                        file_size=size, sha256=sha, stats=stats, warnings=warnings, invalid_samples=samples)
