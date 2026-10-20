"""
parser.py
---------
STEP 1a: Raw log ingestion.

Responsibility:
    Read the raw log file and yield individual JSON records, one per line
    (this dataset is JSON-Lines / NDJSON, NOT a single JSON array -- each
    line is one independently-parseable JSON object, e.g. Winlogbeat/ELK
    style export).

    This module does NOT interpret event semantics. It only:
        - opens the file safely
        - splits it into candidate records
        - attempts json.loads() on each candidate
        - separates valid vs malformed records
        - returns counters describing what happened

    Normalization (turning a raw record into the common schema) is the
    job of normalizer.py, not this file.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ParseReport:
    """Summary statistics for a parsing run."""
    total_lines: int = 0
    blank_lines: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    invalid_samples: List[str] = field(default_factory=list)  # first few bad lines, for debugging
    record_lines: List[int] = field(default_factory=list)     # 1-based source line of each returned record (provenance)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_lines": self.total_lines,
            "blank_lines": self.blank_lines,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
        }


def parse_log_file(path: str, max_invalid_samples: int = 5):
    """
    Read `path` as JSON-Lines and return (records, report).

    Design choices:
      - We parse line-by-line rather than json.load(whole_file) because the
        source export is NDJSON. A single json.load() would fail on the
        very first line ending, which is why the raw file "is not a clean
        JSON array" as observed in the project brief.
      - A malformed line does NOT abort the run. It is counted and (up to
        `max_invalid_samples`) stored for debugging, then skipped.
      - We do not attempt to "fix" malformed JSON (e.g. string surgery).
        Guessing at corrupted data would invent information that was not
        actually in the log.

    Returns:
        records: List[dict]  -- successfully parsed raw JSON objects
        report:  ParseReport
    """
    records: List[Dict[str, Any]] = []
    report = ParseReport()

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            report.total_lines += 1
            line = raw_line.strip()

            if not line:
                report.blank_lines += 1
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                report.invalid_records += 1
                if len(report.invalid_samples) < max_invalid_samples:
                    report.invalid_samples.append(line[:200])
                continue

            if not isinstance(obj, dict):
                # A valid JSON value that isn't an object (e.g. a bare number
                # or string) is not a usable log record.
                report.invalid_records += 1
                continue

            report.valid_records += 1
            records.append(obj)
            report.record_lines.append(report.total_lines)

    return records, report


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/apt29_evals_day1_manual_2020-05-01225525.json"
    recs, rep = parse_log_file(path)
    print("Parse report:", rep.as_dict())
    print("Sample record keys:", sorted(recs[0].keys()) if recs else "NONE")
