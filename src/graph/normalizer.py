"""
normalizer.py
-------------
STEP 1b: Normalization into a common event schema.

Responsibility:
    Take the list of raw, valid JSON records produced by parser.py and turn
    each one into a NormalizedEvent -- a flat dictionary with a fixed set of
    fields, regardless of the source EventID.

Design principles (per project brief):
    - Never invent data. If a field does not exist in the raw record, the
      normalized field is set to None.
    - Preserve the original record under `raw_event` so nothing is lost.
    - Every valid event gets a unique, DETERMINISTIC `event_uid` -- the same
      raw record always normalizes to the same event_uid, on any run, on any
      machine (see `_deterministic_event_uid` below). This matters beyond
      Step 1/2: the GAE stage's `gae_node_mapping.json` and Neo4j's
      `MERGE (e:Event {event_uid: ...})` both depend on event_uid being
      stable across independent runs of the same input file -- a random
      uuid4() would silently break both (re-running the pipeline would
      create duplicate Neo4j nodes instead of updating existing ones, and
      embeddings saved on one run couldn't be traced back to nodes from a
      later re-parse).
    - Timestamps are normalized to UTC ISO-8601 where a parseable timestamp
      field exists.
    - EventID 10 (Sysmon ProcessAccess) and EventID 5156 (WFP connection)
      get their full, event-specific fields populated because those are the
      two event types analyzed in this project's first review. All other
      EventIDs are still normalized into the SAME schema (so they can be
      correlated by host/time in Step 2), but only the fields that
      generically exist across Windows event logs are populated:
      event_id, timestamp, hostname, user, record_number, severity,
      event_type, and process_id (when the raw record has an unambiguous
      ProcessId-style field). This avoids inventing process-access or
      network semantics for event types this phase does not analyze.
"""

import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

EVENTID_PROCESS_ACCESS = 10
EVENTID_WFP_CONNECTION = 5156


@dataclass
class NormalizedEvent:
    event_uid: str
    event_id: Optional[int]
    timestamp: Optional[str]        # ISO-8601 UTC string, or None
    timestamp_epoch: Optional[float]  # seconds since epoch, for fast math in graph_builder
    hostname: Optional[str]

    # Process-access fields (populated for EventID 10; process_id populated
    # generically for other event types when present)
    source_process_guid: Optional[str] = None
    source_process_id: Optional[str] = None
    source_image: Optional[str] = None
    target_process_guid: Optional[str] = None
    target_process_id: Optional[str] = None
    target_image: Optional[str] = None
    process_id: Optional[str] = None   # generic PID used for SAME_PID correlation

    user: Optional[str] = None

    # Network fields (populated for EventID 5156)
    source_ip: Optional[str] = None
    source_port: Optional[str] = None
    destination_ip: Optional[str] = None
    destination_port: Optional[str] = None
    protocol: Optional[str] = None
    direction: Optional[str] = None

    record_number: Optional[str] = None
    severity: Optional[str] = None
    event_type: Optional[str] = None

    raw_event: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Fixed namespace so uuid5 output is stable across machines/runs (any
# constant UUID works here -- what matters is that it never changes).
_EVENT_UID_NAMESPACE = uuid.UUID("d3f6a2b0-6b0e-4e1a-9f7a-1a2b3c4d5e6f")


def _deterministic_event_uid(record: Dict[str, Any]) -> str:
    """
    Deterministic event_uid derived from the raw record's own content, so
    the same raw log line always normalizes to the same event_uid -- no
    randomness, no dependence on line order or run number.

    We hash the full raw record (sorted keys, so key order in the source
    JSON doesn't matter) rather than a handpicked subset of fields -- this
    guarantees uniqueness follows directly from "the raw records are
    different" without us having to decide, per EventID, which fields are
    "identifying enough." In the rare case two raw lines are byte-for-byte
    identical (e.g. a genuine duplicate log entry), they deterministically
    collapse to the same event_uid on every run -- which is the correct,
    reproducible behavior (not a bug), and is a legitimate signal worth
    surfacing via the parser/normalizer stats if it ever happens at scale.
    """
    canonical = json.dumps(record, sort_keys=True, default=str)
    return str(uuid.uuid5(_EVENT_UID_NAMESPACE, canonical))


def _first_present(record: Dict[str, Any], keys) -> Optional[Any]:
    """Return the value of the first key in `keys` that exists AND is not None."""
    for k in keys:
        if k in record and record[k] is not None:
            return record[k]
    return None


def _parse_timestamp(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    """
    Try known timestamp fields, in order of trustworthiness for this dataset:
        UtcTime (Sysmon, e.g. "2020-05-02 02:55:23.551")
        @timestamp (ISO-8601 with Z, e.g. "2020-05-02T02:55:26.493Z")
        EventTime (local collector time, e.g. "2020-05-01 22:55:23")

    Returns (iso_string, epoch_seconds) or (None, None) if nothing parseable
    was found. We never fabricate a timestamp.
    """
    candidates = [
        ("UtcTime", "%Y-%m-%d %H:%M:%S.%f"),
        ("@timestamp", None),  # ISO8601, handled specially below
        ("EventTime", "%Y-%m-%d %H:%M:%S"),
    ]

    for field_name, fmt in candidates:
        val = record.get(field_name)
        if not val:
            continue
        try:
            if field_name == "@timestamp":
                # e.g. 2020-05-02T02:55:26.493Z
                v = val.replace("Z", "+00:00")
                dt = datetime.fromisoformat(v)
            else:
                dt = datetime.strptime(val, fmt).replace(tzinfo=timezone.utc)
            return dt.isoformat(), dt.timestamp()
        except (ValueError, TypeError):
            continue

    return None, None


def normalize_record(record: Dict[str, Any]) -> NormalizedEvent:
    """Convert a single raw record into a NormalizedEvent. Never raises on
    missing fields -- missing means None."""

    event_id = record.get("EventID")
    ts_iso, ts_epoch = _parse_timestamp(record)
    hostname = _first_present(record, ["Hostname", "host"])
    user = _first_present(record, ["AccountName", "UserID"])
    record_number = record.get("RecordNumber")
    severity = record.get("Severity")
    event_type = record.get("EventType")

    ev = NormalizedEvent(
        event_uid=_deterministic_event_uid(record),
        event_id=event_id,
        timestamp=ts_iso,
        timestamp_epoch=ts_epoch,
        hostname=hostname,
        user=user,
        record_number=str(record_number) if record_number is not None else None,
        severity=severity,
        event_type=event_type,
        raw_event=record,
    )

    if event_id == EVENTID_PROCESS_ACCESS:
        ev.source_process_guid = record.get("SourceProcessGUID")
        ev.source_process_id = record.get("SourceProcessId")
        ev.source_image = record.get("SourceImage")
        ev.target_process_guid = record.get("TargetProcessGUID")
        ev.target_process_id = record.get("TargetProcessId")
        ev.target_image = record.get("TargetImage")
        # For SAME_PID correlation, the "owning" process of a ProcessAccess
        # event is the SOURCE process (the one that reached out and touched
        # another process's memory).
        ev.process_id = record.get("SourceProcessId")

    elif event_id == EVENTID_WFP_CONNECTION:
        ev.source_ip = record.get("SourceAddress")
        ev.source_port = record.get("SourcePort")
        ev.destination_ip = record.get("DestAddress")
        ev.destination_port = record.get("DestPort")
        ev.protocol = record.get("Protocol")
        ev.direction = record.get("Direction")
        ev.process_id = record.get("ProcessId")

    else:
        # Generic fallback: many other Windows event types (e.g. 4688, 4663,
        # 4690) carry a ProcessId-shaped field. We only use it if present
        # under an unambiguous name -- we do not guess.
        ev.process_id = _first_present(record, ["ProcessId", "NewProcessId"])

    return ev


def normalize_records(records: List[Dict[str, Any]]) -> Tuple[List[NormalizedEvent], Dict[str, int]]:
    """
    Normalize a list of raw records and return (normalized_events, stats).

    stats includes the counts required by the project brief:
        eventid_10_count, eventid_5156_count, other_eventid_count,
        normalized_count, skipped_count
    """
    normalized: List[NormalizedEvent] = []
    stats = {
        "eventid_10_count": 0,
        "eventid_5156_count": 0,
        "other_eventid_count": 0,
        "normalized_count": 0,
        "skipped_count": 0,
    }

    for record in records:
        # A record with no EventID at all cannot be meaningfully classified.
        # We still normalize it (event_id=None) rather than silently
        # dropping data, but we count it under "skipped" only if it fails
        # to even become a dict -- in practice normalize_record never raises,
        # so skipped_count stays 0 unless we extend validation later.
        try:
            ev = normalize_record(record)
        except Exception:
            stats["skipped_count"] += 1
            continue

        if ev.event_id == EVENTID_PROCESS_ACCESS:
            stats["eventid_10_count"] += 1
        elif ev.event_id == EVENTID_WFP_CONNECTION:
            stats["eventid_5156_count"] += 1
        else:
            stats["other_eventid_count"] += 1

        normalized.append(ev)
        stats["normalized_count"] += 1

    return normalized, stats


if __name__ == "__main__":
    from parser import parse_log_file

    records, report = parse_log_file("data/apt29_evals_day1_manual_2020-05-01225525.json")
    normalized, stats = normalize_records(records)
    print("Parse report:", report.as_dict())
    print("Normalize stats:", stats)
    print("Sample normalized event:", normalized[0].as_dict() if normalized else None)
