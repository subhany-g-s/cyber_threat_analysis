"""
src/ontology/contract.py
--------------------------
Versioned JSON contracts between this pipeline and the SEPARATE ontology
creation/population module (a teammate's work - it does not exist in this
repository and its schema is not assumed).

Two documents, both versioned with semver:

  subhany.ontology-entities  v1.x   ontology module  ->  this pipeline
      entities + typed relationships extracted from CTI / documents, each with
      identity, provenance and evidence. Validated here; ingested into the graph
      store as generic OntologyEntity nodes (ontology/ingest.py).

  subhany.pipeline-findings  v1.x   this pipeline    ->  ontology module
      per-event cluster/noise membership, novelty, behaviour summary, ATT&CK
      candidates and supporting events (ontology/export.py).

What the contract deliberately does NOT do:
  * it enumerates NO entity classes and NO relationship types. `entity_type`,
    `relationship_type` and `ontology_uri` are opaque strings owned by the
    ontology module; they are carried through verbatim;
  * it does not define MITRE-derived ontology classes;
  * `confidence.value` (0..1) is the producer's own extraction/linking
    confidence. Its meaning is whatever `confidence.method` says; it is
    treated as calibrated only if `confidence.calibrated` is true, and it is
    never combined with this pipeline's cosine similarity scores.
See docs/ontology_contract.md for the field-by-field description and
src/ontology/schemas/*.json for machine-readable JSON Schemas.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

ONTOLOGY_CONTRACT = "subhany.ontology-entities"
ONTOLOGY_CONTRACT_VERSION = "1.0.0"
FINDINGS_CONTRACT = "subhany.pipeline-findings"
FINDINGS_CONTRACT_VERSION = "1.0.0"

STATUSES = ("observed", "inferred")
_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


class ContractError(Exception):
    """The document cannot be read as the expected contract at all."""


@dataclass
class ValidationResult:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    entities: int = 0
    relationships: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def _nonempty_str(v: Any) -> bool:
    return isinstance(v, str) and v.strip() != ""


def _check_version(doc: Dict[str, Any], name: str, current: str, res: ValidationResult) -> None:
    if doc.get("contract") != name:
        res.errors.append(f"contract must be '{name}', got {doc.get('contract')!r}")
    v = doc.get("contract_version")
    m = _SEMVER.match(v) if isinstance(v, str) else None
    if not m:
        res.errors.append(f"contract_version must be semver 'MAJOR.MINOR.PATCH', got {v!r}")
        return
    cur = _SEMVER.match(current)
    if m.group(1) != cur.group(1):
        res.errors.append(f"unsupported contract major version {m.group(1)} (this reader supports {cur.group(1)}.x)")
    elif (int(m.group(2)), int(m.group(3))) > (int(cur.group(2)), int(cur.group(3))):
        res.warnings.append(f"document is contract_version {v}, newer than this reader's {current}; unknown fields ignored")


def _check_confidence(c: Any, where: str, res: ValidationResult) -> None:
    if c is None:
        return
    if not isinstance(c, dict):
        res.errors.append(f"{where}.confidence must be an object")
        return
    val = c.get("value")
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not (0.0 <= float(val) <= 1.0):
        res.errors.append(f"{where}.confidence.value must be a number in [0, 1], got {val!r}")
    if not _nonempty_str(c.get("method")):
        res.errors.append(f"{where}.confidence.method is required (it defines what the number means)")
    if "calibrated" in c and not isinstance(c["calibrated"], bool):
        res.errors.append(f"{where}.confidence.calibrated must be true/false")


def _check_evidence(ev: Any, where: str, res: ValidationResult) -> bool:
    """returns True when at least one usable evidence item exists"""
    if ev is None:
        return False
    if not isinstance(ev, list):
        res.errors.append(f"{where}.evidence must be a list")
        return False
    ok = False
    for i, item in enumerate(ev):
        if not isinstance(item, dict) or not any(_nonempty_str(item.get(k)) for k in ("text", "reference", "event_uid")):
            res.errors.append(f"{where}.evidence[{i}] needs at least one of text / reference / event_uid")
        else:
            ok = True
    return ok


def validate_ontology_document(doc: Any) -> ValidationResult:
    res = ValidationResult()
    if not isinstance(doc, dict):
        res.errors.append("document must be a JSON object")
        return res
    _check_version(doc, ONTOLOGY_CONTRACT, ONTOLOGY_CONTRACT_VERSION, res)
    prod = doc.get("producer")
    if not isinstance(prod, dict) or not _nonempty_str(prod.get("name")):
        res.errors.append("producer.name is required (who produced this document)")
    entities, rels = doc.get("entities"), doc.get("relationships", [])
    if not isinstance(entities, list):
        res.errors.append("entities must be a list (may be empty)")
        return res
    if not isinstance(rels, list):
        res.errors.append("relationships must be a list")
        rels = []

    ids, names = {}, {}
    for i, e in enumerate(entities):
        w = f"entities[{i}]"
        if not isinstance(e, dict):
            res.errors.append(f"{w} must be an object")
            continue
        eid = e.get("entity_id")
        if not _nonempty_str(eid):
            res.errors.append(f"{w}.entity_id is required")
        elif eid in ids:
            res.errors.append(f"{w}.entity_id '{eid}' duplicates entities[{ids[eid]}]")
        else:
            ids[eid] = i
        for req in ("entity_type", "canonical_name", "source_id"):
            if not _nonempty_str(e.get(req)):
                res.errors.append(f"{w}.{req} is required")
        if e.get("status") not in STATUSES:
            res.errors.append(f"{w}.status must be one of {STATUSES}, got {e.get('status')!r}")
        _check_confidence(e.get("confidence"), w, res)
        has_ev = _check_evidence(e.get("evidence"), w, res)
        if e.get("status") == "inferred" and not has_ev and not (e.get("confidence") or {}).get("method"):
            res.errors.append(f"{w}: an 'inferred' entity needs evidence or a confidence.method")
        if "ontology_uri" in e and e["ontology_uri"] is not None and not _nonempty_str(e["ontology_uri"]):
            res.errors.append(f"{w}.ontology_uri must be a non-empty string when present")
        if "identifiers" in e:
            idl = e["identifiers"]
            if not isinstance(idl, list) or any(not (isinstance(x, dict) and _nonempty_str(x.get("scheme"))
                                                      and _nonempty_str(x.get("value"))) for x in idl):
                res.errors.append(f"{w}.identifiers must be a list of {{scheme, value}} objects")
        if "aliases" in e and (not isinstance(e["aliases"], list) or any(not _nonempty_str(a) for a in e["aliases"])):
            res.errors.append(f"{w}.aliases must be a list of non-empty strings")
        if _nonempty_str(e.get("entity_type")) and _nonempty_str(e.get("canonical_name")):
            k = (e["entity_type"].strip().lower(), e["canonical_name"].strip().lower())
            if k in names and names[k] != eid:
                res.warnings.append(f"{w}: same type+name as entity '{names[k]}' but a different entity_id; "
                                    f"they are NOT merged (similar names alone do not establish identity)")
            names.setdefault(k, eid)
    res.entities = len(entities)

    for i, r in enumerate(rels):
        w = f"relationships[{i}]"
        if not isinstance(r, dict):
            res.errors.append(f"{w} must be an object")
            continue
        for req in ("source_entity_id", "target_entity_id", "relationship_type", "source_id"):
            if not _nonempty_str(r.get(req)):
                res.errors.append(f"{w}.{req} is required")
        for end in ("source_entity_id", "target_entity_id"):
            if _nonempty_str(r.get(end)) and r[end] not in ids:
                res.errors.append(f"{w}.{end} '{r[end]}' is not an entity_id in this document")
        if r.get("status") not in STATUSES:
            res.errors.append(f"{w}.status must be one of {STATUSES}, got {r.get('status')!r}")
        _check_confidence(r.get("confidence"), w, res)
        has_ev = _check_evidence(r.get("evidence"), w, res)
        if r.get("status") == "inferred" and not has_ev and not (r.get("confidence") or {}).get("method"):
            res.errors.append(f"{w}: an 'inferred' relationship needs evidence or a confidence.method")
    res.relationships = len(rels)
    return res


def validate_findings_document(doc: Any) -> ValidationResult:
    """Checks a document produced by ontology/export.py (also usable by the receiving side)."""
    res = ValidationResult()
    if not isinstance(doc, dict):
        res.errors.append("document must be a JSON object")
        return res
    _check_version(doc, FINDINGS_CONTRACT, FINDINGS_CONTRACT_VERSION, res)
    mp = doc.get("mapping")
    if not isinstance(mp, dict) or not _nonempty_str(mp.get("corpus_version")):
        res.errors.append("mapping.corpus_version is required")
    recs = doc.get("records")
    if not isinstance(recs, list):
        res.errors.append("records must be a list")
        return res
    seen = set()
    for i, r in enumerate(recs):
        w = f"records[{i}]"
        if not isinstance(r, dict):
            res.errors.append(f"{w} must be an object")
            continue
        for req in ("event_uid", "source_id", "unit_scope"):
            if not _nonempty_str(r.get(req)):
                res.errors.append(f"{w}.{req} is required")
        if r.get("event_uid") in seen:
            res.errors.append(f"{w}.event_uid duplicates an earlier record")
        seen.add(r.get("event_uid"))
        if r.get("unit_scope") not in ("cluster", "noise_group", "noise_event", "unreviewed_noise"):
            res.errors.append(f"{w}.unit_scope invalid: {r.get('unit_scope')!r}")
        n = r.get("novelty") or {}
        sc = n.get("score")
        if sc is not None and not (isinstance(sc, (int, float)) and 0.0 <= sc <= 1.0):
            res.errors.append(f"{w}.novelty.score must be in [0, 1]")
        for j, t in enumerate(r.get("technique_candidates") or []):
            s = t.get("similarity_score")
            if not _nonempty_str(t.get("technique_id")) or not isinstance(s, (int, float)) or not (0.0 <= s <= 1.0):
                res.errors.append(f"{w}.technique_candidates[{j}] needs technique_id and similarity_score in [0, 1]")
    res.entities = len(recs)
    return res


def load_ontology_document(path: str):
    """Returns (document, ValidationResult). Raises ContractError for unreadable files."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except OSError as e:
        raise ContractError(f"Cannot read ontology file {path}: {e}")
    except json.JSONDecodeError as e:
        raise ContractError(f"Ontology file {path} is not valid JSON: {e}")
    return doc, validate_ontology_document(doc)
