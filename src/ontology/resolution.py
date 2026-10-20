"""
src/ontology/resolution.py
----------------------------
Evidence-based linking of ontology-module entities to what THIS pipeline
observed. Identity is only asserted from STRONG, explicitly declared
identifiers; similar names are never enough.

    explicit event_uid in an entity/relationship evidence item
        -> EVIDENCED_BY edge to that event (the ontology cites the event)
    declared identifier with a strong scheme that exactly matches observed
    data (file hash / process GUID / IP address / event_uid)
        -> link: events containing the identifier get EVIDENCED_BY edges;
           an IP also resolves to the observed ip_address Entity (RESOLVED_TO)
    canonical_name / alias equals an observed entity's name, but no strong
    identifier
        -> reported under `possible_name_matches` ONLY. No edge. Identity
           is NOT asserted.
    an identifier that matches more than one observed entity of the same
    scheme, or an entity whose identifiers point to conflicting objects
        -> reported under `ambiguous`. No edge.

Entities with no link stay in the graph as separate, unresolved nodes and are
counted in the report - never silently merged.
"""

import re
from typing import Any, Dict, List

from src.graphrag import schema as S
from src.graphrag.store import GraphStore

STRONG_SCHEMES = ("event_uid", "ip_address", "file_sha256", "file_sha1", "file_md5", "process_guid")
_HASH_RE = re.compile(r"(SHA256|SHA1|MD5)=([0-9A-Fa-f]+)")


def _identifier_index(events_by_uid: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, set]]:
    """scheme -> normalised value -> set(event_uid). Built from observed log attributes."""
    idx: Dict[str, Dict[str, set]] = {s: {} for s in STRONG_SCHEMES}

    def add(scheme, value, uid):
        if value in (None, ""):
            return
        idx[scheme].setdefault(str(value).strip().lower(), set()).add(uid)

    for uid, ev in events_by_uid.items():
        raw = ev.get("raw_event") or {}
        add("event_uid", uid, uid)
        for f in ("source_ip", "destination_ip"):
            add("ip_address", ev.get(f), uid)
        for f in ("ProcessGuid", "SourceProcessGUID", "TargetProcessGUID", "ParentProcessGuid"):
            add("process_guid", raw.get(f), uid)
        for algo, val in _HASH_RE.findall(str(raw.get("Hashes") or "")):
            add({"SHA256": "file_sha256", "SHA1": "file_sha1", "MD5": "file_md5"}[algo], val, uid)
    return idx


def resolve_ontology(doc: Dict[str, Any], events_by_uid: Dict[str, Dict[str, Any]],
                     store: GraphStore) -> Dict[str, Any]:
    # the ontology nodes must exist before edges can reference them (ingest is idempotent)
    ingest_ontology(store, doc)
    idx = _identifier_index(events_by_uid)
    observed_names: Dict[str, List[str]] = {}
    for uid, ev in events_by_uid.items():
        raw = ev.get("raw_event") or {}
        for etype, val in (("host", ev.get("hostname")), ("process_image", ev.get("source_image") or raw.get("Image")),
                           ("process_image", ev.get("target_image")), ("user", ev.get("user"))):
            if val:
                observed_names.setdefault(str(val).strip().lower(), [])
                ek = S.entity_key(etype, val)
                if ek not in observed_names[str(val).strip().lower()]:
                    observed_names[str(val).strip().lower()].append(ek)

    resolved, ambiguous, name_matches = [], [], []
    event_links, edges, dangling = 0, [], 0
    linked_entities = set()

    def link_event(entity_id: str, uid: str, how: str):
        nonlocal event_links
        edges.append({"src_kind": S.ONTOLOGY, "src_key": entity_id, "rel": S.EVIDENCED_BY, "dst_kind": S.EVENT,
                      "dst_key": uid, "props": {"status": S.STATUS_OBSERVED, "provenance": f"ontology-link:{how}"}})
        event_links += 1
        linked_entities.add(entity_id)

    for e in doc.get("entities", []):
        eid = e["entity_id"]
        # 1. explicit event citations in evidence
        for item in e.get("evidence") or []:
            uid = item.get("event_uid")
            if uid:
                if uid in events_by_uid:
                    link_event(eid, uid, "evidence.event_uid")
                else:
                    dangling += 1
        # 2. strong identifiers
        for ident in e.get("identifiers") or []:
            scheme, value = ident["scheme"], str(ident["value"]).strip().lower()
            if scheme not in STRONG_SCHEMES:
                continue
            hits = sorted(idx[scheme].get(value, set()))
            if not hits:
                continue
            if scheme == "ip_address":
                ek = S.entity_key("ip_address", value)
                if store.get_node(S.ENTITY, ek) is not None:
                    edges.append({"src_kind": S.ONTOLOGY, "src_key": eid, "rel": S.RESOLVED_TO, "dst_kind": S.ENTITY,
                                  "dst_key": ek, "props": {"status": S.STATUS_OBSERVED,
                                                           "provenance": "ontology-link:ip_address exact match"}})
                    resolved.append({"entity_id": eid, "scheme": scheme, "value": value, "observed": ek,
                                     "events": len(hits)})
            else:
                resolved.append({"entity_id": eid, "scheme": scheme, "value": value, "observed": "events",
                                 "events": len(hits)})
            for uid in hits[:200]:
                link_event(eid, uid, f"identifier:{scheme}")
        # 3. name-only similarity: report, never link
        names = [e.get("canonical_name", "")] + list(e.get("aliases") or [])
        for nm in names:
            keys = observed_names.get(str(nm).strip().lower())
            if keys and eid not in linked_entities:
                name_matches.append({"entity_id": eid, "name": nm, "observed_entities": keys[:3],
                                     "identity_asserted": False,
                                     "reason": "name matches only; no strong identifier supplied"})
                if len(keys) > 1:
                    ambiguous.append({"entity_id": eid, "name": nm, "candidates": keys[:5],
                                      "reason": "name matches several observed entity kinds"})
                break
    store.add_edges(edges)
    n_ent = len(doc.get("entities", []))
    return {
        "entities": n_ent, "linked_to_events": len(linked_entities), "unresolved": n_ent - len(linked_entities),
        "event_links": event_links, "resolved": resolved[:50], "possible_name_matches": name_matches[:50],
        "ambiguous": ambiguous[:50], "dangling_event_references": dangling,
        "policy": "identity asserted only from strong identifiers or explicit event citations; names alone never merge",
    }


def ingest_ontology(store: GraphStore, doc: Dict[str, Any]) -> Dict[str, int]:
    """Adds the ontology module's entities / relationships as generic OntologyEntity nodes and
    ONTOLOGY_RELATION edges. entity_type / relationship_type strings are stored verbatim."""
    producer = (doc.get("producer") or {}).get("name")
    rows = []
    for e in doc.get("entities", []):
        conf = e.get("confidence") or {}
        rows.append({
            "entity_id": e["entity_id"], "entity_type": e["entity_type"], "canonical_name": e["canonical_name"],
            "ontology_uri": e.get("ontology_uri"), "status": e["status"], "source_id": e["source_id"],
            "confidence_value": conf.get("value"), "confidence_method": conf.get("method"),
            "confidence_calibrated": bool(conf.get("calibrated", False)), "producer": producer,
            "contract_version": doc.get("contract_version"),
        })
    store.add_nodes(S.ONTOLOGY, rows)
    rel_rows = []
    for r in doc.get("relationships", []):
        conf = r.get("confidence") or {}
        rel_rows.append({
            "src_kind": S.ONTOLOGY, "src_key": r["source_entity_id"], "rel": S.ONTOLOGY_RELATION,
            "dst_kind": S.ONTOLOGY, "dst_key": r["target_entity_id"],
            "edge_key": f"{S.ONTOLOGY_RELATION}:{r['relationship_type']}",
            "props": {"relationship_type": r["relationship_type"], "status": r["status"], "source_id": r["source_id"],
                      "confidence_value": conf.get("value"), "confidence_method": conf.get("method"),
                      "provenance": f"ontology:{producer}"}})
    store.add_edges(rel_rows)
    return {"entities": len(rows), "relationships": len(rel_rows)}
