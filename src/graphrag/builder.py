"""
src/graphrag/builder.py
-------------------------
Populates a GraphStore from one pipeline run. Every node/edge carries
provenance and a status:

    observed  - taken directly from log attributes / the adjacency rules of src/graph/graph_builder.py
    computed  - produced by the GAE + HDBSCAN + novelty computation
    inferred  - semantic-similarity technique links (CANDIDATE, never confirmed)

Stable identifiers: events keep their deterministic uuid5 event_uid; entities
use "<type>:<normalised value>"; clusters/noise units use "<run_id>:<unit_id>"
where run_id is a deterministic hash of (input content, key parameters), so
re-running identical input updates the same nodes (MERGE semantics) instead of
duplicating them.
"""

import hashlib
from typing import Any, Dict, Iterable, List, Optional

from src.graphrag import schema as S
from src.graphrag.store import GraphStore


def make_run_id(source_sha256: str, params: Dict[str, Any], checkpoint_fingerprint: str = "") -> str:
    blob = repr((source_sha256, sorted(params.items()), checkpoint_fingerprint))
    return "run-" + hashlib.sha256(blob.encode()).hexdigest()[:12]


def _first_line(raw: Dict[str, Any]) -> Optional[str]:
    msg = raw.get("Message")
    if not msg:
        return None
    return str(msg).replace("\r", "\n").split("\n")[0].strip()[:120] or None


def build_knowledge_graph(store: GraphStore, run_id: str, events_by_uid: Dict[str, Dict[str, Any]],
                          provenance_by_uid: Dict[str, Dict[str, Any]], G, labels_by_uid: Dict[str, int],
                          novelty_by_uid: Dict[str, Dict[str, Any]], units: List[Dict[str, Any]],
                          mitre_corpus=None) -> Dict[str, Any]:
    """units: [{unit_id, scope, cluster_id, size, narrative, member_event_uids, representative_event_uids,
                mapping (ClusterMapping.as_dict() or None)}]  - real clusters AND reviewed noise units."""
    unit_of_uid: Dict[str, Dict[str, Any]] = {}
    for u in units:
        for uid in u["member_event_uids"]:
            unit_of_uid[uid] = u

    # ---------------------------------------------------------------- events
    ev_rows, ent_rows, ent_seen, ent_edges = [], [], set(), []
    for uid, ev in events_by_uid.items():
        raw = ev.get("raw_event") or {}
        nv = novelty_by_uid.get(uid, {})
        prov = provenance_by_uid.get(uid, {})
        label = labels_by_uid.get(uid, -1)
        ev_rows.append({
            "event_uid": uid, "event_id": ev.get("event_id"), "timestamp": ev.get("timestamp"),
            "hostname": ev.get("hostname"), "description": _first_line(raw),
            "cluster_id": int(label), "is_noise": int(label) == -1,
            "novelty_score": nv.get("novelty_score"), "novelty_label": nv.get("novelty_label"),
            "source_id": prov.get("source_id"), "source_ref": prov.get("source_ref"),
            "last_run_id": run_id,
        })
        attrs = [
            (S.ON_HOST, "host", ev.get("hostname")),
            (S.SOURCE_PROCESS, "process_image", ev.get("source_image") or raw.get("Image")),
            (S.TARGET_PROCESS, "process_image", ev.get("target_image")),
            (S.BY_USER, "user", ev.get("user")),
            (S.CONNECTED_TO, "ip_address", ev.get("destination_ip")),
        ]
        for rel, etype, val in attrs:
            if val in (None, ""):
                continue
            ek = S.entity_key(etype, val)
            if ek not in ent_seen:
                ent_seen.add(ek)
                ent_rows.append({"entity_key": ek, "entity_type": etype, "name": str(val)})
            ent_edges.append({"src_kind": S.EVENT, "src_key": uid, "rel": rel, "dst_kind": S.ENTITY, "dst_key": ek,
                              "props": {"status": S.STATUS_OBSERVED, "provenance": "pipeline:normalizer"}})
    store.add_nodes(S.EVENT, ev_rows)
    store.add_nodes(S.ENTITY, ent_rows)
    store.add_edges(ent_edges)

    # ------------------------------------------------- event <-> event edges
    ee = []
    for a, b, d in G.edges(data=True):
        if a not in events_by_uid or b not in events_by_uid:
            continue
        for reason in d.get("reasons") or [d.get("relationship")]:
            if reason in S.EVENT_EVENT_RELS:
                props = {"status": S.STATUS_OBSERVED, "provenance": "pipeline:graph_builder"}
                if reason == S.SAME_PID and d.get("pid") is not None:
                    props["pid"] = str(d["pid"])
                if reason == S.SAME_HOST_TEMPORAL and d.get("time_difference") is not None:
                    props["time_difference_s"] = d["time_difference"]
                ee.append({"src_kind": S.EVENT, "src_key": a, "rel": reason, "dst_kind": S.EVENT, "dst_key": b,
                           "props": props})
    store.add_edges(ee)

    # ------------------------------------------ clusters / noise units + links
    cl_rows, member_edges, rep_edges = [], [], []
    for u in units:
        ck = f"{run_id}:{u['unit_id']}"
        cl_rows.append({"cluster_key": ck, "unit_id": u["unit_id"], "scope": u["scope"],
                        "cluster_id": u["cluster_id"], "size": u["size"], "narrative": u["narrative"][:1500],
                        "run_id": run_id})
        rel = S.IN_CLUSTER if u["scope"] == "cluster" else S.IN_NOISE_UNIT
        prov = "pipeline:gae+hdbscan" if u["scope"] == "cluster" else "pipeline:noise-review"
        for uid in u["member_event_uids"]:
            member_edges.append({"src_kind": S.EVENT, "src_key": uid, "rel": rel, "dst_kind": S.CLUSTER,
                                 "dst_key": ck, "props": {"status": S.STATUS_COMPUTED, "provenance": prov}})
        for uid in u["representative_event_uids"]:
            if uid in events_by_uid:
                rep_edges.append({"src_kind": S.CLUSTER, "src_key": ck, "rel": S.HAS_REPRESENTATIVE,
                                  "dst_kind": S.EVENT, "dst_key": uid,
                                  "props": {"status": S.STATUS_COMPUTED, "provenance": prov}})
    store.add_nodes(S.CLUSTER, cl_rows)
    store.add_edges(member_edges)
    store.add_edges(rep_edges)

    # ------------------- technique candidates + trusted reference documents
    tc_rows, doc_rows, cand_edges, doc_edges = {}, {}, [], []
    version = (mitre_corpus.version_label if mitre_corpus is not None else "unknown")
    chash = (mitre_corpus.content_hash[:8] if mitre_corpus is not None else "unknown")
    for u in units:
        m = u.get("mapping")
        if not m:
            continue
        ck = f"{run_id}:{u['unit_id']}"
        for c in m["top_k_techniques"]:
            tid = c["technique_id"]
            if tid not in tc_rows:
                tc_rows[tid] = {"technique_id": tid, "technique_name": c["technique_name"], "tactic": c["tactic"],
                                "is_subtechnique": c.get("is_subtechnique", False), "url": c.get("url"),
                                "corpus_version": version, "note": "ATT&CK technique proposed as a CANDIDATE"}
                tech = mitre_corpus.by_id(tid) if mitre_corpus is not None else None
                passages = []
                if tech is not None and tech.get("description"):
                    passages.append(("description", tech["description"][:1500]))
                    det = _detection_passage(tech)
                    if det:
                        passages.append(("detection", det))
                else:
                    passages.append(("description", c.get("description_excerpt", "")))
                for ptype, text in passages:
                    did = f"mitre:{tid}:{chash}:{ptype}"
                    doc_rows[did] = {"doc_id": did, "technique_id": tid, "passage_type": ptype, "text": text,
                                     "source": "mitre-attack-stix", "corpus_version": version,
                                     "url": c.get("url"), "title": f"{tid} {c['technique_name']} ({ptype})"}
                    doc_edges.append({"src_kind": S.TECHNIQUE, "src_key": tid, "rel": S.DESCRIBED_BY,
                                      "dst_kind": S.REFERENCE, "dst_key": did,
                                      "props": {"status": S.STATUS_OBSERVED, "provenance": f"mitre-attack-stix:{version}"}})
            cand_edges.append({"src_kind": S.CLUSTER, "src_key": ck, "rel": S.CANDIDATE_TECHNIQUE,
                               "dst_kind": S.TECHNIQUE, "dst_key": tid,
                               "props": {"similarity_score": c["similarity_score"], "status": c["mapping_status"],
                                         "rank": c.get("rank", 0), "mapping_method": c.get("mapping_method", "semantic-cosine"),
                                         "corpus_version": version, "inferred": True,
                                         "edge_status": S.STATUS_INFERRED, "provenance": "pipeline:semantic-mapper"}})
    store.add_nodes(S.TECHNIQUE, tc_rows.values())
    store.add_nodes(S.REFERENCE, doc_rows.values())
    store.add_edges(doc_edges)
    store.add_edges(cand_edges)
    return store.stats()


def _detection_passage(tech: Dict[str, Any]) -> str:
    det = tech.get("detection") or {}
    bits = []
    for s in (det.get("strategies") or [])[:2]:
        bits.append(f"Detection strategy: {s.get('name')}.")
        for a in s.get("analytics") or []:
            if "windows" in [p.lower() for p in a.get("platforms") or []] and a.get("description"):
                bits.append(a["description"][:350])
                break
    if det.get("legacy_text"):
        bits.append(det["legacy_text"][:400])
    return " ".join(bits)[:900]
