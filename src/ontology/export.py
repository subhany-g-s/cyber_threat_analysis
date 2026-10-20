"""
src/ontology/export.py
------------------------
Structured export of THIS pipeline's results for the ontology module
(contract 'subhany.pipeline-findings'). One record per event, carrying the
identifiers needed to join with the ontology module's own records
(`event_uid`, `source_id`, `source_ref`) plus cluster/noise membership,
novelty, the behaviour summary and the ATT&CK candidates with their
similarity scores and the mapping/corpus version that produced them.

No ontology classes are invented: the export contains only this pipeline's
own facts. The ontology module maps these records to its actual schema.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List

from src.mapping.semantic_mapper import SIMILARITY_DISCLAIMER
from src.mapping.text_prep import PREP_VERSION
from src.ontology.contract import FINDINGS_CONTRACT, FINDINGS_CONTRACT_VERSION, validate_findings_document


def export_findings(result) -> Dict[str, Any]:
    """result: service.result.RunResult (needs artifacts for per-event rows)."""
    art = result.artifacts
    if art is None or not art.events_by_uid:
        raise ValueError("RunResult carries no in-memory artifacts; run the pipeline in this process to export.")
    labels = {uid: int(art.labels[i]) for i, uid in enumerate(art.row_to_uid)}
    nov = {r["event_uid"]: r for r in art.novelty_rows}

    unit_by_uid: Dict[str, Any] = {}
    for u in result.units:
        for uid in (u.profile or {}).get("member_event_uids", []):
            unit_by_uid[uid] = u

    mp = result.mapping or {}
    records: List[Dict[str, Any]] = []
    for uid in art.row_to_uid:
        prov = art.provenance_by_uid.get(uid, {})
        n = nov.get(uid, {})
        u = unit_by_uid.get(uid)
        rec: Dict[str, Any] = {
            "event_uid": uid,
            "source_id": prov.get("source_id") or "unknown",
            "source_ref": prov.get("source_ref"),
            "cluster_id": labels.get(uid, -1),
            "unit_id": u.unit_id if u else None,
            "unit_scope": u.scope if u else "unreviewed_noise",
            "novelty": {
                "score": n.get("novelty_score"), "label": n.get("novelty_label"),
                "signals": {"density": n.get("density_score"), "embedding_distance": n.get("embedding_distance_score"),
                            "reconstruction_error": n.get("reconstruction_error"),
                            "graph_rarity": n.get("graph_rarity_score")},
                "meaning": "statistical/structural unusualness within this dataset; not maliciousness",
            },
            "behavior_summary": u.narrative if u else None,
            "technique_candidates": [],
            "supporting_event_uids": [],
        }
        if u and u.mapping:
            rec["supporting_event_uids"] = list(u.mapping.get("representative_event_uids", []))
            rec["technique_candidates"] = [{
                "technique_id": t["technique_id"], "technique_name": t["technique_name"],
                "similarity_score": t["similarity_score"], "status": t["mapping_status"], "rank": t.get("rank"),
                "scope": u.scope, "mapping_method": t.get("mapping_method", "semantic-cosine"),
                "reference_url": t.get("url"),
            } for t in u.mapping.get("top_k_techniques", [])]
        records.append(rec)

    doc = {
        "contract": FINDINGS_CONTRACT,
        "contract_version": FINDINGS_CONTRACT_VERSION,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "run": {"run_id": result.run_id, "source_id": result.metadata.get("source_id"),
                "input_path": result.metadata.get("input_path"), "event_count": len(records)},
        "mapping": {
            "corpus_version": mp.get("corpus_version"), "corpus_hash": mp.get("corpus_hash"),
            "attack_version": mp.get("attack_version"), "embedding_model": mp.get("model_id"),
            "text_preprocessing": PREP_VERSION, "platform_filter": mp.get("platform_filter"),
            "thresholds": {"candidate": mp.get("candidate_threshold"), "weak": mp.get("weak_threshold")},
            "similarity_semantics": SIMILARITY_DISCLAIMER,
            "is_standin_embedder": mp.get("is_standin_embedder"),
        },
        "records": records,
    }
    check = validate_findings_document(doc)
    if not check.ok:
        raise ValueError("Internal error: exported findings violate their own contract: " + "; ".join(check.errors[:3]))
    return doc
