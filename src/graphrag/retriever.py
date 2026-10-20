"""
src/graphrag/retriever.py
---------------------------
GRAPH-AWARE retrieval for one analysis unit (a cluster, or a reviewed noise
group / event). It combines

  1. SEMANTIC retrieval  - (a) narrative-vs-narrative similarity to the other
                           clusters / noise units of the run, (b) similarity of the
                           narrative to ATT&CK reference passages beyond the
                           mapper's own Top-K ("near-miss" references);
  2. GRAPH TRAVERSAL     - from the unit's representative + highest-novelty events,
                           breadth-first over SAME_PID / SAME_HOST_TEMPORAL
                           relationships, then through shared entities (processes,
                           users, IPs - hubs are not expanded), collecting
                           events, relationship paths, the clusters/noise units
                           those events belong to, and the technique candidates
                           attached to those neighbouring units;
  3. REFERENCES          - ATT&CK passages for the candidates, for their parent
                           techniques, and for techniques reached through the graph;
  4. ONTOLOGY FACTS      - when an ontology file was supplied and it explicitly
                           cites events in the traversed neighbourhood.

Everything returned keeps its provenance. `graph_contribution` states honestly
how much of the evidence exists ONLY because of graph traversal - when that is
zero, the CLI says so instead of calling the retrieval GraphRAG.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.graphrag import schema as S
from src.graphrag.store import GraphStore


@dataclass
class RetrievalSettings:
    depth: int = 2
    max_events: int = 40
    hub_limit: int = 60
    related_units_k: int = 3
    semantic_reference_k: int = 3
    max_paths: int = 12
    entity_bridge_per_entity: int = 4
    max_entity_bridged: int = 15


@dataclass
class RetrievedContext:
    unit_id: str
    scope: str
    retrieval_mode: str                         # local-networkx | neo4j
    seed_events: List[Dict[str, Any]] = field(default_factory=list)
    traversal_events: List[Dict[str, Any]] = field(default_factory=list)
    entity_bridged_events: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    hub_entities: List[Dict[str, Any]] = field(default_factory=list)
    paths: List[Dict[str, Any]] = field(default_factory=list)
    related_units: List[Dict[str, Any]] = field(default_factory=list)
    references: List[Dict[str, Any]] = field(default_factory=list)
    ontology_facts: List[Dict[str, Any]] = field(default_factory=list)
    graph_contribution: Dict[str, Any] = field(default_factory=dict)
    truncated: bool = False
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _ev_brief(store: GraphStore, uid: str) -> Dict[str, Any]:
    n = store.get_node(S.EVENT, uid) or {}
    return {"event_uid": uid, "event_id": n.get("event_id"), "timestamp": n.get("timestamp"),
            "hostname": n.get("hostname"), "description": n.get("description"),
            "cluster_id": n.get("cluster_id"), "is_noise": n.get("is_noise"),
            "novelty_score": n.get("novelty_score"), "novelty_label": n.get("novelty_label"),
            "source_ref": n.get("source_ref")}


class GraphRetriever:
    def __init__(self, store: GraphStore, settings: RetrievalSettings, run_id: str,
                 technique_corpus=None, mitre_corpus=None,
                 unit_vectors: Optional[Dict[str, np.ndarray]] = None,
                 unit_meta: Optional[Dict[str, Dict[str, Any]]] = None):
        self.store = store
        self.s = settings
        self.run_id = run_id
        self.tcorpus = technique_corpus
        self.mitre = mitre_corpus
        self.unit_vectors = unit_vectors or {}
        self.unit_meta = unit_meta or {}

    # ------------------------------------------------------------------ helpers
    def _path_to_seed(self, uid: str, reached: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
        chain = []
        cur = uid
        while reached[cur]["parent"] is not None:
            chain.append({"from": reached[cur]["parent"], "rel": reached[cur]["rel"], "to": cur})
            cur = reached[cur]["parent"]
        chain.reverse()
        return chain

    def _reference(self, tid: str, via: str, similarity: Optional[float] = None) -> List[Dict[str, Any]]:
        docs = self.store.technique_references(tid)
        out = []
        if docs:
            for d in docs:
                out.append({"ref_id": f"REF:{d['doc_id']}", "technique_id": tid, "title": d.get("title"),
                            "passage_type": d.get("passage_type"), "text": d.get("text", "")[:700],
                            "url": d.get("url"), "source": d.get("source"), "corpus_version": d.get("corpus_version"),
                            "retrieved_via": via, "similarity": similarity})
            return out
        # not in the graph store (parent / near-miss technique): read the cached corpus passage directly
        if self.mitre is not None:
            t = self.mitre.by_id(tid)
            if t and t.get("description"):
                out.append({"ref_id": f"REF:mitre:{tid}:{self.mitre.content_hash[:8]}:description",
                            "technique_id": tid, "title": f"{tid} {t.get('full_name')} (description)",
                            "passage_type": "description", "text": t["description"][:700], "url": t.get("url"),
                            "source": "mitre-attack-stix", "corpus_version": self.mitre.version_label,
                            "retrieved_via": via, "similarity": similarity})
        return out

    # --------------------------------------------------------------------- main
    def retrieve(self, unit: Dict[str, Any]) -> RetrievedContext:
        s, store = self.s, self.store
        unit_id, members = unit["unit_id"], set(unit["member_event_uids"])
        ctx = RetrievedContext(unit_id=unit_id, scope=unit["scope"], retrieval_mode=store.backend)

        seeds = [u for u in unit["seed_event_uids"] if store.get_node(S.EVENT, u) is not None]
        if not seeds:
            ctx.warnings.append("No seed events found in the graph store; graph traversal skipped.")
        ctx.seed_events = [dict(_ev_brief(store, u), role="seed") for u in seeds]

        # ---- 1. event <-> event traversal --------------------------------------
        exp = store.expand_events(seeds, s.depth, s.max_events) if seeds else {"events": {}, "truncated": False}
        reached, ctx.truncated = exp["events"], exp["truncated"]
        unit_cache: Dict[str, Optional[Dict[str, Any]]] = {}

        def unit_of(uid):
            if uid not in unit_cache:
                unit_cache[uid] = store.unit_of_event(uid)
            return unit_cache[uid]

        for uid, info in sorted(reached.items(), key=lambda kv: (kv[1]["hop"], kv[0])):
            if info["hop"] == 0:
                continue
            u = unit_of(uid)
            rec = _ev_brief(store, uid)
            rec.update({"hop": info["hop"], "via_event": info["parent"], "via_rel": info["rel"],
                        "in_unit": uid in members, "unit_id": u["unit_id"] if u else None,
                        "unit_scope": u["scope"] if u else ("noise_unreviewed" if rec.get("is_noise") else None)})
            ctx.traversal_events.append(rec)

        # ---- paths: cross-unit endpoints first, then novelty ----
        def path_priority(e):
            return (0 if not e["in_unit"] else 1, e["hop"], -(e.get("novelty_score") or 0.0), e["event_uid"])

        for i, e in enumerate(sorted(ctx.traversal_events, key=path_priority)[: s.max_paths], start=1):
            chain = self._path_to_seed(e["event_uid"], reached)
            ctx.paths.append({
                "path_id": f"GP:{unit_id}:{i}", "start_event": chain[0]["from"] if chain else None,
                "end_event": e["event_uid"], "hops": e["hop"], "edges": chain,
                "end_unit": e["unit_id"] or e["unit_scope"], "provenance": "event graph (SAME_PID / SAME_HOST_TEMPORAL)",
                "status": S.STATUS_OBSERVED})

        # ---- 2. entity bridging -------------------------------------------------
        ent_agg: Dict[str, Dict[str, Any]] = {}
        for uid in seeds:
            for e in store.event_entities(uid):
                a = ent_agg.setdefault(e["entity_key"], dict(e, roles=set(), seed_events=[]))
                a["roles"].add(e["role"])
                a["seed_events"].append(uid)
        bridged: List[Dict[str, Any]] = []
        known = set(reached)
        for ek, a in sorted(ent_agg.items(), key=lambda kv: (kv[1]["event_count"], kv[0])):
            rec = {"entity_key": ek, "entity_type": a["entity_type"], "name": a["name"],
                   "roles": sorted(a["roles"]), "event_count": a["event_count"],
                   "seed_events": a["seed_events"][:5]}
            if a["event_count"] > s.hub_limit:
                ctx.hub_entities.append(rec)                  # shown, deliberately not expanded
                continue
            ctx.entities.append(rec)
            if len(bridged) >= s.max_entity_bridged:
                continue
            for uid in store.entity_events(ek, s.entity_bridge_per_entity + len(seeds)):
                if uid in known or len(bridged) >= s.max_entity_bridged:
                    continue
                known.add(uid)
                u = unit_of(uid)
                b = _ev_brief(store, uid)
                b.update({"via_entity": ek, "in_unit": uid in members, "unit_id": u["unit_id"] if u else None,
                          "unit_scope": u["scope"] if u else ("noise_unreviewed" if b.get("is_noise") else None)})
                bridged.append(b)
        ctx.entity_bridged_events = bridged

        # ---- 3. related units (graph + semantic) --------------------------------
        related: Dict[str, Dict[str, Any]] = {}
        for e in ctx.traversal_events + ctx.entity_bridged_events:
            if e["in_unit"] or not e.get("unit_id") or e["unit_id"] == unit_id:
                continue
            r = related.setdefault(e["unit_id"], {"unit_id": e["unit_id"], "scope": e["unit_scope"], "via": ["graph"],
                                                  "graph_links": 0, "example_event": e["event_uid"]})
            r["graph_links"] += 1
        vec = unit.get("vector")
        if vec is not None and self.unit_vectors:
            sims = [(uid2, float(np.dot(vec, v))) for uid2, v in self.unit_vectors.items() if uid2 != unit_id]
            for uid2, sim in sorted(sims, key=lambda t: (-t[1], t[0]))[: s.related_units_k]:
                if sim <= 0:
                    continue
                r = related.setdefault(uid2, {"unit_id": uid2, "scope": self.unit_meta.get(uid2, {}).get("scope"),
                                              "via": [], "graph_links": 0})
                if "semantic" not in r["via"]:
                    r["via"].append("semantic")
                r["narrative_similarity"] = round(sim, 4)
        for uid2, r in related.items():
            meta = self.unit_meta.get(uid2, {})
            r.setdefault("size", meta.get("size"))
            r["narrative"] = (meta.get("narrative") or "")[:220]
            cands = store.unit_candidates(f"{self.run_id}:{uid2}")
            r["candidate_techniques"] = [{"technique_id": c["technique_id"], "similarity_score": c.get("similarity_score"),
                                          "status": c.get("status")} for c in cands[:3]]
        ctx.related_units = sorted(related.values(), key=lambda r: (-r["graph_links"], -(r.get("narrative_similarity") or 0), r["unit_id"]))

        # ---- 4. references -------------------------------------------------------
        refs: Dict[str, Dict[str, Any]] = {}
        cand_ids = []
        for c in unit.get("candidates", []):
            if c["mapping_status"] in ("CANDIDATE", "WEAK_MATCH"):
                cand_ids.append(c["technique_id"])
                for r in self._reference(c["technique_id"], "candidate", c["similarity_score"]):
                    refs[r["ref_id"]] = r
                if c.get("is_subtechnique") and self.mitre is not None and "." in c["technique_id"]:
                    for r in self._reference(c["technique_id"].split(".")[0], f"parent_of:{c['technique_id']}"):
                        refs.setdefault(r["ref_id"], r)
        graph_derived = 0
        for rel_unit in ctx.related_units:
            if "graph" not in rel_unit["via"]:
                continue
            for c in rel_unit["candidate_techniques"]:
                if c["status"] == "CANDIDATE" and c["technique_id"] not in cand_ids:
                    for r in self._reference(c["technique_id"], f"graph:related_unit:{rel_unit['unit_id']}",
                                             c.get("similarity_score")):
                        if r["ref_id"] not in refs:
                            refs[r["ref_id"]] = r
                            graph_derived += 1
        if self.tcorpus is not None and vec is not None and s.semantic_reference_k > 0:
            sims = self.tcorpus.vectors @ vec
            taken = 0
            for idx in np.argsort(-sims, kind="stable"):
                tid = self.tcorpus.techniques[idx]["technique_id"]
                if tid in cand_ids or float(sims[idx]) <= 0:
                    continue
                for r in self._reference(tid, "semantic_near_miss", round(float(sims[idx]), 4)):
                    if r["passage_type"] == "description" and r["ref_id"] not in refs:
                        refs[r["ref_id"]] = r
                        taken += 1
                if taken >= s.semantic_reference_k:
                    break
        ctx.references = list(refs.values())

        # ---- 5. ontology facts ---------------------------------------------------
        facts, seen = [], set()
        for uid in set(seeds) | set(reached) | {e["event_uid"] for e in ctx.entity_bridged_events}:
            for o in store.event_ontology(uid):
                if o["entity_id"] in seen:
                    continue
                seen.add(o["entity_id"])
                facts.append({"ref_id": f"ONT:{o['entity_id']}", "entity_id": o["entity_id"],
                              "entity_type": o.get("entity_type"), "canonical_name": o.get("canonical_name"),
                              "ontology_uri": o.get("ontology_uri"), "status": o.get("status"),
                              "confidence": o.get("confidence_value"), "evidence_event_uid": uid,
                              "provenance": o.get("producer")})
        ctx.ontology_facts = facts

        # ---- graph contribution: evidence that exists only because of traversal ----
        contributed = (len(ctx.traversal_events) + len(ctx.entity_bridged_events) + graph_derived) > 0
        ctx.graph_contribution = {
            "traversal_events": len(ctx.traversal_events),
            "cross_unit_events": sum(1 for e in ctx.traversal_events if not e["in_unit"]),
            "entity_bridged_events": len(ctx.entity_bridged_events),
            "paths": len(ctx.paths),
            "related_units_via_graph": sum(1 for r in ctx.related_units if "graph" in r["via"]),
            "graph_derived_references": graph_derived,
            "hub_entities_not_expanded": len(ctx.hub_entities),
            "ontology_facts": len(facts),
            "status": "graph_traversal_contributed" if contributed else "no_additional_evidence_from_graph",
        }
        return ctx
