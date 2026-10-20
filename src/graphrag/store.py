"""
src/graphrag/store.py
-----------------------
Graph-store abstraction for GraphRAG retrieval.

    GraphStore           abstract interface used by the retriever (backend-neutral)
    LocalGraphStore      NetworkX MultiDiGraph, in memory - the default; needs nothing
    (neo4j_store.py)     Neo4jGraphStore - optional persistence + Cypher retrieval

The retriever only ever calls the *query* methods below, so the same retrieval
code runs on either backend. Edges are idempotent on (src, rel, dst) - the
local equivalent of Cypher MERGE.
"""

import abc
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

import networkx as nx

from src.graphrag import schema as S


class GraphStore(abc.ABC):
    backend: str = "abstract"
    run_id: Optional[str] = None            # set by the service; persistent backends scope cluster queries to it

    def set_run(self, run_id: str) -> None:
        self.run_id = run_id

    # ------------------------------------------------------------- writes
    @abc.abstractmethod
    def add_nodes(self, kind: str, rows: Iterable[Dict[str, Any]]) -> int:
        """rows: dicts that contain the kind's unique-key property (schema.UNIQUE_KEYS)."""

    @abc.abstractmethod
    def add_edges(self, rows: Iterable[Dict[str, Any]]) -> int:
        """rows: {src_kind, src_key, rel, dst_kind, dst_key, props{...}}. Idempotent."""

    # ------------------------------------------------------------ queries
    @abc.abstractmethod
    def get_node(self, kind: str, key: str) -> Optional[Dict[str, Any]]: ...

    @abc.abstractmethod
    def expand_events(self, seed_uids: Sequence[str], depth: int, max_events: int,
                      rels: Sequence[str] = S.EVENT_EVENT_RELS) -> Dict[str, Any]:
        """Breadth-first traversal over event<->event relationships.
        Returns {"events": {uid: {"hop": int, "parent": uid|None, "rel": str|None}}, "truncated": bool}."""

    @abc.abstractmethod
    def event_entities(self, uid: str) -> List[Dict[str, Any]]:
        """[{entity_key, entity_type, name, role, event_count}]"""

    @abc.abstractmethod
    def entity_events(self, entity_key: str, limit: int) -> List[str]: ...

    @abc.abstractmethod
    def unit_of_event(self, uid: str) -> Optional[Dict[str, Any]]:
        """{unit_key, unit_id, scope} of the cluster / noise unit containing the event, else None."""

    @abc.abstractmethod
    def unit_candidates(self, unit_key: str) -> List[Dict[str, Any]]:
        """[{technique_id, similarity_score, status, rank, ...edge props}] (inferred links)"""

    @abc.abstractmethod
    def technique_references(self, technique_id: str) -> List[Dict[str, Any]]: ...

    @abc.abstractmethod
    def event_ontology(self, uid: str) -> List[Dict[str, Any]]:
        """OntologyEntity nodes whose evidence explicitly cites this event."""

    @abc.abstractmethod
    def stats(self) -> Dict[str, Any]: ...

    def close(self) -> None:
        pass


class LocalGraphStore(GraphStore):
    backend = "local-networkx"

    def __init__(self):
        self.g = nx.MultiDiGraph()
        self._entity_degree: Dict[str, int] = defaultdict(int)

    @staticmethod
    def _nid(kind: str, key: str) -> str:
        return f"{kind}\x1f{key}"

    # ------------------------------------------------------------- writes
    def add_nodes(self, kind: str, rows: Iterable[Dict[str, Any]]) -> int:
        ukey = S.UNIQUE_KEYS[kind]
        n = 0
        for r in rows:
            nid = self._nid(kind, str(r[ukey]))
            if nid in self.g:
                self.g.nodes[nid].update(r)
            else:
                self.g.add_node(nid, kind=kind, **r)
            n += 1
        return n

    def add_edges(self, rows: Iterable[Dict[str, Any]]) -> int:
        n = 0
        for r in rows:
            if r["rel"] not in S.ALL_RELS:
                raise ValueError(f"Unknown relationship type {r['rel']!r}")
            u = self._nid(r["src_kind"], str(r["src_key"]))
            v = self._nid(r["dst_kind"], str(r["dst_key"]))
            if u not in self.g or v not in self.g:
                raise KeyError(f"Edge {r['rel']} references a missing node: {r['src_kind']}:{r['src_key']} -> "
                               f"{r['dst_kind']}:{r['dst_key']}")
            ekey = r.get("edge_key") or r["rel"]          # ONTOLOGY_RELATION edges are keyed by their relationship_type
            if self.g.has_edge(u, v, key=ekey):
                self.g.edges[u, v, ekey].update(r.get("props") or {})
            else:
                self.g.add_edge(u, v, key=ekey, rel=r["rel"], **(r.get("props") or {}))
                if r["rel"] in S.EVENT_ENTITY_RELS:
                    self._entity_degree[v] += 1
            n += 1
        return n

    # ------------------------------------------------------------ queries
    def _node(self, kind: str, key: str) -> Optional[Dict[str, Any]]:
        nid = self._nid(kind, key)
        return self.g.nodes[nid] if nid in self.g else None

    def get_node(self, kind: str, key: str) -> Optional[Dict[str, Any]]:
        n = self._node(kind, key)
        return dict(n) if n is not None else None

    def expand_events(self, seed_uids, depth, max_events, rels=S.EVENT_EVENT_RELS):
        rels = set(rels)
        events: Dict[str, Dict[str, Any]] = {}
        frontier = []
        for uid in seed_uids:
            if self._nid(S.EVENT, uid) in self.g and uid not in events:
                events[uid] = {"hop": 0, "parent": None, "rel": None}
                frontier.append(uid)
        truncated = False
        for hop in range(1, depth + 1):
            nxt = []
            for uid in frontier:
                nid = self._nid(S.EVENT, uid)
                cand = []
                for _, v, k, d in self.g.out_edges(nid, keys=True, data=True):
                    if k in rels:
                        cand.append((self.g.nodes[v]["event_uid"], k))
                for u, _, k, d in self.g.in_edges(nid, keys=True, data=True):
                    if k in rels:
                        cand.append((self.g.nodes[u]["event_uid"], k))
                for vuid, rel in sorted(cand):
                    if vuid in events:
                        continue
                    if len(events) >= max_events:
                        truncated = True
                        break
                    events[vuid] = {"hop": hop, "parent": uid, "rel": rel}
                    nxt.append(vuid)
            frontier = nxt
            if not frontier:
                break
        return {"events": events, "truncated": truncated}

    def event_entities(self, uid):
        nid = self._nid(S.EVENT, uid)
        out = []
        if nid not in self.g:
            return out
        for _, v, k, d in self.g.out_edges(nid, keys=True, data=True):
            if k in S.EVENT_ENTITY_RELS:
                node = self.g.nodes[v]
                out.append({"entity_key": node["entity_key"], "entity_type": node.get("entity_type"),
                            "name": node.get("name"), "role": k, "event_count": self._entity_degree.get(v, 0)})
        return sorted(out, key=lambda e: (e["role"], e["entity_key"]))

    def entity_events(self, entity_key, limit):
        nid = self._nid(S.ENTITY, entity_key)
        if nid not in self.g:
            return []
        uids = [self.g.nodes[u]["event_uid"] for u, _, k in self.g.in_edges(nid, keys=True)
                if k in S.EVENT_ENTITY_RELS]
        uids.sort(key=lambda u: (-(self.g.nodes[self._nid(S.EVENT, u)].get("novelty_score") or 0.0), u))
        return uids[:limit]

    def unit_of_event(self, uid):
        nid = self._nid(S.EVENT, uid)
        if nid not in self.g:
            return None
        for _, v, k in self.g.out_edges(nid, keys=True):
            if k in (S.IN_CLUSTER, S.IN_NOISE_UNIT):
                n = self.g.nodes[v]
                return {"unit_key": n["cluster_key"], "unit_id": n.get("unit_id"), "scope": n.get("scope")}
        return None

    def unit_candidates(self, unit_key):
        nid = self._nid(S.CLUSTER, unit_key)
        if nid not in self.g:
            return []
        out = []
        for _, v, k, d in self.g.out_edges(nid, keys=True, data=True):
            if k == S.CANDIDATE_TECHNIQUE:
                row = {kk: vv for kk, vv in d.items() if kk != "rel"}
                row["technique_id"] = self.g.nodes[v]["technique_id"]
                row["technique_name"] = self.g.nodes[v].get("technique_name")
                out.append(row)
        return sorted(out, key=lambda r: (r.get("rank", 99), r["technique_id"]))

    def technique_references(self, technique_id):
        nid = self._nid(S.TECHNIQUE, technique_id)
        if nid not in self.g:
            return []
        docs = [dict(self.g.nodes[v]) for _, v, k in self.g.out_edges(nid, keys=True) if k == S.DESCRIBED_BY]
        return sorted(docs, key=lambda d: d["doc_id"])

    def event_ontology(self, uid):
        nid = self._nid(S.EVENT, uid)
        if nid not in self.g:
            return []
        out = []
        for u, _, k, d in self.g.in_edges(nid, keys=True, data=True):
            if k == S.EVIDENCED_BY and self.g.nodes[u].get("kind") == S.ONTOLOGY:
                out.append(dict(self.g.nodes[u]))
        return sorted(out, key=lambda n: n["entity_id"])

    # ---- export in add_nodes/add_edges row format (used to sync into a persistent store) ----
    def iter_nodes(self):
        """yields (kind, row) for every node"""
        for _, d in self.g.nodes(data=True):
            row = {k: v for k, v in d.items() if k != "kind"}
            yield d["kind"], row

    def iter_edges(self):
        for u, v, k, d in self.g.edges(keys=True, data=True):
            props = {kk: vv for kk, vv in d.items() if kk != "rel"}
            row = {"src_kind": self.g.nodes[u]["kind"], "src_key": self.g.nodes[u][S.UNIQUE_KEYS[self.g.nodes[u]["kind"]]],
                   "rel": d["rel"], "dst_kind": self.g.nodes[v]["kind"],
                   "dst_key": self.g.nodes[v][S.UNIQUE_KEYS[self.g.nodes[v]["kind"]]], "props": props}
            if k != d["rel"]:
                row["edge_key"] = k
            yield row

    def stats(self):
        kinds: Dict[str, int] = defaultdict(int)
        for _, d in self.g.nodes(data=True):
            kinds[d["kind"]] += 1
        rels: Dict[str, int] = defaultdict(int)
        for _, _, d in self.g.edges(data=True):
            rels[d["rel"]] += 1
        return {"backend": self.backend, "nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges(),
                "nodes_by_kind": dict(kinds), "edges_by_type": dict(rels)}
