"""
src/graphrag/neo4j_store.py
-----------------------------
OPTIONAL Neo4j backend (persistent graph + Cypher retrieval). The local CLI run
never needs it: if Neo4j is not requested, not installed or not reachable the
service keeps running on the in-memory NetworkX store and reports plainly that
retrieval ran in LOCAL mode.

Connection settings come ONLY from environment variables - nothing is
hard-coded and no credentials are ever logged:
    NEO4J_URI        e.g. bolt://localhost:7687  (or neo4j+s://...)
    NEO4J_USER       default "neo4j"
    NEO4J_PASSWORD   required
    NEO4J_DATABASE   default "neo4j"

Properties of the write path:
  * unique constraints per node kind (schema.UNIQUE_KEYS), created IF NOT EXISTS
  * every write is an idempotent MERGE on the stable key, then SET of properties
    - re-running the same input updates, never duplicates
  * relationship types and labels are interpolated ONLY from the fixed whitelist
    in graphrag/schema.py (Cypher cannot parameterise them); all values travel as
    query parameters, never string-formatted
  * technique links are written as CANDIDATE_TECHNIQUE edges carrying
    inferred=true, similarity, mapping method and corpus version - never as
    confirmed relationships

STATUS: the Cypher here is exercised in tests/test_neo4j_store.py against a
recording fake driver (statement shape, MERGE/constraints, parameterisation,
disconnected handling). It has NOT been run against a live Neo4j server in the
environment this was built in.
"""

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.graphrag import schema as S
from src.graphrag.store import GraphStore

BATCH = 500


class Neo4jUnavailable(Exception):
    """Neo4j was requested but cannot be used (not configured / driver missing / unreachable)."""


def settings_from_env(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = os.environ if env is None else env
    uri, pw = env.get("NEO4J_URI"), env.get("NEO4J_PASSWORD")
    if not uri or not pw:
        raise Neo4jUnavailable("Neo4j not configured: set NEO4J_URI and NEO4J_PASSWORD "
                               "(optionally NEO4J_USER, NEO4J_DATABASE) in the environment.")
    return {"uri": uri, "user": env.get("NEO4J_USER", "neo4j"), "password": pw,
            "database": env.get("NEO4J_DATABASE", "neo4j")}


def _clean(props: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in props.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (str, int, float, bool)) for x in v):
            out[k] = list(v)
        else:
            out[k] = json.dumps(v, default=str)
    return out


_ENT_RELS = "|".join(S.EVENT_ENTITY_RELS)
_EE_RELS = "|".join(S.EVENT_EVENT_RELS)


class Neo4jGraphStore(GraphStore):
    backend = "neo4j"

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j", driver=None):
        self.database = database
        if driver is not None:                       # injected (tests)
            self.driver = driver
        else:
            try:
                from neo4j import GraphDatabase
            except ImportError as e:
                raise Neo4jUnavailable("The 'neo4j' Python driver is not installed (pip install neo4j).") from e
            try:
                self.driver = GraphDatabase.driver(uri, auth=(user, password))
                self.driver.verify_connectivity()
            except Exception as e:
                raise Neo4jUnavailable(f"Cannot connect to Neo4j at {uri} ({type(e).__name__}: {str(e)[:140]}).") from e
        self.ensure_schema()

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None, driver=None) -> "Neo4jGraphStore":
        s = settings_from_env(env)
        return cls(s["uri"], s["user"], s["password"], s["database"], driver=driver)

    # ----------------------------------------------------------------- plumbing
    def _run(self, query: str, **params) -> List[Dict[str, Any]]:
        try:
            with self.driver.session(database=self.database) as session:
                return [r.data() for r in session.run(query, **params)]
        except Neo4jUnavailable:
            raise
        except Exception as e:
            raise Neo4jUnavailable(f"Neo4j query failed ({type(e).__name__}: {str(e)[:140]})") from e

    def ensure_schema(self) -> List[str]:
        stmts = []
        for kind, key in S.UNIQUE_KEYS.items():
            q = f"CREATE CONSTRAINT {kind.lower()}_{key} IF NOT EXISTS FOR (n:`{kind}`) REQUIRE n.{key} IS UNIQUE"
            self._run(q)
            stmts.append(q)
        return stmts

    def close(self) -> None:
        try:
            self.driver.close()
        except Exception:
            pass

    # ------------------------------------------------------------------- writes
    def add_nodes(self, kind: str, rows: Iterable[Dict[str, Any]]) -> int:
        if kind not in S.UNIQUE_KEYS:
            raise ValueError(f"Unknown node kind {kind!r}")
        ukey = S.UNIQUE_KEYS[kind]
        q = (f"UNWIND $rows AS row MERGE (n:`{kind}` {{{ukey}: row.key}}) SET n += row.props")
        batch, n = [], 0
        for r in rows:
            batch.append({"key": str(r[ukey]), "props": _clean({k: v for k, v in r.items() if k != ukey})})
            if len(batch) >= BATCH:
                self._run(q, rows=batch)
                n += len(batch)
                batch = []
        if batch:
            self._run(q, rows=batch)
            n += len(batch)
        return n

    def add_edges(self, rows: Iterable[Dict[str, Any]]) -> int:
        groups: Dict[tuple, List[Dict[str, Any]]] = {}
        for r in rows:
            if r["rel"] not in S.ALL_RELS:
                raise ValueError(f"Unknown relationship type {r['rel']!r}")
            groups.setdefault((r["src_kind"], r["rel"], r["dst_kind"]), []).append(r)
        total = 0
        for (sk, rel, dk), rs in groups.items():
            sp, dp = S.UNIQUE_KEYS[sk], S.UNIQUE_KEYS[dk]
            if rel == S.ONTOLOGY_RELATION:             # the ontology's own type is a PROPERTY (merge key), not a Cypher type
                merge = f"MERGE (a)-[r:`{rel}` {{relationship_type: row.rtype}}]->(b)"
            else:
                merge = f"MERGE (a)-[r:`{rel}`]->(b)"
            q = (f"UNWIND $rows AS row MATCH (a:`{sk}` {{{sp}: row.src}}) MATCH (b:`{dk}` {{{dp}: row.dst}}) "
                 f"{merge} SET r += row.props")
            batch = []
            for r in rs:
                props = _clean(r.get("props") or {})
                item = {"src": str(r["src_key"]), "dst": str(r["dst_key"]), "props": props}
                if rel == S.ONTOLOGY_RELATION:
                    item["rtype"] = props.get("relationship_type", "")
                batch.append(item)
                if len(batch) >= BATCH:
                    self._run(q, rows=batch)
                    total += len(batch)
                    batch = []
            if batch:
                self._run(q, rows=batch)
                total += len(batch)
        return total

    def sync_from(self, local) -> Dict[str, int]:
        """Copy a LocalGraphStore into Neo4j (idempotent). Nodes first, then edges."""
        by_kind: Dict[str, List[Dict[str, Any]]] = {}
        for kind, row in local.iter_nodes():
            by_kind.setdefault(kind, []).append(row)
        nodes = sum(self.add_nodes(k, rows) for k, rows in by_kind.items())
        edges = self.add_edges(list(local.iter_edges()))
        return {"nodes": nodes, "edges": edges}

    # ------------------------------------------------------------------ queries
    def get_node(self, kind: str, key: str):
        if kind not in S.UNIQUE_KEYS:
            return None
        rows = self._run(f"MATCH (n:`{kind}` {{{S.UNIQUE_KEYS[kind]}: $key}}) RETURN properties(n) AS p LIMIT 1", key=key)
        return rows[0]["p"] if rows else None

    def expand_events(self, seed_uids: Sequence[str], depth: int, max_events: int,
                      rels: Sequence[str] = S.EVENT_EVENT_RELS):
        rel_types = "|".join(r for r in rels if r in S.EVENT_EVENT_RELS)
        q = (f"MATCH (a:`Event`)-[r:{rel_types}]-(b:`Event`) WHERE a.event_uid IN $frontier "
             f"RETURN a.event_uid AS a, b.event_uid AS b, type(r) AS rel ORDER BY b, a")
        existing = {r["uid"] for r in self._run("MATCH (e:`Event`) WHERE e.event_uid IN $uids RETURN e.event_uid AS uid",
                                                uids=list(seed_uids))}
        events = {u: {"hop": 0, "parent": None, "rel": None} for u in seed_uids if u in existing}
        frontier, truncated = list(events), False
        for hop in range(1, depth + 1):
            if not frontier:
                break
            rows = self._run(q, frontier=frontier)
            nxt = []
            for row in sorted(rows, key=lambda r: (r["a"], r["b"], r["rel"])):
                if row["b"] in events:
                    continue
                if len(events) >= max_events:
                    truncated = True
                    break
                events[row["b"]] = {"hop": hop, "parent": row["a"], "rel": row["rel"]}
                nxt.append(row["b"])
            frontier = nxt
        return {"events": events, "truncated": truncated}

    def event_entities(self, uid: str):
        q = (f"MATCH (e:`Event` {{event_uid: $uid}})-[r:{_ENT_RELS}]->(n:`Entity`) "
             f"RETURN n.entity_key AS entity_key, n.entity_type AS entity_type, n.name AS name, type(r) AS role, "
             f"size([(n)<-[x:{_ENT_RELS}]-() | x]) AS event_count ORDER BY role, entity_key")
        return self._run(q, uid=uid)

    def entity_events(self, entity_key: str, limit: int):
        q = (f"MATCH (e:`Event`)-[:{_ENT_RELS}]->(n:`Entity` {{entity_key: $k}}) "
             f"RETURN e.event_uid AS uid ORDER BY coalesce(e.novelty_score, 0.0) DESC, uid LIMIT $limit")
        return [r["uid"] for r in self._run(q, k=entity_key, limit=int(limit))]

    def unit_of_event(self, uid: str):
        q = ("MATCH (e:`Event` {event_uid: $uid})-[:IN_CLUSTER|IN_NOISE_UNIT]->(c:`Cluster`) "
             "WHERE $run IS NULL OR c.run_id = $run RETURN c.cluster_key AS unit_key, c.unit_id AS unit_id, c.scope AS scope LIMIT 1")
        rows = self._run(q, uid=uid, run=self.run_id)
        return rows[0] if rows else None

    def unit_candidates(self, unit_key: str):
        q = ("MATCH (c:`Cluster` {cluster_key: $k})-[r:CANDIDATE_TECHNIQUE]->(t:`TechniqueCandidate`) "
             "RETURN properties(r) AS props, t.technique_id AS technique_id, t.technique_name AS technique_name")
        out = []
        for r in self._run(q, k=unit_key):
            row = dict(r["props"])
            row["technique_id"], row["technique_name"] = r["technique_id"], r["technique_name"]
            out.append(row)
        return sorted(out, key=lambda r: (r.get("rank", 99), r["technique_id"]))

    def technique_references(self, technique_id: str):
        q = ("MATCH (t:`TechniqueCandidate` {technique_id: $tid})-[:DESCRIBED_BY]->(d:`ReferenceDocument`) "
             "RETURN properties(d) AS d ORDER BY d.doc_id")
        return [r["d"] for r in self._run(q, tid=technique_id)]

    def event_ontology(self, uid: str):
        q = ("MATCH (o:`OntologyEntity`)-[:EVIDENCED_BY]->(e:`Event` {event_uid: $uid}) "
             "RETURN properties(o) AS o ORDER BY o.entity_id")
        return [r["o"] for r in self._run(q, uid=uid)]

    def stats(self):
        nodes = {r["label"]: r["n"] for r in self._run("MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n")}
        rels = {r["t"]: r["n"] for r in self._run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n")}
        return {"backend": self.backend, "database": self.database, "nodes": sum(nodes.values()),
                "edges": sum(rels.values()), "nodes_by_kind": nodes, "edges_by_type": rels}
