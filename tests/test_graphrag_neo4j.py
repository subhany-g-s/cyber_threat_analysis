import copy

import pytest

from src.graphrag import schema as S
from src.graphrag.neo4j_store import Neo4jGraphStore, Neo4jUnavailable, settings_from_env
from src.graphrag.store import LocalGraphStore
from src.service.pipeline import PipelineService
from tests.conftest import base_config


# ------------------------------------------------------------------ local store
def _tiny():
    st = LocalGraphStore()
    st.add_nodes(S.EVENT, [{"event_uid": u} for u in "abcde"])
    st.add_edges([{"src_kind": S.EVENT, "src_key": "a", "rel": S.SAME_PID, "dst_kind": S.EVENT, "dst_key": "b"},
                  {"src_kind": S.EVENT, "src_key": "b", "rel": S.SAME_HOST_TEMPORAL, "dst_kind": S.EVENT, "dst_key": "c"},
                  {"src_kind": S.EVENT, "src_key": "c", "rel": S.SAME_PID, "dst_kind": S.EVENT, "dst_key": "d"}])
    return st


def test_local_store_is_idempotent_and_validates():
    st = _tiny()
    n_nodes, n_edges = st.g.number_of_nodes(), st.g.number_of_edges()
    st.add_nodes(S.EVENT, [{"event_uid": "a", "extra": 1}])
    st.add_edges([{"src_kind": S.EVENT, "src_key": "a", "rel": S.SAME_PID, "dst_kind": S.EVENT, "dst_key": "b"}])
    assert (st.g.number_of_nodes(), st.g.number_of_edges()) == (n_nodes, n_edges)
    assert st.get_node(S.EVENT, "a")["extra"] == 1
    with pytest.raises(ValueError):
        st.add_edges([{"src_kind": S.EVENT, "src_key": "a", "rel": "MADE_UP", "dst_kind": S.EVENT, "dst_key": "b"}])
    with pytest.raises(KeyError):
        st.add_edges([{"src_kind": S.EVENT, "src_key": "a", "rel": S.SAME_PID, "dst_kind": S.EVENT, "dst_key": "zzz"}])


def test_traversal_depth_hops_and_truncation():
    st = _tiny()
    r1 = st.expand_events(["a"], depth=1, max_events=10)["events"]
    assert set(r1) == {"a", "b"} and r1["b"] == {"hop": 1, "parent": "a", "rel": S.SAME_PID}
    r2 = st.expand_events(["a"], depth=3, max_events=10)["events"]
    assert set(r2) == {"a", "b", "c", "d"} and r2["d"]["hop"] == 3 and "e" not in r2     # e is disconnected
    t = st.expand_events(["a"], depth=3, max_events=2)
    assert t["truncated"] and len(t["events"]) == 2


# ------------------------------------------------------------------ retrieval on the real run
def test_retrieval_provenance_and_honest_graph_claim(small_run):
    ctxs = [u.retrieval for u in small_run.units]
    assert all(c and c["retrieval_mode"] == "local-networkx" for c in ctxs)
    uid_set = set(small_run.artifacts.row_to_uid)
    for c in ctxs:
        for e in c["seed_events"] + c["traversal_events"]:
            assert e["event_uid"] in uid_set
        for p in c["paths"]:
            assert p.get("status") == "observed" and p.get("provenance")
        gc = c["graph_contribution"]
        assert gc["status"] in ("graph_traversal_contributed", "no_additional_evidence_from_graph")
        if gc["status"] == "graph_traversal_contributed":
            assert c["traversal_events"] or c["entity_bridged_events"] or c["related_units"]
    g = small_run.graphrag
    assert g["units_where_graph_contributed"] + g["units_without_graph_evidence"] == g["units_retrieved"] == len(small_run.units)


def test_knowledge_graph_has_required_node_kinds_and_candidate_links_are_inferred(small_run):
    store = small_run.artifacts.knowledge_graph
    kinds = {k for k, _ in store.iter_nodes()}
    assert {S.EVENT, S.ENTITY, S.CLUSTER, S.REFERENCE} <= kinds
    cand = [e for e in store.iter_edges() if e["rel"] == S.CANDIDATE_TECHNIQUE]
    assert cand and all(e["props"]["edge_status"] == S.STATUS_INFERRED and e["props"]["inferred"] is True for e in cand)
    assert not any(e["rel"] == "CONFIRMED_TECHNIQUE" for e in store.iter_edges())
    noise_edges = [e for e in store.iter_edges() if e["rel"] == S.IN_NOISE_UNIT]
    assert noise_edges and all(e["rel"] != S.IN_CLUSTER or e["dst_key"].split(":")[-1].startswith("C")
                               for e in store.iter_edges() if e["rel"] == S.IN_CLUSTER)
    run_ids = {k.split(":")[0] for kk, row in store.iter_nodes() if kk == S.CLUSTER for k in [row["cluster_key"]]}
    assert run_ids == {small_run.run_id}


def test_run_id_is_deterministic(small_input, mitre_cache, small_run):
    from src.graphrag.builder import make_run_id
    assert make_run_id("x", {"a": 1}, "ck") == make_run_id("x", {"a": 1}, "ck") != make_run_id("y", {"a": 1}, "ck")
    assert small_run.run_id.startswith("run-")


# ------------------------------------------------------------------ neo4j adapter (fake driver; NO live server)
class FakeRecord:
    def __init__(self, d):
        self._d = d

    def data(self):
        return self._d


class FakeSession:
    def __init__(self, drv):
        self.drv = drv

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, query, **params):
        if self.drv.disconnected_for_reads and "MERGE" not in query and "CONSTRAINT" not in query:
            raise RuntimeError("connection lost")
        self.drv.calls.append((query, params))
        return [FakeRecord(r) for r in self.drv.reply]


class FakeDriver:
    def __init__(self, disconnected_for_reads=False, reply=()):
        self.calls, self.disconnected_for_reads, self.reply = [], disconnected_for_reads, list(reply)
        self.closed = False

    def session(self, database=None):
        return FakeSession(self)

    def close(self):
        self.closed = True


def test_neo4j_constraints_for_every_node_kind():
    d = FakeDriver()
    Neo4jGraphStore("bolt://x", "u", "p", driver=d)
    cons = [q for q, _ in d.calls if "CONSTRAINT" in q]
    assert len(cons) == len(S.UNIQUE_KEYS)
    for kind, key in S.UNIQUE_KEYS.items():
        assert any(f"`{kind}`" in q and f"n.{key} IS UNIQUE" in q and "IF NOT EXISTS" in q for q in cons)


def test_neo4j_sync_uses_merge_only_and_is_repeatable(small_run):
    d = FakeDriver()
    neo = Neo4jGraphStore("bolt://x", "u", "p", driver=d)
    local = small_run.artifacts.knowledge_graph
    s1 = neo.sync_from(local)
    first = [(q, copy.deepcopy(p)) for q, p in d.calls if "UNWIND" in q]
    s2 = neo.sync_from(local)
    second = [(q, p) for q, p in d.calls if "UNWIND" in q][len(first):]
    assert s1 == s2 and s1["nodes"] > 0 and s1["edges"] > 0
    assert first == second                                            # identical keys & payloads: re-running cannot create duplicates
    writes = [q for q, _ in first]
    assert all("MERGE" in q for q in writes) and not any(" CREATE (" in q or q.lstrip().startswith("CREATE (") for q in writes)
    assert any("MERGE (n:`Event`" in q for q in writes) and any("`TechniqueCandidate`" in q for q in writes)
    cand_batches = [p for q, p in first if "CANDIDATE_TECHNIQUE" in q]
    assert cand_batches and all(r["props"]["edge_status"] == "inferred" for p in cand_batches for r in p["rows"])


def test_neo4j_rejects_unknown_relationship_and_kind():
    neo = Neo4jGraphStore("bolt://x", "u", "p", driver=FakeDriver())
    with pytest.raises(ValueError):
        neo.add_edges([{"src_kind": S.EVENT, "src_key": "a", "rel": "DROP_ALL", "dst_kind": S.EVENT, "dst_key": "b"}])
    with pytest.raises(ValueError):
        neo.add_nodes("Bogus", [{"x": 1}])


def test_neo4j_query_failure_becomes_unavailable():
    neo = Neo4jGraphStore("bolt://x", "u", "p", driver=FakeDriver())
    neo.driver.disconnected_for_reads = True
    with pytest.raises(Neo4jUnavailable):
        neo.get_node(S.EVENT, "a")


def test_neo4j_settings_come_from_environment_only():
    with pytest.raises(Neo4jUnavailable, match="NEO4J_URI"):
        settings_from_env({})
    s = settings_from_env({"NEO4J_URI": "bolt://h", "NEO4J_PASSWORD": "pw"})
    assert s["user"] == "neo4j" and s["database"] == "neo4j"


def test_pipeline_local_mode_report_when_neo4j_not_configured(small_input, mitre_cache, monkeypatch):
    for v in ("NEO4J_URI", "NEO4J_PASSWORD", "NEO4J_USER"):
        monkeypatch.delenv(v, raising=False)
    res = PipelineService(base_config(small_input, mitre_cache, neo4j_enabled=True)).run()
    s9 = res.stages[8]
    assert s9.status == "WARN" and any("LOCAL mode" in w for w in s9.warnings)
    assert "Neo4j unavailable" in res.graphrag["mode_note"] and res.graphrag["retrieval_mode"] == "local-networkx"
    assert res.ok


def test_pipeline_survives_neo4j_disconnect_mid_run(small_input, mitre_cache):
    d = FakeDriver(disconnected_for_reads=True)
    neo = Neo4jGraphStore("bolt://x", "u", "p", driver=d)
    res = PipelineService(base_config(small_input, mitre_cache), neo4j_store=neo).run()
    assert res.ok and res.stages[8].status == "WARN"
    assert any("switching to LOCAL" in w or "LOCAL mode" in w for w in res.stages[8].warnings)
    assert res.graphrag["units_retrieved"] == len(res.units)
    assert any("MERGE" in q for q, _ in d.calls)                      # the sync did reach the (fake) server
