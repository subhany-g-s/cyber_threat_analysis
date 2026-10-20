"""
src/service/result.py
-----------------------
The structured result of one pipeline run. Everything the CLI prints and
everything a dashboard would show is read from this object; nothing is
re-derived from files.

    RunResult.stages        10 StageResult rows (status, timings, metrics, warnings)
    RunResult.<section>     parsing / normalization / graph / gae / clustering /
                            novelty / mapping / graphrag / ontology (plain dicts of
                            JSON-able metrics)
    RunResult.units         one UnitResult per analysed unit: every real cluster
                            AND every reviewed noise unit (scope tells them apart)
    RunResult.noise         noise-event review (all noise events, ranked)
    RunResult.report        run-level evidence report (the technical appendix index)
    RunResult.analyst_report  prioritized, de-duplicated findings for analysts (primary report)
    RunResult.artifacts     heavy in-memory objects (graph, embeddings, ...);
                            NOT serialised - for UI use (graph visualisation).

`to_dict()` is JSON-safe (numpy scalars converted, artifacts omitted).
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

STAGE_NAMES = [
    "Input validation and parsing",
    "Normalization",
    "Event graph construction",
    "GAE checkpoint validation and embeddings",
    "HDBSCAN clustering",
    "Novelty detection",
    "Cluster profiling",
    "Semantic MITRE ATT&CK mapping",
    "GraphRAG retrieval",
    "Evidence-grounded threat report",
]

OK, WARN, FAILED, SKIPPED = "OK", "WARN", "FAILED", "SKIPPED"


@dataclass
class StageResult:
    number: int
    name: str
    status: str = SKIPPED                  # OK | WARN | FAILED | SKIPPED
    elapsed_s: float = 0.0
    explanation: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class UnitResult:
    """One analysed unit: a real cluster, or a noise group / single noise event."""
    unit_id: str                           # C7 | N2 | E5
    scope: str                             # cluster | noise_group | noise_event
    cluster_id: int                        # -1 for noise units
    size: int
    narrative: str
    mapping_text: str
    profile: Dict[str, Any] = field(default_factory=dict)
    novelty_summary: Dict[str, Any] = field(default_factory=dict)
    mapping: Dict[str, Any] = field(default_factory=dict)       # ClusterMapping.as_dict()
    retrieval: Dict[str, Any] = field(default_factory=dict)     # GraphRAG context (RetrievedContext.as_dict())
    report: Dict[str, Any] = field(default_factory=dict)        # per-unit evidence report section
    priority: float = 0.0                                       # triage ordering (max novelty), not a threat score


@dataclass
class RunArtifacts:
    events: List[Any] = field(default_factory=list)             # NormalizedEvent objects
    events_by_uid: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    provenance_by_uid: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    graph: Any = None                                           # networkx.DiGraph (event graph)
    embeddings: Any = None                                      # np.ndarray [N, d]
    row_to_uid: List[str] = field(default_factory=list)
    labels: Any = None
    probabilities: Any = None
    novelty_rows: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_graph: Any = None                                 # GraphRAG store used for retrieval
    mitre_corpus: Any = None
    profiles: List[Any] = field(default_factory=list)


@dataclass
class RunResult:
    run_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    stages: List[StageResult] = field(default_factory=list)

    parsing: Dict[str, Any] = field(default_factory=dict)
    normalization: Dict[str, Any] = field(default_factory=dict)
    graph: Dict[str, Any] = field(default_factory=dict)
    gae: Dict[str, Any] = field(default_factory=dict)
    clustering: Dict[str, Any] = field(default_factory=dict)
    novelty: Dict[str, Any] = field(default_factory=dict)
    mapping: Dict[str, Any] = field(default_factory=dict)
    graphrag: Dict[str, Any] = field(default_factory=dict)
    ontology: Dict[str, Any] = field(default_factory=dict)

    units: List[UnitResult] = field(default_factory=list)
    noise: Dict[str, Any] = field(default_factory=dict)
    report: Dict[str, Any] = field(default_factory=dict)
    analyst_report: Dict[str, Any] = field(default_factory=dict)   # short prioritized findings report (src/rag/findings.py)

    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    artifacts: Optional[RunArtifacts] = None

    # ----------------------------------------------------------------- helpers
    @property
    def ok(self) -> bool:
        return not self.errors and all(s.status != FAILED for s in self.stages)

    def stage(self, number: int) -> StageResult:
        return self.stages[number - 1]

    def unit(self, unit_id: str) -> Optional[UnitResult]:
        return next((u for u in self.units if u.unit_id == unit_id), None)

    def to_dict(self, include_profiles: bool = True) -> Dict[str, Any]:
        d = {}
        for f in dataclasses.fields(self):
            if f.name == "artifacts":
                continue
            d[f.name] = getattr(self, f.name)
        out = jsonable(d)
        out["ok"] = self.ok
        if not include_profiles:
            for u in out.get("units", []):
                u.pop("profile", None)
        return out


def jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses / numpy / tuples / sets to JSON-safe values."""
    try:
        import numpy as np
    except ImportError:                                           # pragma: no cover
        np = None
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in obj]
    if np is not None:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None
    return obj
