"""
gae_features.py
----------------
Converts NetworkX event-graph nodes into a numerical node feature matrix X
for the GAE stage. Does NOT touch parser.py / normalizer.py / graph_builder.py
-- it consumes the existing NormalizedEvent attributes already stored on
each node by graph_builder.build_event_graph().

WHY these design choices (see project brief for the full rationale):

  - event_uid is NEVER a feature. It is only used to fix a deterministic
    row order (event_uid -> row index), saved to gae_node_mapping.json so
    a later embedding row can always be traced back to the original event.

  - raw_event is NEVER a feature in this first implementation. It stays on
    the NetworkX node for investigation/debugging only.

  - process_id / source_process_id / target_process_id are OS-assigned
    integers with no inherent numeric meaning (PID 900 is not "less than"
    PID 2092 in any behaviorally meaningful sense, and PIDs get reused
    across process lifetimes). We never feed the raw PID value as a
    number. Instead we use:
        (a) a presence indicator (has_source_process, has_target_process)
        (b) a frequency-encoded "how often does this PID recur in the
            dataset" score, which IS meaningful (a PID that appears in
            thousands of events is a long-lived, active process; one that
            appears once is not) -- and this information is *distinct*
            from what the SAME_PID edges already encode structurally
            (the edges say "these two events share a PID"; this feature
            says "and that PID is a common/rare one across the dataset").

  - Categorical fields are split into two encoding strategies based on
    cardinality, decided at fit() time from the actual data (not
    hardcoded assumptions):
        LOW cardinality (<= ONEHOT_MAX_CARDINALITY distinct values seen
        during fit, e.g. event_id, event_type, hostname, direction,
        protocol, severity) -> one-hot, with an explicit "OTHER/UNSEEN"
        bucket for values not seen at fit time (keeps inference safe on
        new data without silently producing an all-zero row).

        HIGHER cardinality (source_image, target_image, user, or anything
        that turns out to have more distinct values than the threshold,
        e.g. hostname on a much larger multi-tenant deployment) -> top-K
        one-hot (the K most frequent values, K=TOPK) + an explicit
        "OTHER" bucket, PLUS a single frequency-encoded numeric column
        (log(1+count) normalized to [0,1] by the max seen at fit time).
        This avoids an exploding one-hot matrix on fields like image path
        while keeping the encoding fully deterministic, interpretable,
        and free of hash collisions (which would be hard to explain to a
        reviewer). We deliberately did NOT use hashing here for that
        reason -- explainability matters more than saving a few columns
        for a graph this size.

  - Ports are bucketed into well-known / registered / ephemeral / missing
    rather than fed as raw integers -- port *magnitude* is not
    behaviorally meaningful, port *category* is.

  - Timestamps are converted to cyclic hour-of-day features (sin/cos of
    hour) rather than raw epoch seconds, which have no learnable meaning
    as an absolute number and would let the model implicitly memorize
    dataset order.

  - Missing-value indicator columns are added for every field group that
    is structurally absent for some EventIDs (process fields absent for
    EventID 5156, network fields absent for EventID 10, etc.) so the
    model can distinguish "field doesn't apply to this event type" from
    "field happened to be missing."

  - No label columns (cluster, novelty, MITRE technique) are ever
    included -- those don't exist yet at this stage, and including them
    later would be leakage.
"""

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

ONEHOT_MAX_CARDINALITY = 15   # fields with <= this many distinct values at fit time get full one-hot
TOPK = 10                      # for higher-cardinality fields, keep the K most frequent + "OTHER"

LOW_CARD_CATEGORICAL_FIELDS = ["event_id", "event_type", "hostname", "direction", "protocol", "severity"]
HIGH_CARD_CATEGORICAL_FIELDS = ["source_image", "target_image", "user"]

WELL_KNOWN_PORT_MAX = 1023
REGISTERED_PORT_MAX = 49151


def _safe_str(v) -> Optional[str]:
    """Normalize a raw attribute value to a hashable string category, or
    None if missing. Guards against non-hashable types (e.g. lists)."""
    if v is None:
        return None
    return str(v)


def _port_bucket(v) -> str:
    if v is None:
        return "MISSING"
    try:
        p = int(v)
    except (ValueError, TypeError):
        return "MISSING"
    if p <= WELL_KNOWN_PORT_MAX:
        return "WELL_KNOWN"
    if p <= REGISTERED_PORT_MAX:
        return "REGISTERED"
    return "EPHEMERAL"


@dataclass
class FieldVocab:
    """Vocabulary + (optional) frequency table for one categorical field,
    fixed at fit() time so transform() is deterministic on new data."""
    field_name: str
    strategy: str                     # "onehot" or "topk_freq"
    categories: List[str] = field(default_factory=list)   # column order (excludes OTHER/MISSING, appended separately)
    freq: Dict[str, int] = field(default_factory=dict)     # category -> count at fit time
    max_freq: int = 1

    def column_names(self, prefix: str) -> List[str]:
        cols = [f"{prefix}={c}" for c in self.categories]
        cols.append(f"{prefix}=OTHER_OR_UNSEEN")
        if self.strategy == "topk_freq":
            cols.append(f"{prefix}__freq_encoded")
        return cols

    def encode_value(self, value: Optional[str]) -> List[float]:
        vec = [0.0] * len(self.categories)
        other = 0.0
        if value is not None and value in self.categories:
            vec[self.categories.index(value)] = 1.0
        else:
            other = 1.0
        row = vec + [other]
        if self.strategy == "topk_freq":
            count = self.freq.get(value, 0) if value is not None else 0
            freq_enc = math.log1p(count) / math.log1p(self.max_freq) if self.max_freq > 0 else 0.0
            row.append(freq_enc)
        return row


class NodeFeatureBuilder:
    """
    Fit a deterministic feature schema from a NetworkX event graph, then
    transform any graph with the same node schema into a numeric matrix.

    Usage:
        builder = NodeFeatureBuilder()
        builder.fit(G)
        X, event_uid_order, feature_names = builder.transform(G)
        builder.save("models/gae_feature_schema.json")
    """

    def __init__(self):
        self.vocabs: Dict[str, FieldVocab] = {}
        self.pid_freq: Dict[str, int] = {}
        self.pid_max_freq: int = 1
        self.feature_names: List[str] = []
        self._fitted = False

    # ---------------------------------------------------------------- fit
    def fit(self, G: nx.DiGraph):
        nodes = list(G.nodes(data=True))

        for f in LOW_CARD_CATEGORICAL_FIELDS:
            counts: Dict[str, int] = {}
            for _, data in nodes:
                v = _safe_str(data.get(f))
                if v is None:
                    continue
                counts[v] = counts.get(v, 0) + 1
            categories = sorted(counts.keys())  # deterministic order
            strategy = "onehot" if len(categories) <= ONEHOT_MAX_CARDINALITY else "topk_freq"
            if strategy == "topk_freq":
                categories = sorted(counts, key=lambda c: (-counts[c], c))[:TOPK]
            self.vocabs[f] = FieldVocab(
                field_name=f, strategy=strategy, categories=categories,
                freq=counts, max_freq=max(counts.values()) if counts else 1,
            )

        for f in HIGH_CARD_CATEGORICAL_FIELDS:
            counts: Dict[str, int] = {}
            for _, data in nodes:
                v = _safe_str(data.get(f))
                if v is None:
                    continue
                counts[v] = counts.get(v, 0) + 1
            top_categories = sorted(counts, key=lambda c: (-counts[c], c))[:TOPK]
            self.vocabs[f] = FieldVocab(
                field_name=f, strategy="topk_freq", categories=top_categories,
                freq=counts, max_freq=max(counts.values()) if counts else 1,
            )

        # process-id-family frequency table (shared across process_id,
        # source_process_id, target_process_id -- they're the same kind of
        # OS identifier, so one frequency table over their union is more
        # data-efficient than three separate ones).
        pid_counts: Dict[str, int] = {}
        for _, data in nodes:
            for f in ("process_id", "source_process_id", "target_process_id"):
                v = _safe_str(data.get(f))
                if v is not None:
                    pid_counts[v] = pid_counts.get(v, 0) + 1
        self.pid_freq = pid_counts
        self.pid_max_freq = max(pid_counts.values()) if pid_counts else 1

        self._fitted = True
        self.feature_names = self._build_feature_names()

    # ---------------------------------------------------------- feature names
    def _build_feature_names(self) -> List[str]:
        names: List[str] = []
        for f in LOW_CARD_CATEGORICAL_FIELDS:
            names.extend(self.vocabs[f].column_names(f))
        for f in HIGH_CARD_CATEGORICAL_FIELDS:
            names.extend(self.vocabs[f].column_names(f))

        names.extend([
            "has_source_process", "has_target_process", "has_process_id",
            "source_process_id__freq_encoded", "target_process_id__freq_encoded", "process_id__freq_encoded",
            "has_network_fields",
            "source_port_bucket=WELL_KNOWN", "source_port_bucket=REGISTERED",
            "source_port_bucket=EPHEMERAL", "source_port_bucket=MISSING",
            "dest_port_bucket=WELL_KNOWN", "dest_port_bucket=REGISTERED",
            "dest_port_bucket=EPHEMERAL", "dest_port_bucket=MISSING",
            "has_user",
            "hour_sin", "hour_cos", "has_timestamp",
        ])
        return names

    # ----------------------------------------------------------- transform
    def transform(self, G: nx.DiGraph) -> Tuple[np.ndarray, List[str], List[str]]:
        """Returns (X, event_uid_order, feature_names). X is float32,
        shape [num_nodes, num_features]. event_uid_order[i] is the
        event_uid corresponding to row i of X."""
        if not self._fitted:
            raise RuntimeError("Call fit(G) before transform(G).")

        event_uid_order = sorted(G.nodes())  # deterministic: sort by event_uid string
        rows: List[List[float]] = []

        for node_id in event_uid_order:
            data = G.nodes[node_id]
            row: List[float] = []

            for f in LOW_CARD_CATEGORICAL_FIELDS:
                v = _safe_str(data.get(f))
                row.extend(self.vocabs[f].encode_value(v))
            for f in HIGH_CARD_CATEGORICAL_FIELDS:
                v = _safe_str(data.get(f))
                row.extend(self.vocabs[f].encode_value(v))

            has_source_process = 1.0 if data.get("source_process_id") is not None else 0.0
            has_target_process = 1.0 if data.get("target_process_id") is not None else 0.0
            has_process_id = 1.0 if data.get("process_id") is not None else 0.0
            row.extend([has_source_process, has_target_process, has_process_id])

            def pid_freq_enc(field_val):
                v = _safe_str(field_val)
                count = self.pid_freq.get(v, 0) if v is not None else 0
                return math.log1p(count) / math.log1p(self.pid_max_freq) if self.pid_max_freq > 0 else 0.0

            row.extend([
                pid_freq_enc(data.get("source_process_id")),
                pid_freq_enc(data.get("target_process_id")),
                pid_freq_enc(data.get("process_id")),
            ])

            has_network = 1.0 if data.get("source_ip") is not None else 0.0
            row.append(has_network)

            for portfield in ("source_port", "destination_port"):
                bucket = _port_bucket(data.get(portfield))
                row.extend([
                    1.0 if bucket == "WELL_KNOWN" else 0.0,
                    1.0 if bucket == "REGISTERED" else 0.0,
                    1.0 if bucket == "EPHEMERAL" else 0.0,
                    1.0 if bucket == "MISSING" else 0.0,
                ])

            has_user = 1.0 if data.get("user") is not None else 0.0
            row.append(has_user)

            ts_epoch = data.get("timestamp_epoch")
            if ts_epoch is not None:
                hour = (ts_epoch / 3600.0) % 24.0
                hour_sin = math.sin(2 * math.pi * hour / 24.0)
                hour_cos = math.cos(2 * math.pi * hour / 24.0)
                has_ts = 1.0
            else:
                hour_sin, hour_cos, has_ts = 0.0, 0.0, 0.0
            row.extend([hour_sin, hour_cos, has_ts])

            rows.append(row)

        X = np.asarray(rows, dtype=np.float32)
        return X, event_uid_order, self.feature_names

    # -------------------------------------------------------------- I/O
    def save(self, path: str):
        payload = {
            "onehot_max_cardinality": ONEHOT_MAX_CARDINALITY,
            "topk": TOPK,
            "low_card_fields": LOW_CARD_CATEGORICAL_FIELDS,
            "high_card_fields": HIGH_CARD_CATEGORICAL_FIELDS,
            "vocabs": {
                k: {
                    "field_name": v.field_name, "strategy": v.strategy,
                    "categories": v.categories, "freq": v.freq, "max_freq": v.max_freq,
                } for k, v in self.vocabs.items()
            },
            "pid_freq": self.pid_freq,
            "pid_max_freq": self.pid_max_freq,
            "feature_names": self.feature_names,
        }
        with open(path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "NodeFeatureBuilder":
        with open(path) as f:
            payload = json.load(f)
        builder = cls()
        builder.vocabs = {
            k: FieldVocab(**v) for k, v in payload["vocabs"].items()
        }
        builder.pid_freq = payload["pid_freq"]
        builder.pid_max_freq = payload["pid_max_freq"]
        builder.feature_names = payload["feature_names"]
        builder._fitted = True
        return builder


def networkx_to_pyg(G: nx.DiGraph, builder: NodeFeatureBuilder, symmetrize: bool = True):
    """
    Convert G (+ its fitted feature builder) into a PyTorch Geometric Data
    object with data.x and data.edge_index.

    DIRECTION DECISION (documented, not silent):
        The event graph is directed (edges point from the chronologically
        earlier event to the later one). Standard GCN layers (Kipf &
        Welling, used here) assume a symmetric, normalized adjacency and
        propagate information along BOTH directions of an edge implicitly.
        Feeding a purely directed edge_index into a plain GCNConv would
        silently only let information flow forward in time, which is not
        what we want for a first structural-embedding model (we want each
        node's embedding to reflect its full local context, not just what
        happened after it).

        We therefore symmetrize by default: for every directed edge (u, v)
        we add both (u, v) and (v, u) to edge_index before encoding. This
        is a deliberate simplification for the first GAE version -- it
        means the model does NOT learn from edge direction (i.e. it can't
        distinguish "A happened before B" from "B happened before A" at
        the structural level). The chronological ordering is not lost
        from the project overall -- it still lives in the original,
        UNCHANGED directed NetworkX graph `G`, and in the `time_difference`
        edge attribute -- it is simply not fed to this first encoder. A
        future iteration could switch to a directed-GNN layer or add
        edge_weight/edge_attr (we already export edge_type and
        edge_time_diff onto the Data object, unused by the current
        GCNConv encoder, for exactly that future extension).

        Pass symmetrize=False to get the raw directed edge_index instead
        (not recommended for the first GAE version -- see above).

    The input NetworkX graph G is NOT modified by this function.
    """
    import torch
    from torch_geometric.data import Data

    X, event_uid_order, feature_names = builder.transform(G)
    index_of = {uid: i for i, uid in enumerate(event_uid_order)}

    rel_to_code = {"SAME_PID": 0, "SAME_HOST_TEMPORAL": 1, "SAME_PID_AND_SAME_HOST_TEMPORAL": 2}

    src, dst, etype, etime = [], [], [], []
    for u, v, data in G.edges(data=True):
        rel = data.get("relationship", "").replace("+", "_AND_")
        code = rel_to_code.get(rel, -1)
        td = data.get("time_difference")
        td = float(td) if td is not None else -1.0

        src.append(index_of[u]); dst.append(index_of[v]); etype.append(code); etime.append(td)
        if symmetrize:
            src.append(index_of[v]); dst.append(index_of[u]); etype.append(code); etime.append(td)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(etype, dtype=torch.long)
    edge_time_diff = torch.tensor(etime, dtype=torch.float32)

    data_obj = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=edge_index,
    )
    # Stored but NOT consumed by the v1 GCNConv encoder -- see docstring.
    data_obj.edge_type = edge_type
    data_obj.edge_time_diff = edge_time_diff

    return data_obj, event_uid_order, feature_names


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from parser import parse_log_file
    from normalizer import normalize_records
    from graph_builder import build_event_graph

    records, _ = parse_log_file("data/sample_2000_events.json")
    events, _ = normalize_records(records)
    G = build_event_graph(events)

    builder = NodeFeatureBuilder()
    builder.fit(G)
    X, order, names = builder.transform(G)
    print("X shape:", X.shape)
    print("num feature names:", len(names))
    print("first 10 feature names:", names[:10])

    data_obj, order2, names2 = networkx_to_pyg(G, builder)
    print("PyG Data:", data_obj)
