"""
src/noise/review.py
---------------------
The noise-event review path. HDBSCAN noise (cluster_id == -1) is NOT dropped:
every unclustered event is kept with its attributes, graph neighbours,
novelty signals and the measurable properties that are consistent with it
falling outside every dense region.

What this module does NOT do:
  * assume noise is malicious - novelty/structural unusualness is not intent;
  * invent a per-event cause: HDBSCAN emits no reason for noise, so the
    "signals" listed are MEASURED properties (isolated in the graph, far from
    every cluster medoid, poorly reconstructed by the GAE, a rare EventID...)
    that are consistent with - not proof of - why no cluster formed;
  * assign noise to a normal cluster. Noise units carry scope "noise_group"
    or "noise_event" for the rest of the pipeline.

Review units (what later stages analyse semantically):
  * noise_group : >= noise_group_min_size unclustered events connected by the
                  chosen relationship type(s) in the event graph (default
                  SAME_PID: the same process id), up to noise_group_max_size.
  * noise_event : any other single unclustered event (isolated, or part of a
                  related set too large to be treated as one coherent group).
Units are ranked by the existing novelty score (ties: structural signals).
"""

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import networkx as nx
import numpy as np

EXPLANATION = (
    "HDBSCAN labels a point as noise when it is not part of any sufficiently dense region "
    "(min_cluster_size={mcs}, min_samples={ms}); it does not report a per-point reason. The signals "
    "below are measured properties consistent with that outcome, not proof of cause, and noise is "
    "not evidence of malicious activity."
)


def _percentile_threshold(values: Iterable[float], pct: float = 90.0) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.percentile(arr, pct)) if arr.size else 1.0


def _edge_has_type(data: Dict[str, Any], types: Sequence[str]) -> bool:
    rel = str(data.get("relationship", ""))
    return any(t in rel for t in types)


def review_noise(row_to_uid: List[str], labels: np.ndarray, G: nx.DiGraph, events_by_uid: Dict[str, Dict],
                 novelty_rows: List[Dict[str, Any]], min_cluster_size: int, min_samples: int,
                 schema_builder=None, group_edge_types: Sequence[str] = ("SAME_PID",),
                 group_min_size: int = 2, group_max_size: int = 30) -> Dict[str, Any]:
    labels = np.asarray(labels)
    uid_to_label = {uid: int(labels[i]) for i, uid in enumerate(row_to_uid)}
    noise_uids = [u for u in row_to_uid if uid_to_label[u] == -1]
    nov = {r["event_uid"]: r for r in novelty_rows}
    U = G.to_undirected()
    eid_counts = Counter(events_by_uid[u].get("event_id") for u in row_to_uid if u in events_by_uid)

    # run-wide percentile cut-offs used to word the "signals"
    p_dist = _percentile_threshold(r.get("embedding_distance_score", 0.0) for r in novelty_rows)
    p_recon = _percentile_threshold(r.get("reconstruction_error", 0.0) for r in novelty_rows)
    p_rarity = _percentile_threshold(r.get("graph_rarity_score", 0.0) for r in novelty_rows)

    events: List[Dict[str, Any]] = []
    for uid in noise_uids:
        ev = events_by_uid.get(uid, {})
        n = nov.get(uid, {})
        nbrs = []
        for v in U.neighbors(uid):
            ed = U.get_edge_data(uid, v) or {}
            nbrs.append({"event_uid": v, "relationship": ed.get("relationship"),
                         "cluster_id": uid_to_label.get(v), "is_noise": uid_to_label.get(v) == -1})
        degree = len(nbrs)
        signals: List[Dict[str, str]] = [{"signal": "no_dense_region",
                                          "detail": "membership probability 0: not part of any HDBSCAN cluster"}]
        if degree == 0:
            signals.append({"signal": "graph_isolated", "detail": "no SAME_PID / SAME_HOST_TEMPORAL edges"})
        if n and n.get("embedding_distance_score", 0.0) >= p_dist and p_dist > 0:
            signals.append({"signal": "far_from_cluster_medoids",
                            "detail": f"embedding distance score {n['embedding_distance_score']:.2f} "
                                      f"(run 90th percentile {p_dist:.2f})"})
        if n and n.get("reconstruction_error", 0.0) >= p_recon and p_recon > 0:
            signals.append({"signal": "poorly_reconstructed",
                            "detail": f"GAE reconstruction error {n['reconstruction_error']:.2f} "
                                      f"(run 90th percentile {p_recon:.2f})"})
        if n and n.get("graph_rarity_score", 0.0) >= p_rarity and p_rarity > 0:
            signals.append({"signal": "structurally_rare",
                            "detail": f"graph rarity score {n['graph_rarity_score']:.2f} "
                                      f"(run 90th percentile {p_rarity:.2f})"})
        cnt = eid_counts.get(ev.get("event_id"), 0)
        if ev.get("event_id") is not None and cnt < min_cluster_size:
            signals.append({"signal": "rare_event_type",
                            "detail": f"only {cnt} event(s) with EventID {ev.get('event_id')} in this run "
                                      f"(< min_cluster_size={min_cluster_size}), so no cluster of this type can form"})
        if schema_builder is not None:
            unseen = []
            for f in ("event_id", "source_image", "target_image", "user", "hostname"):
                vocab = schema_builder.vocabs.get(f)
                val = ev.get(f)
                if vocab is not None and val is not None and str(val) not in set(vocab.categories):
                    unseen.append(f)
            if unseen:
                signals.append({"signal": "unseen_feature_values",
                                "detail": "value(s) outside the pretrained schema's vocabulary: " + ", ".join(unseen)})
        events.append({
            "event_uid": uid, "event_id": ev.get("event_id"), "timestamp": ev.get("timestamp"),
            "hostname": ev.get("hostname"), "source_image": ev.get("source_image"),
            "target_image": ev.get("target_image"), "destination_ip": ev.get("destination_ip"),
            "user": ev.get("user"),
            "novelty_score": n.get("novelty_score"), "novelty_label": n.get("novelty_label"),
            "density_score": n.get("density_score"), "embedding_distance_score": n.get("embedding_distance_score"),
            "reconstruction_error": n.get("reconstruction_error"), "graph_rarity_score": n.get("graph_rarity_score"),
            "graph_degree": degree, "neighbors": nbrs[:12], "neighbor_count": degree,
            "neighbor_clusters": sorted({x["cluster_id"] for x in nbrs if x["cluster_id"] not in (None, -1)}),
            "signals": signals,
        })

    events.sort(key=lambda e: (-(e["novelty_score"] or 0.0), -(e["graph_rarity_score"] or 0.0),
                               -(e["embedding_distance_score"] or 0.0), e["event_uid"]))
    for rank, e in enumerate(events, start=1):
        e["review_rank"] = rank

    # ---- review units: related groups + singletons ----
    noise_set: Set[str] = set(noise_uids)
    H = nx.Graph()
    H.add_nodes_from(noise_set)
    for u, v, d in G.edges(data=True):
        if u in noise_set and v in noise_set and _edge_has_type(d, group_edge_types):
            H.add_edge(u, v)
    comps = [sorted(c) for c in nx.connected_components(H)]
    rank_of = {e["event_uid"]: e["review_rank"] for e in events}
    group_sets, loose = [], []
    oversized = 0
    if group_min_size <= 1:                                   # degenerate setting: every event is its own unit
        loose = sorted(noise_set)
    else:
        for c in comps:
            if len(c) > group_max_size:
                oversized += 1                                # too large to be one coherent "small group"
                loose.extend(c)
            elif len(c) >= group_min_size:
                group_sets.append(c)
            else:
                loose.extend(c)

    def unit_record(uids: List[str], scope: str) -> Dict[str, Any]:
        scores = [nov.get(u, {}).get("novelty_score", 0.0) for u in uids]
        ncl = Counter()
        for u in uids:
            for v in U.neighbors(u):
                if uid_to_label.get(v, -1) != -1:
                    ncl[uid_to_label[v]] += 1
        best = min(uids, key=lambda u: rank_of[u])
        return {"scope": scope, "event_uids": sorted(uids, key=lambda u: rank_of[u]), "size": len(uids),
                "max_novelty": round(max(scores), 4) if scores else 0.0,
                "mean_novelty": round(float(np.mean(scores)), 4) if scores else 0.0,
                "top_review_rank": rank_of[best], "adjacent_clusters": dict(sorted(ncl.items()))}

    units = [unit_record(c, "noise_group") for c in group_sets] + [unit_record([u], "noise_event") for u in loose]
    units.sort(key=lambda r: (-r["max_novelty"], r["top_review_rank"]))
    gi = ei = 0
    for r in units:
        if r["scope"] == "noise_group":
            gi += 1
            r["unit_id"] = f"N{gi}"
        else:
            ei += 1
            r["unit_id"] = f"E{ei}"
    label_counts = Counter(e["novelty_label"] for e in events)
    sig_counts = Counter(s["signal"] for e in events for s in e["signals"])
    return {
        "events": events,
        "units": units,
        "stats": {
            "noise_events": len(events),
            "noise_share_pct": round(100.0 * len(events) / max(len(row_to_uid), 1), 2),
            "groups": sum(1 for r in units if r["scope"] == "noise_group"),
            "grouped_events": sum(r["size"] for r in units if r["scope"] == "noise_group"),
            "single_event_units": sum(1 for r in units if r["scope"] == "noise_event"),
            "oversized_related_sets_treated_as_individual_events": oversized,
            "group_edge_types": list(group_edge_types),
            "group_size_range": [group_min_size, group_max_size],
            "novelty_label_counts": dict(label_counts),
            "signal_counts": dict(sig_counts),
            "isolated_in_graph": sum(1 for e in events if e["graph_degree"] == 0),
            "with_clustered_neighbors": sum(1 for e in events if e["neighbor_clusters"]),
        },
        "explanation": EXPLANATION.format(mcs=min_cluster_size, ms=min_samples),
    }
