"""
novelty_detection.py
----------------------
STAGE D: Novelty / anomaly detection.

Produces a defensible, multi-signal novelty score per event -- NOT a
claim of malicious activity. "Noise" from HDBSCAN, a low reconstruction
score, or an unusual graph neighborhood are all EVIDENCE OF STATISTICAL/
STRUCTURAL UNUSUALNESS, nothing more. This module never writes "ATTACK"
anywhere in its output.

Four signals are computed, each independently defensible and documented
below, then combined by an UNWEIGHTED AVERAGE by default (weights are
configurable, but arbitrary hand-tuned weights are not used without
justification -- see docstrings on each score for why equal weighting is
the principled default absent labeled ground truth to calibrate against).

    1. density_score            -- from existing clustering output
    2. embedding_distance_score -- from existing embeddings + cluster medoids
    3. reconstruction_score      -- from the existing (not retrained) GAE decoder
    4. graph_rarity_score        -- from the existing (read-only) NetworkX graph

Usage:
    python src/novelty_detection.py \
        --embeddings output/gae_embeddings.npy \
        --mapping output/gae_node_mapping.json \
        --clusters output/embedding_clusters.csv \
        --input data/sample_2000_events.json
"""

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np


# ============================================================
# 1. density_score -- reuses clustering output, invents nothing new
# ============================================================

def compute_density_score(cluster_rows: List[Dict]) -> Dict[str, float]:
    """
    density_score = 1 - membership_probability.

    Rationale: HDBSCAN's membership_probability already IS a
    density-based confidence that a point belongs to its assigned dense
    region (0 for noise, up to 1 for a point deep inside a cluster). We
    do not recompute density ourselves -- we reuse the clustering
    algorithm's own output, inverted so that higher = more novel/unusual,
    consistent with the other three scores below.
    """
    return {r["event_uid"]: 1.0 - r["membership_probability"] for r in cluster_rows}


# ============================================================
# 2. embedding_distance_score -- distance to nearest cluster medoid
# ============================================================

def compute_embedding_distance_score(Z: np.ndarray, row_to_uid: List[str],
                                       cluster_rows: List[Dict],
                                       medoid_row_by_cluster: Dict[int, int]) -> Dict[str, float]:
    """
    For every event, distance (Euclidean, in the raw 32D latent space) to
    the NEAREST cluster medoid, min-max normalized across all events to
    [0, 1].

    Rationale: even a point technically assigned to a cluster can sit
    near that cluster's edge; and noise points still have a "nearest
    normal behavior" reference point. This is a distance-based novelty
    signal independent of the clustering algorithm's own internal
    probability estimate (signal 1 above), so combining the two isn't
    circular -- one comes from HDBSCAN's density estimate, the other from
    a plain geometric distance in the embedding space.
    """
    if not medoid_row_by_cluster:
        return {uid: 0.0 for uid in row_to_uid}

    medoid_rows = list(medoid_row_by_cluster.values())
    medoid_vectors = Z[medoid_rows]  # [n_clusters, latent_dim]

    # distance from every row to every medoid, then take the min per row
    # (O(N * n_clusters), not O(N^2) -- n_clusters is small)
    dists = np.linalg.norm(Z[:, None, :] - medoid_vectors[None, :, :], axis=2)  # [N, n_clusters]
    min_dist = dists.min(axis=1)  # [N]

    d_min, d_max = float(min_dist.min()), float(min_dist.max())
    if d_max > d_min:
        norm = (min_dist - d_min) / (d_max - d_min)
    else:
        norm = np.zeros_like(min_dist)

    return {row_to_uid[i]: float(norm[i]) for i in range(len(row_to_uid))}


# ============================================================
# 3. reconstruction_score -- from the EXISTING, NOT retrained, GAE decoder
# ============================================================

def compute_reconstruction_score(Z: np.ndarray, row_to_uid: List[str], G) -> Dict[str, Tuple[float, bool]]:
    """
    Per-node reconstruction score using the SAME inner-product decoder
    the trained GAE already uses (A_hat[i,j] = sigmoid(z_i . z_j)) -- we
    do NOT retrain anything; the decoder has no learnable parameters of
    its own (it's a fixed inner product), so we can apply it directly to
    the already-saved Z from gae_embeddings.npy.

    For each node, we look at its REAL edges in the existing (unmodified)
    NetworkX graph and compute how well the decoder predicts them:

        node_reconstruction = mean( sigmoid(z_i . z_j) for each real
                                     neighbor j of node i )

    A node whose real edges are all predicted with high probability is
    "well explained" by the graph structure the GAE learned -- reusing
    the SAME reconstruction concept the GAE's AUC/AP metric already
    evaluates (per-edge), just aggregated per-node instead of over the
    whole graph. This is NOT the same as, and should not be reported as,
    the model's graph-level AUC/AP.

    reconstruction_score returned = 1 - node_reconstruction (so higher =
    less well explained = more novel, consistent with the other 3 scores).

    Nodes with zero edges (isolated) get reconstruction_score = 1.0
    (maximally novel by this signal) with has_edges=False flagged, since
    "no evidence this node fits any known relationship" is itself a
    (weak) novelty signal, not an error.
    """
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    uid_to_row = {uid: i for i, uid in enumerate(row_to_uid)}
    scores: Dict[str, Tuple[float, bool]] = {}

    # Use the undirected view so a node's reconstruction reflects ALL its
    # correlations regardless of which direction the original directed
    # edge pointed (consistent with how the GAE was trained on the
    # symmetrized graph -- see gae_features.networkx_to_pyg).
    G_undirected = G.to_undirected()

    for uid in row_to_uid:
        i = uid_to_row[uid]
        neighbors = list(G_undirected.neighbors(uid))
        if not neighbors:
            scores[uid] = (1.0, False)
            continue
        neighbor_rows = [uid_to_row[n] for n in neighbors if n in uid_to_row]
        if not neighbor_rows:
            scores[uid] = (1.0, False)
            continue
        z_i = Z[i]
        z_js = Z[neighbor_rows]
        probs = sigmoid(z_js @ z_i)
        mean_prob = float(np.mean(probs))
        scores[uid] = (1.0 - mean_prob, True)

    return scores


# ============================================================
# 4. graph_rarity_score -- from the EXISTING, READ-ONLY NetworkX graph
# ============================================================

def compute_graph_rarity_score(G) -> Dict[str, float]:
    """
    Structural rarity from the existing graph, computed on an undirected
    VIEW (G.to_undirected() creates a new graph object; the original
    directed G passed in is never mutated).

    Two components, averaged:
      (a) degree_rarity: |degree - median_degree| / max(|degree - median_degree|)
          -- both unusually LOW-degree (isolated-ish) and unusually
          HIGH-degree (hub) nodes are structurally atypical relative to
          the typical node in this chain-based, low-degree-by-design
          graph (recall: avg degree ~2.26 by construction).
      (b) neighbor_event_type_diversity: number of DISTINCT event_ids
          among a node's neighbors, normalized by the max observed --
          a node whose neighborhood mixes many different event types is
          structurally more heterogeneous (e.g. bridges process-access
          and network-connection activity) than one whose neighbors are
          all the same EventID.
    """
    G_undirected = G.to_undirected()
    degrees = dict(G_undirected.degree())
    degree_values = np.array(list(degrees.values()), dtype=float)
    median_degree = float(np.median(degree_values)) if len(degree_values) else 0.0
    abs_dev = np.abs(degree_values - median_degree)
    max_dev = float(abs_dev.max()) if abs_dev.max() > 0 else 1.0

    diversity_raw = {}
    for node in G_undirected.nodes():
        neighbors = list(G_undirected.neighbors(node))
        event_ids = set()
        for n in neighbors:
            eid = G_undirected.nodes[n].get("event_id")
            if eid is not None:
                event_ids.add(eid)
        diversity_raw[node] = len(event_ids)
    max_diversity = max(diversity_raw.values()) if diversity_raw else 1
    max_diversity = max(max_diversity, 1)

    scores = {}
    node_list = list(G_undirected.nodes())
    for idx, node in enumerate(node_list):
        deg_rarity = abs(degrees[node] - median_degree) / max_dev
        diversity_score = diversity_raw[node] / max_diversity
        scores[node] = float(0.5 * deg_rarity + 0.5 * diversity_score)

    return scores


# ============================================================
# combination + labeling
# ============================================================

def combine_novelty_scores(density: Dict[str, float], emb_dist: Dict[str, float],
                            recon: Dict[str, Tuple[float, bool]], rarity: Dict[str, float],
                            weights: Dict[str, float]) -> List[Dict]:
    rows = []
    all_uids = set(density) | set(emb_dist) | set(recon) | set(rarity)
    for uid in all_uids:
        d = density.get(uid, 0.0)
        e = emb_dist.get(uid, 0.0)
        r, has_edges = recon.get(uid, (0.0, False))
        g = rarity.get(uid, 0.0)

        total_weight = sum(weights.values())
        score = (weights["density"] * d + weights["embedding_distance"] * e +
                 weights["reconstruction"] * r + weights["graph_rarity"] * g) / total_weight
        score = max(0.0, min(1.0, score))

        rows.append({
            "event_uid": uid,
            "density_score": round(d, 4),
            "embedding_distance_score": round(e, 4),
            "reconstruction_error": round(r, 4),
            "reconstruction_has_edges": has_edges,
            "graph_rarity_score": round(g, 4),
            "novelty_score": round(score, 4),
        })
    return rows


def label_novelty(score: float, thresholds: Tuple[float, float, float]) -> str:
    low, mid, high = thresholds
    if score < low:
        return "NORMAL"
    if score < mid:
        return "LOW_NOVELTY"
    if score < high:
        return "SUSPICIOUS"
    return "HIGH_NOVELTY"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="output/gae_embeddings.npy")
    ap.add_argument("--mapping", default="output/gae_node_mapping.json")
    ap.add_argument("--clusters", default="output/embedding_clusters.csv")
    ap.add_argument("--cluster-summary", default="output/cluster_summary.json")
    ap.add_argument("--input", default="data/sample_2000_events.json",
                     help="Same raw input used for train_gae.py, to rebuild the read-only NetworkX graph.")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--weight-density", type=float, default=1.0)
    ap.add_argument("--weight-embedding-distance", type=float, default=1.0)
    ap.add_argument("--weight-reconstruction", type=float, default=1.0)
    ap.add_argument("--weight-graph-rarity", type=float, default=1.0)
    ap.add_argument("--threshold-low", type=float, default=0.25)
    ap.add_argument("--threshold-mid", type=float, default=0.50)
    ap.add_argument("--threshold-high", type=float, default=0.75)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    weights = {
        "density": args.weight_density,
        "embedding_distance": args.weight_embedding_distance,
        "reconstruction": args.weight_reconstruction,
        "graph_rarity": args.weight_graph_rarity,
    }
    print(f"Novelty score weights (unweighted-equal by default, documented in module docstring): {weights}")
    print(f"Novelty labeling thresholds: NORMAL < {args.threshold_low} <= LOW_NOVELTY < {args.threshold_mid} "
          f"<= SUSPICIOUS < {args.threshold_high} <= HIGH_NOVELTY")

    print("\nLoading embeddings + mapping ...")
    Z = np.load(args.embeddings)
    with open(args.mapping) as f:
        mapping = json.load(f)
    row_to_uid = mapping["row_to_event_uid"]

    print("Loading clustering output ...")
    from profile_clusters import load_clusters
    cluster_rows = load_clusters(args.clusters)
    with open(args.cluster_summary) as f:
        cluster_summary = json.load(f)
    medoid_row_by_cluster = {int(k): v for k, v in cluster_summary.get("cluster_medoid_row_index", {}).items()}

    print(f"Re-deriving the read-only NetworkX graph from {args.input} "
          f"(same unmodified parser/normalizer/graph_builder -- this graph is NOT mutated) ...")
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from parser import parse_log_file
    from normalizer import normalize_records
    from graph_builder import build_event_graph

    records, _ = parse_log_file(args.input)
    events, _ = normalize_records(records)
    G = build_event_graph(events)

    print("\n[1/4] Computing density_score (from clustering membership probability) ...")
    density = compute_density_score(cluster_rows)

    print("[2/4] Computing embedding_distance_score (distance to nearest cluster medoid) ...")
    emb_dist = compute_embedding_distance_score(Z, row_to_uid, cluster_rows, medoid_row_by_cluster)

    print("[3/4] Computing reconstruction_score (existing GAE decoder, NOT retrained) ...")
    recon = compute_reconstruction_score(Z, row_to_uid, G)

    print("[4/4] Computing graph_rarity_score (existing, read-only NetworkX graph) ...")
    rarity = compute_graph_rarity_score(G)

    print("\nCombining signals ...")
    rows = combine_novelty_scores(density, emb_dist, recon, rarity, weights)

    cluster_by_uid = {r["event_uid"]: r for r in cluster_rows}
    for row in rows:
        c = cluster_by_uid.get(row["event_uid"], {})
        row["embedding_index"] = c.get("embedding_index")
        row["cluster_id"] = c.get("cluster_id")
        row["is_noise"] = c.get("is_noise")
        row["novelty_label"] = label_novelty(row["novelty_score"], (args.threshold_low, args.threshold_mid, args.threshold_high))

    rows.sort(key=lambda r: -r["novelty_score"])

    csv_path = os.path.join(args.output_dir, "novelty_scores.csv")
    fieldnames = ["event_uid", "embedding_index", "cluster_id", "is_noise",
                  "density_score", "embedding_distance_score", "reconstruction_error",
                  "reconstruction_has_edges", "graph_rarity_score", "novelty_score", "novelty_label"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    print(f"\nWrote {csv_path}")

    label_counts = {}
    for row in rows:
        label_counts[row["novelty_label"]] = label_counts.get(row["novelty_label"], 0) + 1

    print("\nNovelty label distribution:")
    for label in ["NORMAL", "LOW_NOVELTY", "SUSPICIOUS", "HIGH_NOVELTY"]:
        print(f"  {label:14s} {label_counts.get(label, 0)}")

    print("\nTop 10 highest novelty_score events:")
    for row in rows[:10]:
        print(f"  {row['event_uid']}  score={row['novelty_score']:.3f}  label={row['novelty_label']}  "
              f"cluster={row['cluster_id']}  is_noise={row['is_noise']}")

    print("\nIMPORTANT: novelty_score reflects statistical/structural unusualness relative to this "
          "dataset. It is NOT a maliciousness classification. See docs/phase2.md for full discussion.")


if __name__ == "__main__":
    main()
