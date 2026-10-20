"""
cluster_embeddings.py
-----------------------
STAGE A: Load and validate the GAE embeddings + event_uid mapping.
STAGE B: Density-based clustering (HDBSCAN, DBSCAN fallback) on the raw
         32-dimensional latent embeddings -- NEVER on the 2D PCA
         projection, which is for visualization only.

Consumes existing outputs unchanged:
    output/gae_embeddings.npy
    output/gae_node_mapping.json
    output/normalized_events.json

Does NOT modify graph_builder.py, gae_features.py, gae_model.py, or
train_gae.py, and does NOT retrain the GAE.

Usage:
    python src/cluster_embeddings.py \
        --embeddings output/gae_embeddings.npy \
        --mapping output/gae_node_mapping.json \
        --events output/normalized_events.json \
        --min-cluster-size 20 \
        --min-samples 10
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np


# ============================================================
# STAGE A -- load + validate
# ============================================================

class ValidationError(Exception):
    pass


def load_and_validate(embeddings_path: str, mapping_path: str, events_path: str) -> Tuple[np.ndarray, Dict, List[Dict]]:
    """
    Loads the three existing artifacts and cross-checks their row counts
    and event_uid sets agree. Fails LOUDLY (raises ValidationError) on any
    mismatch -- this function never silently truncates or reorders data
    to make counts agree, per the project brief.
    """
    if not os.path.exists(embeddings_path):
        raise ValidationError(f"Embeddings file not found: {embeddings_path}")
    if not os.path.exists(mapping_path):
        raise ValidationError(f"Node mapping file not found: {mapping_path}")
    if not os.path.exists(events_path):
        raise ValidationError(f"Normalized events file not found: {events_path}")

    Z = np.load(embeddings_path)
    with open(mapping_path) as f:
        mapping = json.load(f)
    with open(events_path) as f:
        events = json.load(f)

    row_to_uid = mapping["row_to_event_uid"]
    event_uid_to_row = mapping["event_uid_to_row"]

    report_lines = []
    report_lines.append(f"Embedding shape: {Z.shape}")
    report_lines.append(f"Number of mapped event_uids: {len(row_to_uid)}")
    report_lines.append(f"Number of normalized events: {len(events)}")

    problems = []

    if Z.shape[0] != len(row_to_uid):
        problems.append(
            f"Embedding row count ({Z.shape[0]}) != mapping length ({len(row_to_uid)})"
        )

    if len(row_to_uid) != len(events):
        problems.append(
            f"Mapping length ({len(row_to_uid)}) != normalized event count ({len(events)})"
        )

    mapping_uid_set = set(row_to_uid)
    event_uid_set = set(e["event_uid"] for e in events)
    if mapping_uid_set != event_uid_set:
        only_in_mapping = mapping_uid_set - event_uid_set
        only_in_events = event_uid_set - mapping_uid_set
        problems.append(
            f"event_uid sets differ: {len(only_in_mapping)} uids only in mapping, "
            f"{len(only_in_events)} uids only in normalized_events.json"
        )

    # event_uid_to_row must be the exact inverse of row_to_event_uid
    for uid, row in event_uid_to_row.items():
        if row_to_uid[row] != uid:
            problems.append(
                f"event_uid_to_row is not the inverse of row_to_event_uid at uid={uid} (row {row})"
            )
            break

    if np.isnan(Z).any():
        problems.append("Embeddings contain NaN values")

    if problems:
        report_lines.append("Mapping consistency: FAIL")
        report_lines.append("Problems:")
        for p in problems:
            report_lines.append(f"  - {p}")
        print("\n".join(report_lines))
        raise ValidationError("Embedding/mapping/event validation failed -- see report above. "
                               "Refusing to proceed with clustering on inconsistent data.")

    report_lines.append("Mapping consistency: PASS")
    print("\n".join(report_lines))

    return Z, mapping, events


# ============================================================
# STAGE B -- density-based clustering
# ============================================================

def cluster_embeddings(Z: np.ndarray, min_cluster_size: int, min_samples: int,
                        algorithm: str = "hdbscan", dbscan_eps: float = 0.5,
                        random_state: int = 42):
    """
    Cluster the raw latent embeddings Z (NOT a PCA projection).

    Returns (labels, probabilities, medoids, algorithm_used).
        labels: np.ndarray[int], -1 = noise
        probabilities: np.ndarray[float] in [0,1], cluster membership
            strength (HDBSCAN only; DBSCAN doesn't produce this, so we
            fill with 1.0 for cluster members / 0.0 for noise -- documented,
            not a fabricated confidence value)
        medoids: dict[cluster_id -> row index of the medoid embedding],
            or {} if not computable
    """
    np.random.seed(random_state)  # HDBSCAN/DBSCAN in sklearn are deterministic given data+params,
                                    # this seed only guards any incidental randomness in tie-breaking.

    algorithm_used = algorithm
    medoids: Dict[int, int] = {}

    if algorithm == "hdbscan":
        try:
            from sklearn.cluster import HDBSCAN
        except ImportError:
            print("sklearn.cluster.HDBSCAN not available in this environment -- falling back to DBSCAN.")
            algorithm_used = "dbscan"

    if algorithm_used == "hdbscan":
        from sklearn.cluster import HDBSCAN
        model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                         store_centers="medoid", copy=True)
        labels = model.fit_predict(Z)
        probabilities = model.probabilities_
        # medoids_ is an array of shape [n_clusters, n_features] in cluster-label
        # order (0..n_clusters-1, noise excluded) -- map back to a representative
        # ROW INDEX in Z (nearest embedding to each medoid vector), since that's
        # what downstream stages (profiling, MITRE mapping) need to look up an
        # actual event_uid.
        unique_clusters = sorted(c for c in set(labels) if c != -1)
        for i, cluster_id in enumerate(unique_clusters):
            medoid_vec = model.medoids_[i]
            cluster_rows = np.where(labels == cluster_id)[0]
            dists = np.linalg.norm(Z[cluster_rows] - medoid_vec, axis=1)
            medoids[int(cluster_id)] = int(cluster_rows[np.argmin(dists)])
    else:
        from sklearn.cluster import DBSCAN
        model = DBSCAN(eps=dbscan_eps, min_samples=min_samples)
        labels = model.fit_predict(Z)
        # DBSCAN has no native membership-probability concept. We do NOT
        # invent one; membership is binary (assigned vs. noise), so we
        # report 1.0/0.0 and document this explicitly rather than
        # pretending it's a calibrated probability.
        probabilities = np.where(labels == -1, 0.0, 1.0)

        unique_clusters = sorted(c for c in set(labels) if c != -1)
        for cluster_id in unique_clusters:
            cluster_rows = np.where(labels == cluster_id)[0]
            centroid = Z[cluster_rows].mean(axis=0)
            dists = np.linalg.norm(Z[cluster_rows] - centroid, axis=1)
            medoids[int(cluster_id)] = int(cluster_rows[np.argmin(dists)])

    return labels, probabilities, medoids, algorithm_used


def summarize_clusters(labels: np.ndarray, probabilities: np.ndarray) -> Dict:
    n_total = len(labels)
    unique_clusters = sorted(c for c in set(labels) if c != -1)
    n_clusters = len(unique_clusters)
    n_noise = int(np.sum(labels == -1))

    sizes = {int(c): int(np.sum(labels == c)) for c in unique_clusters}
    sorted_sizes = sorted(sizes.items(), key=lambda kv: -kv[1])

    return {
        "total_events": int(n_total),
        "number_of_clusters": n_clusters,
        "number_of_noise_points": n_noise,
        "percentage_noise": round(100.0 * n_noise / n_total, 2) if n_total else 0.0,
        "percentage_clustered": round(100.0 * (n_total - n_noise) / n_total, 2) if n_total else 0.0,
        "cluster_sizes": sizes,
        "largest_clusters": sorted_sizes[:5],
        "smallest_clusters": sorted_sizes[-5:] if sorted_sizes else [],
        "mean_membership_probability_clustered": (
            round(float(np.mean(probabilities[labels != -1])), 4) if n_total > n_noise else None
        ),
    }


def compute_cluster_quality_metrics(Z: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Silhouette / Davies-Bouldin / Calinski-Harabasz, computed ONLY on
    non-noise points, and ONLY if there are >= 2 clusters with >= 2 points
    each (all three metrics are mathematically undefined or meaningless
    otherwise -- e.g. silhouette requires at least 2 clusters; a single
    cluster or all-noise result would either crash or produce a number
    that looks like a score but means nothing).
    """
    non_noise_mask = labels != -1
    non_noise_labels = labels[non_noise_mask]
    unique_clusters = set(non_noise_labels)

    result = {
        "points_evaluated": int(non_noise_mask.sum()),
        "points_excluded_as_noise": int((~non_noise_mask).sum()),
        "num_clusters_evaluated": len(unique_clusters),
    }

    if len(unique_clusters) < 2 or non_noise_mask.sum() < 3:
        result["status"] = "NOT_COMPUTED"
        result["reason"] = (
            f"Only {len(unique_clusters)} non-noise cluster(s) with {int(non_noise_mask.sum())} "
            "total non-noise points -- silhouette/Davies-Bouldin/Calinski-Harabasz all require "
            "at least 2 clusters to be mathematically meaningful. Reporting a number here would "
            "be misleading, not informative."
        )
        return result

    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    Z_nn = Z[non_noise_mask]
    result["status"] = "COMPUTED"
    result["silhouette_score"] = round(float(silhouette_score(Z_nn, non_noise_labels)), 4)
    result["davies_bouldin_index"] = round(float(davies_bouldin_score(Z_nn, non_noise_labels)), 4)
    result["calinski_harabasz_index"] = round(float(calinski_harabasz_score(Z_nn, non_noise_labels)), 4)
    result["interpretation_note"] = (
        "These describe how geometrically separated the learned clusters are in the 32D latent "
        "space -- they say NOTHING about attack-detection accuracy or whether a cluster is "
        "malicious. Noise points were excluded from all three metrics (see points_excluded_as_noise)."
    )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="output/gae_embeddings.npy")
    ap.add_argument("--mapping", default="output/gae_node_mapping.json")
    ap.add_argument("--events", default="output/normalized_events.json")
    ap.add_argument("--algorithm", choices=["hdbscan", "dbscan"], default="hdbscan")
    ap.add_argument("--min-cluster-size", type=int, default=20)
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--dbscan-eps", type=float, default=0.5, help="Only used if --algorithm dbscan")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="output")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("STAGE A: Load + validate embeddings/mapping/events")
    print("=" * 60)
    Z, mapping, events = load_and_validate(args.embeddings, args.mapping, args.events)
    row_to_uid = mapping["row_to_event_uid"]

    print("\n" + "=" * 60)
    print(f"STAGE B: Density-based clustering ({args.algorithm})")
    print("=" * 60)
    print(f"Parameters: min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples}"
          + (f", eps={args.dbscan_eps}" if args.algorithm == "dbscan" else ""))

    labels, probabilities, medoids, algorithm_used = cluster_embeddings(
        Z, args.min_cluster_size, args.min_samples,
        algorithm=args.algorithm, dbscan_eps=args.dbscan_eps, random_state=args.seed,
    )
    if algorithm_used != args.algorithm:
        print(f"NOTE: requested '{args.algorithm}' but actually used '{algorithm_used}' (fallback).")

    summary = summarize_clusters(labels, probabilities)
    summary["algorithm"] = algorithm_used
    summary["parameters"] = {
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "dbscan_eps": args.dbscan_eps if algorithm_used == "dbscan" else None,
        "seed": args.seed,
    }
    summary["cluster_medoid_row_index"] = medoids

    print(f"\nComputing cluster quality metrics (excluding noise) ...")
    summary["quality_metrics"] = compute_cluster_quality_metrics(Z, labels)
    qm = summary["quality_metrics"]
    if qm["status"] == "COMPUTED":
        print(f"  silhouette={qm['silhouette_score']}  davies_bouldin={qm['davies_bouldin_index']}  "
              f"calinski_harabasz={qm['calinski_harabasz_index']}  (n={qm['points_evaluated']} non-noise points)")
    else:
        print(f"  NOT_COMPUTED: {qm['reason']}")


    print(f"\nTotal events:        {summary['total_events']}")
    print(f"Clusters found:      {summary['number_of_clusters']}")
    print(f"Noise points:        {summary['number_of_noise_points']} "
          f"({summary['percentage_noise']}%)")
    print(f"Clustered:           {summary['percentage_clustered']}%")
    print(f"Largest clusters:    {summary['largest_clusters']}")

    # ---- write embedding_clusters.csv ----
    import csv
    csv_path = os.path.join(args.output_dir, "embedding_clusters.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["embedding_index", "event_uid", "cluster_id", "membership_probability", "is_noise"])
        for i in range(len(labels)):
            writer.writerow([i, row_to_uid[i], int(labels[i]), round(float(probabilities[i]), 4), labels[i] == -1])
    print(f"\nWrote {csv_path}")

    # ---- write cluster_summary.json ----
    summary_path = os.path.join(args.output_dir, "cluster_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    return labels, probabilities, medoids, row_to_uid


if __name__ == "__main__":
    main()
