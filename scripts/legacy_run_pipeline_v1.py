"""
scripts/legacy_run_pipeline_v1.py  (ORIGINAL driver, kept for reference; uses the legacy 15-entry corpus. Use run_pipeline.py)
---------------------------
End-to-end driver: parse -> normalize -> graph -> GAE encode (pretrained,
NOT retrained) -> cluster -> novelty -> profile -> semantic MITRE map ->
RAG export.

Usage:
    python scripts/run_pipeline.py --input data/sample_2000_events.json
    python scripts/run_pipeline.py --input data/sample_2000_events.json --embedder tfidf   # offline dev/test only, see README
"""

import argparse
import csv
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "graph"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "gae"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "clustering"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "novelty"))

import torch
import numpy as np

from parser import parse_log_file
from normalizer import normalize_records
from graph_builder import build_event_graph
from gae_features import NodeFeatureBuilder, networkx_to_pyg
from gae_model import build_gae
import cluster_embeddings as cluster_mod
import novelty_detection as novelty_mod

from src.profiling.cluster_profiler import build_cluster_profiles
from src.mapping.semantic_mapper import SemanticMitreMapper, TechniqueCorpus
from src.mapping.embedders import SBertEmbedder
from src.rag.rag_export import build_rag_bundle, write_rag_bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/sample_2000_events.json")
    ap.add_argument("--pretrained-model", default="models/gae_model.pt")
    ap.add_argument("--pretrained-schema", default="models/gae_feature_schema.json")
    ap.add_argument("--min-cluster-size", type=int, default=20)
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--corpus", default="config/mitre_technique_corpus.json")
    ap.add_argument("--cache-vectors", default="cache/technique_embeddings.npy")
    ap.add_argument("--cache-ids", default="cache/technique_ids.json")
    ap.add_argument("--embedder", choices=["sbert", "tfidf"], default="sbert",
                     help="'tfidf' is an OFFLINE DEV/TEST stand-in only (see README) -- production default is sbert.")
    ap.add_argument("--output-dir", default="output")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/7] Parse + normalize + build graph ...")
    records, parse_report = parse_log_file(args.input)
    events, norm_stats = normalize_records(records)
    events_by_uid = {e.event_uid: e.as_dict() for e in events}
    G = build_event_graph(events)
    print(f"      {parse_report.valid_records} valid records, graph: "
          f"{G.number_of_nodes()} nodes / {G.number_of_edges()} edges")

    print("[2/7] Load pretrained GAE and encode (no retraining) ...")
    builder = NodeFeatureBuilder.load(args.pretrained_schema)
    checkpoint = torch.load(args.pretrained_model, map_location="cpu", weights_only=False)
    model = build_gae(checkpoint["in_channels"], checkpoint["hidden_channels"], checkpoint["latent_channels"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data_obj, row_to_uid, _ = networkx_to_pyg(G, builder, symmetrize=True)
    with torch.no_grad():
        Z = model.encode(data_obj.x, data_obj.edge_index).numpy()
    print(f"      Embeddings: {Z.shape}")

    print("[3/7] Cluster (HDBSCAN) ...")
    labels, probabilities, medoids, algo = cluster_mod.cluster_embeddings(
        Z, min_cluster_size=args.min_cluster_size, min_samples=args.min_samples, algorithm="hdbscan",
    )
    summary = cluster_mod.summarize_clusters(labels, probabilities)
    print(f"      {summary['number_of_clusters']} clusters, {summary['number_of_noise_points']} noise points")

    cluster_assignments = [
        {"event_uid": row_to_uid[i], "cluster_id": int(labels[i]), "membership_probability": float(probabilities[i])}
        for i in range(len(row_to_uid))
    ]
    medoid_event_uid_by_cluster = {cid: row_to_uid[row] for cid, row in medoids.items()}

    # Persisted to disk (not just kept in memory) because the RAG retriever
    # (src/rag/retriever.py) needs full cluster membership -- not just the
    # representative events already carried in cluster_profiles_v2.json --
    # to compute a cluster's high-novelty member events at query time.
    assignments_path = os.path.join(args.output_dir, "embedding_clusters.csv")
    with open(assignments_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["embedding_index", "event_uid", "cluster_id", "membership_probability", "is_noise"])
        for i, row in enumerate(cluster_assignments):
            writer.writerow([i, row["event_uid"], row["cluster_id"], row["membership_probability"],
                              row["cluster_id"] == -1])
    print(f"      Wrote {assignments_path}")

    print("[4/7] Novelty scoring (reused, unmodified novelty_detection.py) ...")
    cluster_rows_for_novelty = [
        {"event_uid": row_to_uid[i], "membership_probability": float(probabilities[i])}
        for i in range(len(row_to_uid))
    ]
    density = novelty_mod.compute_density_score(cluster_rows_for_novelty)
    emb_dist = novelty_mod.compute_embedding_distance_score(Z, row_to_uid, cluster_rows_for_novelty, medoids)
    recon = novelty_mod.compute_reconstruction_score(Z, row_to_uid, G)
    rarity = novelty_mod.compute_graph_rarity_score(G)
    weights = {"density": 1.0, "embedding_distance": 1.0, "reconstruction": 1.0, "graph_rarity": 1.0}
    novelty_rows = novelty_mod.combine_novelty_scores(density, emb_dist, recon, rarity, weights)

    cluster_by_uid = {row["event_uid"]: row["cluster_id"] for row in cluster_assignments}
    for row in novelty_rows:
        row["cluster_id"] = cluster_by_uid.get(row["event_uid"])
        row["novelty_label"] = novelty_mod.label_novelty(row["novelty_score"], (0.25, 0.5, 0.75))
    novelty_rows.sort(key=lambda r: -r["novelty_score"])

    novelty_path = os.path.join(args.output_dir, "novelty_scores.csv")
    with open(novelty_path, "w", newline="") as f:
        fieldnames = ["event_uid", "cluster_id", "density_score", "embedding_distance_score",
                      "reconstruction_error", "graph_rarity_score", "novelty_score", "novelty_label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in novelty_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    label_counts = {}
    for row in novelty_rows:
        label_counts[row["novelty_label"]] = label_counts.get(row["novelty_label"], 0) + 1
    print(f"      Wrote {novelty_path}  ({label_counts})")

    print("[5/7] Build cluster profiles (structured + text narrative) ...")
    profiles = build_cluster_profiles(cluster_assignments, events_by_uid, medoid_event_uid_by_cluster)
    print(f"      {len(profiles)} non-noise cluster profiles built")

    print(f"[6/7] Semantic MITRE mapping (embedder={args.embedder}) ...")
    with open(args.corpus) as f:
        techniques = json.load(f)["techniques"]

    if args.embedder == "sbert":
        embedder = SBertEmbedder()
        if os.path.exists(args.cache_vectors) and os.path.exists(args.cache_ids):
            corpus = TechniqueCorpus.load_cached(techniques, args.cache_vectors, args.cache_ids)
        else:
            print("      No cache found -- embedding corpus now (run embed_technique_corpus.py to cache it for next time).")
            corpus = TechniqueCorpus.build(techniques, embedder)
    else:
        print("      WARNING: --embedder tfidf is an offline dev/test stand-in, NOT the production path. See README.")
        from tests.tfidf_embedder import TfidfEmbedder
        embedder = TfidfEmbedder()
        texts_for_fit = [f"{t['technique_id']} {t['technique_name']}: {t['description']}" for t in techniques]
        texts_for_fit += [p.text for p in profiles]
        embedder.fit(texts_for_fit)
        corpus = TechniqueCorpus.build(techniques, embedder)

    mapper = SemanticMitreMapper(corpus, embedder, top_k=3, min_similarity=0.35)
    mappings = mapper.map_clusters(profiles)

    print("[7/7] RAG export ...")
    bundle = build_rag_bundle(profiles, mappings)
    write_rag_bundle(bundle, os.path.join(args.output_dir, "rag_ready_clusters.json"))

    with open(os.path.join(args.output_dir, "cluster_profiles_v2.json"), "w") as f:
        json.dump([p.as_dict() for p in profiles], f, indent=2, default=str)
    with open(os.path.join(args.output_dir, "semantic_mitre_mappings.json"), "w") as f:
        json.dump([m.as_dict() for m in mappings], f, indent=2)

    print(f"\nWrote {args.output_dir}/cluster_profiles_v2.json")
    print(f"Wrote {args.output_dir}/semantic_mitre_mappings.json")
    print(f"Wrote {args.output_dir}/rag_ready_clusters.json")

    print("\nSample mapping:")
    for m in mappings[:3]:
        print(f"  cluster {m.cluster_id}: {m.cluster_text[:100]}...")
        for t in m.top_k_techniques:
            print(f"    - {t.technique_id} {t.technique_name}  sim={t.similarity_score}  {t.mapping_status}")


if __name__ == "__main__":
    main()
