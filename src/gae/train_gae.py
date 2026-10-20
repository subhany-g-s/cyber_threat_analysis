"""
train_gae.py
------------
End-to-end script: raw logs -> (existing parser/normalizer/graph_builder,
UNCHANGED) -> node features -> PyG Data -> train GAE -> save latent
embeddings + event_uid mapping.

Usage:
    python src/train_gae.py
    python src/train_gae.py --input data/sample_2000_events.json --epochs 100 --latent-dim 32
    python src/train_gae.py --input data/apt29_evals_day1_manual_....json --epochs 200 --hidden-dim 64 --latent-dim 32

This script does NOT change parser.py, normalizer.py, or graph_builder.py.
The NetworkX event graph they build is treated as the unmodified source of
truth; this script only reads it.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch_geometric.transforms import RandomLinkSplit

from parser import parse_log_file
from normalizer import normalize_records
from graph_builder import build_event_graph
from gae_features import NodeFeatureBuilder, networkx_to_pyg
from gae_model import build_gae

DEFAULT_INPUT = "data/sample_2000_events.json"
OUTPUT_DIR = "output"
MODEL_DIR = "models"      # the pretrained checkpoint + feature schema read by the pipeline live here


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Steps 1 & 2 (existing, unmodified pipeline) ----
    print(f"[1/6] Parsing + normalizing + building event graph from {args.input} ...")
    records, parse_report = parse_log_file(args.input)
    events, norm_stats = normalize_records(records)
    G = build_event_graph(events)
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    print(f"      Event graph: {num_nodes} nodes, {num_edges} edges "
          f"(directed; symmetrized to {2 * num_edges} for GCN input, see gae_features.networkx_to_pyg docstring)")

    # ---- Node features ----
    print("[2/6] Building node feature matrix ...")
    builder = NodeFeatureBuilder()
    builder.fit(G)
    schema_path = os.path.join(MODEL_DIR, "gae_feature_schema.json")
    builder.save(schema_path)
    print(f"      Feature schema saved to {schema_path}")

    data_obj, event_uid_order, feature_names = networkx_to_pyg(G, builder, symmetrize=True)
    in_channels = data_obj.x.shape[1]
    print(f"      Feature matrix X: {tuple(data_obj.x.shape)} ({len(feature_names)} named features)")

    # ---- Save event_uid <-> row index mapping (critical for later stages) ----
    mapping_path = os.path.join(OUTPUT_DIR, "gae_node_mapping.json")
    node_mapping = {
        "row_to_event_uid": event_uid_order,                              # list: row index -> event_uid
        "event_uid_to_row": {uid: i for i, uid in enumerate(event_uid_order)},  # dict: event_uid -> row index
        "feature_names": feature_names,
    }
    with open(mapping_path, "w") as f:
        json.dump(node_mapping, f)
    print(f"      Node mapping saved to {mapping_path} "
          f"(row index <-> event_uid, so any embedding row can be traced back to its original event)")

    # ---- Train/val/test edge split (for reconstruction AUC/AP, not node labels) ----
    print("[3/6] Splitting edges for train/validation/test reconstruction ...")
    splitter = RandomLinkSplit(
        num_val=args.val_ratio, num_test=args.test_ratio,
        is_undirected=True,           # matches the symmetrized graph we built
        add_negative_train_samples=False,  # GAE.recon_loss samples negatives itself, on the fly, each epoch
        split_labels=True,
    )
    train_data, val_data, test_data = splitter(data_obj)
    print(f"      Train positive edges: {train_data.pos_edge_label_index.shape[1]}")
    print(f"      Val   positive/negative edges: {val_data.pos_edge_label_index.shape[1]} / {val_data.neg_edge_label_index.shape[1]}")
    print(f"      Test  positive/negative edges: {test_data.pos_edge_label_index.shape[1]} / {test_data.neg_edge_label_index.shape[1]}")

    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    # ---- Model ----
    print(f"[4/6] Building GAE (in_dim={in_channels}, hidden_dim={args.hidden_dim}, latent_dim={args.latent_dim}) ...")
    model = build_gae(in_channels, args.hidden_dim, args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- Training loop ----
    print(f"[5/6] Training for {args.epochs} epochs ...")
    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
        loss = model.recon_loss(z, train_data.pos_edge_label_index)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            z_val = model.encode(train_data.x, train_data.edge_index)
            val_auc, val_ap = model.test(z_val, val_data.pos_edge_label_index, val_data.neg_edge_label_index)

        history.append({"epoch": epoch, "train_loss": float(loss.item()), "val_auc": float(val_auc), "val_ap": float(val_ap)})
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            print(f"      epoch {epoch:4d}/{args.epochs}  train_loss={loss.item():.4f}  val_auc={val_auc:.4f}  val_ap={val_ap:.4f}")

    train_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        z_full = model.encode(train_data.x, train_data.edge_index)
        test_auc, test_ap = model.test(z_full, test_data.pos_edge_label_index, test_data.neg_edge_label_index)
    print(f"      Final test reconstruction: auc={test_auc:.4f} ap={test_ap:.4f}  ({train_time:.1f}s total)")

    # ---- Extract embeddings over the FULL graph (not just train split) ----
    print("[6/6] Extracting latent embeddings for all nodes and saving outputs ...")
    model.eval()
    with torch.no_grad():
        data_obj_dev = data_obj.to(device)
        z_all = model.encode(data_obj_dev.x, data_obj_dev.edge_index).cpu().numpy()

    embeddings_path = os.path.join(OUTPUT_DIR, "gae_embeddings.npy")
    np.save(embeddings_path, z_all)

    history_path = os.path.join(OUTPUT_DIR, "gae_training_history.json")
    with open(history_path, "w") as f:
        json.dump({
            "config": vars(args),
            "num_nodes": num_nodes,
            "num_edges_directed": num_edges,
            "num_edges_symmetrized": int(data_obj.edge_index.shape[1]),
            "in_channels": in_channels,
            "history": history,
            "final_test_auc": float(test_auc),
            "final_test_ap": float(test_ap),
            "train_time_seconds": train_time,
            "device": str(device),
        }, f, indent=2)

    model_path = os.path.join(MODEL_DIR, "gae_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "in_channels": in_channels,
        "hidden_channels": args.hidden_dim,
        "latent_channels": args.latent_dim,
    }, model_path)

    metadata_path = os.path.join(OUTPUT_DIR, "gae_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({
            "input_file": args.input,
            "seed": args.seed,
            "num_nodes": num_nodes,
            "num_edges_directed": num_edges,
            "embedding_shape": list(z_all.shape),
            "note": (
                "Unsupervised structural+feature representation learning. "
                "No attack/MITRE/cluster labels were used anywhere in this "
                "training run. z_all[i] corresponds to event_uid "
                "gae_node_mapping.json['row_to_event_uid'][i]."
            ),
        }, f, indent=2)

    print(f"      Wrote {embeddings_path}  shape={z_all.shape}")
    print(f"      Wrote {history_path}")
    print(f"      Wrote {model_path}")
    print(f"      Wrote {metadata_path}")
    print("\nDone. Next stage (not implemented yet): density-based clustering on gae_embeddings.npy.")


if __name__ == "__main__":
    main()
