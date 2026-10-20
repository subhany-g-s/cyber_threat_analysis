"""
src/service/gae_stage.py
--------------------------
STAGE 4: validate the PRETRAINED GAE checkpoint against its feature schema
and the current data, then produce node embeddings. Nothing is retrained.

Checks (each failure is a clear GaeError, not a traceback):
  * checkpoint + feature-schema files exist and load
  * torch and torch_geometric import
  * checkpoint carries in/hidden/latent channel sizes and a state dict whose
    layer shapes agree with them
  * schema feature count == checkpoint in_channels == width of the feature
    matrix actually built from the current graph
  * embeddings are finite and one row per event
Also reported (as warnings, not failures): how much of the current data falls
into the schema's "OTHER_OR_UNSEEN" buckets - a drift signal telling the
analyst the pretrained model is seeing values it never trained on.

Alternative: `precomputed_embeddings` + mapping JSON can be loaded instead of
running the GAE ("embedding loading").
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class GaeError(Exception):
    """The GAE stage cannot produce valid embeddings."""


@dataclass
class GaeOutcome:
    Z: np.ndarray
    row_to_uid: List[str]
    info: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


def _import_torch():
    try:
        import torch
        import torch_geometric  # noqa: F401
        return torch
    except ImportError as e:
        raise GaeError(f"PyTorch / PyTorch Geometric are required to run the pretrained GAE "
                       f"({e}). Install: pip install torch torch-geometric")


def _load_checkpoint(torch, path: str) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)      # safe tensor-only load
    except Exception as first:
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            warnings.append("Checkpoint required full pickle loading (weights_only=False). Only use "
                            "checkpoints you trust.")
        except Exception as e:
            raise GaeError(f"Checkpoint {path} could not be loaded ({type(e).__name__}: {str(e)[:160]}).")
    if not isinstance(ck, dict):
        raise GaeError(f"Checkpoint {path} is not a dict of weights/metadata.")
    missing = [k for k in ("in_channels", "hidden_channels", "latent_channels", "model_state_dict") if k not in ck]
    if missing:
        raise GaeError(f"Checkpoint {path} is missing required key(s): {missing}.")
    sd = ck["model_state_dict"]
    expect = {"encoder.conv1.lin.weight": (ck["hidden_channels"], ck["in_channels"]),
              "encoder.conv2.lin.weight": (ck["latent_channels"], ck["hidden_channels"])}
    for k, shape in expect.items():
        if k not in sd:
            raise GaeError(f"Checkpoint state dict lacks '{k}'; it was not produced by src/gae/train_gae.py.")
        if tuple(sd[k].shape) != tuple(shape):
            raise GaeError(f"Checkpoint layer '{k}' has shape {tuple(sd[k].shape)} but the recorded channel "
                           f"sizes imply {shape}: inconsistent checkpoint.")
    return ck, warnings


def schema_coverage(G, builder) -> Dict[str, float]:
    """Share of events whose value for a key field is NOT in the schema's vocabulary."""
    out = {}
    n = max(G.number_of_nodes(), 1)
    for f in ("event_id", "hostname", "source_image", "target_image", "user"):
        vocab = builder.vocabs.get(f)
        if vocab is None:
            continue
        cats = set(vocab.categories)
        unseen = 0
        present = 0
        for _, d in G.nodes(data=True):
            v = d.get(f)
            if v is None:
                continue
            present += 1
            if str(v) not in cats:
                unseen += 1
        out[f] = round(unseen / present, 4) if present else 0.0
    return out


def run_gae(G, checkpoint_path: str, schema_path: str) -> GaeOutcome:
    for label, p in (("GAE checkpoint", checkpoint_path), ("GAE feature schema", schema_path)):
        if not os.path.exists(p):
            raise GaeError(f"{label} not found: {p}. The pretrained model ships in models/; "
                           f"this tool does not retrain it. Pass --pretrained-model / --pretrained-schema, or "
                           f"--embeddings-npy to supply embeddings instead.")
    torch = _import_torch()
    from src.gae.gae_features import NodeFeatureBuilder, networkx_to_pyg
    from src.gae.gae_model import build_gae

    ck, warnings = _load_checkpoint(torch, checkpoint_path)
    try:
        builder = NodeFeatureBuilder.load(schema_path)
    except Exception as e:
        raise GaeError(f"Feature schema {schema_path} could not be loaded ({type(e).__name__}: {e}).")
    if len(builder.feature_names) != ck["in_channels"]:
        raise GaeError(
            f"Incompatible feature schema: the schema defines {len(builder.feature_names)} features but the "
            f"checkpoint expects in_channels={ck['in_channels']}. Schema and checkpoint must come from the "
            f"same training run.")

    try:
        data_obj, row_to_uid, names = networkx_to_pyg(G, builder, symmetrize=True)
    except Exception as e:
        raise GaeError(f"Feature extraction failed ({type(e).__name__}: {e}).")
    if data_obj.x.shape[1] != ck["in_channels"]:
        raise GaeError(f"Feature matrix has {data_obj.x.shape[1]} columns but the checkpoint expects "
                       f"{ck['in_channels']}.")
    if not torch.isfinite(data_obj.x).all():
        raise GaeError("Feature matrix contains NaN/inf values.")

    model = build_gae(ck["in_channels"], ck["hidden_channels"], ck["latent_channels"])
    try:
        model.load_state_dict(ck["model_state_dict"])
    except Exception as e:
        raise GaeError(f"Checkpoint weights do not fit the model architecture ({type(e).__name__}: {str(e)[:160]}).")
    model.eval()
    with torch.no_grad():
        Z = model.encode(data_obj.x, data_obj.edge_index).numpy()
    if Z.shape != (len(row_to_uid), ck["latent_channels"]) or not np.isfinite(Z).all():
        raise GaeError(f"Embedding output invalid: shape {Z.shape}, finite={bool(np.isfinite(Z).all())}.")

    cover = schema_coverage(G, builder)
    for f, share in cover.items():
        if share >= 0.25:
            warnings.append(f"{round(100 * share)}% of events have a '{f}' value the pretrained schema never saw "
                            f"(OTHER_OR_UNSEEN bucket): embeddings for them are less informative.")
    info = {
        "source": "pretrained_checkpoint", "checkpoint": checkpoint_path, "feature_schema": schema_path,
        "in_channels": int(ck["in_channels"]), "hidden_channels": int(ck["hidden_channels"]),
        "latent_channels": int(ck["latent_channels"]), "schema_feature_count": len(builder.feature_names),
        "feature_matrix_shape": list(data_obj.x.shape), "edge_index_shape": list(data_obj.edge_index.shape),
        "embedding_shape": list(Z.shape), "retrained": False, "schema_unseen_share": cover,
        "checkpoint_loaded_weights_only": not any("pickle" in w for w in warnings),
        "embedding_norm_mean": round(float(np.linalg.norm(Z, axis=1).mean()), 4) if len(Z) else 0.0,
    }
    return GaeOutcome(Z=Z, row_to_uid=list(row_to_uid), info=info, warnings=warnings)


def load_precomputed(G, npy_path: str, mapping_path: str) -> GaeOutcome:
    """Use embeddings computed elsewhere. The mapping JSON must contain
    "row_to_event_uid"; its uid set must equal the graph's - otherwise the
    rows cannot be trusted to belong to these events."""
    for label, p in (("embeddings", npy_path), ("mapping", mapping_path)):
        if not p or not os.path.exists(p):
            raise GaeError(f"Precomputed {label} file not found: {p}")
    try:
        Z = np.load(npy_path, allow_pickle=False)
        with open(mapping_path, "r", encoding="utf-8") as f:
            row_to_uid = json.load(f)["row_to_event_uid"]
    except Exception as e:
        raise GaeError(f"Could not read precomputed embeddings/mapping ({type(e).__name__}: {e}).")
    if Z.ndim != 2 or Z.shape[0] != len(row_to_uid):
        raise GaeError(f"Embedding rows ({Z.shape[0] if Z.ndim else 'n/a'}) != mapping length ({len(row_to_uid)}).")
    if set(row_to_uid) != set(G.nodes()):
        only_map = len(set(row_to_uid) - set(G.nodes()))
        only_graph = len(set(G.nodes()) - set(row_to_uid))
        raise GaeError(f"Precomputed embeddings do not match this input: {only_map} uid(s) only in the mapping, "
                       f"{only_graph} only in the current graph. Refusing to align rows by guesswork.")
    if not np.isfinite(Z).all():
        raise GaeError("Precomputed embeddings contain NaN/inf.")
    info = {"source": "precomputed", "embeddings_file": npy_path, "mapping_file": mapping_path,
            "embedding_shape": list(Z.shape), "latent_channels": int(Z.shape[1]), "retrained": False}
    return GaeOutcome(Z=Z.astype(np.float32), row_to_uid=list(row_to_uid), info=info)
