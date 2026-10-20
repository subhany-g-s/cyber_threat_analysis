"""
gae_model.py
------------
Graph Autoencoder model definition.

IMPORTANT CONCEPTUAL NOTE (see also train_gae.py and the project README):
This is UNSUPERVISED representation learning. There are no attack labels
anywhere in this file or its training loop. The model learns:

    "which nodes have compatible latent representations, given their own
    features AND their graph neighborhood" (i.e. can the decoder predict
    the real edges from the learned Z?)

It does NOT learn "is this event malicious" -- that question is out of
scope for this stage entirely. The output Z is consumed by later,
separate stages (density-based clustering, novelty scoring, MITRE
mapping) that this module knows nothing about.

Architecture (deliberately simple, per project brief -- start simple,
not unnecessarily complex):

    X (node features)              A (symmetrized adjacency, via edge_index)
            \\                              /
             \\                            /
              GCNConv(in_dim -> hidden_dim)
                          |
                        ReLU
                          |
              GCNConv(hidden_dim -> latent_dim)
                          |
                          Z   (Z in R^[N x latent_dim])
                          |
              InnerProductDecoder: A_hat = sigmoid(Z Z^T)

hidden_dim / latent_dim are configurable (not hardcoded) -- see
train_gae.py's --hidden-dim / --latent-dim CLI args, defaults 64 / 32.

We use PyTorch Geometric's built-in GAE wrapper (torch_geometric.nn.GAE),
which already provides:
    - the inner-product decoder
    - recon_loss(z, pos_edge_index) with automatic negative sampling
      (handles the sparse-graph class-imbalance issue named in the brief
      -- without negative sampling, "always predict no edge" would
      trivially minimize the loss on a graph this sparse)
    - test(z, pos_edge_index, neg_edge_index) -> (auc, ap) for validation

We do not reimplement any of that by hand; the value we add here is the
encoder and its integration with our feature/graph pipeline.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE


class GCNEncoder(torch.nn.Module):
    """Two-layer GCN encoder: (X, edge_index) -> Z."""

    def __init__(self, in_channels: int, hidden_channels: int, latent_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, latent_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        z = self.conv2(h, edge_index)
        return z


def build_gae(in_channels: int, hidden_channels: int = 64, latent_channels: int = 32) -> GAE:
    """Construct the full GAE model (encoder + PyG's inner-product decoder)."""
    encoder = GCNEncoder(in_channels, hidden_channels, latent_channels)
    return GAE(encoder)
