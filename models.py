"""
models.py -- the two model families compared in the experiments.

MLP        flattens a (20, 52) window and classifies it; it sees no structure.
SensorGNN  encodes each of the 52 sensors with one shared encoder plus a learned
           sensor embedding, mixes sensors along a fixed graph with two GNN layers
           (GCN, GAT or R-GCN, optionally residual), and classifies the
           concatenated node vectors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, RGCNConv


class MLP(nn.Module):
    """Flat baseline: window -> 1040 numbers -> 128 -> 128 -> 22 class scores."""

    def __init__(self, n_inputs=20 * 52, n_hidden=128, n_classes=22, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.Linear(n_inputs, n_hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(n_hidden, n_classes))

    def forward(self, x):
        return self.net(x)


class SensorGNN(nn.Module):
    """GNN over a fixed sensor graph. Input: windows of shape (batch, window, 52).

    edge_index : (2, E) array of directed edges between the 52 variables
    conv       : "gcn", "gat" or "rgcn"
    residual   : if True, each graph layer adds to the node's vector (h + f(h))
                 instead of replacing it (f(h))
    readout    : "flatten" concatenates the 52 node vectors in a fixed order;
                 "mean" averages them (permutation-invariant)
    edge_type  : (E,) relation index per edge, used by "rgcn"
    """

    def __init__(self, edge_index, conv="gcn", residual=False, readout="flatten", edge_type=None,
                 window=20, n_nodes=52, hidden=32, emb_dim=8, n_classes=22, dropout=0.2):
        super().__init__()
        self.n_nodes, self.conv_type, self.residual, self.readout = n_nodes, conv, residual, readout
        # the graph is stored in the model as buffers: they move with it to the GPU but are not trained
        self.register_buffer("edge_index", torch.as_tensor(edge_index, dtype=torch.long))
        et = torch.zeros(self.edge_index.shape[1], dtype=torch.long) if edge_type is None \
            else torch.as_tensor(edge_type, dtype=torch.long)
        self.register_buffer("edge_type", et)

        self.encoder = nn.Linear(window, hidden)            # shared by all sensors: 20 values -> 32
        self.node_emb = nn.Embedding(n_nodes, emb_dim)      # a learned identity vector per sensor
        self.mix = nn.Linear(hidden + emb_dim, hidden)      # combine signal and identity
        if conv == "gcn":
            self.convs = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(2)])
        elif conv == "gat":                                 # 4 attention heads of 8 numbers each
            self.convs = nn.ModuleList([GATConv(hidden, hidden // 4, heads=4) for _ in range(2)])
        elif conv == "rgcn":                                # one weight matrix per relation type
            n_rel = int(self.edge_type.max()) + 1
            self.convs = nn.ModuleList([RGCNConv(hidden, hidden, n_rel) for _ in range(2)])
        else:
            raise ValueError(f"unknown conv {conv!r}")
        self.drop = nn.Dropout(dropout)
        n_read = hidden * n_nodes if readout == "flatten" else hidden
        self.head = nn.Sequential(nn.Linear(n_read, 128), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(128, n_classes))

    def batch_graph(self, B):
        """Copy the graph B times with node numbers shifted by 52 (what PyG's Batch does)."""
        offsets = torch.arange(B, device=self.edge_index.device).repeat_interleave(self.edge_index.shape[1])
        return self.edge_index.repeat(1, B) + offsets * self.n_nodes, self.edge_type.repeat(B)

    def forward(self, x):                                   # x: (B, 20, 52)
        B = x.shape[0]
        h = x.transpose(1, 2).reshape(B * self.n_nodes, -1) # one row per sensor: (B*52, 20)
        h = F.relu(self.encoder(h))
        ids = torch.arange(self.n_nodes, device=x.device).repeat(B)
        h = self.mix(torch.cat([h, self.node_emb(ids)], dim=1))
        ei, et = self.batch_graph(B)
        for i, conv in enumerate(self.convs):
            m = conv(h, ei, et) if self.conv_type == "rgcn" else conv(h, ei)
            m = F.relu(m)
            h = h + m if self.residual else m
            if i == 0:
                h = self.drop(h)
        h = h.view(B, self.n_nodes, -1)
        h = h.flatten(1) if self.readout == "flatten" else h.mean(dim=1)
        return self.head(h)
