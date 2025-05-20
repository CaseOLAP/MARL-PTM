# agents/graph_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from agents.base_agent import BaseAgent

class GraphAgent(BaseAgent):
    """
    Learns contextual residue embeddings from residue-residue graph.
    Input: residue node features [L, F] and edge_index [2, E]
    Output: node embeddings [L, D]
    """

    def __init__(self, input_feat_dim=1, hidden_dim=64, output_dim=128, num_layers=2, device='cpu'):
        super().__init__(input_dim=input_feat_dim, output_dim=output_dim, device=device)

        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_feat_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.layers.append(GCNConv(hidden_dim, output_dim))

    def forward(self, node_features, edge_index):
        """
        node_features: FloatTensor [L, F]
        edge_index: LongTensor [2, E] (graph edges)
        returns: FloatTensor [L, D]
        """
        x = node_features
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, edge_index))
        x = self.layers[-1](x, edge_index)
        return x  # [L, D]
