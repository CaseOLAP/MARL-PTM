import os
import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.utils import from_networkx

class GraphEmbedder(torch.nn.Module):
    """
    Graph Neural Network model for embedding protein-pathway graphs.
    Supports GraphSAGE and GAT convolutional layers.
    """

    def __init__(self, input_dim=64, hidden_dim=128, output_dim=128, model_type='sage'):
        super(GraphEmbedder, self).__init__()
        if model_type == 'gat':
            self.conv1 = GATConv(input_dim, hidden_dim)
            self.conv2 = GATConv(hidden_dim, output_dim)
        else:
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GraphLoader:
    """
    Loads and embeds protein-pathway graphs using a GNN.
    Builds a fixed embedding for each protein node for use by the Graph Agent.
    """

    def __init__(self, graph_file, model_type='sage', device='cpu'):
        self.graph_file = graph_file
        self.model_type = model_type
        self.device = device

    def load_graph(self):
        """
        Loads a NetworkX graph from edge list or GML.
        Assumes protein and pathway nodes are already encoded as integers or strings.
        """
        if self.graph_file.endswith('.edgelist'):
            G = nx.read_edgelist(self.graph_file)
        elif self.graph_file.endswith('.gml'):
            G = nx.read_gml(self.graph_file)
        else:
            raise ValueError("Unsupported graph file format.")
        return G

    def build_embeddings(self, input_dim=64):
        """
        Computes GNN embeddings for all nodes in the graph.
        Returns a dictionary {node_id: embedding_vector}
        """
        G = self.load_graph()
        for node in G.nodes:
            G.nodes[node]['x'] = np.random.rand(input_dim)  # Initial features

        pyg_data = from_networkx(G)
        pyg_data.x = torch.FloatTensor([G.nodes[n]['x'] for n in G.nodes])
        pyg_data.edge_index = pyg_data.edge_index.to(self.device)
        pyg_data.x = pyg_data.x.to(self.device)

        model = GraphEmbedder(input_dim=input_dim, model_type=self.model_type).to(self.device)
        model.eval()
        with torch.no_grad():
            embeddings = model(pyg_data.x, pyg_data.edge_index)

        node_ids = list(G.nodes)
        result = {node_ids[i]: embeddings[i].cpu() for i in range(len(node_ids))}
        return result
