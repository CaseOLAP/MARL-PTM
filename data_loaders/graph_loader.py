import pandas as pd
import torch

class GraphLoader:
    def __init__(self, filepath):
        self.edge_data = self._load_data(filepath)

    def _load_data(self, filepath):
        df = pd.read_csv(filepath)
        grouped = df.groupby("protein_id")
        return {
            pid: list(zip(gdf["residue_1"], gdf["residue_2"]))
            for pid, gdf in grouped
        }

    def get_graph_edge_index(self, protein_id):
        edges = self.edge_data.get(protein_id, [])
        if not edges:
            return torch.empty((2, 0), dtype=torch.long)
        edge_index = torch.tensor(edges, dtype=torch.long).T  # shape: [2, num_edges]
        return edge_index
