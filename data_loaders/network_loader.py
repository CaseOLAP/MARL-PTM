import pandas as pd

class NetworkLoader:
    def __init__(self, filepath):
        self.data = self._load_data(filepath)

    def _load_data(self, filepath):
        df = pd.read_csv(filepath)
        grouped = df.groupby("source_node")
        return {
            pid: list(g["target_node"])
            for pid, g in grouped
        }

    def get_pathway_nodes(self, protein_id):
        return self.data.get(protein_id, [])  # List of pathway node IDs
