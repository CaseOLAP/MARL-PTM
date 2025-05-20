import pandas as pd
import torch

class ExpressionLoader:
    def __init__(self, filepath, max_len=100):
        self.max_len = max_len
        self.data = self._load_data(filepath)

    def _load_data(self, filepath):
        df = pd.read_csv(filepath)
        grouped = df.groupby("protein_id")
        return {
            pid: [row["expression"] for _, row in g.iterrows()]
            for pid, g in grouped
        }

    def get_expression_tensor(self, protein_id):
        values = self.data.get(protein_id, [])
        if len(values) < self.max_len:
            values += [0.0] * (self.max_len - len(values))
        return torch.tensor(values[:self.max_len], dtype=torch.float)  # shape: [L]
