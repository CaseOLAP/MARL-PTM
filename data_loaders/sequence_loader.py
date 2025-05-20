import pandas as pd
import torch

class SequenceLoader:
    def __init__(self, filepath, max_len=100):
        self.max_len = max_len
        self.aa_vocab = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}  # 1-indexed
        self.pad_idx = 0
        self.data = self._load_data(filepath)

    def _load_data(self, filepath):
        df = pd.read_csv(filepath)
        return {row['protein_id']: row['sequence'] for _, row in df.iterrows()}

    def get_sequence_tensor(self, protein_id):
        seq = self.data.get(protein_id, "")
        encoded = [self.aa_vocab.get(aa, self.pad_idx) for aa in seq[:self.max_len]]
        if len(encoded) < self.max_len:
            encoded += [self.pad_idx] * (self.max_len - len(encoded))
        return torch.tensor(encoded, dtype=torch.long)  # shape: [L]
