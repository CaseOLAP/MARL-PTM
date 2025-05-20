import pandas as pd
import torch

class StructureLoader:
    def __init__(self, filepath, max_len=100):
        self.max_len = max_len
        self.struct_vocab = {'H': 1, 'E': 2, 'C': 3}
        self.pad_idx = 0
        self.data = self._load_data(filepath)

    def _load_data(self, filepath):
        df = pd.read_csv(filepath)
        return {row['protein_id']: row['dssp'] for _, row in df.iterrows()}

    def get_structure_tensor(self, protein_id):
        dssp = self.data.get(protein_id, "")
        encoded = [self.struct_vocab.get(ss, self.pad_idx) for ss in dssp[:self.max_len]]
        if len(encoded) < self.max_len:
            encoded += [self.pad_idx] * (self.max_len - len(encoded))
        return torch.tensor(encoded, dtype=torch.long)  # shape: [L]
