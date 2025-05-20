import pandas as pd

class ProteoformLoader:
    def __init__(self, filepath):
        self.data = self._load_data(filepath)

    def _load_data(self, filepath):
        df = pd.read_csv(filepath)
        grouped = df.groupby("protein_id")
        return {
            pid: [(row["residue"], row["prior_ptm"]) for _, row in g.iterrows()]
            for pid, g in grouped
        }

    def get_proteoform_sites(self, protein_id):
        return self.data.get(protein_id, [])  # List of (residue_idx, ptm_type)
