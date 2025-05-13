import os
import numpy as np
import torch
from Bio import SeqIO

class ProteoformLoader:
    """
    Loads isoform-specific sequence features for PTM prediction.
    Aligns canonical and isoform sequences to identify accessibility differences.
    Converts isoform-specific regions into residue-wise masks and domain indicators.
    """

    def __init__(self, canonical_fasta, isoform_fasta, max_len=1024, domain_dim=64):
        self.canonical_fasta = canonical_fasta
        self.isoform_fasta = isoform_fasta
        self.max_len = max_len
        self.domain_dim = domain_dim

    def load_sequences(self, fasta_path):
        """
        Loads sequences from a FASTA file.
        Returns a dictionary: {protein_id: sequence_str}
        """
        seqs = {}
        for record in SeqIO.parse(fasta_path, "fasta"):
            seq = str(record.seq)
            if len(seq) > 10:
                seqs[record.id] = seq[:self.max_len]
        return seqs

    def align_sequences(self, canonical_seq, isoform_seq):
        """
        Compares canonical and isoform sequences.
        Returns a binary mask of accessible (present) residues in isoform.
        """
        min_len = min(len(canonical_seq), len(isoform_seq), self.max_len)
        mask = np.zeros(self.max_len)
        for i in range(min_len):
            mask[i] = 1.0 if canonical_seq[i] == isoform_seq[i] else 0.0
        return mask

    def generate_domain_embeddings(self):
        """
        Placeholder: Simulates domain encoding.
        Returns random domain vectors of fixed dimension per residue.
        """
        return np.random.rand(self.max_len, self.domain_dim)

    def build_features(self):
        """
        Main function to load, align, and format proteoform features.
        Returns: {protein_id: tensor of shape (max_len, feature_dim)}
        """
        canonical = self.load_sequences(self.canonical_fasta)
        isoforms = self.load_sequences(self.isoform_fasta)

        feature_map = {}

        for pid, can_seq in canonical.items():
            isoform_seq = isoforms.get(pid, can_seq)
            mask = self.align_sequences(can_seq, isoform_seq).reshape(-1, 1)
            domains = self.generate_domain_embeddings()

            feature = np.concatenate([mask, domains], axis=1)  # shape: (max_len, 1 + domain_dim)
            feature_map[pid] = torch.FloatTensor(feature)

        return feature_map
