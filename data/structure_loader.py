import os
import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from collections import defaultdict

class StructureLoader:
    """
    Loads structural features for each protein.
    Extracts residue-level features from DSSP and AlphaFold confidence scores.
    """

    def __init__(self, structure_dir, max_len=1024):
        self.structure_dir = structure_dir
        self.max_len = max_len
        self.parser = PDBParser(QUIET=True)

    def parse_structure(self, pdb_file):
        """
        Parses a PDB file and returns DSSP-derived structural features.
        Returns a list of tuples: (RSA, secondary_structure, pLDDT or placeholder)
        """
        structure_id = os.path.basename(pdb_file).split('.')[0]
        structure = self.parser.get_structure(structure_id, pdb_file)

        model = structure[0]
        dssp = DSSP(model, pdb_file)

        features = []
        for key in dssp.keys():
            rsa = dssp[key][3]        # Relative solvent accessibility
            ss = dssp[key][2]         # Secondary structure
            ss_onehot = self.encode_secondary_structure(ss)
            plddt = 0.7               # Placeholder for experimental structures

            feature_vector = np.concatenate(([rsa], ss_onehot, [plddt]))
            features.append(feature_vector)

        if len(features) < self.max_len:
            pad_len = self.max_len - len(features)
            features.extend([np.zeros_like(features[0])] * pad_len)
        else:
            features = features[:self.max_len]

        return np.array(features)

    def encode_secondary_structure(self, ss):
        """
        One-hot encode secondary structure:
        H = alpha helix, E = beta sheet, C = coil/loop
        """
        categories = {'H': [1, 0, 0], 'E': [0, 1, 0], '-': [0, 0, 1]}
        return categories.get(ss, [0, 0, 1])

    def load_batch_structures(self):
        """
        Loads all structure files in the structure_dir and returns
        a dictionary {protein_id: structure_feature_tensor}
        """
        features = {}
        for file in os.listdir(self.structure_dir):
            if file.endswith('.pdb'):
                protein_id = file.split('.')[0]
                try:
                    path = os.path.join(self.structure_dir, file)
                    mat = self.parse_structure(path)
                    features[protein_id] = torch.FloatTensor(mat)
                except Exception as e:
                    print(f"Failed to process {file}: {e}")
        return features
