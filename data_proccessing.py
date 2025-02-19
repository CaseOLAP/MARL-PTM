import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from Bio import SeqIO

def preprocess_gene_expression(file_path):
    df = pd.read_csv(file_path, index_col=0)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return torch.tensor(df_scaled, dtype=torch.float32)

def extract_sequence_features(fasta_file):
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq)
        sequences.append(seq)
    return sequences

if __name__ == "__main__":
    gene_expression_tensor = preprocess_gene_expression("data/gene_expression.csv")
    sequences = extract_sequence_features("data/proteins.fasta")
    print("Gene Expression Shape:", gene_expression_tensor.shape)
    print("Sample Protein Sequence:", sequences[0])
