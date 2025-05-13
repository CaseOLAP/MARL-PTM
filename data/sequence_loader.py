import os
import torch
import numpy as np
from transformers import EsmModel, EsmTokenizer
from Bio import SeqIO

class SequenceLoader:
    """
    Loads protein sequences from FASTA files and generates ESM embeddings.
    Designed to preprocess input for the Sequence Agent.
    """

    def __init__(self, model_name='facebook/esm2_t33_650M_UR50D', max_len=1024, device='cpu'):
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name).to(device)
        self.device = device
        self.max_len = max_len

    def load_fasta_sequences(self, fasta_path):
        """
        Loads protein sequences from a FASTA file.
        Returns a dictionary {protein_id: sequence_str}
        """
        sequences = {}
        for record in SeqIO.parse(fasta_path, "fasta"):
            seq = str(record.seq)
            if len(seq) > 10:
                sequences[record.id] = seq[:self.max_len]  # truncate long sequences
        return sequences

    def embed_sequence(self, sequence):
        """
        Converts a raw protein sequence into ESM embedding.
        Returns a tensor of shape (seq_len, embedding_dim)
        """
        inputs = self.tokenizer(sequence, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            embeddings = outputs.last_hidden_state.squeeze(0)[1:-1]  # remove [CLS] and [EOS]

        return embeddings.cpu()

    def batch_embed_sequences(self, sequences_dict):
        """
        Converts a dictionary of {protein_id: sequence} into
        {protein_id: embedding_tensor}
        """
        embeddings = {}
        for pid, seq in sequences_dict.items():
            try:
                emb = self.embed_sequence(seq)
                embeddings[pid] = emb
            except Exception as e:
                print(f"Failed to embed {pid}: {e}")
        return embeddings
