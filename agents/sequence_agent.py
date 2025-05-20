# agents/sequence_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.base_agent import BaseAgent

class SequenceAgent(BaseAgent):
    """
    Learns attention-aware embeddings from amino acid sequences.
    Input: tokenized sequence tensor [L]
    Output: contextual embedding [L, D]
    """

    def __init__(self, vocab_size=21, embed_dim=64, hidden_dim=128, output_dim=128, max_len=100, device='cpu'):
        super().__init__(input_dim=max_len, output_dim=output_dim, device=device)

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.positional_encoding = nn.Parameter(torch.randn(max_len, embed_dim))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=hidden_dim, dropout=0.1, batch_first=True),
            num_layers=2
        )
        self.proj = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        """
        x: LongTensor of shape [L] (amino acid tokens)
        returns: FloatTensor of shape [L, D]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, L]

        x = self.embedding(x) + self.positional_encoding[:x.size(1)]  # [B, L, E]
        x = self.encoder(x)  # [B, L, E]
        x = self.proj(x)     # [B, L, D]
        return x.squeeze(0)  # [L, D]
