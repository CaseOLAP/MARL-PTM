# agents/structure_agent.py

import torch
import torch.nn as nn
from agents.base_agent import BaseAgent

class StructureAgent(BaseAgent):
    """
    Learns contextual embeddings from secondary structure (DSSP) codes.
    Input: LongTensor [L] (encoded DSSP symbols)
    Output: FloatTensor [L, D] (contextual structural embedding)
    """

    def __init__(self, vocab_size=4, embed_dim=32, hidden_dim=64, output_dim=128, max_len=100, device='cpu'):
        super().__init__(input_dim=max_len, output_dim=output_dim, device=device)

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.positional_encoding = nn.Parameter(torch.randn(max_len, embed_dim))
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        """
        x: LongTensor [L] (DSSP encoding)
        returns: FloatTensor [L, D]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, L]

        x = self.embedding(x) + self.positional_encoding[:x.size(1)]  # [B, L, E]
        output, _ = self.encoder(x)  # [B, L, 2H]
        x = self.proj(output)        # [B, L, D]
        return x.squeeze(0)          # [L, D]
