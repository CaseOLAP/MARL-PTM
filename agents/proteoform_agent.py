# agents/proteoform_agent.py

import torch
import torch.nn as nn
from agents.base_agent import BaseAgent

class ProteoformAgent(BaseAgent):
    """
    Learns embeddings from residue-wise PTM priors (e.g., historical modifications).
    Input: LongTensor [L] (one-hot or index per residue indicating prior PTM)
    Output: FloatTensor [L, D]
    """

    def __init__(self, ptm_vocab_size=14, embed_dim=32, hidden_dim=64, output_dim=128, max_len=100, device='cpu'):
        super().__init__(input_dim=max_len, output_dim=output_dim, device=device)

        self.embedding = nn.Embedding(num_embeddings=ptm_vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.positional_encoding = nn.Parameter(torch.randn(max_len, embed_dim))
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        """
        x: LongTensor [L] (prior PTM token per residue; 0 if none)
        returns: FloatTensor [L, D]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, L]

        x = self.embedding(x) + self.positional_encoding[:x.size(1)]  # [1, L, E]
        x, _ = self.encoder(x)                                        # [1, L, 2H]
        x = self.proj(x)                                              # [1, L, D]
        return x.squeeze(0)                                           # [L, D]
