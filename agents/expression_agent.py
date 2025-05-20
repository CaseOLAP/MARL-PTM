# agents/expression_agent.py

import torch
import torch.nn as nn
from agents.base_agent import BaseAgent

class ExpressionAgent(BaseAgent):
    """
    Learns contextual embeddings from per-residue expression levels.
    Input: FloatTensor [L] (expression per residue)
    Output: FloatTensor [L, D] (context-aware embeddings)
    """

    def __init__(self, input_dim=1, hidden_dim=64, output_dim=128, max_len=100, device='cpu'):
        super().__init__(input_dim=max_len, output_dim=output_dim, device=device)

        self.positional_encoding = nn.Parameter(torch.randn(max_len, input_dim))
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        """
        x: FloatTensor [L] (expression values)
        returns: FloatTensor [L, D]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(-1)  # [1, L, 1]
        elif x.dim() == 2:
            x = x.unsqueeze(0)  # [1, L, 1]

        L = x.size(1)
        x = x + self.positional_encoding[:L]  # Add positional encoding
        output, _ = self.gru(x)               # [1, L, 2H]
        x = self.proj(output)                 # [1, L, D]
        return x.squeeze(0)                   # [L, D]
