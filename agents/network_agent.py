# agents/network_agent.py

import torch
import torch.nn as nn
from agents.base_agent import BaseAgent

class NetworkAgent(BaseAgent):
    """
    Learns pathway-contextual embeddings per residue by broadcasting pathway identity.
    Input: List of pathway indices → embedded, broadcasted to all residues
    Output: FloatTensor [L, D]
    """

    def __init__(self, num_pathways=20, embed_dim=64, output_dim=128, max_len=100, device='cpu'):
        super().__init__(input_dim=max_len, output_dim=output_dim, device=device)

        self.pathway_embedding = nn.Embedding(num_pathways + 1, embed_dim, padding_idx=0)
        self.positional_encoding = nn.Parameter(torch.randn(max_len, embed_dim))
        self.proj = nn.Linear(embed_dim, output_dim)

    def forward(self, pathway_ids, residue_len=100):
        """
        pathway_ids: list of ints (pathway indices for this protein)
        residue_len: int (number of residues, typically max_len)
        returns: FloatTensor [L, D]
        """
        if not pathway_ids:
            pathway_ids = [0]  # padding / unknown

        pathway_ids = torch.tensor(pathway_ids, dtype=torch.long).to(self.device)  # [P]
        pathway_embed = self.pathway_embedding(pathway_ids).mean(dim=0)            # [E]
        broadcast = pathway_embed.unsqueeze(0).repeat(residue_len, 1)              # [L, E]

        x = broadcast + self.positional_encoding[:residue_len]                     # [L, E]
        x = self.proj(x)                                                           # [L, D]
        return x
