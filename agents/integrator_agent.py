# agents/integrator_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.base_agent import BaseAgent

class IntegratorAgent(BaseAgent):
    """
    Learns to integrate embeddings from multiple data agents using attention.
    Input: Dict[str → Tensor[L, D]] from 6 data agents
    Output: Tensor[L, D] fused representation
    """

    def __init__(self, input_modalities=6, embed_dim=128, device='cpu'):
        super().__init__(input_dim=embed_dim, output_dim=embed_dim, device=device)

        self.modalities = input_modalities
        self.attention_layer = nn.Sequential(
            nn.Linear(embed_dim * input_modalities, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, input_modalities),  # produce per-modality weights
            nn.Softmax(dim=-1)
        )

    def forward(self, modality_embeddings):
        """
        modality_embeddings: dict of {modality_name: [L, D]}
        returns: Tensor [L, D]
        """
        assert len(modality_embeddings) == self.modalities, \
            f"Expected {self.modalities} modalities, got {len(modality_embeddings)}"

        # Stack modality tensors into one tensor: [L, M, D]
        tensors = torch.stack(list(modality_embeddings.values()), dim=1)  # [L, M, D]

        # Flatten for attention input: [L, M*D]
        concat_flat = tensors.view(tensors.size(0), -1)  # [L, M*D]

        # Compute attention weights across modalities: [L, M]
        weights = self.attention_layer(concat_flat)  # [L, M]
        weights = weights.unsqueeze(-1)              # [L, M, 1]

        # Weighted sum over modalities: [L, D]
        fused = torch.sum(weights * tensors, dim=1)  # [L, D]
        return fused
