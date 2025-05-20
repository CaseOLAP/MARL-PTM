# agents/ptm_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
#from agents.base_agent import BaseAgent

class PTMAgent(BaseAgent):
    """
    Learns to predict PTM probability scores at each residue.
    Input: integrated embedding tensor [L, D]
    Output: PTM scores per residue [L]
    """

    def __init__(self, input_dim=128, hidden_dim=64, device='cpu'):
        super().__init__(input_dim=input_dim, output_dim=1, device=device)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)  # Binary classification per residue
        )

    def forward(self, x):
        """
        x: Tensor [L, D] — integrated input from the IntegratorAgent
        returns: Tensor [L] — PTM probabilities per residue
        """
        logits = self.classifier(x).squeeze(-1)        # [L]
        probs = torch.sigmoid(logits)                  # [L]
        return probs
