import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.base_agent import BaseAgent

class ProteoformQNetwork(nn.Module):
    """
    Q-network for the Proteoform Agent.
    Learns to interpret isoform-specific features such as domain presence or residue masking.
    """

    def __init__(self, input_dim, output_dim):
        super(ProteoformQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        """
        Forward pass through the proteoform Q-network.
        Input: (batch_size, sequence_len, input_dim)
        Output: (batch_size, sequence_len)
        """
        batch_size, seq_len, dim = x.shape
        x = x.view(-1, dim)  # Flatten to (batch_size * sequence_len, input_dim)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x = x.view(batch_size, seq_len)  # Reshape to (batch_size, sequence_len)
        return x


class ProteoformAgent(BaseAgent):
    """
    Proteoform Agent for isoform-aware PTM prediction.
    Focuses on alternative splicing, domain-level masking, and accessibility restrictions.
    """

    def __init__(self, input_dim=64, action_dim=1024, **kwargs):
        super().__init__(input_dim, action_dim, **kwargs)

    def build_model(self):
        """
        Constructs the Q-network for proteoform domain features.
        """
        return ProteoformQNetwork(input_dim=self.input_dim, output_dim=self.action_dim)
