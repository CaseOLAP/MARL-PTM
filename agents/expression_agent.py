import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.base_agent import BaseAgent

class ExpressionQNetwork(nn.Module):
    """
    Q-network for the Expression Agent.
    Projects compressed expression embeddings into residue-level PTM predictions.
    """

    def __init__(self, input_dim, output_dim):
        super(ExpressionQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        """
        Forward pass through the expression Q-network.
        Input: (batch_size, input_dim)
        Output: (batch_size, sequence_len)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ExpressionAgent(BaseAgent):
    """
    Expression Agent for integrating transcriptomic context into PTM prediction.
    Accepts PCA-reduced expression vectors and expands to residue-level output.
    """

    def __init__(self, input_dim=128, action_dim=1024, **kwargs):
        super().__init__(input_dim, action_dim, **kwargs)

    def build_model(self):
        """
        Constructs the Q-network for gene expression embeddings.
        """
        return ExpressionQNetwork(input_dim=self.input_dim, output_dim=self.action_dim)
