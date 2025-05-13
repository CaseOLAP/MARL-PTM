import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.base_agent import BaseAgent

class GraphQNetwork(nn.Module):
    """
    Q-network for the Graph Agent.
    Processes graph-based node embeddings (e.g., from GraphSAGE or GAT)
    to predict PTM likelihoods per residue.
    """

    def __init__(self, input_dim, output_dim):
        super(GraphQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        """
        Forward pass through the graph Q-network.
        Input: (batch_size, sequence_len, input_dim)
        Output: (batch_size, sequence_len)
        """
        batch_size, seq_len, dim = x.shape
        x = x.view(-1, dim)  # Flatten to (batch_size * sequence_len, input_dim)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x = x.view(batch_size, seq_len)  # Reshape back to (batch_size, sequence_len)
        return x


class GraphAgent(BaseAgent):
    """
    Graph Agent specialized for interpreting pathway-based node embeddings.
    Predicts PTM probability based on context from protein interaction networks.
    """

    def __init__(self, input_dim=128, action_dim=1024, **kwargs):
        super().__init__(input_dim, action_dim, **kwargs)

    def build_model(self):
        """
        Constructs the Q-network to process graph-based embeddings.
        """
        return GraphQNetwork(input_dim=self.input_dim, output_dim=self.action_dim)
