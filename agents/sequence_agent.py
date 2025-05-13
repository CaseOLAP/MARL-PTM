import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.base_agent import BaseAgent

class SequenceQNetwork(nn.Module):
    """
    Q-network for the Sequence Agent.
    Processes residue-wise transformer embeddings using 1D convolution
    followed by fully connected layers to output PTM action scores.
    """

    def __init__(self, input_dim, output_dim):
        super(SequenceQNetwork, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        """
        Forward pass through the Q-network.
        Input shape: (batch_size, sequence_len, embedding_dim)
        Output shape: (batch_size, sequence_len)
        """
        # Rearrange to (batch_size, embedding_dim, sequence_len)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # Back to (batch_size, sequence_len, features)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x.squeeze(-1)  # Output PTM score per residue


class SequenceAgent(BaseAgent):
    """
    Sequence Agent specialized for processing protein sequence embeddings.
    Uses a convolutional neural network to learn residue-level PTM predictions.
    """

    def __init__(self, input_dim=1280, action_dim=1024, **kwargs):
        super().__init__(input_dim, action_dim, **kwargs)

    def build_model(self):
        """
        Constructs the modality-specific Q-network for sequence features.
        """
        return SequenceQNetwork(input_dim=self.input_dim, output_dim=self.action_dim)
