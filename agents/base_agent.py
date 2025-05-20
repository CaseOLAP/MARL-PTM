# agents/base_agent.py

import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseAgent(nn.Module, ABC):
    """
    Base class for all agents in MARL-PTM.
    Enforces a consistent interface and shared utilities.
    """

    def __init__(self, input_dim, output_dim, device='cpu'):
        super(BaseAgent, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.to(device)

    @abstractmethod
    def forward(self, x, *args, **kwargs):
        """
        Each agent must implement its own forward method.
        """
        pass

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to(self.device)
