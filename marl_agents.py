import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SequenceAgent, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class StructureAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(StructureAgent, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class GraphAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphAgent, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class GeneExpressionAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GeneExpressionAgent, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
