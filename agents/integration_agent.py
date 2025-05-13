import torch
import torch.nn as nn
import torch.nn.functional as F

class PTMIntegrationNetwork(nn.Module):
    """
    Meta-policy network for integrating outputs from all modality-specific agents.
    Applies attention or gating mechanisms to weigh and combine agent predictions.
    """

    def __init__(self, num_agents, action_dim):
        super(PTMIntegrationNetwork, self).__init__()
        self.num_agents = num_agents
        self.action_dim = action_dim

        # Attention weights for agent contributions (learnable)
        self.attention = nn.Parameter(torch.ones(num_agents))

        # Optional MLP for fusion
        self.fc1 = nn.Linear(num_agents, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, agent_outputs):
        """
        Forward pass through integration network.
        agent_outputs: List of (batch_size, sequence_len) tensors from each agent
        """
        # Stack all agent outputs into a tensor of shape (batch_size, sequence_len, num_agents)
        stacked = torch.stack(agent_outputs, dim=-1)  # shape: (B, L, N)

        # Normalize attention weights
        alpha = F.softmax(self.attention, dim=0)  # shape: (N,)

        # Weighted sum over agent outputs
        fused = torch.einsum('bln,n->bl', stacked, alpha)  # shape: (B, L)

        # Optional post-processing with MLP
        fused_expanded = fused.unsqueeze(-1)  # shape: (B, L, 1)
        fused_mlp = F.relu(self.fc1(stacked))  # shape: (B, L, 64)
        fused_mlp = self.fc2(fused_mlp).squeeze(-1)  # shape: (B, L)

        return fused + fused_mlp  # residual combination
        

class PTMIntegrationAgent:
    """
    The PTM Integration Agent aggregates predictions from all core agents
    to produce the final PTM prediction at each residue.
    """

    def __init__(self, num_agents=5, action_dim=1024, device='cpu'):
        self.model = PTMIntegrationNetwork(num_agents, action_dim).to(device)
        self.device = device

    def integrate(self, agent_outputs):
        """
        Accepts a list of predictions from each agent and returns final output.
        Each prediction should be of shape (batch_size, sequence_len)
        """
        agent_outputs = [torch.FloatTensor(out).to(self.device) for out in agent_outputs]
        return self.model(agent_outputs).detach().cpu().numpy()
