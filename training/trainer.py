import torch
import torch.nn as nn
import torch.optim as optim

from agents.sequence_agent import SequenceAgent
from agents.structure_agent import StructureAgent
from agents.graph_agent import GraphAgent
from agents.expression_agent import ExpressionAgent
from agents.proteoform_agent import ProteoformAgent
from agents.integration_agent import PTMIntegrationAgent


def initialize_agents(agent_config, device='cpu'):
    """
    Initializes all core agents using provided configuration.

    Parameters:
        agent_config (dict): Configuration dict with agent-specific hyperparameters
        device (str): 'cpu' or 'cuda'

    Returns:
        dict: {agent_name: agent_instance}
    """
    agents = {
        'sequence': SequenceAgent(**agent_config['sequence'], device=device),
        'structure': StructureAgent(**agent_config['structure'], device=device),
        'graph': GraphAgent(**agent_config['graph'], device=device),
        'expression': ExpressionAgent(**agent_config['expression'], device=device),
        'proteoform': ProteoformAgent(**agent_config['proteoform'], device=device)
    }
    return agents


def initialize_ptm_agent(num_agents=5, action_dim=1024, device='cpu'):
    """
    Initializes the PTMIntegrationAgent.

    Parameters:
        num_agents (int): Number of core agents to integrate
        action_dim (int): Sequence length of predictions
        device (str): Torch device

    Returns:
        PTMIntegrationAgent
    """
    return PTMIntegrationAgent(num_agents=num_agents, action_dim=action_dim, device=device)


def configure_optimizer(model, lr=1e-4, weight_decay=1e-5):
    """
    Returns a standard Adam optimizer for a given model.

    Parameters:
        model (torch.nn.Module)
        lr (float): Learning rate
        weight_decay (float): L2 regularization

    Returns:
        torch.optim.Optimizer
    """
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def configure_scheduler(optimizer, decay_rate=0.95, step_size=10):
    """
    Configures a learning rate scheduler.

    Parameters:
        optimizer (torch.optim.Optimizer)
        decay_rate (float)
        step_size (int)

    Returns:
        torch.optim.lr_scheduler.StepLR
    """
    return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=decay_rate)
