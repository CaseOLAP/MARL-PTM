import numpy as np
import torch

from reward.reward_agent import RewardAgent
from env.marl_env import MultiAgentEnvironment

class Trainer:
    """
    Coordinates training of all MARL-PTM agents in a unified loop.
    Calls agent methods for action selection, reward computation, and learning.
    """

    def __init__(self, agents, ptm_agent, env, data_loader, reward_config, device='cpu'):
        """
        Parameters:
            agents (dict): {agent_name: agent_instance}
            ptm_agent (PTMIntegrationAgent): Aggregates per-agent predictions
            env (MultiAgentEnvironment): Agent environment interface
            data_loader (iterable): Yields (protein_id, ground_truth, context)
            reward_config (dict): Parameters for reward agent
        """
        self.agents = agents
        self.ptm_agent = ptm_agent
        self.env = env
        self.data_loader = data_loader
        self.reward_agent = RewardAgent(**reward_config)
        self.device = device

    def train(self, num_epochs=10, epsilon=0.1, update_target_every=5):
        """
        Executes full multi-agent training loop.
        """
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}")
            total_loss = 0.0
            for step, (protein_id, ground_truth, context_info) in enumerate(self.data_loader):

                # Step 1: Get current states per agent
                states = self.env.get_states(protein_id)

                # Step 2: Each agent selects an action
                predictions = {}
                confidences = {}
                for name, agent in self.agents.items():
                    raw = agent.act(states[name], epsilon)
                    predictions[name] = raw
                    confidences[name] = np.clip(raw, 0, 1)

                # Step 3: Integration agent generates final prediction
                agent_outputs = [predictions[k] for k in sorted(self.agents.keys())]
                final_pred = self.ptm_agent.integrate(agent_outputs)
                predictions["final"] = final_pred
                confidences["final"] = final_pred  # for now use same

                # Step 4: Compute rewards using reward agent
                reward_inputs = {
                    name: (predictions[name], confidences[name]) for name in predictions
                }
                rewards = self.reward_agent.compute(reward_inputs, ground_truth, context_info)

                # Step 5: Update each agent
                for name, agent in self.agents.items():
                    state = torch.FloatTensor(states[name])
                    action = torch.FloatTensor(predictions[name])
                    reward = torch.FloatTensor(rewards[name])
                    next_state = state.clone()  # Static input; no temporal transitions

                    agent.store_transition(state, action, reward.mean(), next_state)
                    agent.learn()

                    if epoch % update_target_every == 0:
                        agent.update_target_network()
