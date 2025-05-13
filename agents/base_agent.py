import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class BaseAgent:
    """
    Abstract base class for all MARL-PTM agents.
    Implements shared reinforcement learning logic, including Q-network,
    target network, action selection, experience replay, and learning step.
    """

    def __init__(self, input_dim, action_dim, lr=1e-4, gamma=0.95, buffer_size=10000, batch_size=64, device='cpu'):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

        self.q_network = self.build_model().to(self.device)
        self.target_network = self.build_model().to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)

        self.replay_buffer = deque(maxlen=buffer_size)
        self.loss_fn = nn.MSELoss()

    def build_model(self):
        """
        Override this method in child classes.
        Should return a PyTorch model that maps input_dim → action_dim.
        """
        raise NotImplementedError("Each agent must implement its own model.")

    def act(self, state, epsilon=0.1):
        """
        Select action based on ε-greedy strategy.
        Returns a vector of action probabilities or scores.
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if np.random.rand() < epsilon:
            return np.random.rand(self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
            return q_values.cpu().numpy().squeeze()

    def store_transition(self, state, action, reward, next_state):
        """
        Stores a single transition in the replay buffer.
        """
        self.replay_buffer.append((state, action, reward, next_state))

    def sample_batch(self):
        """
        Samples a mini-batch from the replay buffer.
        """
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        return (
            torch.FloatTensor(states).to(self.device),
            torch.FloatTensor(actions).to(self.device),
            torch.FloatTensor(rewards).unsqueeze(1).to(self.device),
            torch.FloatTensor(next_states).to(self.device)
        )

    def update_target_network(self):
        """
        Copies weights from q_network to target_network.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def learn(self):
        """
        Perform one learning step using a sampled batch.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states = self.sample_batch()

        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(dim=1, keepdim=True)[0]

        target_q = rewards + self.gamma * max_next_q
        predicted_q = self.q_network(states).gather(1, actions.long().unsqueeze(1))

        loss = self.loss_fn(predicted_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
