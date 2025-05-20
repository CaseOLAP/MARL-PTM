# training/replay_buffer.py

import random
from collections import deque
import torch

class ReplayBuffer:
    """
    Generic experience replay buffer for MARL PTM agents.
    Stores (state, action, reward, next_state, done) tuples.
    """

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Stores a single transition tuple.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Randomly samples a batch of transitions from the buffer.
        Returns separate batches for each component.
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(list, zip(*batch))

        return (
            torch.stack(state),
            torch.stack(action),
            torch.tensor(reward, dtype=torch.float),
            torch.stack(next_state),
            torch.tensor(done, dtype=torch.float)
        )

    def __len__(self):
        return len(self.buffer)
