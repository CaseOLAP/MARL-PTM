# agents/aggregator_agent.py

import torch
import torch.nn as nn
from agents.base_agent import BaseAgent

class AggregatorAgent(BaseAgent):
    """
    Aggregates predictions from 13 PTM agents.
    Input: Dict[str → Tensor[L]] — per-PTM residue scores
    Output: Dict[int → (ptm_type, score)] — final PTM per residue
    """

    def __init__(self, ptm_types, threshold=0.5, device='cpu'):
        super().__init__(input_dim=len(ptm_types), output_dim=1, device=device)
        self.ptm_types = ptm_types
        self.threshold = threshold

    def forward(self, ptm_outputs):
        """
        ptm_outputs: dict of {ptm_type: Tensor[L]} from PTMAgents
        returns: dict of {residue_index: (ptm_type, score)}
        """
        L = list(ptm_outputs.values())[0].shape[0]
        result = {}

        # For each residue, find the PTM with max score
        for i in range(L):
            best_ptm = None
            best_score = 0.0
            for ptm, scores in ptm_outputs.items():
                score = scores[i].item()
                if score > best_score and score >= self.threshold:
                    best_score = score
                    best_ptm = ptm
            if best_ptm:
                result[i] = (best_ptm, best_score)

        return result  # {residue_idx: (ptm_type, score)}
