import numpy as np
from reward.reward_matrix import compute_reward_matrix

class RewardAgent:
    """
    RewardAgent dynamically computes per-agent reward values based on 
    correctness, confidence, context support, and agreement with other agents.
    """

    def __init__(self, agreement_threshold=0.6):
        self.agreement_threshold = agreement_threshold

    def compute(self, predictions_dict, ground_truth, context_info):
        """
        Computes reward for each agent based on predictions and context.

        Parameters:
            predictions_dict (dict): 
                {
                    'sequence': (pred_scores_seq, confidence_seq),
                    'structure': (pred_scores_struct, confidence_struct),
                    ...,
                    'final': (pred_scores_final, confidence_final)
                }
            ground_truth (np.array): Binary labels for each residue (shape: [batch, L])
            context_info (dict): {
                'is_critical': [bool], 
                'context_supported': {
                    'sequence': [...],
                    'structure': [...],
                    ...
                }
            }

        Returns:
            rewards (dict): Mapping of agent_name → reward array [batch, L]
        """

        agent_names = list(predictions_dict.keys())
        num_agents = len(agent_names)
        batch_size, seq_len = ground_truth.shape

        # Initialize reward dictionary
        rewards = {name: np.zeros((batch_size, seq_len)) for name in agent_names}

        # Collect binary decisions per agent to evaluate agreement
        binary_decisions = {}
        for name, (scores, _) in predictions_dict.items():
            binary_decisions[name] = (scores > 0.5).astype(int)

        # Compute majority agreement at each residue
        agreement_map = np.zeros((batch_size, seq_len))
        for b in range(batch_size):
            for l in range(seq_len):
                votes = [binary_decisions[name][b, l] for name in agent_names if name != 'final']
                agreement_ratio = sum(votes) / len(votes)
                agreement_map[b, l] = 1 if agreement_ratio >= self.agreement_threshold else 0

        # Compute rewards per agent and per residue
        for name in agent_names:
            preds, confs = predictions_dict[name]
            for b in range(batch_size):
                for l in range(seq_len):
                    is_correct = int(preds[b, l] > 0.5) == int(ground_truth[b, l])
                    is_critical = context_info['is_critical'][b][l]
                    context_match = context_info['context_supported'].get(name, [[False]*seq_len]*batch_size)[b][l]
                    agreement = bool(agreement_map[b, l])

                    rewards[name][b, l] = compute_reward_matrix(
                        agent_name=name,
                        is_correct=is_correct,
                        is_critical=is_critical,
                        confidence=confs[b, l],
                        context_supported=context_match,
                        agent_agreement=agreement
                    )

        return rewards
