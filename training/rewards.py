# training/rewards.py

def compute_per_residue_reward(predictions, ground_truth, ptm_type, motif_mask=None, prior_mask=None):
    """
    Computes reward for a single PTM agent over a sequence.
    
    Args:
        predictions: Tensor [L], predicted probabilities
        ground_truth: Dict[int → str], true labels {res_idx: ptm_type}
        ptm_type: str, this agent's PTM type
        motif_mask: Optional Tensor [L] — 1 if matches known motif
        prior_mask: Optional Tensor [L] — 1 if matches proteoform prior
    
    Returns:
        Tensor [L], reward per residue
    """
    L = predictions.shape[0]
    reward = [0.0] * L

    for i in range(L):
        pred_score = predictions[i].item()
        true_label = ground_truth.get(i, None)

        if true_label == ptm_type:
            reward[i] += 1.0  # correct PTM at correct site
        elif true_label is not None:
            reward[i] += 0.5  # site is correct, but PTM type is wrong
        elif pred_score > 0.5:
            reward[i] -= 1.0  # false positive

        if motif_mask is not None and motif_mask[i] == 1:
            reward[i] += 0.25  # bonus: motif match

        if prior_mask is not None and prior_mask[i] == 1:
            reward[i] += 0.25  # bonus: proteoform consistency

    return torch.tensor(reward, dtype=torch.float)
