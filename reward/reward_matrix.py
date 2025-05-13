def compute_reward_matrix(agent_name, is_correct, is_critical, confidence, context_supported, agent_agreement):
    """
    Computes reward value for a single agent based on biological and statistical criteria.
    
    Parameters:
        agent_name (str): One of ['sequence', 'structure', 'graph', 'expression', 'proteoform', 'final']
        is_correct (bool): Whether the agent's prediction matches the ground truth
        is_critical (bool): Whether the site is a biologically critical residue
        confidence (float): Confidence score from the agent's Q-network
        context_supported (bool): Whether biological context supports the prediction
        agent_agreement (bool): Whether this agent agrees with other agents

    Returns:
        reward (float): Scalar reward for this agent
    """

    reward = 0.0

    if is_correct:
        if context_supported:
            reward += 3.0 if agent_name == 'structure' or agent_name == 'graph' else 2.0
        else:
            reward += 1.0
    else:
        if is_critical:
            reward -= 3.0 if agent_name == 'structure' else 2.0
        else:
            reward -= 1.0

    # Confidence reward shaping
    if confidence >= 0.9:
        reward += 1.0
    elif confidence <= 0.5:
        reward -= 1.0

    # Context bonus
    if context_supported:
        if agent_name in ['structure', 'graph', 'expression', 'proteoform']:
            reward += 2.0

    # Agreement bonus
    if agent_agreement:
        reward += 1.0

    # Final integration bonus
    if agent_name == 'final' and is_correct:
        reward += 5.0 if context_supported else 2.0
        if is_critical:
            reward += 2.0
        if not is_correct:
            reward -= 5.0

    return reward
