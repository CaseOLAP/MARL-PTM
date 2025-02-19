def reward_function(predicted_site, actual_site, confidence_score, pathway_support):
    base_reward = 2 if predicted_site == actual_site else -1
    confidence_bonus = confidence_score * 0.5
    pathway_bonus = 2 if pathway_support else 0
    return base_reward + confidence_bonus + pathway_bonus
