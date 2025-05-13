import numpy as np
from evaluation.interpretability import InterpretabilityTools

class CaseStudyRunner:
    """
    Performs detailed prediction and interpretability analysis for selected proteins.
    Intended for validation against experimentally validated PTM sites.
    """

    def __init__(self, agents, ptm_agent, environment, known_ptm_dict):
        """
        Parameters:
            agents (dict): Trained agent instances
            ptm_agent (PTMIntegrationAgent): Meta-policy for final prediction
            environment (MultiAgentEnvironment): Access to protein state inputs
            known_ptm_dict (dict): {protein_id: list of validated PTM positions}
        """
        self.agents = agents
        self.ptm_agent = ptm_agent
        self.environment = environment
        self.known_ptm_dict = known_ptm_dict
        self.viz = InterpretabilityTools(ptm_agent)

    def analyze_protein(self, protein_id):
        """
        Analyzes and visualizes predictions for a single protein.

        Parameters:
            protein_id (str): ID of the protein to analyze

        Returns:
            dict: {
                'predicted_sites': list of high-confidence PTM positions,
                'matched_sites': intersection with validated sites,
                'false_positives': predicted but not validated
            }
        """
        # Step 1: Collect agent states
        states = self.environment.get_states(protein_id)

        # Step 2: Collect agent predictions
        agent_outputs = {}
        for name, agent in self.agents.items():
            agent_outputs[name] = agent.act(states[name], epsilon=0.0)

        # Step 3: Get integrated prediction
        sorted_outputs = [agent_outputs[k] for k in sorted(agent_outputs)]
        y_pred = self.ptm_agent.integrate(sorted_outputs)

        # Step 4: Get known PTM sites
        validated = set(self.known_ptm_dict.get(protein_id, []))
        predicted = set(np.where(y_pred > 0.5)[0])

        matched = predicted.intersection(validated)
        false_positives = predicted.difference(validated)

        # Step 5: Visualizations
        self.viz.plot_residue_prediction_map(y_pred, [1 if i in validated else 0 for i in range(len(y_pred))])
        self.viz.plot_agent_outputs(agent_outputs)

        return {
            'predicted_sites': sorted(predicted),
            'matched_sites': sorted(matched),
            'false_positives': sorted(false_positives)
        }
