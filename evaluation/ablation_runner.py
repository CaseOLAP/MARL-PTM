import copy
from evaluation.evaluator import Evaluator

class AblationRunner:
    """
    Automates ablation testing by disabling one or more agents and re-evaluating model performance.
    """

    def __init__(self, base_agents, ptm_agent, env, test_loader, reward_agent, device='cpu'):
        """
        Parameters:
            base_agents (dict): All trained agents {agent_name: agent_instance}
            ptm_agent (PTMIntegrationAgent): Integration agent
            env (MultiAgentEnvironment): Evaluation environment
            test_loader (iterable): Iterable yielding (protein_id, ground_truth, context)
            reward_agent (RewardAgent): Used for computing alignment/penalty bonuses
            device (str): Compute device
        """
        self.base_agents = base_agents
        self.ptm_agent = ptm_agent
        self.env = env
        self.test_loader = test_loader
        self.reward_agent = reward_agent
        self.evaluator = Evaluator()
        self.device = device

    def run_ablation(self, disable_agents):
        """
        Disables specific agents and evaluates overall performance.

        Parameters:
            disable_agents (list): List of agent names to exclude

        Returns:
            dict: Evaluation metrics from Evaluator
        """
        agents = {k: v for k, v in self.base_agents.items() if k not in disable_agents}
        results = {}
        prediction_dict = {}
        ground_truth_dict = {}

        for protein_id, y_true, context in self.test_loader:
            states = self.env.get_states(protein_id)

            agent_outputs = []
            for name in sorted(agents.keys()):
                state = states[name]
                pred = agents[name].act(state, epsilon=0.0)
                agent_outputs.append(pred)

            # Use integration agent to predict
            y_pred = self.ptm_agent.integrate(agent_outputs)
            prediction_dict[protein_id] = y_pred
            ground_truth_dict[protein_id] = y_true

        results = self.evaluator.evaluate_all(prediction_dict, ground_truth_dict)
        return results

    def run_all_ablations(self):
        """
        Runs full ablation study, disabling one agent at a time.

        Returns:
            dict: {agent_name: metric_results}
        """
        ablation_results = {}
        agent_names = list(self.base_agents.keys())

        for name in agent_names:
            print(f"Running ablation: removing {name}")
            result = self.run_ablation(disable_agents=[name])
            ablation_results[name] = result

        return ablation_results
