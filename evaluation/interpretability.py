import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class InterpretabilityTools:
    """
    Provides interpretability tools for analyzing agent contributions and prediction rationale.
    Visualizes attention weights, per-residue predictions, and modality contributions.
    """

    def __init__(self, ptm_agent):
        """
        Parameters:
            ptm_agent (PTMIntegrationAgent): Integration agent with attention weights
        """
        self.ptm_agent = ptm_agent

    def get_attention_weights(self):
        """
        Extracts normalized attention weights from the integration model.
        Returns:
            np.ndarray of shape (num_agents,)
        """
        alpha = self.ptm_agent.model.attention.detach().cpu().numpy()
        weights = np.exp(alpha) / np.sum(np.exp(alpha))
        return weights

    def plot_attention_weights(self, agent_names):
        """
        Plots attention weights assigned to each agent.
        """
        weights = self.get_attention_weights()
        plt.figure(figsize=(8, 4))
        sns.barplot(x=agent_names, y=weights)
        plt.title("Attention Weights per Agent")
        plt.ylabel("Weight")
        plt.xlabel("Agent")
        plt.tight_layout()
        plt.show()

    def plot_residue_prediction_map(self, y_pred, y_true, title="PTM Prediction vs Ground Truth"):
        """
        Plots a heatmap comparing predicted and true PTM sites for a single protein.

        Parameters:
            y_pred (np.ndarray): Predicted PTM probabilities (length L)
            y_true (np.ndarray): Binary ground truth labels (length L)
        """
        heatmap_data = np.vstack([y_pred, y_true])
        plt.figure(figsize=(10, 2))
        sns.heatmap(heatmap_data, cmap="YlGnBu", cbar=True, xticklabels=False, yticklabels=["Pred", "True"])
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_agent_outputs(self, agent_outputs, residue_range=None):
        """
        Plots raw outputs of each agent for a single protein.

        Parameters:
            agent_outputs (dict): {agent_name: np.ndarray of shape (L,)}
            residue_range (tuple): Optional (start, end) to slice sequence
        """
        plt.figure(figsize=(12, 4))
        for name, scores in agent_outputs.items():
            if residue_range:
                scores = scores[residue_range[0]:residue_range[1]]
            plt.plot(scores, label=name)

        plt.title("Agent Output Scores per Residue")
        plt.xlabel("Residue Index")
        plt.ylabel("PTM Likelihood")
        plt.legend()
        plt.tight_layout()
        plt.show()
