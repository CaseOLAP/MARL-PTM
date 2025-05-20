# utils/visualization.py

import matplotlib.pyplot as plt

def plot_loss_curve(losses, save_path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_ptm_scores(ptm_scores, true_sites=None):
    """
    ptm_scores: Tensor [L], predicted PTM probabilities
    true_sites: List[int], optional list of true PTM indices
    """
    L = len(ptm_scores)
    plt.figure(figsize=(10, 2))
    plt.plot(ptm_scores.detach().cpu(), label="PTM Score")
    if true_sites:
        for idx in true_sites:
            plt.axvline(x=idx, color='red', linestyle='--', alpha=0.5, label='True PTM')
    plt.xlabel("Residue Index")
    plt.ylabel("Score")
    plt.title("PTM Site Prediction")
    plt.legend()
    plt.show()
