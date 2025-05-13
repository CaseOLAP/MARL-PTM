import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

class Evaluator:
    """
    Evaluates trained MARL-PTM agents on a test dataset.
    Computes standard classification metrics per residue and per protein.
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def evaluate_predictions(self, y_true, y_pred):
        """
        Evaluates a batch of PTM predictions.
        
        Parameters:
            y_true (np.ndarray): Ground truth binary labels (shape: [N, L])
            y_pred (np.ndarray): Predicted scores (shape: [N, L])

        Returns:
            dict: Evaluation metrics
        """
        # Flatten predictions and ground truth
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        y_pred_bin = (y_pred_flat >= self.threshold).astype(int)

        results = {
            'accuracy': accuracy_score(y_true_flat, y_pred_bin),
            'precision': precision_score(y_true_flat, y_pred_bin, zero_division=0),
            'recall': recall_score(y_true_flat, y_pred_bin, zero_division=0),
            'f1_score': f1_score(y_true_flat, y_pred_bin, zero_division=0),
            'auroc': roc_auc_score(y_true_flat, y_pred_flat) if len(np.unique(y_true_flat)) > 1 else None,
            'auprc': average_precision_score(y_true_flat, y_pred_flat)
        }

        return results

    def evaluate_all(self, prediction_dict, ground_truth_dict):
        """
        Evaluates multiple proteins across the test set.

        Parameters:
            prediction_dict (dict): {protein_id: predicted_scores}
            ground_truth_dict (dict): {protein_id: true_labels}

        Returns:
            dict: Average metrics across proteins
        """
        metrics_accum = []
        for pid in prediction_dict:
            if pid in ground_truth_dict:
                pred = prediction_dict[pid]
                true = ground_truth_dict[pid]
                metrics = self.evaluate_predictions(np.array([true]), np.array([pred]))
                metrics_accum.append(metrics)

        # Average results across all proteins
        avg_results = {}
        for key in metrics_accum[0].keys():
            values = [m[key] for m in metrics_accum if m[key] is not None]
            avg_results[key] = float(np.mean(values)) if values else None

        return avg_results
