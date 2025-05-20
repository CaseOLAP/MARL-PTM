# utils/metrics.py

import torch
from sklearn.metrics import precision_score, recall_score, f1_score

def compute_binary_metrics(preds, targets, threshold=0.5):
    """
    Computes binary classification metrics for a single PTM prediction vector.
    
    Args:
        preds: Tensor [L] — predicted scores
        targets: Tensor [L] — true binary labels
        threshold: float — threshold to binarize predictions

    Returns:
        dict: precision, recall, f1
    """
    preds_bin = (preds > threshold).long().cpu().numpy()
    targets = targets.long().cpu().numpy()

    precision = precision_score(targets, preds_bin, zero_division=0)
    recall = recall_score(targets, preds_bin, zero_division=0)
    f1 = f1_score(targets, preds_bin, zero_division=0)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }
