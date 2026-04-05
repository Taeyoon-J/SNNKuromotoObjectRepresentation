import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score

def compute_metrics(y_true, probs, average='macro'):
    """
    Computes classification metrics for validation.

    Args:
        y_true (np.array): Ground truth labels (flattened).
        probs (np.array): Prediction probabilities [N, NumClasses].
        average (str): Averaging method for multi-class metrics ('macro', 'micro', or 'weighted').
    
    Returns:
        tuple: (accuracy, precision, recall, f1_score, auc_score)
    """
    y_pred = probs.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    
    uniq = np.unique(y_true)
    if uniq.size < 2:
        # Unable to compute AUC if only one class is present in the batch
        auc = float("nan")
    else:
        try:
            if probs.shape[1] == 2: 
                # Binary classification case
                auc = roc_auc_score(y_true, probs[:, 1])
            else: 
                # Multi-class case (One-vs-Rest)
                auc = roc_auc_score(y_true, probs, multi_class='ovr', average=average)
        except ValueError:
            auc = float("nan")
            
    return acc, precision, recall, f1, auc