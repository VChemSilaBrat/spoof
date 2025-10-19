import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(labels, scores):
    """
    Compute Equal Error Rate (EER)
    
    Args:
        labels: Ground truth labels (0 for bonafide, 1 for spoof)
        scores: Prediction scores
    
    Returns:
        eer: Equal Error Rate
        threshold: Threshold at EER
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    # Find the threshold where FPR and FNR are equal (or closest)
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    
    return eer * 100, eer_threshold


def compute_min_dcf(labels, scores, p_target=0.05, c_miss=1, c_fa=1):
    """
    Compute minimum Detection Cost Function (min-DCF)
    
    Args:
        labels: Ground truth labels
        scores: Prediction scores
        p_target: Prior probability of target
        c_miss: Cost of miss
        c_fa: Cost of false alarm
    
    Returns:
        min_dcf: Minimum DCF value
    """
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    min_dcf = np.min(dcf)
    
    return min_dcf
