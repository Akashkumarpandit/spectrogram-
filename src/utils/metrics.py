import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_recall_fscore_support


def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and the optimal decision threshold.
    
    In ASVspoof conventions:
    - High score represents Bona fide (human voice).
    - Low score represents Spoof (synthetic/replay/voice clone).

    Args:
        bonafide_scores: 1D array of scores for genuine human utterances.
        spoof_scores: 1D array of scores for spoofed / deepfake utterances.
    Returns:
        eer: Equal Error Rate in percentage (e.g., 2.35%).
        threshold: Decision threshold where FAR == FRR.
    """
    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return 0.0, 0.0

    labels = np.concatenate([np.ones_like(bonafide_scores), np.zeros_like(spoof_scores)])
    scores = np.concatenate([bonafide_scores, spoof_scores])

    # Compute False Acceptance Rate (FAR) and False Rejection Rate (FRR)
    # y_true = 1 for bonafide, 0 for spoof
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    # Find the threshold where FAR (fpr) == FRR (fnr)
    diff = np.abs(fpr - fnr)
    idx = np.nanargmin(diff)

    eer = (fpr[idx] + fnr[idx]) / 2.0 * 100.0
    threshold = thresholds[idx]

    return float(eer), float(threshold)


def compute_min_tdcf(
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
    p_spoof: float = 0.05,
    p_tar: float = 0.90,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
    c_fa_asv: float = 10.0,
) -> float:
    """
    Simplified tandem Detection Cost Function (min t-DCF) for ASVspoof.
    Evaluates countermeasure reliability under tandem operation with an ASV system.
    """
    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return 0.0

    labels = np.concatenate([np.ones_like(bonafide_scores), np.zeros_like(spoof_scores)])
    scores = np.concatenate([bonafide_scores, spoof_scores])

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    # Simplified cost formulation
    beta = (c_fa * (1 - p_tar)) / (c_miss * p_tar)
    t_dcf_norm = fnr + beta * fpr
    min_t_dcf = np.min(t_dcf_norm)

    return float(min_t_dcf)


def compute_all_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive metrics: EER, optimal threshold, AUC, Accuracy, Precision, Recall, F1.
    
    Args:
        y_true: 1D array of 0 (spoof) and 1 (bonafide).
        y_scores: 1D array of continuous scores (higher = more likely bonafide).
        threshold: Optional preset threshold. If None, uses EER threshold.
    """
    bonafide_scores = y_scores[y_true == 1]
    spoof_scores = y_scores[y_true == 0]

    eer, opt_thresh = compute_eer(bonafide_scores, spoof_scores)
    eval_thresh = threshold if threshold is not None else opt_thresh

    min_tdcf = compute_min_tdcf(bonafide_scores, spoof_scores)

    # Predictions based on threshold
    y_pred = (y_scores >= eval_thresh).astype(int)

    acc = accuracy_score(y_true, y_pred) * 100.0
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_scores) * 100.0
    except Exception:
        auc = 50.0

    return {
        "eer": round(eer, 4),
        "threshold": round(eval_thresh, 4),
        "min_tdcf": round(min_tdcf, 4),
        "auc": round(auc, 2),
        "accuracy": round(acc, 2),
        "precision": round(prec * 100.0, 2),
        "recall": round(rec * 100.0, 2),
        "f1": round(f1 * 100.0, 2),
    }
