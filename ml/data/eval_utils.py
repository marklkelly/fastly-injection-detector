"""
Evaluation utilities for prevalence-adjusted metric estimation.

The evaluate_at_prior() function estimates PPV and F1 under a specified prior
(base rate of positives), which is typically very different from the balanced
50/50 distribution in training data.
"""
from typing import Tuple, List
import math


def evaluate_at_prior(
    y_true: List[int],
    scores: List[float],
    threshold: float,
    prior: float = 0.02,
) -> Tuple[float, float, float, float]:
    """
    Prevalence-adjusted metric estimation.

    Uses class weights proportional to:
        w_pos = prior / observed_positive_rate
        w_neg = (1 - prior) / observed_negative_rate

    This reweights balanced evaluation data to estimate performance under
    the specified real-world prior (e.g., 2% injection rate in production).

    Args:
        y_true: True binary labels (0=SAFE, 1=INJECTION)
        scores: Model scores/probabilities for the positive class
        threshold: Decision threshold; score >= threshold → predict INJECTION
        prior: Expected real-world prevalence of positives (default: 0.02 = 2%)

    Returns:
        (estimated_ppv, estimated_f1, tpr, fpr)
        - estimated_ppv: Estimated positive predictive value (precision) at prior
        - estimated_f1: Estimated F1 at prior
        - tpr: True positive rate (recall) — does not depend on prior
        - fpr: False positive rate — does not depend on prior

    Raises:
        ValueError: If only one class is present in y_true.
    """
    if len(set(y_true)) < 2:
        raise ValueError(
            f"evaluate_at_prior requires both classes in y_true. "
            f"Got only: {set(y_true)}"
        )

    pos_indices = [i for i, y in enumerate(y_true) if y == 1]
    neg_indices = [i for i, y in enumerate(y_true) if y == 0]

    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    n_total = len(y_true)

    observed_pos_rate = n_pos / n_total
    observed_neg_rate = n_neg / n_total

    w_pos = prior / observed_pos_rate
    w_neg = (1.0 - prior) / observed_neg_rate

    # Threshold predictions
    y_pred = [1 if s >= threshold else 0 for s in scores]

    # Compute TP, FP, FN, TN
    tp = sum(1 for i in pos_indices if y_pred[i] == 1)
    fn = n_pos - tp
    fp = sum(1 for i in neg_indices if y_pred[i] == 1)
    tn = n_neg - fp

    tpr = tp / n_pos if n_pos > 0 else 0.0
    fpr = fp / n_neg if n_neg > 0 else 0.0

    # Weighted TP, FP for prevalence-adjusted PPV
    w_tp = tp * w_pos
    w_fp = fp * w_neg
    w_fn = fn * w_pos

    estimated_ppv = w_tp / (w_tp + w_fp) if (w_tp + w_fp) > 0 else 0.0
    estimated_recall = tpr  # same as TPR, weights cancel out

    if estimated_ppv + estimated_recall > 0:
        estimated_f1 = 2 * estimated_ppv * estimated_recall / (estimated_ppv + estimated_recall)
    else:
        estimated_f1 = 0.0

    return {
        "estimated_ppv": float(estimated_ppv),
        "estimated_f1": float(estimated_f1),
        "tpr": float(tpr),
        "fpr": float(fpr),
    }
