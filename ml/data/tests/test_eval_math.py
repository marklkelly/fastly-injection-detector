"""Unit tests for evaluate_at_prior() math correctness."""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from ml.data.eval_utils import evaluate_at_prior


def test_perfect_classifier_at_prior():
    """Perfect classifier: estimated_ppv should be 1.0."""
    y_true = [0] * 100 + [1] * 100
    # Perfect scores: positives get 1.0, negatives get 0.0
    scores = [0.0] * 100 + [1.0] * 100
    threshold = 0.5
    result = evaluate_at_prior(y_true, scores, threshold, prior=0.02)
    assert result["estimated_ppv"] == pytest.approx(1.0, abs=1e-9), f"Expected ppv=1.0, got {result['estimated_ppv']}"
    assert result["tpr"] == pytest.approx(1.0, abs=1e-9), f"Expected tpr=1.0, got {result['tpr']}"
    assert result["fpr"] == pytest.approx(0.0, abs=1e-9), f"Expected fpr=0.0, got {result['fpr']}"


def test_random_classifier_at_prior():
    """Random classifier predicts positive for ~50% of all examples.
    Estimated F1 should be near 2*prior/(1+prior) at prior=0.02."""
    import random
    random.seed(42)
    n = 10000
    y_true = [0] * (n // 2) + [1] * (n // 2)
    # Random scores
    scores = [random.random() for _ in range(n)]
    threshold = 0.5  # approx 50% flagged
    result = evaluate_at_prior(y_true, scores, threshold, prior=0.02)
    expected_f1 = 2 * 0.02 / (1 + 0.02)
    # Allow generous tolerance since it's random
    assert abs(result["estimated_f1"] - expected_f1) < 0.02, f"Expected f1 ≈ {expected_f1:.4f}, got {result['estimated_f1']:.4f}"


def test_weights_correct_for_balanced_input():
    """On 50/50 input with prior=0.5: results should match standard (unweighted) metrics."""
    y_true = [0] * 50 + [1] * 50
    # scores: positives score 0.9, negatives score 0.1
    scores = [0.1] * 50 + [0.9] * 50
    threshold = 0.5
    result = evaluate_at_prior(y_true, scores, threshold, prior=0.5)
    # With perfect separation and prior=0.5, PPV should be 1.0
    assert result["estimated_ppv"] == pytest.approx(1.0, abs=1e-9), f"Expected ppv=1.0 at prior=0.5, got {result['estimated_ppv']}"
    assert result["tpr"] == pytest.approx(1.0, abs=1e-9), f"Expected tpr=1.0, got {result['tpr']}"


def test_raises_on_missing_class():
    """Raises ValueError if only one class is present."""
    y_true = [0, 0, 0, 0, 0]
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    with pytest.raises(ValueError, match="requires both classes"):
        evaluate_at_prior(y_true, scores, threshold=0.5)


def test_low_prior_reduces_ppv():
    """Lower prior should reduce estimated PPV (more negatives → more FPs count more)."""
    # Use imperfect classifier: some negatives are scored above threshold (FPs)
    y_true = [0] * 90 + [1] * 10
    # 80 negatives below threshold, 10 negatives above (FPs), 10 positives above (TPs)
    scores = [0.1] * 80 + [0.9] * 10 + [0.9] * 10
    threshold = 0.5
    result_high = evaluate_at_prior(y_true, scores, threshold, prior=0.5)
    result_low = evaluate_at_prior(y_true, scores, threshold, prior=0.01)
    assert result_low["estimated_ppv"] < result_high["estimated_ppv"], (
        f"Expected lower PPV at lower prior: {result_low['estimated_ppv']} vs {result_high['estimated_ppv']}"
    )


def test_imbalanced_observed_set():
    """Imbalanced observed set (10% positives, prior=0.01) tests w_pos = prior / observed_positive_rate."""
    # 90 negatives, 10 positives (10% observed positive rate)
    # Classifier: all 10 positives correctly flagged (TP=10, FN=0)
    #             10 negatives incorrectly flagged (FP=10, TN=80)
    n_neg, n_pos = 90, 10
    y_true = [0] * n_neg + [1] * n_pos
    scores = [0.1] * 80 + [0.9] * 10 + [0.9] * n_pos  # 10 FP + 10 TP above threshold
    threshold = 0.5
    prior = 0.01

    result = evaluate_at_prior(y_true, scores, threshold, prior=prior)

    # Manual calculation:
    # observed_pos_rate = 10/100 = 0.10
    # observed_neg_rate = 90/100 = 0.90
    # w_pos = 0.01 / 0.10 = 0.1
    # w_neg = 0.99 / 0.90 = 1.1
    # TP=10, FP=10, FN=0
    # w_tp = 10 * 0.1 = 1.0
    # w_fp = 10 * 1.1 = 11.0
    # estimated_ppv = 1.0 / (1.0 + 11.0) = 1/12
    # tpr = 10/10 = 1.0
    # estimated_f1 = 2 * (1/12) * 1.0 / (1/12 + 1.0) = (2/12) / (13/12) = 2/13
    expected_ppv = 1.0 / 12.0
    expected_f1 = 2.0 / 13.0
    expected_tpr = 1.0
    expected_fpr = 10.0 / 90.0

    assert result["estimated_ppv"] == pytest.approx(expected_ppv, abs=1e-9), (
        f"Expected ppv={expected_ppv:.6f}, got {result['estimated_ppv']:.6f}"
    )
    assert result["estimated_f1"] == pytest.approx(expected_f1, abs=1e-9), (
        f"Expected f1={expected_f1:.6f}, got {result['estimated_f1']:.6f}"
    )
    assert result["tpr"] == pytest.approx(expected_tpr, abs=1e-9), (
        f"Expected tpr={expected_tpr}, got {result['tpr']}"
    )
    assert result["fpr"] == pytest.approx(expected_fpr, abs=1e-9), (
        f"Expected fpr={expected_fpr:.6f}, got {result['fpr']:.6f}"
    )
