import numpy as np
import pytest
from src.evaluation.metrics import compute_accuracy, compute_auc, compute_ece


def test_compute_accuracy_perfect():
    preds = np.array([0, 1, 0, 1])
    labels = np.array([0, 1, 0, 1])
    assert compute_accuracy(preds, labels) == 1.0


def test_compute_accuracy_zero():
    preds = np.array([1, 0, 1, 0])
    labels = np.array([0, 1, 0, 1])
    assert compute_accuracy(preds, labels) == 0.0


def test_compute_accuracy_half():
    preds = np.array([0, 1, 1, 0])
    labels = np.array([0, 1, 0, 1])
    assert compute_accuracy(preds, labels) == 0.5


def test_compute_auc_perfect():
    probs = np.array([
        [0.9, 0.1],
        [0.1, 0.9],
        [0.8, 0.2],
        [0.2, 0.8],
    ])
    labels = np.array([0, 1, 0, 1])
    assert compute_auc(probs, labels) == 1.0


def test_compute_auc_random():
    rng = np.random.RandomState(42)
    probs = rng.rand(1000, 2)
    probs = probs / probs.sum(axis=1, keepdims=True)
    labels = rng.randint(0, 2, size=1000)
    auc = compute_auc(probs, labels)
    assert 0.3 < auc < 0.7, "Random predictions should yield AUC near 0.5"


def test_ece_perfect_calibration():
    """When confidence matches accuracy exactly, ECE should be ~0."""
    probs = np.array([
        [0.1, 0.9],
        [0.2, 0.8],
        [0.3, 0.7],
        [0.05, 0.95],
    ])
    labels = np.array([1, 1, 1, 1])
    ece = compute_ece(probs, labels, n_bins=10)
    assert ece < 0.05, f"ECE should be near 0 for perfect calibration, got {ece}"


def test_ece_worst_case():
    """High-confidence wrong predictions should yield large ECE."""
    probs = np.array([
        [0.05, 0.95],
        [0.05, 0.95],
        [0.05, 0.95],
        [0.05, 0.95],
    ])
    labels = np.array([0, 0, 0, 0])
    ece = compute_ece(probs, labels, n_bins=10)
    assert ece > 0.8, f"ECE should be near 1 for fully wrong predictions, got {ece}"
