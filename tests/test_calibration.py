import numpy as np
import torch
import torch.nn as nn
import pytest
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.calibration import (
    apply_temperature,
    TemperatureScaler,
    find_temperature,
)


def _softmax(logits):
    """Reference numpy softmax for comparison."""
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


class TestApplyTemperature:

    def test_identity_at_t1(self):
        logits = np.array([[2.0, 0.5], [0.1, 1.8], [-0.3, 0.9]])
        result = apply_temperature(logits, temperature=1.0)
        expected = _softmax(logits)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_softens_at_t2(self):
        logits = np.array([[3.0, 0.0], [0.0, 3.0]])
        probs_t1 = apply_temperature(logits, temperature=1.0)
        probs_t2 = apply_temperature(logits, temperature=2.0)
        assert probs_t2.max() < probs_t1.max(), (
            "T=2 should produce lower max confidence than T=1"
        )

    def test_sharpens_at_t05(self):
        logits = np.array([[3.0, 0.0], [0.0, 3.0]])
        probs_t1 = apply_temperature(logits, temperature=1.0)
        probs_t05 = apply_temperature(logits, temperature=0.5)
        assert probs_t05.max() > probs_t1.max(), (
            "T=0.5 should produce higher max confidence than T=1"
        )

    def test_output_sums_to_one(self):
        logits = np.random.randn(50, 2)
        for t in [0.5, 1.0, 2.0, 5.0]:
            probs = apply_temperature(logits, temperature=t)
            row_sums = probs.sum(axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


class TestTemperatureScaler:

    def test_forward_divides_logits(self):
        scaler = TemperatureScaler()
        scaler.temperature = nn.Parameter(torch.tensor(2.0))
        logits = torch.tensor([[4.0, 2.0]])
        result = scaler(logits)
        expected = torch.tensor([[2.0, 1.0]])
        torch.testing.assert_close(result, expected)

    def test_temperature_is_learnable(self):
        scaler = TemperatureScaler()
        assert scaler.temperature.requires_grad


class TestFindTemperature:

    def test_learns_temperature_for_overconfident_model(self):
        """A model that outputs large logits (overconfident) should yield T > 1."""
        torch.manual_seed(42)
        np.random.seed(42)

        n_samples = 200
        model = nn.Linear(10, 2)

        # Amplify weights to make model overconfident
        with torch.no_grad():
            model.weight.mul_(5.0)

        features = torch.randn(n_samples, 10)
        labels = torch.randint(0, 2, (n_samples,))

        dataset = TensorDataset(features, labels)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        device = torch.device("cpu")

        learned_t = find_temperature(model, loader, device)

        assert learned_t > 1.0, (
            f"Overconfident model should need T > 1 to soften, got T={learned_t:.4f}"
        )

    def test_ece_decreases_after_calibration(self):
        """ECE on the validation set should not increase after temperature scaling."""
        torch.manual_seed(123)
        np.random.seed(123)
        from src.evaluation.metrics import compute_ece

        n_samples = 300
        model = nn.Linear(10, 2)
        with torch.no_grad():
            model.weight.mul_(8.0)

        features = torch.randn(n_samples, 10)
        labels = torch.randint(0, 2, (n_samples,))

        dataset = TensorDataset(features, labels)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        device = torch.device("cpu")

        # Collect logits for ECE comparison
        model.eval()
        with torch.no_grad():
            all_logits = model(features)
        probs_before = torch.softmax(all_logits, dim=1).numpy()
        ece_before = compute_ece(probs_before, labels.numpy())

        learned_t = find_temperature(model, loader, device)

        probs_after = apply_temperature(all_logits.numpy(), learned_t)
        ece_after = compute_ece(probs_after, labels.numpy())

        assert ece_after <= ece_before + 0.01, (
            f"ECE should not increase: before={ece_before:.4f}, after={ece_after:.4f}"
        )
