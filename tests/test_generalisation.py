"""Unit tests for the generalisation evaluation module.

Tests cover:
- Data loader labeling and discovery
- Metric computation on synthetic logits with known answers
- Degradation math
- Results JSON schema
- End-to-end smoke test with dummy images
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import (
    discover_generator_families,
    get_generalisation_loader,
    _FolderDataset,
)
from src.evaluation.generalisation import (
    evaluate_generator,
    compute_degradation,
    save_generalisation_results,
    plot_cross_generator_accuracy,
    plot_degradation_waterfall,
    plot_confidence_distributions,
    plot_ece_comparison,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_family_dir(tmp_path):
    """Create a temporary family directory with dummy images."""
    family = tmp_path / "test_family"
    fake_dir = family / "FAKE"
    fake_dir.mkdir(parents=True)

    # Create 8 dummy FAKE images
    for i in range(8):
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        img.save(fake_dir / f"{i:04d}.png")

    return family


@pytest.fixture
def tmp_real_dir(tmp_path):
    """Create a temporary REAL reference directory with dummy images."""
    real_dir = tmp_path / "real_ref"
    real_dir.mkdir(parents=True)

    for i in range(8):
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        img.save(real_dir / f"{i:04d}.png")

    return real_dir


@pytest.fixture
def tmp_families_root(tmp_path):
    """Create a root dir with multiple family folders."""
    for name in ["family_a", "family_b"]:
        fake_dir = tmp_path / name / "FAKE"
        fake_dir.mkdir(parents=True)
        for i in range(4):
            img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
            img.save(fake_dir / f"{i:04d}.png")

    # Add a non-family directory (no FAKE/ subfolder)
    (tmp_path / "not_a_family").mkdir()

    return tmp_path


class DummyModel(nn.Module):
    """Minimal model that returns fixed logits for testing."""

    def __init__(self, fake_confidence=0.9):
        super().__init__()
        self.fake_confidence = fake_confidence
        self.linear = nn.Linear(3 * 224 * 224, 2)

    def forward(self, x):
        batch_size = x.shape[0]
        # Return logits that classify everything as FAKE
        logits = torch.zeros(batch_size, 2)
        logits[:, 0] = 2.0  # high logit for FAKE class
        logits[:, 1] = -2.0
        return logits


# ---------------------------------------------------------------------------
# Tests: discover_generator_families
# ---------------------------------------------------------------------------

class TestDiscoverFamilies:
    def test_finds_valid_families(self, tmp_families_root):
        families = discover_generator_families(str(tmp_families_root))
        assert len(families) == 2
        names = [Path(f).name for f in families]
        assert "family_a" in names
        assert "family_b" in names

    def test_excludes_non_families(self, tmp_families_root):
        families = discover_generator_families(str(tmp_families_root))
        names = [Path(f).name for f in families]
        assert "not_a_family" not in names

    def test_empty_dir(self, tmp_path):
        families = discover_generator_families(str(tmp_path / "nonexistent"))
        assert families == []


# ---------------------------------------------------------------------------
# Tests: get_generalisation_loader
# ---------------------------------------------------------------------------

class TestGeneralisationLoader:
    def test_fake_only_loader(self, tmp_family_dir):
        loader = get_generalisation_loader(
            family_dir=str(tmp_family_dir),
            real_reference_dir=None,
            batch_size=4,
            num_workers=0,
        )
        all_labels = []
        for _, labels in loader:
            all_labels.extend(labels.tolist())
        # All labels should be 0 (FAKE)
        assert all(l == 0 for l in all_labels)
        assert len(all_labels) == 8

    def test_with_real_reference(self, tmp_family_dir, tmp_real_dir):
        loader = get_generalisation_loader(
            family_dir=str(tmp_family_dir),
            real_reference_dir=str(tmp_real_dir),
            batch_size=4,
            num_workers=0,
        )
        all_labels = []
        for _, labels in loader:
            all_labels.extend(labels.tolist())
        # Should have both FAKE (0) and REAL (1)
        assert 0 in all_labels
        assert 1 in all_labels
        # 8 fake + up to 8 real (balanced)
        assert len(all_labels) == 16

    def test_missing_fake_dir_raises(self, tmp_path):
        empty_family = tmp_path / "empty"
        empty_family.mkdir()
        with pytest.raises(FileNotFoundError):
            get_generalisation_loader(str(empty_family))

    def test_label_convention(self, tmp_family_dir, tmp_real_dir):
        """Labels: 0=FAKE, 1=REAL (matching CIFAKE)."""
        loader = get_generalisation_loader(
            family_dir=str(tmp_family_dir),
            real_reference_dir=str(tmp_real_dir),
            batch_size=16,
            num_workers=0,
        )
        for _, labels in loader:
            for label in labels.tolist():
                assert label in [0, 1]


# ---------------------------------------------------------------------------
# Tests: evaluate_generator
# ---------------------------------------------------------------------------

class TestEvaluateGenerator:
    def test_returns_expected_keys(self, tmp_family_dir, tmp_real_dir):
        model = DummyModel()
        loader = get_generalisation_loader(
            family_dir=str(tmp_family_dir),
            real_reference_dir=str(tmp_real_dir),
            batch_size=8,
            num_workers=0,
        )
        results = evaluate_generator(
            model=model,
            loader=loader,
            device=torch.device("cpu"),
            temperature=1.2189,
            family_name="test",
        )

        expected_keys = [
            "family", "n_samples", "n_fake", "n_real",
            "accuracy", "fake_detection_rate", "auc",
            "precision_fake", "recall_fake", "f1_fake",
            "ece_pre_calibration", "ece_post_calibration",
            "confusion_matrix", "confidence_stats",
            "temperature_used", "per_image",
        ]
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"

    def test_perfect_detector(self, tmp_family_dir):
        """A model that always predicts FAKE should get 100% on fake-only set."""
        model = DummyModel()
        loader = get_generalisation_loader(
            family_dir=str(tmp_family_dir),
            real_reference_dir=None,
            batch_size=8,
            num_workers=0,
        )
        results = evaluate_generator(
            model=model,
            loader=loader,
            device=torch.device("cpu"),
            temperature=1.0,
            family_name="test",
        )
        assert results["fake_detection_rate"] == 1.0
        assert results["accuracy"] == 1.0

    def test_metrics_range(self, tmp_family_dir, tmp_real_dir):
        model = DummyModel()
        loader = get_generalisation_loader(
            family_dir=str(tmp_family_dir),
            real_reference_dir=str(tmp_real_dir),
            batch_size=8,
            num_workers=0,
        )
        results = evaluate_generator(
            model=model, loader=loader,
            device=torch.device("cpu"), temperature=1.0,
            family_name="test",
        )
        assert 0.0 <= results["accuracy"] <= 1.0
        assert 0.0 <= results["ece_pre_calibration"] <= 1.0
        assert 0.0 <= results["ece_post_calibration"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: compute_degradation
# ---------------------------------------------------------------------------

class TestComputeDegradation:
    def test_no_degradation(self):
        baseline = {"test_accuracy": 0.97, "test_auc": 0.99}
        gen_results = {"accuracy": 0.97, "auc": 0.99, "family": "test"}
        deg = compute_degradation(baseline, gen_results)
        assert abs(deg["accuracy_drop_absolute"]) < 1e-9
        assert abs(deg["accuracy_drop_relative"]) < 1e-9

    def test_50_percent_degradation(self):
        baseline = {"test_accuracy": 0.96, "test_auc": 0.99}
        gen_results = {"accuracy": 0.48, "auc": 0.50, "family": "test"}
        deg = compute_degradation(baseline, gen_results)
        assert abs(deg["accuracy_drop_absolute"] - 0.48) < 1e-9
        assert abs(deg["accuracy_drop_relative"] - 0.50) < 0.01

    def test_handles_none_auc(self):
        baseline = {"test_accuracy": 0.96, "test_auc": 0.99}
        gen_results = {"accuracy": 0.60, "auc": None, "family": "test"}
        deg = compute_degradation(baseline, gen_results)
        assert deg["auc_drop_absolute"] is None
        assert deg["auc_drop_relative"] is None


# ---------------------------------------------------------------------------
# Tests: save_generalisation_results
# ---------------------------------------------------------------------------

class TestSaveResults:
    def test_creates_json_files(self, tmp_path):
        all_results = {
            "fam_a": {
                "family": "fam_a", "accuracy": 0.80, "auc": 0.85,
                "per_image": {"probs_fake": [0.8], "labels": [0], "preds": [0]},
                "fake_detection_rate": 0.8, "f1_fake": 0.75,
                "ece_pre_calibration": 0.05, "ece_post_calibration": 0.04,
            }
        }
        degradation = {
            "fam_a": {"family": "fam_a", "accuracy_drop_absolute": 0.17}
        }
        save_generalisation_results(all_results, degradation, str(tmp_path))

        assert (tmp_path / "generalisation_results.json").exists()
        assert (tmp_path / "degradation_summary.json").exists()

        # Check per_image is stripped from summary
        with open(tmp_path / "generalisation_results.json") as f:
            saved = json.load(f)
        assert "per_image" not in saved["fam_a"]

    def test_json_valid(self, tmp_path):
        all_results = {"x": {"family": "x", "accuracy": 0.5, "per_image": {}}}
        degradation = {"x": {"family": "x"}}
        save_generalisation_results(all_results, degradation, str(tmp_path))

        with open(tmp_path / "generalisation_results.json") as f:
            data = json.load(f)
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# Tests: plotting (smoke tests — just verify no exceptions)
# ---------------------------------------------------------------------------

class TestPlots:
    def test_accuracy_plot(self, tmp_path):
        results = {"fam_a": {"accuracy": 0.7}, "fam_b": {"accuracy": 0.5}}
        plot_cross_generator_accuracy(results, 0.97, str(tmp_path / "acc.png"))
        assert (tmp_path / "acc.png").exists()

    def test_waterfall_plot(self, tmp_path):
        deg = {
            "fam_a": {"generator_accuracy": 0.7},
            "fam_b": {"generator_accuracy": 0.5},
        }
        plot_degradation_waterfall(deg, 0.97, str(tmp_path / "wf.png"))
        assert (tmp_path / "wf.png").exists()

    def test_confidence_plot(self, tmp_path):
        results = {
            "fam_a": {"per_image": {"probs_fake": [0.8, 0.9], "labels": [0, 0]}},
        }
        plot_confidence_distributions(results, str(tmp_path / "conf.png"))
        assert (tmp_path / "conf.png").exists()

    def test_ece_plot(self, tmp_path):
        results = {
            "fam_a": {"ece_pre_calibration": 0.05, "ece_post_calibration": 0.04},
            "fam_b": {"ece_pre_calibration": 0.15, "ece_post_calibration": 0.12},
        }
        plot_ece_comparison(results, str(tmp_path / "ece.png"))
        assert (tmp_path / "ece.png").exists()


# ---------------------------------------------------------------------------
# Smoke test: end-to-end with dummy images
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_pipeline(self, tmp_path):
        """Fabricate 8 dummy images, run loader → eval → plot end to end."""
        # Create family with fake images
        family_dir = tmp_path / "smoke_family"
        fake_dir = family_dir / "FAKE"
        fake_dir.mkdir(parents=True)
        for i in range(8):
            img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
            img.save(fake_dir / f"{i:04d}.png")

        # Create real reference
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        for i in range(8):
            img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
            img.save(real_dir / f"{i:04d}.png")

        # Load
        loader = get_generalisation_loader(
            family_dir=str(family_dir),
            real_reference_dir=str(real_dir),
            batch_size=8,
            num_workers=0,
        )

        # Evaluate
        model = DummyModel()
        results = evaluate_generator(
            model=model, loader=loader,
            device=torch.device("cpu"),
            temperature=1.2189,
            family_name="smoke",
        )

        # Degradation
        baseline = {"test_accuracy": 0.9696, "test_auc": 0.9971}
        deg = compute_degradation(baseline, results)

        # Save
        output_dir = tmp_path / "outputs"
        save_generalisation_results(
            {"smoke": results}, {"smoke": deg}, str(output_dir)
        )

        # Plot
        plot_cross_generator_accuracy(
            {"smoke": results}, 0.9696,
            str(output_dir / "acc.png"),
        )

        # Verify
        assert (output_dir / "generalisation_results.json").exists()
        assert (output_dir / "degradation_summary.json").exists()
        assert (output_dir / "acc.png").exists()
        assert results["n_samples"] == 16
        assert results["family"] == "smoke"
