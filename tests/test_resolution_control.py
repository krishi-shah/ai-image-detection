"""Unit tests for the resolution-control experiment."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.analysis.resolution_control import (
    evaluate_condition,
    get_matched_lowres_transform,
    get_native_eval_transform,
    plot_before_after_matching,
    plot_fake_rate_bars,
    plot_p_fake_distributions,
    run_resolution_control,
)


class DummyModel(nn.Module):
    """Returns fixed logits that classify everything as FAKE."""

    def __init__(self, fake_logit: float = 2.0):
        super().__init__()
        self.fake_logit = fake_logit
        self.linear = nn.Linear(1, 1)  # so parameters exist

    def forward(self, x):
        batch = x.shape[0]
        logits = torch.zeros(batch, 2)
        logits[:, 0] = self.fake_logit
        logits[:, 1] = 0.0
        return logits


class AlwaysRealModel(nn.Module):
    """Returns fixed logits that classify everything as REAL."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        batch = x.shape[0]
        logits = torch.zeros(batch, 2)
        logits[:, 0] = 0.0
        logits[:, 1] = 2.0
        return logits


@pytest.fixture
def image_dir(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    for i in range(8):
        # High-res-ish images so the 32x32 force transform has an effect
        arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        Image.fromarray(arr).save(d / f"{i:04d}.png")
    return d


@pytest.fixture
def cifake_real_dir(tmp_path):
    d = tmp_path / "cifake_real"
    d.mkdir()
    for i in range(8):
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(arr).save(d / f"{i:04d}.png")
    return d


@pytest.fixture
def fake_dirs(tmp_path):
    families = {}
    for name in ["stylegan", "gpt4o"]:
        d = tmp_path / name / "FAKE"
        d.mkdir(parents=True)
        for i in range(8):
            arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{i:04d}.png")
        families[name] = d
    return families


# ---------------------------------------------------------------------------
# Transform tests
# ---------------------------------------------------------------------------

class TestTransforms:
    def test_matched_lowres_output_shape(self):
        t = get_matched_lowres_transform()
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        out = t(img)
        assert out.shape == (3, 224, 224)

    def test_native_output_shape(self):
        t = get_native_eval_transform()
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        out = t(img)
        assert out.shape == (3, 224, 224)

    def test_matched_starts_with_32x32(self):
        """First step must be Resize((32, 32))."""
        t = get_matched_lowres_transform()
        first = t.transforms[0]
        assert isinstance(first, type(get_matched_lowres_transform().transforms[0]))
        assert first.size == (32, 32)


# ---------------------------------------------------------------------------
# evaluate_condition
# ---------------------------------------------------------------------------

class TestEvaluateCondition:
    def test_always_fake_model(self, image_dir):
        model = DummyModel(fake_logit=5.0)
        result = evaluate_condition(
            model, image_dir, true_label=1,
            transform=get_native_eval_transform(),
            device=torch.device("cpu"),
            condition_name="test",
        )
        assert result["fake_rate"] == 1.0
        assert result["n_images"] == 8
        assert result["mean_p_fake"] > 0.9
        assert len(result["p_fake"]) == 8

    def test_always_real_model(self, image_dir):
        model = AlwaysRealModel()
        result = evaluate_condition(
            model, image_dir, true_label=1,
            transform=get_native_eval_transform(),
            device=torch.device("cpu"),
            condition_name="test",
        )
        assert result["fake_rate"] == 0.0
        assert result["mean_p_fake"] < 0.15

    def test_max_images_cap(self, image_dir):
        model = AlwaysRealModel()
        result = evaluate_condition(
            model, image_dir, true_label=1,
            transform=get_native_eval_transform(),
            device=torch.device("cpu"),
            max_images=3,
            condition_name="capped",
        )
        assert result["n_images"] == 3

    def test_missing_dir_raises(self, tmp_path):
        model = AlwaysRealModel()
        with pytest.raises(FileNotFoundError):
            evaluate_condition(
                model, tmp_path / "nope", true_label=1,
                transform=get_native_eval_transform(),
                device=torch.device("cpu"),
            )


# ---------------------------------------------------------------------------
# Full grid + JSON schema
# ---------------------------------------------------------------------------

class TestRunResolutionControl:
    def test_full_pipeline(self, cifake_real_dir, image_dir, fake_dirs, tmp_path):
        model = AlwaysRealModel()
        out = tmp_path / "outputs"
        results = run_resolution_control(
            model=model,
            device=torch.device("cpu"),
            cifake_real_dir=cifake_real_dir,
            hires_real_dir=image_dir,
            generator_fake_dirs=fake_dirs,
            temperature=1.0,
            n_images=8,
            batch_size=4,
            output_dir=out,
        )

        assert results["experiment"] == "resolution_control"
        assert "conditions" in results
        assert "A_cifake_real_native" in results["conditions"]
        assert "B_hires_real_native" in results["conditions"]
        assert "C_hires_real_matched" in results["conditions"]
        assert "D_generator_fakes_matched" in results["conditions"]
        assert "interpretation" in results

        json_path = out / "results" / "resolution_control.json"
        assert json_path.exists()
        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded["experiment"] == "resolution_control"
        assert "fake_rate" in loaded["conditions"]["A_cifake_real_native"]
        # Per-image lists live under distributions, not inside condition summaries
        assert "p_fake" not in loaded["conditions"]["A_cifake_real_native"]
        assert "A_cifake_real_native" in loaded["distributions"]

        # Plots written
        plots = out / "plots" / "resolution_control"
        assert (plots / "fake_rate_by_condition.png").exists()
        assert (plots / "p_fake_distributions.png").exists()
        assert (plots / "before_after_resolution_matching.png").exists()

    def test_always_real_interpretation(self, cifake_real_dir, image_dir, fake_dirs, tmp_path):
        model = AlwaysRealModel()
        results = run_resolution_control(
            model=model,
            device=torch.device("cpu"),
            cifake_real_dir=cifake_real_dir,
            hires_real_dir=image_dir,
            generator_fake_dirs=fake_dirs,
            n_images=8,
            batch_size=4,
            output_dir=tmp_path / "outputs",
        )
        interp = results["interpretation"]
        assert interp["B_hires_real_fake_rate"] == 0.0
        assert interp["resolution_confound_suspected"] is False


class TestPlotHelpers:
    def test_plot_helpers_do_not_crash(self, tmp_path):
        conditions = {
            "A_cifake_real_native": {
                "fake_rate": 0.05, "p_fake": [0.01, 0.02, 0.1],
            },
            "B_hires_real_native": {
                "fake_rate": 0.15, "p_fake": [0.1, 0.2, 0.3],
            },
            "C_hires_real_matched": {
                "fake_rate": 0.08, "p_fake": [0.05, 0.07, 0.09],
            },
            "D_generator_fakes_matched": {
                "stylegan": {"fake_rate": 0.9},
                "gpt4o": {"fake_rate": 0.7},
            },
        }
        plot_fake_rate_bars(conditions, tmp_path / "bars.png")
        plot_p_fake_distributions(conditions, tmp_path / "dist.png")
        plot_before_after_matching(
            {"stylegan": {"fake_rate": 0.94}, "gpt4o": {"fake_rate": 0.86}},
            {"stylegan": {"fake_rate": 0.80}, "gpt4o": {"fake_rate": 0.60}},
            tmp_path / "ba.png",
        )
        assert (tmp_path / "bars.png").exists()
        assert (tmp_path / "dist.png").exists()
        assert (tmp_path / "ba.png").exists()
