"""Tests for the Gradio interactive demo (app.py).

All tests use a dummy model with random weights so they work without
the trained checkpoint.
"""

import numpy as np
import pytest
import torch
from PIL import Image

import gradio as gr

from src.model.detector import build_detector
from src.utils.data_loader import get_transforms
from app import analyse_image, build_demo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def dummy_model():
    model = build_detector(pretrained=False)
    model.eval()
    return model


@pytest.fixture()
def device():
    return torch.device("cpu")


@pytest.fixture()
def transform():
    return get_transforms("test")


def _make_rgb_image(width=256, height=256):
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

class TestOutputStructure:

    def test_returns_four_elements(self, dummy_model, device, transform):
        img = _make_rgb_image()
        result = analyse_image(img, model=dummy_model, device=device,
                               temperature=1.0, transform=transform)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_verdict_is_string(self, dummy_model, device, transform):
        img = _make_rgb_image()
        verdict, _, _, _ = analyse_image(
            img, model=dummy_model, device=device,
            temperature=1.0, transform=transform,
        )
        assert isinstance(verdict, str)
        assert len(verdict) > 0

    def test_confidence_dict_keys(self, dummy_model, device, transform):
        img = _make_rgb_image()
        _, conf, _, _ = analyse_image(
            img, model=dummy_model, device=device,
            temperature=1.0, transform=transform,
        )
        assert "AI-Generated" in conf
        assert "Authentic" in conf

    def test_confidence_sums_to_one(self, dummy_model, device, transform):
        img = _make_rgb_image()
        _, conf, _, _ = analyse_image(
            img, model=dummy_model, device=device,
            temperature=1.0, transform=transform,
        )
        total = sum(conf.values())
        assert abs(total - 1.0) < 1e-5, f"Confidence sum = {total}"

    def test_heatmap_shape_and_range(self, dummy_model, device, transform):
        img = _make_rgb_image()
        _, _, overlay, _ = analyse_image(
            img, model=dummy_model, device=device,
            temperature=1.0, transform=transform,
        )
        assert isinstance(overlay, np.ndarray)
        assert overlay.shape == (224, 224, 3)
        assert overlay.dtype == np.uint8
        assert overlay.min() >= 0
        assert overlay.max() <= 255

    def test_details_is_string(self, dummy_model, device, transform):
        img = _make_rgb_image()
        _, _, _, details = analyse_image(
            img, model=dummy_model, device=device,
            temperature=1.0, transform=transform,
        )
        assert isinstance(details, str)
        assert "Temperature" in details


# ---------------------------------------------------------------------------
# Image format edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_grayscale_input(self, dummy_model, device, transform):
        gray = Image.fromarray(
            np.random.randint(0, 256, (100, 100), dtype=np.uint8), "L"
        )
        verdict, conf, overlay, _ = analyse_image(
            gray, model=dummy_model, device=device,
            temperature=1.0, transform=transform,
        )
        assert overlay.shape == (224, 224, 3)
        assert abs(sum(conf.values()) - 1.0) < 1e-5

    def test_rgba_input(self, dummy_model, device, transform):
        rgba = Image.fromarray(
            np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8), "RGBA"
        )
        verdict, conf, overlay, _ = analyse_image(
            rgba, model=dummy_model, device=device,
            temperature=1.0, transform=transform,
        )
        assert overlay.shape == (224, 224, 3)

    def test_small_image(self, dummy_model, device, transform):
        small = _make_rgb_image(16, 16)
        _, conf, overlay, _ = analyse_image(
            small, model=dummy_model, device=device,
            temperature=1.0, transform=transform,
        )
        assert overlay.shape == (224, 224, 3)
        assert abs(sum(conf.values()) - 1.0) < 1e-5

    def test_large_image(self, dummy_model, device, transform):
        large = _make_rgb_image(1024, 1024)
        _, conf, overlay, _ = analyse_image(
            large, model=dummy_model, device=device,
            temperature=1.0, transform=transform,
        )
        assert overlay.shape == (224, 224, 3)
        assert abs(sum(conf.values()) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Temperature scaling integration
# ---------------------------------------------------------------------------

class TestCalibration:

    def test_temperature_changes_probabilities(self, dummy_model, device, transform):
        img = _make_rgb_image()
        _, conf_t1, _, _ = analyse_image(
            img, model=dummy_model, device=device,
            temperature=1.0, transform=transform,
        )
        _, conf_t2, _, _ = analyse_image(
            img, model=dummy_model, device=device,
            temperature=2.0, transform=transform,
        )
        # T=2 should soften probabilities (push closer to 0.5)
        max_t1 = max(conf_t1.values())
        max_t2 = max(conf_t2.values())
        assert max_t2 <= max_t1 + 1e-6, (
            f"T=2 should soften: max@T1={max_t1:.4f}, max@T2={max_t2:.4f}"
        )


# ---------------------------------------------------------------------------
# Missing checkpoint
# ---------------------------------------------------------------------------

class TestMissingCheckpoint:

    def test_returns_error_message_when_model_none(self, device, transform):
        verdict, conf, overlay, _ = analyse_image(
            _make_rgb_image(), model=None, device=device,
            temperature=1.0, transform=transform,
        )
        assert "Model not loaded" in verdict
        assert conf == {}
        assert overlay is None


# ---------------------------------------------------------------------------
# Gradio Blocks construction
# ---------------------------------------------------------------------------

class TestGradioBlocks:

    def test_build_demo_returns_blocks(self):
        demo = build_demo()
        assert isinstance(demo, gr.Blocks)
