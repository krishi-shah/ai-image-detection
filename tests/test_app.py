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
from app import analyse_image, build_demo, get_labeled_examples


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


def _run(dummy_model, device, transform, img=None, temperature=1.0):
    if img is None:
        img = _make_rgb_image()
    return analyse_image(img, model=dummy_model, device=device,
                         temperature=temperature, transform=transform)


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

class TestOutputStructure:

    def test_returns_five_elements(self, dummy_model, device, transform):
        result = _run(dummy_model, device, transform)
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_verdict_is_string(self, dummy_model, device, transform):
        verdict, _, _, _, _ = _run(dummy_model, device, transform)
        assert isinstance(verdict, str)
        assert len(verdict) > 0

    def test_confidence_dict_keys(self, dummy_model, device, transform):
        _, conf, _, _, _ = _run(dummy_model, device, transform)
        assert "AI-Generated" in conf
        assert "Authentic" in conf

    def test_confidence_sums_to_one(self, dummy_model, device, transform):
        _, conf, _, _, _ = _run(dummy_model, device, transform)
        total = sum(conf.values())
        assert abs(total - 1.0) < 1e-5, f"Confidence sum = {total}"

    def test_original_image_shape(self, dummy_model, device, transform):
        _, _, original, _, _ = _run(dummy_model, device, transform)
        assert isinstance(original, np.ndarray)
        assert original.ndim == 3 and original.shape[2] == 3
        assert original.dtype == np.uint8
        assert max(original.shape[:2]) <= 512

    def test_heatmap_shape_and_range(self, dummy_model, device, transform):
        _, _, original, overlay, _ = _run(dummy_model, device, transform)
        assert isinstance(overlay, np.ndarray)
        assert overlay.shape == original.shape
        assert overlay.dtype == np.uint8
        assert overlay.min() >= 0
        assert overlay.max() <= 255

    def test_details_is_string(self, dummy_model, device, transform):
        _, _, _, _, details = _run(dummy_model, device, transform)
        assert isinstance(details, str)


# ---------------------------------------------------------------------------
# Image format edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_grayscale_input(self, dummy_model, device, transform):
        gray = Image.fromarray(
            np.random.randint(0, 256, (100, 100), dtype=np.uint8), "L"
        )
        _, conf, original, overlay, _ = _run(dummy_model, device, transform, img=gray)
        assert overlay.ndim == 3 and overlay.shape[2] == 3
        assert overlay.shape == original.shape
        assert abs(sum(conf.values()) - 1.0) < 1e-5

    def test_rgba_input(self, dummy_model, device, transform):
        rgba = Image.fromarray(
            np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8), "RGBA"
        )
        _, conf, original, overlay, _ = _run(dummy_model, device, transform, img=rgba)
        assert overlay.shape == original.shape
        assert overlay.shape[2] == 3

    def test_small_image(self, dummy_model, device, transform):
        small = _make_rgb_image(16, 16)
        _, conf, original, overlay, _ = _run(dummy_model, device, transform, img=small)
        assert overlay.shape == original.shape
        assert min(overlay.shape[:2]) >= 224
        assert abs(sum(conf.values()) - 1.0) < 1e-5

    def test_large_image(self, dummy_model, device, transform):
        large = _make_rgb_image(1024, 1024)
        _, conf, original, overlay, _ = _run(dummy_model, device, transform, img=large)
        assert overlay.shape == original.shape
        assert max(overlay.shape[:2]) == 512
        assert abs(sum(conf.values()) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Temperature scaling integration
# ---------------------------------------------------------------------------

class TestCalibration:

    def test_temperature_changes_probabilities(self, dummy_model, device, transform):
        img = _make_rgb_image()
        _, conf_t1, _, _, _ = _run(dummy_model, device, transform, img=img,
                                   temperature=1.0)
        _, conf_t2, _, _, _ = _run(dummy_model, device, transform, img=img,
                                   temperature=2.0)
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
        import app as app_module
        original_model = app_module.MODEL
        original_loaded = app_module.MODEL_LOADED
        try:
            app_module.MODEL = None
            app_module.MODEL_LOADED = False
            result = analyse_image(
                _make_rgb_image(), device=device,
                temperature=1.0, transform=transform,
            )
            assert len(result) == 5
            verdict = result[0]
            conf = result[1]
            assert "Model not loaded" in verdict or "not loaded" in verdict.lower()
            assert conf == {}
        finally:
            app_module.MODEL = original_model
            app_module.MODEL_LOADED = original_loaded


# ---------------------------------------------------------------------------
# Gradio Blocks construction
# ---------------------------------------------------------------------------

class TestGradioBlocks:

    def test_build_demo_returns_blocks(self):
        demo = build_demo()
        assert isinstance(demo, gr.Blocks)

    def test_labeled_examples_use_research_names(self):
        pairs = get_labeled_examples()
        labels = {label for _, label in pairs}
        expected = {"CIFAKE Real", "CIFAKE Fake", "StyleGAN", "Midjourney", "GPT-4o"}
        assert expected.issubset(labels) or len(pairs) == 0

    def test_none_image_returns_empty_state(self, dummy_model, device, transform):
        verdict, conf, original, overlay, details = analyse_image(
            None, model=dummy_model, device=device,
            temperature=1.0, transform=transform,
        )
        assert "Awaiting" in verdict or "awaiting" in verdict.lower()
        assert conf == {}
        assert original is None
        assert overlay is None
        assert isinstance(details, str)
