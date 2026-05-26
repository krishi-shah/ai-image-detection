import os
import tempfile
import numpy as np
import torch
import pytest
from src.model.detector import build_detector
from src.explainability.gradcam import run_gradcam, save_heatmap


@pytest.fixture(scope="module")
def model():
    m = build_detector(pretrained=False, num_classes=2)
    m.eval()
    return m


@pytest.fixture(scope="module")
def dummy_input():
    return torch.randn(1, 3, 224, 224)


@pytest.fixture(scope="module")
def heatmap(model, dummy_input):
    target_layer = model.conv_head
    return run_gradcam(model, dummy_input, target_layer)


def test_run_gradcam_output_shape(heatmap):
    assert heatmap.shape == (224, 224), f"Expected (224, 224), got {heatmap.shape}"


def test_run_gradcam_value_range(heatmap):
    assert heatmap.min() >= 0.0, f"Heatmap min {heatmap.min()} is below 0"
    assert heatmap.max() <= 1.0, f"Heatmap max {heatmap.max()} is above 1"


def test_save_heatmap_creates_file(heatmap):
    original_image = np.random.rand(224, 224, 3).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "subdir", "test_heatmap.png")
        save_heatmap(heatmap, original_image, save_path)
        assert os.path.exists(save_path), "Heatmap file was not created"
        assert os.path.getsize(save_path) > 0, "Heatmap file is empty"
