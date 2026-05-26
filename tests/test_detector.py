import os
import tempfile
import torch
import pytest
from src.model.detector import build_detector, save_checkpoint, load_checkpoint


@pytest.fixture(scope="module")
def model():
    """Build a detector once for the whole module (no pretrained weights to save time)."""
    return build_detector(pretrained=False, num_classes=2)


def test_build_detector_output_shape(model):
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy)
    assert output.shape == (1, 2), f"Expected (1, 2), got {output.shape}"


def test_build_detector_batch(model):
    dummy = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy)
    assert output.shape == (4, 2), f"Expected (4, 2), got {output.shape}"


def test_build_detector_custom_num_classes():
    m = build_detector(pretrained=False, num_classes=5)
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = m(dummy)
    assert output.shape == (1, 5), f"Expected (1, 5), got {output.shape}"


def test_save_and_load_checkpoint(model):
    metrics = {"val_acc": 0.95, "val_loss": 0.12}
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        tmp_path = f.name

    try:
        save_checkpoint(model, tmp_path, epoch=10, metrics=metrics)
        assert os.path.exists(tmp_path), "Checkpoint file was not created"

        loaded_model = build_detector(pretrained=False, num_classes=2)
        loaded_epoch, loaded_metrics = load_checkpoint(loaded_model, tmp_path)

        assert loaded_epoch == 10
        assert loaded_metrics["val_acc"] == pytest.approx(0.95)
        assert loaded_metrics["val_loss"] == pytest.approx(0.12)

        for p_orig, p_loaded in zip(model.parameters(), loaded_model.parameters()):
            assert torch.equal(p_orig, p_loaded), "Weights do not match after loading"
    finally:
        os.unlink(tmp_path)
