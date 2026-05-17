import torch
import numpy as np
from PIL import Image
from src.utils.data_loader import get_transforms


def _make_dummy_image(size=(32, 32)):
    """Create a random RGB PIL image."""
    arr = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_transforms_train_output_shape():
    transform = get_transforms("train")
    img = _make_dummy_image()
    tensor = transform(img)
    assert tensor.shape == (3, 224, 224), f"Expected (3, 224, 224), got {tensor.shape}"


def test_transforms_val_output_shape():
    transform = get_transforms("val")
    img = _make_dummy_image(size=(256, 256))
    tensor = transform(img)
    assert tensor.shape == (3, 224, 224), f"Expected (3, 224, 224), got {tensor.shape}"


def test_transforms_test_output_shape():
    transform = get_transforms("test")
    img = _make_dummy_image(size=(256, 256))
    tensor = transform(img)
    assert tensor.shape == (3, 224, 224), f"Expected (3, 224, 224), got {tensor.shape}"


def test_transforms_normalization_range():
    """After ImageNet normalization, values should not all be in [0, 1]."""
    transform = get_transforms("val")
    img = _make_dummy_image(size=(256, 256))
    tensor = transform(img)
    assert tensor.min() < 0.0 or tensor.max() > 1.0, (
        "Normalized tensor should have values outside [0, 1]"
    )
