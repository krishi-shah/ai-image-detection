import torch
import torch.nn as nn
import timm


def build_detector(pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    """Create an EfficientNet-B3 binary classifier.

    Loads ImageNet-pretrained weights via timm, replaces the classifier head
    with a Linear layer mapping to `num_classes` outputs.
    """
    model = timm.create_model("efficientnet_b3", pretrained=pretrained)
    in_features = model.classifier.in_features  # 1536 for EfficientNet-B3
    model.classifier = nn.Linear(in_features, num_classes)
    return model


def save_checkpoint(
    model: nn.Module,
    path: str,
    epoch: int,
    metrics: dict,
) -> None:
    """Save model weights, epoch, and metrics to a .pth file."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(model: nn.Module, path: str) -> tuple:
    """Load weights from a .pth checkpoint.

    Returns (epoch, metrics) after restoring model state.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint["epoch"], checkpoint["metrics"]
