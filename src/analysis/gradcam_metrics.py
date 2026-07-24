"""Quantitative Grad-CAM attention metrics for cross-generator comparison.

Measures how focused or diffuse the model's attention is, giving numerical
evidence for whether the detector has a clear signal for a given generator.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def gradcam_entropy(heatmap: np.ndarray) -> float:
    """Compute Shannon entropy of a Grad-CAM heatmap.

    High entropy = diffuse, unfocused attention (model uncertain where to look).
    Low entropy = concentrated attention on specific artefacts.

    Args:
        heatmap: 2-D array with values in [0, 1].

    Returns:
        Entropy value (nats).  Higher = more diffuse.
    """
    flat = heatmap.flatten().astype(np.float64)
    flat = flat / (flat.sum() + 1e-10)
    flat = flat[flat > 0]
    return float(-np.sum(flat * np.log(flat)))


def peak_activation_ratio(heatmap: np.ndarray, top_frac: float = 0.1) -> float:
    """Fraction of total activation contained in the hottest pixels.

    High ratio = attention concentrated in a small region (strong signal).
    Low ratio = activation spread evenly (weak/absent signal).

    Args:
        heatmap: 2-D array with values in [0, 1].
        top_frac: Fraction of pixels to consider as "top" (default 10%).

    Returns:
        Ratio in [0, 1].
    """
    flat = heatmap.flatten()
    total = flat.sum() + 1e-10
    k = max(1, int(len(flat) * top_frac))
    top_k = np.partition(flat, -k)[-k:]
    return float(top_k.sum() / total)


def gini_coefficient(heatmap: np.ndarray) -> float:
    """Compute the Gini coefficient of heatmap activation values.

    Measures inequality of activation distribution.
    High Gini (-> 1) = very unequal, concentrated attention.
    Low Gini (-> 0) = uniform activation, no focus.

    Args:
        heatmap: 2-D array with values in [0, 1].

    Returns:
        Gini coefficient in [0, 1].
    """
    flat = np.sort(heatmap.flatten().astype(np.float64))
    n = len(flat)
    if n == 0 or flat.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * flat) / (n * np.sum(flat))) - (n + 1) / n)


def batch_gradcam_metrics(
    model: nn.Module,
    loader: DataLoader,
    target_layer,
    device: torch.device,
    max_images: int = 150,
) -> list[dict]:
    """Compute Grad-CAM attention metrics for a batch of images.

    Args:
        model: Trained detector.
        loader: DataLoader yielding (images, labels).
        target_layer: Conv layer for Grad-CAM (e.g. model.conv_head).
        device: Torch device.
        max_images: Cap on number of images to process.

    Returns:
        List of dicts, one per image, with keys:
        entropy, peak_ratio, gini, label, pred, p_fake.
    """
    model.eval()
    cam = GradCAM(model=model, target_layers=[target_layer])

    results = []
    count = 0

    for images, labels in loader:
        if count >= max_images:
            break

        images_dev = images.to(device)
        with torch.no_grad():
            logits = model(images_dev)
            probs = torch.softmax(logits, dim=1).cpu()

        for i in range(images.size(0)):
            if count >= max_images:
                break

            img_tensor = images[i].unsqueeze(0).to(device)
            grayscale_cam = cam(input_tensor=img_tensor)
            heatmap = grayscale_cam[0, :]

            results.append({
                "entropy": gradcam_entropy(heatmap),
                "peak_ratio": peak_activation_ratio(heatmap),
                "gini": gini_coefficient(heatmap),
                "label": int(labels[i].item()),
                "pred": int(probs[i].argmax().item()),
                "p_fake": float(probs[i, 0].item()),
            })
            count += 1

    return results


def plot_gradcam_metrics(
    family_metrics: dict[str, list[dict]],
    save_dir: str = "outputs/plots/gpt4o_investigation",
) -> None:
    """Create violin plots comparing Grad-CAM metrics across families.

    Generates three plots: entropy, peak activation ratio, and Gini coefficient.

    Args:
        family_metrics: Dict mapping family name -> list of metric dicts
                        (from batch_gradcam_metrics, filtered to FAKE images only).
        save_dir: Directory for output plots.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    metric_configs = [
        ("entropy", "Grad-CAM Entropy (higher = more diffuse)", "gradcam_entropy_comparison.png"),
        ("peak_ratio", "Peak Activation Ratio (higher = more focused)", "gradcam_peak_ratio_comparison.png"),
        ("gini", "Gini Coefficient (higher = more concentrated)", "gradcam_gini_comparison.png"),
    ]

    palette = ["#332288", "#88CCEE", "#44AA99", "#117733", "#999933",
               "#CC6677", "#882255"]

    for metric_key, ylabel, filename in metric_configs:
        families = list(family_metrics.keys())
        data = []
        for fam in families:
            values = [m[metric_key] for m in family_metrics[fam]]
            data.append(values)

        fig, ax = plt.subplots(figsize=(10, 6))
        parts = ax.violinplot(data, positions=range(len(families)), showmeans=True, showmedians=True)

        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(palette[i % len(palette)])
            pc.set_alpha(0.7)

        ax.set_xticks(range(len(families)))
        ax.set_xticklabels(families, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Grad-CAM {metric_key.replace('_', ' ').title()} by Generator Family\n(FAKE images only)")
        ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        save_path = str(Path(save_dir) / filename)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"  {metric_key} plot saved to: {save_path}")
