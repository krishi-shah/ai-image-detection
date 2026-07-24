"""Feature-space visualization via EfficientNet-B3 penultimate embeddings.

Extracts 1536-dim embeddings from the global average pooling layer and
projects them to 2-D using t-SNE for cross-generator comparison.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract penultimate-layer (1536-d) embeddings from EfficientNet-B3.

    Uses a forward hook on the global_pool layer to capture the feature
    vector before the classifier head.

    Args:
        model: Trained EfficientNet-B3 detector.
        loader: DataLoader yielding (images, labels).
        device: Torch device.

    Returns:
        (embeddings, labels) — numpy arrays of shape (N, 1536) and (N,).
    """
    model.eval()
    embeddings_list = []
    labels_list = []
    hook_output = {}

    def _hook_fn(module, input, output):
        hook_output["embedding"] = output.detach().cpu()

    handle = model.global_pool.register_forward_hook(_hook_fn)

    try:
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                _ = model(images)
                embeddings_list.append(hook_output["embedding"])
                labels_list.append(labels)
    finally:
        handle.remove()

    return (
        torch.cat(embeddings_list, dim=0).numpy(),
        torch.cat(labels_list, dim=0).numpy(),
    )


def compute_tsne(
    embeddings: np.ndarray,
    perplexity: float = 30.0,
    seed: int = 42,
) -> np.ndarray:
    """Project high-dimensional embeddings to 2-D via t-SNE.

    Args:
        embeddings: Array of shape (N, D).
        perplexity: t-SNE perplexity parameter.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (N, 2).
    """
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(embeddings) - 1),
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(embeddings)


def plot_tsne(
    coords_2d: np.ndarray,
    family_labels: list[str],
    save_path: str = "outputs/plots/gpt4o_investigation/feature_space_tsne.png",
    title: str = "Feature-Space t-SNE: Cross-Generator Embeddings",
) -> None:
    """Scatter plot of t-SNE projections colored by generator family.

    Args:
        coords_2d: Array of shape (N, 2) from compute_tsne.
        family_labels: List of length N with the family name per sample.
        save_path: Output file path.
        title: Plot title.
    """
    palette = {
        "CIFAKE_REAL": "#117733",
        "CIFAKE_FAKE": "#332288",
        "gpt4o": "#CC6677",
        "janus_pro": "#88CCEE",
        "midjourney_v6": "#DDCC77",
        "sd3_flux": "#44AA99",
        "stylegan": "#882255",
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_families = list(dict.fromkeys(family_labels))

    for family in unique_families:
        mask = np.array([f == family for f in family_labels])
        color = palette.get(family, "#999999")
        marker = "x" if family == "CIFAKE_REAL" else "o"
        alpha = 0.4 if family.startswith("CIFAKE") else 0.7
        size = 15 if family.startswith("CIFAKE") else 25

        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=color,
            label=family,
            marker=marker,
            alpha=alpha,
            s=size,
            edgecolors="none",
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.legend(loc="best", markerscale=2, fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  t-SNE plot saved to: {save_path}")
