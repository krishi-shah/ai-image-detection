import os
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_gradcam(model, image_tensor, target_layer) -> np.ndarray:
    """Generate a Grad-CAM heatmap for a single image.

    Args:
        model: The EfficientNet-B3 detector (nn.Module).
        image_tensor: Preprocessed image tensor of shape (1, 3, 224, 224).
        target_layer: The convolutional layer to visualise
                      (e.g. model.conv_head for EfficientNet-B3).

    Returns:
        Heatmap as a numpy array of shape (224, 224) with values in [0, 1].
    """
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=image_tensor)
    return grayscale_cam[0, :]


def save_heatmap(
    heatmap: np.ndarray,
    original_image: np.ndarray,
    save_path: str,
) -> None:
    """Overlay a Grad-CAM heatmap on the original image and save.

    Args:
        heatmap: Array of shape (H, W) with values in [0, 1].
        original_image: RGB image as numpy array with values in [0, 1],
                        shape (H, W, 3).
        save_path: Destination file path (e.g. 'outputs/heatmaps/sample.png').
    """
    overlay = show_cam_on_image(original_image, heatmap, use_rgb=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# Batch failure analysis for generalisation study
# ---------------------------------------------------------------------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized image tensor (C,H,W) back to [0,1] RGB numpy."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1).astype(np.float32)


def batch_gradcam_failures(
    model: nn.Module,
    loader,
    target_layer,
    family: str,
    device: torch.device,
    k: int = 16,
    output_dir: str = "outputs/heatmaps",
) -> list[dict]:
    """Generate Grad-CAM heatmaps for failure cases in a generator family.

    Selects the k most confident FALSE NEGATIVES (fakes classified as real)
    and k lowest-confidence correct detections.

    Args:
        model: Trained detector.
        loader: DataLoader for the generator family.
        target_layer: Conv layer for Grad-CAM (e.g. model.conv_head).
        family: Generator family name.
        device: Torch device.
        k: Number of samples to visualise per category.
        output_dir: Base output directory for heatmaps.

    Returns:
        List of dicts with metadata for each generated heatmap.
    """
    model.eval()
    cam = GradCAM(model=model, target_layers=[target_layer])

    all_items = []

    with torch.no_grad():
        for images, labels in loader:
            images_dev = images.to(device)
            logits = model(images_dev)
            probs = torch.softmax(logits, dim=1).cpu()

            for i in range(images.size(0)):
                label = labels[i].item()
                pred = probs[i].argmax().item()
                p_fake = probs[i, 0].item()

                all_items.append({
                    "image": images[i],
                    "label": label,
                    "pred": pred,
                    "p_fake": p_fake,
                    "is_false_negative": (label == 0 and pred == 1),
                    "is_correct_fake": (label == 0 and pred == 0),
                })

    # Select k most confident false negatives (lowest P(FAKE) among FN)
    false_negatives = sorted(
        [x for x in all_items if x["is_false_negative"]],
        key=lambda x: x["p_fake"],
    )[:k]

    # Select k lowest-confidence correct detections (P(FAKE) closest to 0.5 among correct)
    correct_fakes = sorted(
        [x for x in all_items if x["is_correct_fake"]],
        key=lambda x: x["p_fake"],
    )[:k]

    save_dir = Path(output_dir) / f"gradcam_{family}_failures"
    save_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for category, items, prefix in [
        ("false_negative", false_negatives, "fn"),
        ("low_confidence_correct", correct_fakes, "lc"),
    ]:
        for idx, item in enumerate(items):
            img_tensor = item["image"].unsqueeze(0)
            grayscale_cam = cam(input_tensor=img_tensor.to(device))
            heatmap = grayscale_cam[0, :]

            original = _denormalize(item["image"])
            filename = f"{prefix}_{idx:02d}_p{item['p_fake']:.3f}.png"
            save_path = str(save_dir / filename)
            save_heatmap(heatmap, original, save_path)

            results.append({
                "family": family,
                "category": category,
                "index": idx,
                "p_fake": item["p_fake"],
                "save_path": save_path,
            })

    print(f"  [Grad-CAM] {family}: {len(false_negatives)} false negatives, "
          f"{len(correct_fakes)} low-confidence correct → {save_dir}")

    return results


def comparative_grid(
    families: list[str],
    heatmap_dir: str = "outputs/heatmaps",
    n_per_family: int = 4,
    save_path: str = "outputs/plots/gradcam_comparison_grid.png",
) -> None:
    """Create a single figure comparing failure heatmaps across families.

    Rows = families, columns = failure heatmaps (false negatives).

    Args:
        families: List of family names.
        heatmap_dir: Directory containing gradcam_{family}_failures/ subdirs.
        n_per_family: Number of heatmaps per family to show.
        save_path: Output path for the comparison grid.
    """
    fig, axes = plt.subplots(
        len(families), n_per_family,
        figsize=(3 * n_per_family, 3 * len(families)),
    )
    if len(families) == 1:
        axes = axes[np.newaxis, :]
    if n_per_family == 1:
        axes = axes[:, np.newaxis]

    for row, family in enumerate(families):
        family_dir = Path(heatmap_dir) / f"gradcam_{family}_failures"
        # Get false negative heatmaps (fn_*.png)
        heatmap_files = sorted(family_dir.glob("fn_*.png")) if family_dir.exists() else []

        for col in range(n_per_family):
            ax = axes[row, col]
            if col < len(heatmap_files):
                img = cv2.imread(str(heatmap_files[col]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                # Extract confidence from filename
                fname = heatmap_files[col].stem
                p_val = fname.split("_p")[-1] if "_p" in fname else ""
                ax.set_title(f"P(FAKE)={p_val}", fontsize=8)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12)
            ax.axis("off")

            if col == 0:
                ax.set_ylabel(family, fontsize=11, rotation=0, labelpad=60, va="center")

    fig.suptitle("Grad-CAM Failure Analysis: False Negatives by Generator", fontsize=13)
    fig.tight_layout(rect=[0.05, 0, 1, 0.96])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  [Grad-CAM] Comparison grid saved to: {save_path}")


def write_attention_notes(
    families: list[str],
    output_dir: str = "outputs/heatmaps",
) -> None:
    """Write a template markdown file for qualitative attention observations."""
    notes_path = Path(output_dir) / "attention_notes.md"
    lines = [
        "# Grad-CAM Attention Analysis Notes\n\n",
        "Fill in qualitative observations for each generator family.\n\n",
    ]
    for family in families:
        lines.append(f"## {family}\n\n")
        lines.append("**False negatives (fakes classified as real):**\n\n")
        lines.append("- TODO: Describe where the model attends on these failure cases\n")
        lines.append("- TODO: What artifacts are missing compared to CIFAKE fakes?\n\n")
        lines.append("**Low-confidence correct detections:**\n\n")
        lines.append("- TODO: What features is the model uncertain about?\n")
        lines.append("- TODO: How does the attention pattern differ from high-confidence cases?\n\n")
        lines.append("---\n\n")

    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text("".join(lines))
    print(f"  Attention notes template: {notes_path}")
