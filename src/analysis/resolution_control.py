"""Resolution-matched control experiment.

Tests whether the generalisation evaluation confounds class labels with
resolution: CIFAKE reals are 32x32 upscaled to 224, while generator fakes
are high-resolution images downscaled to 224.

Conditions:
  A  CIFAKE REAL, native eval pipeline (reference real)
  B  High-res real photographs, native eval pipeline
  C  Same high-res reals forced to 32x32, then native pipeline
  D  Generator fakes forced to 32x32, then native pipeline
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.data_loader import IMAGENET_MEAN, IMAGENET_STD, _FolderDataset


PALETTE = ["#332288", "#88CCEE", "#44AA99", "#117733", "#CC6677", "#882255"]


def get_matched_lowres_transform() -> transforms.Compose:
    """Eval transform that first forces images to 32x32 (CIFAKE resolution).

    Reproduces the effective resolution of CIFAKE images before the standard
    Resize(256) + CenterCrop(224) + ImageNet normalize pipeline.
    """
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_native_eval_transform() -> transforms.Compose:
    """Standard eval transform used by the rest of the project."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def evaluate_condition(
    model: nn.Module,
    image_dir: str | Path,
    true_label: int,
    transform: transforms.Compose,
    device: torch.device,
    temperature: float = 1.0,
    batch_size: int = 32,
    max_images: Optional[int] = None,
    condition_name: str = "unknown",
) -> dict:
    """Evaluate the detector on a single folder of images with a fixed label.

    Args:
        model: Trained detector.
        image_dir: Folder of RGB images.
        true_label: 0=FAKE, 1=REAL (ground-truth for this condition).
        transform: Preprocessing pipeline to apply.
        device: Torch device.
        temperature: Softmax temperature (for calibrated confidence only).
        batch_size: DataLoader batch size.
        max_images: Optional cap on number of images.
        condition_name: Label for logging / JSON.

    Returns:
        Dict with FAKE-rate, mean/median P(FAKE), per-image scores, etc.
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    dataset = _FolderDataset(image_dir, transform=transform, label=true_label)
    if len(dataset) == 0:
        raise ValueError(f"No images found in {image_dir}")

    if max_images is not None and len(dataset) > max_images:
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(dataset), generator=generator)[:max_images].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model.eval()
    all_probs = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc=f"Evaluating {condition_name}", leave=False):
            images = images.to(device)
            logits = model(images)
            scaled = logits / temperature
            probs = torch.softmax(scaled, dim=1)
            all_probs.append(probs.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    p_fake = probs[:, 0]
    preds = np.argmax(probs, axis=1)
    fake_rate = float((preds == 0).mean())

    return {
        "condition": condition_name,
        "true_label": int(true_label),
        "n_images": int(len(p_fake)),
        "fake_rate": fake_rate,
        "mean_p_fake": float(p_fake.mean()),
        "median_p_fake": float(np.median(p_fake)),
        "std_p_fake": float(p_fake.std()),
        "p_fake": p_fake.tolist(),
    }


def run_resolution_control(
    model: nn.Module,
    device: torch.device,
    cifake_real_dir: str | Path,
    hires_real_dir: str | Path,
    generator_fake_dirs: dict[str, str | Path],
    temperature: float = 1.0,
    n_images: int = 300,
    batch_size: int = 32,
    output_dir: str | Path = "outputs",
) -> dict:
    """Run the full 2x2 resolution-control grid and write results + plots.

    Args:
        model: Trained detector.
        device: Torch device.
        cifake_real_dir: Path to CIFAKE test/REAL images (condition A).
        hires_real_dir: Path to high-resolution real photographs (B, C).
        generator_fake_dirs: Mapping family_name -> FAKE folder path (D).
        temperature: Softmax temperature.
        n_images: Cap per condition.
        batch_size: DataLoader batch size.
        output_dir: Root output directory.

    Returns:
        Full results dict (also written to JSON).
    """
    output_dir = Path(output_dir)
    results_dir = output_dir / "results"
    plots_dir = output_dir / "plots" / "resolution_control"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    native = get_native_eval_transform()
    matched = get_matched_lowres_transform()

    conditions = {}

    print("\n=== Condition A: CIFAKE REAL (native) ===")
    conditions["A_cifake_real_native"] = evaluate_condition(
        model, cifake_real_dir, true_label=1, transform=native,
        device=device, temperature=temperature, batch_size=batch_size,
        max_images=n_images, condition_name="A_cifake_real_native",
    )

    print("\n=== Condition B: High-res REAL (native) ===")
    conditions["B_hires_real_native"] = evaluate_condition(
        model, hires_real_dir, true_label=1, transform=native,
        device=device, temperature=temperature, batch_size=batch_size,
        max_images=n_images, condition_name="B_hires_real_native",
    )

    print("\n=== Condition C: High-res REAL (forced 32x32) ===")
    conditions["C_hires_real_matched"] = evaluate_condition(
        model, hires_real_dir, true_label=1, transform=matched,
        device=device, temperature=temperature, batch_size=batch_size,
        max_images=n_images, condition_name="C_hires_real_matched",
    )

    print("\n=== Condition D: Generator FAKEs (forced 32x32) ===")
    condition_d = {}
    for family, fake_dir in generator_fake_dirs.items():
        fake_path = Path(fake_dir)
        if not fake_path.exists():
            print(f"  Skipping {family}: {fake_path} not found")
            continue
        print(f"  --- {family} ---")
        condition_d[family] = evaluate_condition(
            model, fake_path, true_label=0, transform=matched,
            device=device, temperature=temperature, batch_size=batch_size,
            max_images=n_images, condition_name=f"D_{family}_matched",
        )
    conditions["D_generator_fakes_matched"] = condition_d

    # Also evaluate generator fakes at native resolution for before/after
    print("\n=== Generator FAKEs (native, for comparison) ===")
    native_fakes = {}
    for family, fake_dir in generator_fake_dirs.items():
        fake_path = Path(fake_dir)
        if not fake_path.exists():
            continue
        native_fakes[family] = evaluate_condition(
            model, fake_path, true_label=0, transform=native,
            device=device, temperature=temperature, batch_size=batch_size,
            max_images=n_images, condition_name=f"{family}_native",
        )
    conditions["generator_fakes_native"] = native_fakes

    interpretation = _interpret(conditions)

    results = {
        "experiment": "resolution_control",
        "n_images_per_condition": n_images,
        "temperature": temperature,
        "conditions": {
            k: _strip_p_fake(v) if not isinstance(v, dict) or "fake_rate" in v
            else {fk: _strip_p_fake(fv) for fk, fv in v.items()}
            for k, v in conditions.items()
        },
        "distributions": {
            "A_cifake_real_native": conditions["A_cifake_real_native"]["p_fake"],
            "B_hires_real_native": conditions["B_hires_real_native"]["p_fake"],
            "C_hires_real_matched": conditions["C_hires_real_matched"]["p_fake"],
        },
        "interpretation": interpretation,
    }

    json_path = results_dir / "resolution_control.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    plot_fake_rate_bars(conditions, plots_dir / "fake_rate_by_condition.png")
    plot_p_fake_distributions(conditions, plots_dir / "p_fake_distributions.png")
    plot_before_after_matching(
        native_fakes, condition_d, plots_dir / "before_after_resolution_matching.png"
    )

    return results


def _strip_p_fake(cond: dict) -> dict:
    """Return condition dict without the full per-image list (kept separately)."""
    return {k: v for k, v in cond.items() if k != "p_fake"}


def _interpret(conditions: dict) -> dict:
    """Derive a concise interpretation from the 2x2 grid."""
    a = conditions["A_cifake_real_native"]["fake_rate"]
    b = conditions["B_hires_real_native"]["fake_rate"]
    c = conditions["C_hires_real_matched"]["fake_rate"]

    d_rates = {
        fam: r["fake_rate"]
        for fam, r in conditions.get("D_generator_fakes_matched", {}).items()
    }
    native_rates = {
        fam: r["fake_rate"]
        for fam, r in conditions.get("generator_fakes_native", {}).items()
    }

    # High false-positive on high-res reals => resolution confound
    confound_suspected = b > 0.20  # >20% of high-res reals called FAKE

    return {
        "A_cifake_real_fake_rate": a,
        "B_hires_real_fake_rate": b,
        "C_hires_real_matched_fake_rate": c,
        "D_matched_fake_detection_rates": d_rates,
        "native_fake_detection_rates": native_rates,
        "resolution_confound_suspected": confound_suspected,
        "summary": (
            "High false-positive rate on high-resolution reals suggests the "
            "detector partly relies on resolution/resampling cues."
            if confound_suspected else
            "Low false-positive rate on high-resolution reals suggests the "
            "detector is not primarily using resolution as a class cue."
        ),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_fake_rate_bars(conditions: dict, save_path: str | Path) -> None:
    """Grouped bar chart of FAKE-rate by condition."""
    labels = ["A: CIFAKE REAL\n(native)", "B: Hi-res REAL\n(native)", "C: Hi-res REAL\n(32x32)"]
    rates = [
        conditions["A_cifake_real_native"]["fake_rate"],
        conditions["B_hires_real_native"]["fake_rate"],
        conditions["C_hires_real_matched"]["fake_rate"],
    ]

    # Average D across families
    d = conditions.get("D_generator_fakes_matched", {})
    if d:
        labels.append("D: Gen. FAKEs\n(32x32, mean)")
        rates.append(float(np.mean([r["fake_rate"] for r in d.values()])))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, rates, color=PALETTE[:len(rates)], edgecolor="black", linewidth=0.5)
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{rate:.1%}", ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylabel("Fraction predicted FAKE")
    ax.set_title("Resolution Control: FAKE Prediction Rate by Condition")
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def plot_p_fake_distributions(conditions: dict, save_path: str | Path) -> None:
    """Box/violin of P(FAKE) for conditions A, B, C."""
    data = [
        conditions["A_cifake_real_native"]["p_fake"],
        conditions["B_hires_real_native"]["p_fake"],
        conditions["C_hires_real_matched"]["p_fake"],
    ]
    labels = ["A: CIFAKE REAL", "B: Hi-res REAL", "C: Hi-res→32×32"]

    fig, ax = plt.subplots(figsize=(9, 5))
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(PALETTE[i])
        body.set_alpha(0.7)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels)
    ax.set_ylabel("P(FAKE)")
    ax.set_title("P(FAKE) Distributions — Real-Image Conditions")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def plot_before_after_matching(
    native_fakes: dict,
    matched_fakes: dict,
    save_path: str | Path,
) -> None:
    """Per-family fake detection rate: native vs resolution-matched."""
    families = sorted(set(native_fakes) & set(matched_fakes))
    if not families:
        return

    native_rates = [native_fakes[f]["fake_rate"] for f in families]
    matched_rates = [matched_fakes[f]["fake_rate"] for f in families]

    x = np.arange(len(families))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, native_rates, width, label="Native (high-res→224)", color=PALETTE[0])
    ax.bar(x + width / 2, matched_rates, width, label="Matched (→32×32→224)", color=PALETTE[4])
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=15, ha="right")
    ax.set_ylabel("Fake Detection Rate")
    ax.set_title("Generator Fake Detection: Native vs Resolution-Matched")
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")
