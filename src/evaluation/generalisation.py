"""Cross-generator generalisation evaluation module.

Evaluates the CIFAKE-trained detector on images from different generator
families and measures performance degradation relative to the baseline.
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.metrics import compute_ece
from src.evaluation.calibration import apply_temperature, collect_logits


# ---------------------------------------------------------------------------
# Colorblind-safe palette (Tol's qualitative)
# ---------------------------------------------------------------------------

PALETTE = ["#332288", "#88CCEE", "#44AA99", "#117733", "#999933",
           "#DDCC77", "#CC6677", "#882255", "#AA4499"]


def _set_plot_style():
    """Apply shared plotting defaults."""
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_generator(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float = 1.0,
    family_name: str = "unknown",
) -> dict:
    """Run full evaluation on a single generator family's images.

    Args:
        model: Trained EfficientNet-B3 detector.
        loader: DataLoader for the generator family (labels: 0=FAKE, 1=REAL).
        device: Torch device.
        temperature: Learned temperature for calibration (default 1.0 = uncalibrated).
        family_name: Name for this generator family.

    Returns:
        Dict with all evaluation metrics and per-image scores.
    """
    logits_t, labels_t = collect_logits(model, loader, device)
    logits = logits_t.numpy()
    labels = labels_t.numpy()

    # Probabilities: uncalibrated and calibrated
    probs_uncal = torch.softmax(logits_t, dim=1).numpy()
    probs_cal = apply_temperature(logits, temperature)

    preds = np.argmax(probs_uncal, axis=1)

    # --- Metrics ---
    acc = float(accuracy_score(labels, preds))

    # FAKE detection rate: accuracy on FAKE images only (label=0)
    fake_mask = labels == 0
    fake_detection_rate = float((preds[fake_mask] == 0).mean()) if fake_mask.any() else None

    # AUC (only meaningful if both classes present)
    has_both_classes = len(np.unique(labels)) > 1
    auc = float(roc_auc_score(labels, probs_uncal[:, 1])) if has_both_classes else None

    # Precision, recall, F1 for FAKE class (label=0)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0], average=None, zero_division=0.0
    )
    precision_fake = float(precision[0])
    recall_fake = float(recall[0])
    f1_fake = float(f1[0])

    # ECE before and after temperature scaling
    ece_pre = float(compute_ece(probs_uncal, labels))
    ece_post = float(compute_ece(probs_cal, labels))

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()

    # Confidence statistics on FAKE images
    p_fake_on_fakes = probs_uncal[fake_mask, 0] if fake_mask.any() else np.array([])
    confidence_stats = {
        "mean_p_fake": float(p_fake_on_fakes.mean()) if len(p_fake_on_fakes) > 0 else None,
        "median_p_fake": float(np.median(p_fake_on_fakes)) if len(p_fake_on_fakes) > 0 else None,
    }

    # Per-image scores (for later analysis)
    per_image = {
        "probs_fake": probs_uncal[:, 0].tolist(),
        "labels": labels.tolist(),
        "preds": preds.tolist(),
    }

    return {
        "family": family_name,
        "n_samples": int(len(labels)),
        "n_fake": int(fake_mask.sum()),
        "n_real": int((labels == 1).sum()),
        "accuracy": acc,
        "fake_detection_rate": fake_detection_rate,
        "auc": auc,
        "precision_fake": precision_fake,
        "recall_fake": recall_fake,
        "f1_fake": f1_fake,
        "ece_pre_calibration": ece_pre,
        "ece_post_calibration": ece_post,
        "confusion_matrix": cm,
        "confidence_stats": confidence_stats,
        "temperature_used": temperature,
        "per_image": per_image,
    }


# ---------------------------------------------------------------------------
# Degradation computation
# ---------------------------------------------------------------------------

def compute_degradation(baseline_results: dict, gen_results: dict) -> dict:
    """Compute absolute and relative performance drop vs CIFAKE baseline.

    Args:
        baseline_results: Dict from baseline_results.json (keys: test_accuracy, test_auc).
        gen_results: Dict returned by evaluate_generator().

    Returns:
        Dict with degradation metrics.
    """
    baseline_acc = baseline_results["test_accuracy"]
    baseline_auc = baseline_results.get("test_auc")

    gen_acc = gen_results["accuracy"]
    gen_auc = gen_results.get("auc")

    acc_drop_abs = baseline_acc - gen_acc
    acc_drop_rel = acc_drop_abs / baseline_acc if baseline_acc > 0 else 0.0

    auc_drop_abs = None
    auc_drop_rel = None
    if baseline_auc is not None and gen_auc is not None:
        auc_drop_abs = baseline_auc - gen_auc
        auc_drop_rel = auc_drop_abs / baseline_auc if baseline_auc > 0 else 0.0

    return {
        "family": gen_results["family"],
        "baseline_accuracy": baseline_acc,
        "generator_accuracy": gen_acc,
        "accuracy_drop_absolute": acc_drop_abs,
        "accuracy_drop_relative": acc_drop_rel,
        "baseline_auc": baseline_auc,
        "generator_auc": gen_auc,
        "auc_drop_absolute": auc_drop_abs,
        "auc_drop_relative": auc_drop_rel,
    }


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def save_generalisation_results(
    all_results: dict[str, dict],
    degradation: dict[str, dict],
    output_dir: str = "outputs/results",
) -> None:
    """Save full results and degradation summary as JSON.

    Args:
        all_results: Dict mapping family name -> evaluate_generator() output.
        degradation: Dict mapping family name -> compute_degradation() output.
        output_dir: Directory for output JSON files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Full results (strip per_image for the summary file)
    summary = {}
    for fam, res in all_results.items():
        entry = {k: v for k, v in res.items() if k != "per_image"}
        summary[fam] = entry

    (out / "generalisation_results.json").write_text(
        json.dumps(summary, indent=2)
    )

    # Degradation summary
    (out / "degradation_summary.json").write_text(
        json.dumps(degradation, indent=2)
    )

    print(f"Results saved to: {out / 'generalisation_results.json'}")
    print(f"Degradation saved to: {out / 'degradation_summary.json'}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_cross_generator_accuracy(
    all_results: dict[str, dict],
    baseline_acc: float,
    save_path: str = "outputs/plots/cross_generator_accuracy.png",
) -> None:
    """Bar chart of accuracy per generator with baseline reference line."""
    _set_plot_style()
    families = list(all_results.keys())
    accs = [all_results[f]["accuracy"] for f in families]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(families, accs, color=PALETTE[:len(families)], width=0.6)
    ax.axhline(baseline_acc, color="black", linestyle="--", linewidth=1.2,
               label=f"CIFAKE baseline ({baseline_acc:.1%})")
    ax.set_ylabel("Accuracy")
    ax.set_title("Cross-Generator Detection Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_cross_generator_auc(
    all_results: dict[str, dict],
    baseline_auc: float,
    save_path: str = "outputs/plots/cross_generator_auc.png",
) -> None:
    """Bar chart of AUC per generator with baseline reference line."""
    _set_plot_style()
    families = [f for f in all_results if all_results[f].get("auc") is not None]
    aucs = [all_results[f]["auc"] for f in families]

    if not families:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(families, aucs, color=PALETTE[:len(families)], width=0.6)
    ax.axhline(baseline_auc, color="black", linestyle="--", linewidth=1.2,
               label=f"CIFAKE baseline ({baseline_auc:.4f})")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Cross-Generator AUC-ROC")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")

    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{auc:.4f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_degradation_waterfall(
    degradation: dict[str, dict],
    baseline_acc: float,
    save_path: str = "outputs/plots/degradation_waterfall.png",
) -> None:
    """Waterfall chart showing accuracy drop from baseline per family."""
    _set_plot_style()

    # Sort by accuracy (worst to best degradation)
    sorted_fams = sorted(
        degradation.keys(),
        key=lambda f: degradation[f]["generator_accuracy"],
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    x_labels = ["CIFAKE\n(baseline)"] + sorted_fams
    values = [baseline_acc] + [degradation[f]["generator_accuracy"] for f in sorted_fams]
    colors = ["#117733"] + [PALETTE[i % len(PALETTE)] for i in range(len(sorted_fams))]

    bars = ax.bar(x_labels, values, color=colors, width=0.6)
    ax.axhline(baseline_acc, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Accuracy")
    ax.set_title("Performance Degradation Across Generators")
    ax.set_ylim(0, 1.05)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_confidence_distributions(
    all_results: dict[str, dict],
    save_path: str = "outputs/plots/confidence_distributions_by_generator.png",
) -> None:
    """Overlaid histograms of P(FAKE) on fake images per generator."""
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (fam, res) in enumerate(all_results.items()):
        per_image = res.get("per_image", {})
        probs = np.array(per_image.get("probs_fake", []))
        labels = np.array(per_image.get("labels", []))

        # Only plot FAKE images (label=0)
        fake_probs = probs[labels == 0] if len(labels) > 0 else probs
        if len(fake_probs) == 0:
            continue

        ax.hist(fake_probs, bins=30, alpha=0.5, label=fam,
                color=PALETTE[i % len(PALETTE)], density=True)

    ax.set_xlabel("P(FAKE)")
    ax.set_ylabel("Density")
    ax.set_title("Confidence Distribution on AI-Generated Images by Generator")
    ax.legend(loc="upper center")
    ax.set_xlim(0, 1)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_ece_comparison(
    all_results: dict[str, dict],
    save_path: str = "outputs/plots/ece_comparison_by_generator.png",
) -> None:
    """Grouped bar chart of ECE pre/post temperature scaling per generator."""
    _set_plot_style()
    families = list(all_results.keys())
    ece_pre = [all_results[f]["ece_pre_calibration"] for f in families]
    ece_post = [all_results[f]["ece_post_calibration"] for f in families]

    x = np.arange(len(families))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, ece_pre, width, label="Before T-scaling", color=PALETTE[0])
    ax.bar(x + width / 2, ece_post, width, label="After T-scaling", color=PALETTE[2])
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_ylabel("Expected Calibration Error")
    ax.set_title("Calibration Under Distribution Shift")
    ax.legend()

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_generalisation_eval(
    model: nn.Module,
    families_root: str,
    real_reference_dir: str,
    device: torch.device,
    temperature: float = 1.0,
    baseline_path: str = "outputs/results/baseline_results.json",
    output_dir: str = "outputs",
    batch_size: int = 32,
) -> dict[str, dict]:
    """Run the full generalisation evaluation pipeline.

    Args:
        model: Trained detector.
        families_root: Root directory containing generator family folders.
        real_reference_dir: Path to real reference images (e.g., CIFAKE test/REAL).
        device: Torch device.
        temperature: Learned temperature.
        baseline_path: Path to baseline_results.json.
        output_dir: Root output directory.
        batch_size: Batch size for evaluation.

    Returns:
        Dict mapping family name -> evaluation results.
    """
    from src.utils.data_loader import (
        discover_generator_families,
        get_generalisation_loader,
    )

    # Load baseline
    with open(baseline_path) as f:
        baseline = json.load(f)

    family_dirs = discover_generator_families(families_root)
    if not family_dirs:
        raise FileNotFoundError(f"No generator families found in {families_root}")

    all_results = {}
    all_degradation = {}

    for fdir in family_dirs:
        family_name = Path(fdir).name
        print(f"\n{'='*50}")
        print(f"Evaluating: {family_name}")
        print(f"{'='*50}")

        loader = get_generalisation_loader(
            family_dir=fdir,
            real_reference_dir=real_reference_dir,
            batch_size=batch_size,
            matched_real=(Path(fdir) / "REAL").exists(),
        )

        results = evaluate_generator(
            model=model,
            loader=loader,
            device=device,
            temperature=temperature,
            family_name=family_name,
        )
        all_results[family_name] = results

        deg = compute_degradation(baseline, results)
        all_degradation[family_name] = deg

        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Fake detection rate: {results['fake_detection_rate']}")
        print(f"  Drop from baseline: {deg['accuracy_drop_absolute']:.4f} "
              f"({deg['accuracy_drop_relative']:.1%} relative)")

    # Save results
    results_dir = os.path.join(output_dir, "results")
    save_generalisation_results(all_results, all_degradation, results_dir)

    # Generate plots
    plots_dir = os.path.join(output_dir, "plots")
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    plot_cross_generator_accuracy(
        all_results, baseline["test_accuracy"],
        os.path.join(plots_dir, "cross_generator_accuracy.png"),
    )
    if baseline.get("test_auc"):
        plot_cross_generator_auc(
            all_results, baseline["test_auc"],
            os.path.join(plots_dir, "cross_generator_auc.png"),
        )
    plot_degradation_waterfall(
        all_degradation, baseline["test_accuracy"],
        os.path.join(plots_dir, "degradation_waterfall.png"),
    )
    plot_confidence_distributions(
        all_results,
        os.path.join(plots_dir, "confidence_distributions_by_generator.png"),
    )
    plot_ece_comparison(
        all_results,
        os.path.join(plots_dir, "ece_comparison_by_generator.png"),
    )

    return all_results
