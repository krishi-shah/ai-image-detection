import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def compute_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of predictions that match the true labels."""
    return float((preds == labels).mean())


def compute_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """AUC-ROC using the positive-class probability (column 1)."""
    return float(roc_auc_score(labels, probs[:, 1]))


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (Guo et al., 2017).

    Bins predictions by their maximum confidence, then computes the
    weighted average of |accuracy - confidence| per bin.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)

    return float(ece / len(labels))


def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    n_bins: int = 10,
) -> None:
    """Save a reliability diagram comparing per-bin accuracy to confidence."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs, bin_confs, bin_sizes = [], [], []
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            bin_sizes.append(0)
        else:
            bin_accs.append(float(accuracies[mask].mean()))
            bin_confs.append(float(confidences[mask].mean()))
            bin_sizes.append(int(mask.sum()))

    bin_centres = [(bin_boundaries[i] + bin_boundaries[i + 1]) / 2 for i in range(n_bins)]
    width = 1.0 / n_bins

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(bin_centres, bin_accs, width=width * 0.9, alpha=0.7, label="Accuracy")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    save_path: str,
) -> None:
    """Save an ROC curve with AUC annotated in the legend."""
    fpr, tpr, _ = roc_curve(labels, probs[:, 1])
    auc_score = roc_auc_score(labels, probs[:, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
