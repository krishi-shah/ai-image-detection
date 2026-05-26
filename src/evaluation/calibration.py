import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Scale logits by temperature and return softmax probabilities.

    Args:
        logits: Raw model outputs of shape (N, C).
        temperature: Scalar T > 0. T=1 is identity, T>1 softens, T<1 sharpens.

    Returns:
        Probability array of shape (N, C).
    """
    scaled = logits / temperature
    exp = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


class TemperatureScaler(nn.Module):
    """Learns a single temperature parameter to calibrate model confidence.

    Follows the post-hoc calibration method from Guo et al. (2017).
    The temperature is optimised on a held-out validation set after training.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1.5))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a single forward pass over a DataLoader and collect raw logits.

    Returns:
        (logits, labels) as tensors on CPU.
    """
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Collecting logits", leave=False):
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def find_temperature(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_iter: int = 50,
    lr: float = 0.01,
) -> float:
    """Learn the optimal temperature on a validation set using L-BFGS.

    Args:
        model: Trained classifier (kept frozen).
        val_loader: Validation DataLoader.
        device: Torch device.
        max_iter: Maximum L-BFGS iterations.
        lr: L-BFGS learning rate.

    Returns:
        The learned temperature value as a float.
    """
    logits, labels = collect_logits(model, val_loader, device)

    scaler = TemperatureScaler()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)

    def _eval():
        optimizer.zero_grad()
        loss = criterion(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(_eval)

    learned_t = scaler.temperature.item()

    # Print before/after ECE for immediate feedback
    from src.evaluation.metrics import compute_ece

    probs_before = torch.softmax(logits, dim=1).numpy()
    probs_after = apply_temperature(logits.numpy(), learned_t)

    ece_before = compute_ece(probs_before, labels.numpy())
    ece_after = compute_ece(probs_after, labels.numpy())

    print(f"Temperature: {learned_t:.4f}")
    print(f"ECE before:  {ece_before:.4f}")
    print(f"ECE after:   {ece_after:.4f}")

    return learned_t
