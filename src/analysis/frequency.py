"""Frequency-domain analysis for cross-generator comparison.

Computes radial power spectra via 2D FFT to compare frequency signatures
across generator families. Based on the approach from Corvi et al. (2023).
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_radial_spectrum(image_gray: np.ndarray) -> np.ndarray:
    """Compute the 1-D radial average of the 2-D power spectrum.

    Args:
        image_gray: Grayscale image as float array, shape (H, W).

    Returns:
        1-D array of length max_radius, where each entry is the mean
        log-power at that radial frequency.
    """
    h, w = image_gray.shape
    f_transform = np.fft.fft2(image_gray)
    f_shift = np.fft.fftshift(f_transform)
    power = np.abs(f_shift) ** 2

    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

    max_radius = min(cy, cx)
    radial_mean = np.zeros(max_radius)
    for r in range(max_radius):
        mask = radius == r
        if mask.any():
            radial_mean[r] = power[mask].mean()

    # Log scale (avoid log(0))
    radial_mean = np.log10(radial_mean + 1e-10)
    return radial_mean


def batch_radial_spectra(images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and std of radial spectra over a batch of grayscale images.

    Args:
        images: List of grayscale float arrays, all same (H, W).

    Returns:
        (mean_spectrum, std_spectrum) each of shape (max_radius,).
    """
    spectra = [compute_radial_spectrum(img) for img in images]
    min_len = min(len(s) for s in spectra)
    spectra = np.array([s[:min_len] for s in spectra])
    return spectra.mean(axis=0), spectra.std(axis=0)


def plot_radial_spectra(
    family_spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    save_path: str = "outputs/plots/gpt4o_investigation/radial_power_spectra.png",
    title: str = "Radial Power Spectrum by Generator Family",
) -> None:
    """Overlay radial power spectra with shaded confidence bands.

    Args:
        family_spectra: Dict mapping family name -> (mean, std) arrays.
        save_path: Output file path.
        title: Plot title.
    """
    palette = ["#332288", "#88CCEE", "#44AA99", "#117733", "#999933",
               "#CC6677", "#882255", "#AA4499", "#DDCC77"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (name, (mean, std)) in enumerate(family_spectra.items()):
        freqs = np.arange(len(mean))
        color = palette[i % len(palette)]
        ax.plot(freqs, mean, label=name, color=color, linewidth=1.5)
        ax.fill_between(freqs, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Radial Frequency (pixels)")
    ax.set_ylabel("Log₁₀ Power")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Radial spectrum plot saved to: {save_path}")
