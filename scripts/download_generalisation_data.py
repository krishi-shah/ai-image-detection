"""Download generalisation evaluation images from HuggingFace datasets.

Uses streaming mode to avoid downloading full datasets. Samples N images
per generator family with a fixed seed for reproducibility. Idempotent:
skips families whose folder already contains >= N images.

Usage (CLI):
    python scripts/download_generalisation_data.py --output-dir data/generalisation --per-generator 300

Usage (from notebook):
    from scripts.download_generalisation_data import download_all_families
    download_all_families(output_dir="data/generalisation", per_generator=300)
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Family download functions
# ---------------------------------------------------------------------------

def _save_image(img: Image.Image, path: Path, idx: int, fmt: str = "png") -> str:
    """Save a PIL image and return the filename."""
    filename = f"{idx:04d}.{fmt}"
    filepath = path / filename
    img.save(filepath)
    return filename


def _count_existing(folder: Path) -> int:
    """Count image files already present in a folder."""
    if not folder.exists():
        return 0
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in exts)


def download_stylegan(
    output_dir: Path,
    n: int = 300,
    seed: int = 42,
) -> dict:
    """Download StyleGAN images from ForenSynths / CommunityForensics.

    Primary: OwensLab/CommunityForensics on HuggingFace filtered to GAN family.
    Fallback: manual instructions printed.
    """
    from datasets import load_dataset

    family_dir = output_dir / "stylegan"
    fake_dir = family_dir / "FAKE"
    fake_dir.mkdir(parents=True, exist_ok=True)

    existing = _count_existing(fake_dir)
    if existing >= n:
        print(f"  [stylegan] Already have {existing} images, skipping.")
        return {"family": "stylegan", "count": existing, "skipped": True}

    print(f"  [stylegan] Downloading {n} images from wang-research/CNNDetection...")
    try:
        ds = load_dataset(
            "wang-research/CNNDetection",
            split="test",
            streaming=True,
        )
        rng = random.Random(seed)

        reservoir = []
        for i, sample in enumerate(tqdm(ds, desc="stylegan", total=n * 5)):
            label = sample.get("label", None)
            # CNNDetection: label 1 = fake for progan/stylegan subsets
            if label == 1:
                reservoir.append(sample)
                if len(reservoir) >= n * 3:
                    break

        rng.shuffle(reservoir)
        selected = reservoir[:n]

        for idx, sample in enumerate(selected):
            img = sample["image"] if isinstance(sample["image"], Image.Image) else Image.open(BytesIO(sample["image"]))
            img = img.convert("RGB")
            _save_image(img, fake_dir, idx)

        count = _count_existing(fake_dir)
        print(f"  [stylegan] Saved {count} images.")
        return {"family": "stylegan", "count": count, "source": "wang-research/CNNDetection"}

    except Exception as e:
        print(f"  [stylegan] Primary source failed: {e}")
        print("  [stylegan] Trying fallback: OwensLab/CommunityForensics...")
        try:
            ds = load_dataset(
                "OwensLab/CommunityForensics",
                split="train",
                streaming=True,
            )
            rng = random.Random(seed)
            reservoir = []
            for sample in tqdm(ds, desc="stylegan-fallback", total=n * 5):
                gen_type = sample.get("generator", "") or sample.get("model", "")
                if "gan" in str(gen_type).lower() or "stylegan" in str(gen_type).lower():
                    reservoir.append(sample)
                    if len(reservoir) >= n * 3:
                        break

            rng.shuffle(reservoir)
            selected = reservoir[:n]

            for idx, sample in enumerate(selected):
                img = sample["image"] if isinstance(sample["image"], Image.Image) else Image.open(BytesIO(sample["image"]))
                img = img.convert("RGB")
                _save_image(img, fake_dir, idx)

            count = _count_existing(fake_dir)
            print(f"  [stylegan] Fallback saved {count} images.")
            return {"family": "stylegan", "count": count, "source": "OwensLab/CommunityForensics"}

        except Exception as e2:
            print(f"  [stylegan] Fallback also failed: {e2}")
            print("  [stylegan] MANUAL DOWNLOAD REQUIRED:")
            print("    1. Download ForenSynths test set from: https://github.com/peterwang512/CNNDetection")
            print("    2. Extract StyleGAN/StyleGAN2 fake images")
            print(f"    3. Place {n}+ images in: {fake_dir}")
            return {"family": "stylegan", "count": 0, "error": str(e2), "manual_required": True}


def download_sd3_flux(
    output_dir: Path,
    n: int = 300,
    seed: int = 42,
) -> dict:
    """Download modern diffusion images (SD3/Flux) from GenImage++."""
    from datasets import load_dataset

    family_dir = output_dir / "sd3_flux"
    fake_dir = family_dir / "FAKE"
    fake_dir.mkdir(parents=True, exist_ok=True)

    existing = _count_existing(fake_dir)
    if existing >= n:
        print(f"  [sd3_flux] Already have {existing} images, skipping.")
        return {"family": "sd3_flux", "count": existing, "skipped": True}

    print(f"  [sd3_flux] Downloading {n} images from Lunahera/genimagepp...")
    try:
        ds = load_dataset(
            "Lunahera/genimagepp",
            split="test",
            streaming=True,
        )
        rng = random.Random(seed)
        reservoir = []

        for sample in tqdm(ds, desc="sd3_flux", total=n * 5):
            label = sample.get("label", None)
            gen = str(sample.get("generator", "") or sample.get("source", "")).lower()
            # Look for stable diffusion 3, flux, or any modern diffusion model
            is_fake = (label == 1) or ("fake" in str(label).lower())
            is_target = ("sd3" in gen or "flux" in gen or "stable" in gen or
                        "diffusion" in gen or is_fake)
            if is_target:
                reservoir.append(sample)
                if len(reservoir) >= n * 3:
                    break

        rng.shuffle(reservoir)
        selected = reservoir[:n]

        for idx, sample in enumerate(selected):
            img = sample["image"] if isinstance(sample["image"], Image.Image) else Image.open(BytesIO(sample["image"]))
            img = img.convert("RGB")
            _save_image(img, fake_dir, idx)

        count = _count_existing(fake_dir)
        print(f"  [sd3_flux] Saved {count} images.")
        return {"family": "sd3_flux", "count": count, "source": "Lunahera/genimagepp"}

    except Exception as e:
        print(f"  [sd3_flux] Failed: {e}")
        print("  [sd3_flux] MANUAL DOWNLOAD REQUIRED:")
        print("    1. Visit: https://huggingface.co/datasets/Lunahera/genimagepp")
        print(f"    2. Download {n}+ test images and place in: {fake_dir}")
        return {"family": "sd3_flux", "count": 0, "error": str(e), "manual_required": True}


def download_midjourney_v6(
    output_dir: Path,
    n: int = 300,
    seed: int = 42,
) -> dict:
    """Download Midjourney v6 images from CortexLM/midjourney-v6."""
    from datasets import load_dataset

    family_dir = output_dir / "midjourney_v6"
    fake_dir = family_dir / "FAKE"
    fake_dir.mkdir(parents=True, exist_ok=True)

    existing = _count_existing(fake_dir)
    if existing >= n:
        print(f"  [midjourney_v6] Already have {existing} images, skipping.")
        return {"family": "midjourney_v6", "count": existing, "skipped": True}

    print(f"  [midjourney_v6] Downloading {n} images from CortexLM/midjourney-v6...")
    try:
        ds = load_dataset(
            "CortexLM/midjourney-v6",
            split="train",
            streaming=True,
        )
        rng = random.Random(seed)
        reservoir = []

        for sample in tqdm(ds, desc="midjourney_v6", total=n * 3):
            reservoir.append(sample)
            if len(reservoir) >= n * 3:
                break

        rng.shuffle(reservoir)
        selected = reservoir[:n]

        for idx, sample in enumerate(selected):
            img = sample["image"] if isinstance(sample["image"], Image.Image) else Image.open(BytesIO(sample["image"]))
            img = img.convert("RGB")
            _save_image(img, fake_dir, idx)

        count = _count_existing(fake_dir)
        print(f"  [midjourney_v6] Saved {count} images.")
        return {"family": "midjourney_v6", "count": count, "source": "CortexLM/midjourney-v6"}

    except Exception as e:
        print(f"  [midjourney_v6] Failed: {e}")
        print("  [midjourney_v6] Trying fallback: terminusresearch/midjourney-v6-520k-raw...")
        try:
            ds = load_dataset(
                "terminusresearch/midjourney-v6-520k-raw",
                split="train",
                streaming=True,
            )
            rng = random.Random(seed)
            reservoir = []
            for sample in tqdm(ds, desc="midjourney_v6-fallback", total=n * 3):
                reservoir.append(sample)
                if len(reservoir) >= n * 3:
                    break

            rng.shuffle(reservoir)
            selected = reservoir[:n]

            for idx, sample in enumerate(selected):
                img = sample["image"] if isinstance(sample["image"], Image.Image) else Image.open(BytesIO(sample["image"]))
                img = img.convert("RGB")
                _save_image(img, fake_dir, idx)

            count = _count_existing(fake_dir)
            print(f"  [midjourney_v6] Fallback saved {count} images.")
            return {"family": "midjourney_v6", "count": count, "source": "terminusresearch/midjourney-v6-520k-raw"}

        except Exception as e2:
            print(f"  [midjourney_v6] Fallback failed: {e2}")
            print("  [midjourney_v6] MANUAL DOWNLOAD REQUIRED:")
            print("    1. Visit: https://huggingface.co/datasets/CortexLM/midjourney-v6")
            print(f"    2. Download {n}+ images and place in: {fake_dir}")
            return {"family": "midjourney_v6", "count": 0, "error": str(e2), "manual_required": True}


def download_gpt4o(
    output_dir: Path,
    n: int = 300,
    seed: int = 42,
) -> dict:
    """Download GPT-4o generated images from Yejy53/GPT-ImgEval."""
    from datasets import load_dataset

    family_dir = output_dir / "gpt4o"
    fake_dir = family_dir / "FAKE"
    fake_dir.mkdir(parents=True, exist_ok=True)

    existing = _count_existing(fake_dir)
    if existing >= n:
        print(f"  [gpt4o] Already have {existing} images, skipping.")
        return {"family": "gpt4o", "count": existing, "skipped": True}

    print(f"  [gpt4o] Downloading {n} images from Yejy53/GPT-ImgEval...")
    try:
        ds = load_dataset(
            "Yejy53/GPT-ImgEval",
            split="train",
            streaming=True,
        )
        rng = random.Random(seed)
        reservoir = []

        for sample in tqdm(ds, desc="gpt4o", total=n * 3):
            reservoir.append(sample)
            if len(reservoir) >= n * 3:
                break

        rng.shuffle(reservoir)
        selected = reservoir[:n]

        for idx, sample in enumerate(selected):
            img = sample["image"] if isinstance(sample["image"], Image.Image) else Image.open(BytesIO(sample["image"]))
            img = img.convert("RGB")
            _save_image(img, fake_dir, idx)

        count = _count_existing(fake_dir)
        print(f"  [gpt4o] Saved {count} images.")
        return {"family": "gpt4o", "count": count, "source": "Yejy53/GPT-ImgEval"}

    except Exception as e:
        print(f"  [gpt4o] Primary failed: {e}")
        print("  [gpt4o] Trying fallback: Yejy53/Echo-4o-Image...")
        try:
            ds = load_dataset(
                "Yejy53/Echo-4o-Image",
                split="train",
                streaming=True,
            )
            rng = random.Random(seed)
            reservoir = []
            for sample in tqdm(ds, desc="gpt4o-fallback", total=n * 3):
                reservoir.append(sample)
                if len(reservoir) >= n * 3:
                    break

            rng.shuffle(reservoir)
            selected = reservoir[:n]

            for idx, sample in enumerate(selected):
                img = sample["image"] if isinstance(sample["image"], Image.Image) else Image.open(BytesIO(sample["image"]))
                img = img.convert("RGB")
                _save_image(img, fake_dir, idx)

            count = _count_existing(fake_dir)
            print(f"  [gpt4o] Fallback saved {count} images.")
            return {"family": "gpt4o", "count": count, "source": "Yejy53/Echo-4o-Image"}

        except Exception as e2:
            print(f"  [gpt4o] Fallback failed: {e2}")
            print("  [gpt4o] MANUAL DOWNLOAD REQUIRED:")
            print("    1. Generate images using GPT-4o (ChatGPT image generation)")
            print(f"    2. Save {n}+ images in: {fake_dir}")
            return {"family": "gpt4o", "count": 0, "error": str(e2), "manual_required": True}


# ---------------------------------------------------------------------------
# Manual GPT-4o folder setup
# ---------------------------------------------------------------------------

def setup_gpt4o_manual(output_dir: Path) -> None:
    """Create the gpt4o_manual folder with a README for manually collected images."""
    manual_dir = output_dir / "gpt4o_manual"
    manual_dir.mkdir(parents=True, exist_ok=True)
    fake_dir = manual_dir / "FAKE"
    fake_dir.mkdir(exist_ok=True)

    readme_path = manual_dir / "README.md"
    if not readme_path.exists():
        readme_path.write_text(
            "# GPT-4o Manual Image Collection\n\n"
            "Place manually collected GPT-4o generated images in the `FAKE/` folder.\n\n"
            "## Instructions\n\n"
            "1. Use ChatGPT (GPT-4o) to generate images with varied prompts\n"
            "2. Save the generated images as PNG or JPEG\n"
            "3. Aim for 200-500 images covering diverse subjects\n"
            "4. Name them sequentially: 0000.png, 0001.png, ...\n\n"
            "## Suggested prompt categories\n\n"
            "- Landscapes and nature scenes\n"
            "- Portraits and people\n"
            "- Animals\n"
            "- Architecture and buildings\n"
            "- Food and objects\n"
            "- Abstract and artistic\n\n"
            "These images supplement the `gpt4o/` folder (auto-downloaded from HuggingFace).\n"
        )


# ---------------------------------------------------------------------------
# Manifest and orchestration
# ---------------------------------------------------------------------------

def _write_manifest(output_dir: Path, results: list[dict]) -> None:
    """Write/update manifest.json with download metadata."""
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "seed": 42,
        "families": {r["family"]: r for r in results},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written to: {manifest_path}")


def download_all_families(
    output_dir: str = "data/generalisation",
    per_generator: int = 300,
    seed: int = 42,
    base_path: Optional[str] = None,
) -> dict:
    """Download images for all generator families.

    Args:
        output_dir: Relative path for output (from project root or base_path).
        per_generator: Number of images to download per family.
        seed: Random seed for reproducible sampling.
        base_path: Optional base path (e.g., Google Drive mount point).
                   If provided, output_dir is relative to this.

    Returns:
        Dict with per-family download results.
    """
    if base_path:
        out = Path(base_path) / output_dir
    else:
        out = Path(output_dir)

    out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out.resolve()}")
    print(f"Target: {per_generator} images per family, seed={seed}\n")

    results = []

    families = [
        ("StyleGAN", download_stylegan),
        ("SD3/Flux", download_sd3_flux),
        ("Midjourney v6", download_midjourney_v6),
        ("GPT-4o", download_gpt4o),
    ]

    for name, fn in families:
        print(f"--- {name} ---")
        result = fn(out, n=per_generator, seed=seed)
        results.append(result)
        print()

    setup_gpt4o_manual(out)
    _write_manifest(out, results)

    # Summary
    print("=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)
    for r in results:
        status = "OK" if r.get("count", 0) >= per_generator else "INCOMPLETE"
        if r.get("manual_required"):
            status = "MANUAL REQUIRED"
        elif r.get("skipped"):
            status = "SKIPPED (already present)"
        print(f"  {r['family']:15s} | {r.get('count', 0):4d} images | {status}")

    return {"output_dir": str(out), "results": results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download generalisation evaluation images from HuggingFace."
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/generalisation",
        help="Output directory for downloaded images (default: data/generalisation)",
    )
    parser.add_argument(
        "--per-generator", type=int, default=300,
        help="Number of images per generator family (default: 300)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--base-path", type=str, default=None,
        help="Base path (e.g., /content/drive/MyDrive) for Colab persistence",
    )
    args = parser.parse_args()
    download_all_families(
        output_dir=args.output_dir,
        per_generator=args.per_generator,
        seed=args.seed,
        base_path=args.base_path,
    )


if __name__ == "__main__":
    main()
