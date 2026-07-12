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


def _sample_to_pil(sample: dict) -> Image.Image:
    """Extract a PIL RGB image from various HuggingFace row formats."""
    if sample.get("image_data") is not None:
        raw = sample["image_data"]
        if isinstance(raw, bytes):
            return Image.open(BytesIO(raw)).convert("RGB")

    for key in ("Image", "image", "img", "photo"):
        if sample.get(key) is None:
            continue
        val = sample[key]
        if isinstance(val, Image.Image):
            return val.convert("RGB")
        if isinstance(val, bytes):
            return Image.open(BytesIO(val)).convert("RGB")
        if isinstance(val, dict) and val.get("bytes"):
            return Image.open(BytesIO(val["bytes"])).convert("RGB")

    raise KeyError(f"No image field found in sample keys: {list(sample.keys())}")


def _stream_reservoir(
    stream,
    n: int,
    seed: int,
    filter_fn,
    desc: str,
    max_scan: int | None = None,
) -> list[dict]:
    """Collect up to n*3 matching samples from a streaming dataset, then subsample n."""
    rng = random.Random(seed)
    reservoir = []
    for i, sample in enumerate(tqdm(stream, desc=desc, total=n * 5)):
        if filter_fn(sample):
            reservoir.append(sample)
            if len(reservoir) >= n * 3:
                break
        if max_scan is not None and i >= max_scan:
            break
    rng.shuffle(reservoir)
    return reservoir[:n]


def _save_samples(samples: list[dict], fake_dir: Path) -> int:
    """Save a list of HF samples as numbered PNGs."""
    for idx, sample in enumerate(samples):
        img = _sample_to_pil(sample)
        _save_image(img, fake_dir, idx)
    return _count_existing(fake_dir)


def download_stylegan(
    output_dir: Path,
    n: int = 300,
    seed: int = 42,
) -> dict:
    """Download StyleGAN/GAN images from CommunityForensics-Eval."""
    from datasets import load_dataset

    family_dir = output_dir / "stylegan"
    fake_dir = family_dir / "FAKE"
    fake_dir.mkdir(parents=True, exist_ok=True)

    existing = _count_existing(fake_dir)
    if existing >= n:
        print(f"  [stylegan] Already have {existing} images, skipping.")
        return {"family": "stylegan", "count": existing, "skipped": True}

    print(f"  [stylegan] Downloading {n} GAN images from OwensLab/CommunityForensics-Eval...")
    try:
        ds = load_dataset(
            "OwensLab/CommunityForensics-Eval",
            split="CompEval",
            streaming=True,
        )

        def is_gan_fake(sample):
            arch = str(sample.get("architecture", "")).upper()
            label = sample.get("label")
            return arch == "GAN" and str(label) == "1"

        selected = _stream_reservoir(ds, n, seed, is_gan_fake, "stylegan")
        count = _save_samples(selected, fake_dir)
        print(f"  [stylegan] Saved {count} images.")
        return {
            "family": "stylegan",
            "count": count,
            "source": "OwensLab/CommunityForensics-Eval (architecture=GAN)",
        }

    except Exception as e:
        print(f"  [stylegan] Failed: {e}")
        print("  [stylegan] MANUAL DOWNLOAD REQUIRED:")
        print("    1. Download ForenSynths test set from: https://github.com/peterwang512/CNNDetection")
        print("    2. Extract StyleGAN/StyleGAN2 fake images")
        print(f"    3. Place {n}+ images in: {fake_dir}")
        return {"family": "stylegan", "count": 0, "error": str(e), "manual_required": True}


def download_sd3_flux(
    output_dir: Path,
    n: int = 300,
    seed: int = 42,
) -> dict:
    """Download Stable Diffusion 3 images from Defactify_Image_Dataset."""
    from datasets import load_dataset

    family_dir = output_dir / "sd3_flux"
    fake_dir = family_dir / "FAKE"
    fake_dir.mkdir(parents=True, exist_ok=True)

    existing = _count_existing(fake_dir)
    if existing >= n:
        print(f"  [sd3_flux] Already have {existing} images, skipping.")
        return {"family": "sd3_flux", "count": existing, "skipped": True}

    print(f"  [sd3_flux] Downloading {n} SD3 images from Rajarshi-Roy-research/Defactify_Image_Dataset...")
    try:
        ds = load_dataset(
            "Rajarshi-Roy-research/Defactify_Image_Dataset",
            split="test",
            streaming=True,
        )

        def is_sd3_fake(sample):
            # Label_B: 3 = SD3, Label_A: 1 = AI-generated
            return sample.get("Label_B") == 3 and sample.get("Label_A") == 1

        selected = _stream_reservoir(ds, n, seed, is_sd3_fake, "sd3_flux", max_scan=50000)
        count = _save_samples(selected, fake_dir)
        print(f"  [sd3_flux] Saved {count} images.")
        return {
            "family": "sd3_flux",
            "count": count,
            "source": "Rajarshi-Roy-research/Defactify_Image_Dataset (Label_B=SD3)",
        }

    except Exception as e:
        print(f"  [sd3_flux] Failed: {e}")
        print("  [sd3_flux] Trying fallback: OwensLab/CommunityForensics-Eval (LatDiff)...")
        try:
            ds = load_dataset(
                "OwensLab/CommunityForensics-Eval",
                split="CompEval",
                streaming=True,
            )

            def is_modern_diffusion(sample):
                model = str(sample.get("model_name", "")).lower()
                arch = str(sample.get("architecture", "")).upper()
                label = str(sample.get("label", ""))
                is_fake = label == "1"
                is_modern = (
                    "flux" in model or "sd3" in model or "stable-diffusion-3" in model
                    or (arch == "LATDIFF" and "stable" in model)
                )
                return is_fake and is_modern

            selected = _stream_reservoir(ds, n, seed, is_modern_diffusion, "sd3_flux-fallback")
            count = _save_samples(selected, fake_dir)
            print(f"  [sd3_flux] Fallback saved {count} images.")
            return {
                "family": "sd3_flux",
                "count": count,
                "source": "OwensLab/CommunityForensics-Eval (modern LatDiff)",
            }

        except Exception as e2:
            print(f"  [sd3_flux] Fallback failed: {e2}")
            print("  [sd3_flux] MANUAL DOWNLOAD REQUIRED:")
            print("    1. Visit: https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset")
            print(f"    2. Download {n}+ SD3 images and place in: {fake_dir}")
            return {"family": "sd3_flux", "count": 0, "error": str(e2), "manual_required": True}


def download_midjourney_v6(
    output_dir: Path,
    n: int = 300,
    seed: int = 42,
) -> dict:
    """Download Midjourney v6 images from Defactify or ehristoforu/midjourney-images."""
    from datasets import load_dataset

    family_dir = output_dir / "midjourney_v6"
    fake_dir = family_dir / "FAKE"
    fake_dir.mkdir(parents=True, exist_ok=True)

    existing = _count_existing(fake_dir)
    if existing >= n:
        print(f"  [midjourney_v6] Already have {existing} images, skipping.")
        return {"family": "midjourney_v6", "count": existing, "skipped": True}

    print(f"  [midjourney_v6] Downloading {n} images from Rajarshi-Roy-research/Defactify_Image_Dataset...")
    try:
        ds = load_dataset(
            "Rajarshi-Roy-research/Defactify_Image_Dataset",
            split="test",
            streaming=True,
        )

        def is_mj_fake(sample):
            # Label_B: 5 = Midjourney v6
            return sample.get("Label_B") == 5 and sample.get("Label_A") == 1

        selected = _stream_reservoir(ds, n, seed, is_mj_fake, "midjourney_v6", max_scan=50000)
        count = _save_samples(selected, fake_dir)
        print(f"  [midjourney_v6] Saved {count} images.")
        return {
            "family": "midjourney_v6",
            "count": count,
            "source": "Rajarshi-Roy-research/Defactify_Image_Dataset (Label_B=Midjourney)",
        }

    except Exception as e:
        print(f"  [midjourney_v6] Primary failed: {e}")
        print("  [midjourney_v6] Trying fallback: ehristoforu/midjourney-images...")
        try:
            ds = load_dataset(
                "ehristoforu/midjourney-images",
                split="train",
                streaming=True,
            )
            selected = _stream_reservoir(ds, n, seed, lambda _: True, "midjourney_v6-fallback")
            count = _save_samples(selected, fake_dir)
            print(f"  [midjourney_v6] Fallback saved {count} images.")
            return {
                "family": "midjourney_v6",
                "count": count,
                "source": "ehristoforu/midjourney-images",
            }

        except Exception as e2:
            print(f"  [midjourney_v6] Fallback failed: {e2}")
            print("  [midjourney_v6] MANUAL DOWNLOAD REQUIRED:")
            print("    1. Visit: https://huggingface.co/datasets/ehristoforu/midjourney-images")
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
        selected = _stream_reservoir(ds, n, seed, lambda _: True, "gpt4o")
        count = _save_samples(selected, fake_dir)
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
            selected = _stream_reservoir(ds, n, seed, lambda _: True, "gpt4o-fallback")
            count = _save_samples(selected, fake_dir)
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
