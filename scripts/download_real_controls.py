"""Download high-resolution real photographs for the resolution-control experiment.

Streams real images from HuggingFace datasets into:
  data/generalisation/_controls/{coco_real,defactify_real,communityforensics_real}/REAL/

Usage:
    python scripts/download_real_controls.py --output-dir data/generalisation/_controls --n 300

From a notebook:
    from scripts.download_real_controls import download_all_controls
    download_all_controls(output_dir="data/generalisation/_controls", n=300)
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm


def _save_image(img: Image.Image, path: Path, idx: int, fmt: str = "png") -> str:
    filename = f"{idx:04d}.{fmt}"
    img.save(path / filename)
    return filename


def _count_existing(folder: Path) -> int:
    if not folder.exists():
        return 0
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in exts)


def _sample_to_pil(sample: dict) -> Image.Image:
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


def _stream_reservoir(stream, n: int, seed: int, filter_fn, desc: str, max_scan: int | None = None):
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


def _save_samples(samples: list[dict], out_dir: Path) -> int:
    for idx, sample in enumerate(samples):
        img = _sample_to_pil(sample)
        _save_image(img, out_dir, idx)
    return _count_existing(out_dir)


def download_coco_real(output_dir: Path, n: int = 300, seed: int = 42) -> dict:
    """Download COCO 2017 validation photographs (high-res, no AI content)."""
    from datasets import load_dataset

    family_dir = output_dir / "coco_real"
    real_dir = family_dir / "REAL"
    real_dir.mkdir(parents=True, exist_ok=True)

    existing = _count_existing(real_dir)
    if existing >= n:
        print(f"  [coco_real] Already have {existing} images, skipping.")
        return {"family": "coco_real", "count": existing, "skipped": True}

    print(f"  [coco_real] Downloading {n} images from detection-datasets/coco...")
    try:
        # Prefer detection-datasets/coco (has image field); fall back to common alternatives
        try:
            ds = load_dataset("detection-datasets/coco", split="val", streaming=True)
        except Exception:
            ds = load_dataset(
                "HuggingFaceM4/COCO",
                name="2017_captioning",
                split="validation",
                streaming=True,
            )

        selected = _stream_reservoir(ds, n, seed, lambda _: True, "coco_real", max_scan=10000)
        count = _save_samples(selected, real_dir)
        print(f"  [coco_real] Saved {count} images.")
        return {
            "family": "coco_real",
            "count": count,
            "source": "COCO 2017 val (photographic)",
        }
    except Exception as e:
        print(f"  [coco_real] Failed: {e}")
        print("  [coco_real] MANUAL DOWNLOAD REQUIRED:")
        print("    Place 300+ high-resolution real photographs in:")
        print(f"    {real_dir}")
        return {"family": "coco_real", "count": 0, "error": str(e), "manual_required": True}


def download_defactify_real(output_dir: Path, n: int = 300, seed: int = 42) -> dict:
    """Download real images from Defactify (matched source to SD3/Midjourney fakes)."""
    from datasets import load_dataset

    family_dir = output_dir / "defactify_real"
    real_dir = family_dir / "REAL"
    real_dir.mkdir(parents=True, exist_ok=True)

    existing = _count_existing(real_dir)
    if existing >= n:
        print(f"  [defactify_real] Already have {existing} images, skipping.")
        return {"family": "defactify_real", "count": existing, "skipped": True}

    print(f"  [defactify_real] Downloading {n} real images from Defactify_Image_Dataset...")
    try:
        ds = load_dataset(
            "Rajarshi-Roy-research/Defactify_Image_Dataset",
            split="test",
            streaming=True,
        )

        def is_real(sample):
            # Label_A: 0 = real
            return sample.get("Label_A") == 0

        selected = _stream_reservoir(ds, n, seed, is_real, "defactify_real", max_scan=50000)
        count = _save_samples(selected, real_dir)
        print(f"  [defactify_real] Saved {count} images.")
        return {
            "family": "defactify_real",
            "count": count,
            "source": "Rajarshi-Roy-research/Defactify_Image_Dataset (Label_A=0)",
        }
    except Exception as e:
        print(f"  [defactify_real] Failed: {e}")
        return {"family": "defactify_real", "count": 0, "error": str(e), "manual_required": True}


def download_communityforensics_real(output_dir: Path, n: int = 300, seed: int = 42) -> dict:
    """Download real images from CommunityForensics-Eval (matched source to StyleGAN)."""
    from datasets import load_dataset

    family_dir = output_dir / "communityforensics_real"
    real_dir = family_dir / "REAL"
    real_dir.mkdir(parents=True, exist_ok=True)

    existing = _count_existing(real_dir)
    if existing >= n:
        print(f"  [communityforensics_real] Already have {existing} images, skipping.")
        return {"family": "communityforensics_real", "count": existing, "skipped": True}

    print(f"  [communityforensics_real] Downloading {n} real images...")
    try:
        ds = load_dataset(
            "OwensLab/CommunityForensics-Eval",
            split="CompEval",
            streaming=True,
        )

        def is_real(sample):
            return str(sample.get("label")) == "0"

        selected = _stream_reservoir(
            ds, n, seed, is_real, "communityforensics_real", max_scan=50000
        )
        count = _save_samples(selected, real_dir)
        print(f"  [communityforensics_real] Saved {count} images.")
        return {
            "family": "communityforensics_real",
            "count": count,
            "source": "OwensLab/CommunityForensics-Eval (label=0)",
        }
    except Exception as e:
        print(f"  [communityforensics_real] Failed: {e}")
        return {
            "family": "communityforensics_real",
            "count": 0,
            "error": str(e),
            "manual_required": True,
        }


def download_all_controls(
    output_dir: str = "data/generalisation/_controls",
    n: int = 300,
    seed: int = 42,
    base_path: Optional[str] = None,
) -> dict:
    """Download all real-image control sets."""
    if base_path:
        out = Path(base_path) / output_dir
    else:
        out = Path(output_dir)

    out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out.resolve()}")
    print(f"Target: {n} images per control set, seed={seed}\n")

    results = []
    for name, fn in [
        ("COCO 2017 Real", download_coco_real),
        ("Defactify Real", download_defactify_real),
        ("CommunityForensics Real", download_communityforensics_real),
    ]:
        print(f"--- {name} ---")
        results.append(fn(out, n=n, seed=seed))
        print()

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n": n,
        "families": {r["family"]: r for r in results},
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest written to: {manifest_path}")

    print("=" * 50)
    print("CONTROL DOWNLOAD SUMMARY")
    print("=" * 50)
    for r in results:
        count = r.get("count", 0)
        if r.get("manual_required"):
            status = "MANUAL REQUIRED"
        elif r.get("skipped"):
            status = "SKIPPED"
        elif count >= n:
            status = "OK"
        else:
            status = "INCOMPLETE"
        print(f"  {r['family']:30s} | {count:4d} | {status}")

    return {"output_dir": str(out), "results": results}


def main():
    parser = argparse.ArgumentParser(
        description="Download high-resolution real photographs for resolution control."
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/generalisation/_controls",
    )
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-path", type=str, default=None)
    args = parser.parse_args()
    download_all_controls(
        output_dir=args.output_dir,
        n=args.n,
        seed=args.seed,
        base_path=args.base_path,
    )


if __name__ == "__main__":
    main()
