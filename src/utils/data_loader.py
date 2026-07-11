import os
from pathlib import Path

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(split: str) -> transforms.Compose:
    """Return image transforms for the given split ('train', 'val', or 'test')."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=8),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def get_cifake_loaders(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 2,
    seed: int = 42,
):
    """Load CIFAKE dataset and return (train_loader, val_loader, test_loader).

    Expects data_dir to contain 'train/' and 'test/' subdirectories,
    each with 'REAL/' and 'FAKE/' class folders.
    """
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=get_transforms("train"),
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "test"),
        transform=get_transforms("test"),
    )

    # Split training set into train / val
    n_val = int(len(train_dataset) * val_split)
    n_train = len(train_dataset) - n_val
    import torch
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        train_dataset, [n_train, n_val], generator=generator,
    )

    # Val subset should use eval transforms — wrap with an override
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=get_transforms("val"),
    )
    val_subset = torch.utils.data.Subset(val_dataset, val_subset.indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def discover_generator_families(root_dir: str) -> list[str]:
    """Return list of available generator family directory paths.

    Scans root_dir for subdirectories containing a FAKE/ folder.
    """
    root = Path(root_dir)
    families = []
    if not root.exists():
        return families
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and (entry / "FAKE").is_dir():
            families.append(str(entry))
    return families


def get_generalisation_loader(
    family_dir: str,
    real_reference_dir: str | None = None,
    batch_size: int = 32,
    num_workers: int = 2,
    matched_real: bool = False,
    seed: int = 42,
) -> DataLoader:
    """Load a generalisation test set for a single generator family.

    Labels follow the CIFAKE convention: 0 = FAKE, 1 = REAL.

    Args:
        family_dir: Path to the family folder (e.g., data/generalisation/stylegan/).
                    Must contain a FAKE/ subfolder. May contain a REAL/ subfolder.
        real_reference_dir: Path to a folder of real images (e.g., CIFAKE test/REAL)
                           used when the family has no matched REAL set.
        batch_size: Batch size for the DataLoader.
        num_workers: Number of data loading workers.
        matched_real: If True and family_dir has a REAL/ folder, use it.
                      Otherwise use real_reference_dir.
        seed: Seed for subsampling the real reference set.

    Returns:
        DataLoader yielding (images, labels) with 0=FAKE, 1=REAL.
    """
    family_path = Path(family_dir)
    fake_dir = family_path / "FAKE"
    real_dir = family_path / "REAL"

    if not fake_dir.exists():
        raise FileNotFoundError(f"FAKE directory not found: {fake_dir}")

    eval_transform = get_transforms("test")

    # Determine which real images to use
    use_matched_real = matched_real and real_dir.exists() and any(real_dir.iterdir())
    has_real = use_matched_real or (real_reference_dir is not None)

    if has_real:
        if use_matched_real:
            # Build a combined dataset with both FAKE/ and REAL/ from the family
            combined_dataset = datasets.ImageFolder(
                str(family_path),
                transform=eval_transform,
            )
            # ImageFolder assigns labels alphabetically: FAKE=0, REAL=1
            loader = DataLoader(
                combined_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
        else:
            # Pair fake images with a balanced subsample of the real reference
            import torch

            fake_dataset = datasets.ImageFolder(
                str(fake_dir.parent),
                transform=eval_transform,
            )
            # ImageFolder on parent will pick up FAKE as class 0 (only class if no REAL)
            # Instead, load fake images directly and assign label 0
            fake_imgs = _FolderDataset(fake_dir, transform=eval_transform, label=0)

            real_imgs = _FolderDataset(
                Path(real_reference_dir), transform=eval_transform, label=1
            )
            # Subsample real to match fake count
            n_fake = len(fake_imgs)
            if len(real_imgs) > n_fake:
                generator = torch.Generator().manual_seed(seed)
                indices = torch.randperm(len(real_imgs), generator=generator)[:n_fake].tolist()
                real_imgs = torch.utils.data.Subset(real_imgs, indices)

            combined = torch.utils.data.ConcatDataset([fake_imgs, real_imgs])
            loader = DataLoader(
                combined,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
    else:
        # Fake-only evaluation
        fake_imgs = _FolderDataset(fake_dir, transform=eval_transform, label=0)
        loader = DataLoader(
            fake_imgs,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return loader


class _FolderDataset:
    """Simple dataset loading all images from a folder with a fixed label."""

    EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

    def __init__(self, folder: Path, transform, label: int):
        self.folder = folder
        self.transform = transform
        self.label = label
        self.paths = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in self.EXTENSIONS
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.label


def get_dataset_stats(loader: DataLoader) -> dict:
    """Return basic statistics about the dataset behind a DataLoader."""
    dataset = loader.dataset
    # Unwrap Subset if needed
    if hasattr(dataset, "dataset"):
        base_dataset = dataset.dataset
    else:
        base_dataset = dataset

    class_to_idx = getattr(base_dataset, "class_to_idx", {})

    images, _ = next(iter(loader))
    return {
        "class_to_idx": class_to_idx,
        "total_samples": len(dataset),
        "image_shape": tuple(images.shape[1:]),
    }
