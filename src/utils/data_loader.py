import os
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
