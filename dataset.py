import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import config


def get_transforms():
    """Returns standard data transformations."""
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.Lambda(lambda x: x.view(-1)),  # Flatten 48x48 -> 2304
        ]
    )


def get_dataloaders(batch_size):
    """Loads raw data, applies splits, and returns DataLoader instances.

    batch_size is passed explicitly since it now varies per experiment config.
    """
    transform = get_transforms()

    full_train_dataset = datasets.ImageFolder(config.TRAIN_PATH, transform=transform)
    test_dataset = datasets.ImageFolder(config.TEST_PATH, transform=transform)

    # Train / Validation Split
    val_size = int(len(full_train_dataset) * config.VAL_SPLIT)
    train_size = len(full_train_dataset) - val_size

    generator = torch.Generator().manual_seed(config.SEED)
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, full_train_dataset.classes
