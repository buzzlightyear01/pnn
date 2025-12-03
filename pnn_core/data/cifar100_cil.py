import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Tuple


def get_cifar100_cil_dataloaders(
    data_root: str,
    batch_size: int = 128,
    num_workers: int = 2,
    seed: int = 0,
) -> List[Tuple[DataLoader, DataLoader]]:
    """
    Return list of (train_loader, test_loader) for 10 tasks.
    Task t contains classes [10*t .. 10*t+9].
    """
    torch.manual_seed(seed)

    # Standard CIFAR-100 transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761),
        ),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761),
        ),
    ])

    train_dataset = datasets.CIFAR100(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = datasets.CIFAR100(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform,
    )

    # Map: class_id -> list of indices
    train_class_to_idx = {c: [] for c in range(100)}
    for idx, (_, label) in enumerate(train_dataset):
        train_class_to_idx[label].append(idx)

    test_class_to_idx = {c: [] for c in range(100)}
    for idx, (_, label) in enumerate(test_dataset):
        test_class_to_idx[label].append(idx)

    task_loaders: List[Tuple[DataLoader, DataLoader]] = []

    for t in range(10):
        class_start = 10 * t
        class_end = class_start + 10
        task_classes = list(range(class_start, class_end))

        # collect indices for these classes
        train_indices = []
        test_indices = []
        for c in task_classes:
            train_indices.extend(train_class_to_idx[c])
            test_indices.extend(test_class_to_idx[c])

        train_subset = Subset(train_dataset, train_indices)
        test_subset = Subset(test_dataset, test_indices)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        task_loaders.append((train_loader, test_loader))

    return task_loaders
