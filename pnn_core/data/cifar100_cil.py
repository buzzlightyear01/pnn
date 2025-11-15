"""
CIFAR-100 برای سناریوی class-incremental:
- 100 کلاس → 10 تسک × 10 کلاس
- Task 0: classes 0–9
- Task 1: classes 10–19
- ...
- Task 9: classes 90–99

ایده:
- یک head مشترک با 100 کلاس داریم.
- برای هر تسک t:
  - train_loader_t فقط شامل کلاس‌های آن تسک است.
  - test_loader_t فقط شامل کلاس‌های همان تسک است.
- در حلقه‌ی آموزش:
  - بعد از اتمام تسک t، روی test_loader های 0..t ارزیابی می‌کنیم.
  - این به ما ماتریس دقت [num_tasks, num_tasks] را می‌دهد:
    acc[t_train, t_eval]
"""

from typing import List, Tuple

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


NUM_CLASSES = 100
NUM_TASKS = 10
CLASSES_PER_TASK = NUM_CLASSES // NUM_TASKS


def get_task_class_splits(
    num_classes: int = NUM_CLASSES,
    num_tasks: int = NUM_TASKS
) -> List[List[int]]:
    """
    کلاس‌ها را به num_tasks بلاک مساوی تقسیم می‌کند.
    مثال برای CIFAR-100:
      [[0..9], [10..19], ..., [90..99]]
    """
    assert num_classes % num_tasks == 0, "num_classes باید مضربی از num_tasks باشد."
    classes_per_task = num_classes // num_tasks
    task_classes: List[List[int]] = []
    for t in range(num_tasks):
        start = t * classes_per_task
        end = (t + 1) * classes_per_task
        task_classes.append(list(range(start, end)))
    return task_classes


def _get_targets_as_numpy(dataset) -> np.ndarray:
    """
    targets را به صورت numpy array برمی‌گرداند.
    در نسخه‌های مختلف torchvision ممکن است dataset.targets یا dataset.labels باشد.
    """
    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif hasattr(dataset, "labels"):
        targets = dataset.labels
    else:
        raise AttributeError("Dataset has neither 'targets' nor 'labels' attribute.")
    return np.array(targets)


def _indices_for_classes(targets: np.ndarray, class_list: List[int]) -> np.ndarray:
    """
    برای یک لیست از کلاس‌ها، ایندکس نمونه‌هایی را که در آن کلاس‌ها هستند برمی‌گرداند.
    """
    mask = np.isin(targets, np.array(class_list))
    return np.where(mask)[0]


def get_cifar100_cil_dataloaders(
    data_root: str,
    batch_size: int,
    num_workers: int = 2,
    shuffle_train: bool = True,
    pin_memory: bool = True,
) -> Tuple[List[DataLoader], List[DataLoader], List[List[int]]]:
    """
    دیتالودرهای CIFAR-100 برای سناریوی class-incremental.

    خروجی:
      train_loaders:  لیست طول NUM_TASKS، هرکدام DataLoader برای train روی تسک t
      test_loaders:   لیست طول NUM_TASKS، هرکدام DataLoader برای test روی تسک t
      task_classes:   لیست طول NUM_TASKS، شامل کلاس‌های هر تسک (برای لاگ و sanity check)

    توجه:
      - head مدل 100 کلاس دارد. ما labels را نمی‌چرخانیم/ری‌مپ نمی‌کنیم.
      - برای ارزیابی بعد از تسک t، کافی است روی test_loaders[0..t] ارزیابی کنیم.
    """

    # transform ها مطابق توصیه‌ی سوپروایزر
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR100(
        root=data_root,
        train=True,
        download=True,
        transform=transform_train,
    )
    test_dataset = datasets.CIFAR100(
        root=data_root,
        train=False,
        download=True,
        transform=transform_test,
    )

    task_classes = get_task_class_splits(
        num_classes=NUM_CLASSES,
        num_tasks=NUM_TASKS,
    )

    train_targets = _get_targets_as_numpy(train_dataset)
    test_targets = _get_targets_as_numpy(test_dataset)

    train_loaders: List[DataLoader] = []
    test_loaders: List[DataLoader] = []

    for t, class_list in enumerate(task_classes):
        # ایندکس نمونه‌های train و test مربوط به این تسک
        train_indices = _indices_for_classes(train_targets, class_list)
        test_indices = _indices_for_classes(test_targets, class_list)

        train_subset = Subset(train_dataset, train_indices.tolist())
        test_subset = Subset(test_dataset, test_indices.tolist())

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return train_loaders, test_loaders, task_classes
