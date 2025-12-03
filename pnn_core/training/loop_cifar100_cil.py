import json
from typing import Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from pnn_core.data.cifar100_cil import get_cifar100_cil_dataloaders
from pnn_core.models.resnet_cifar import ResNet18CIFAR
from pnn_core.pnn.stabilizer import PNNStabilizer
from pnn_core.pnn.schedule import linear_warmup
from pnn_core.training.eval_metrics import compute_metrics
from pnn_core.utils.logging import ExperimentLogger
from pnn_core.pnn.utils import set_seed


def train_one_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    device,
    stabilizer: Optional[PNNStabilizer],
    mu_min: float,
    mu_max: float,
    warmup_steps: int,
    epoch: int,
    max_epochs: int,
):
    model.train()
    criterion = nn.CrossEntropyLoss()
    num_steps = len(train_loader)
    global_step = epoch * num_steps

    running_loss = 0.0
    running_reg = 0.0

    for step, (x, y) in enumerate(tqdm(train_loader, leave=False)):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        task_loss = criterion(logits, y)

        # PNN regularization
        reg_loss = 0.0
        if stabilizer is not None:
            current_step = global_step + step
            mu = linear_warmup(
                step=current_step,
                warmup_steps=warmup_steps,
                mu_min=mu_min,
                mu_max=mu_max,
            )
            reg = stabilizer.reg_loss(mu=mu)
            reg_loss = reg
            loss = task_loss + reg
        else:
            loss = task_loss

        loss.backward()

        if stabilizer is not None:
            # Use gradients to accumulate importance
            stabilizer.accumulate_importance()

        optimizer.step()

        running_loss += task_loss.item()
        if stabilizer is not None:
            running_reg += float(reg_loss)

    avg_loss = running_loss / max(1, num_steps)
    avg_reg = running_reg / max(1, num_steps) if stabilizer is not None else 0.0
    return avg_loss, avg_reg


@torch.no_grad()
def evaluate(model: nn.Module, data_loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(1, total)


def train_cifar100_cil_experiment(
    data_root: str,
    run_dir: str,
    method: str,
    seed: int = 0,
    batch_size: int = 128,
    num_workers: int = 2,
    lr: float = 0.1,
    weight_decay: float = 5e-4,
    momentum: float = 0.9,
    epochs_per_task: int = 50,
    beta: float = 0.99,
    eps: float = 1e-8,
    kappa: float = 1.0,
    gamma: float = 1.0,
    lambda_max: float = 100.0,
    mu_min: float = 0.1,
    mu_max: float = 1.0,
    warmup_epochs: int = 2,
):
    """
    Main training loop for CIFAR-100 CIL with ResNet-18 and PNN variants.
    method: "naive", "pnn_param", "pnn_layer"
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    logger = ExperimentLogger(run_dir=run_dir)

    # Prepare data: list of (train_loader, test_loader) for each task
    task_loaders = get_cifar100_cil_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    num_tasks = len(task_loaders)

    # Build model
    model = ResNet18CIFAR(num_classes=100).to(device)

    # Choose stabilizer
    stabilizer: Optional[PNNStabilizer] = None
    grouping = None
    if method == "pnn_param":
        grouping = "param"
    elif method == "pnn_layer":
        grouping = "layer"

    if grouping is not None:
        stabilizer = PNNStabilizer(
            model=model,
            beta=beta,
            eps=eps,
            kappa=kappa,
            gamma=gamma,
            lambda_max=lambda_max,
            grouping=grouping,
            device=device,
            excluded_params=None,
        )

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    # Simple step LR schedule or you can switch to cosine
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(epochs_per_task * 0.5), int(epochs_per_task * 0.75)],
        gamma=0.1,
    )

    # Prepare tracking accuracies
    task_accuracies: List[List[float]] = []

    for task_id in range(num_tasks):
        train_loader, _ = task_loaders[task_id]
        logger.log_text(f"==== Training Task {task_id} ({task_id*10}-{task_id*10+9}) ====")

        # Start of task: reset importance etc.
        if stabilizer is not None:
            stabilizer.begin_task(reset_importance=True)

        for epoch in range(epochs_per_task):
            warmup_steps = warmup_epochs * len(train_loader)

            avg_loss, avg_reg = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                device=device,
                stabilizer=stabilizer,
                mu_min=mu_min,
                mu_max=mu_max,
                warmup_steps=warmup_steps,
                epoch=epoch,
                max_epochs=epochs_per_task,
            )
            lr_scheduler.step()

            # Log reg_loss
            if stabilizer is not None:
                logger.log_reg_loss(
                    task=task_id,
                    epoch=epoch,
                    reg_loss=avg_reg,
                    method=method,
                    seed=seed,
                )

            logger.log_text(
                f"Task {task_id} | Epoch {epoch+1}/{epochs_per_task} "
                f"| Loss={avg_loss:.4f} | Reg={avg_reg:.4f}"
            )

        # End of task: consolidate
        if stabilizer is not None:
            stabilizer.end_task()

        # Evaluate on all tasks seen so far
        acc_row: List[float] = []
        for eval_task in range(task_id + 1):
            _, test_loader = task_loaders[eval_task]
            acc = evaluate(model, test_loader, device)
            acc_row.append(acc)

            logger.log_metric(
                task_train=task_id,
                task_eval=eval_task,
                accuracy=acc,
                epoch=epochs_per_task,
                method=method,
                seed=seed,
            )

        task_accuracies.append(acc_row)

        logger.log_text(
            f"After Task {task_id}, accuracies on seen tasks: {acc_row}"
        )

    # Pad accuracies to full 10 columns with zeros for convenience
    max_len = num_tasks
    padded_task_accuracies: List[List[float]] = []
    for row in task_accuracies:
        row_extended = row + [0.0] * (max_len - len(row))
        padded_task_accuracies.append(row_extended)

    metrics = compute_metrics(padded_task_accuracies)

    # Save metrics.json
    metrics_out = {
        "task_accuracies": padded_task_accuracies,
        "metrics": metrics,
        "method": method,
        "seed": seed,
    }
    metrics_path = f"{run_dir}/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)

    logger.log_text(f"Final metrics: {metrics}")
    return padded_task_accuracies, metrics
