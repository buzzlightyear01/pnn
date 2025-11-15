from typing import Dict, Any, List
import os
import json

import torch
import torch.nn as nn

CURRENT_DIR = os.path.dirname(__file__)
PNN_CORE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

import sys
if PNN_CORE_ROOT not in sys.path:
    sys.path.insert(0, PNN_CORE_ROOT)

from data.cifar100_cil import get_cifar100_cil_dataloaders
from models.resnet18 import build_resnet18
from utils.logging import ExperimentLogger
from utils.seed import set_seed
from training.eval_metrics import compute_metrics


def _evaluate(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    if total == 0:
        return 0.0
    return 100.0 * correct / total


def train_cifar100_cil(config: Dict[str, Any], logger: ExperimentLogger) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method = config["method"]
    seed = config.get("seed", 0)

    set_seed(seed)
    logger.log_text(f"Using device: {device}")
    logger.log_text(f"Method: {method}, Seed: {seed}")

    train_loaders, test_loaders, task_classes = get_cifar100_cil_dataloaders(
        data_root=config["data_root"],
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 2),
    )

    num_tasks = len(train_loaders)
    logger.log_text(f"Number of tasks: {num_tasks}")
    logger.log_text(f"Task class splits: {task_classes}")

    model = build_resnet18(num_classes=100, pretrained=config.get("pretrained", False))
    model = model.to(device)

    opt_cfg = config["optimizer"]
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt_cfg["lr"],
        momentum=opt_cfg.get("momentum", 0.9),
        weight_decay=opt_cfg.get("weight_decay", 5e-4),
    )
    criterion = nn.CrossEntropyLoss()

    stabilizer = None
    linear_warmup = None
    use_pnn = method.startswith("pnn")
    if use_pnn:
        from pnn_core.pnn.stabilizer import PNNStabilizer
        from pnn_core.pnn.schedule import linear_warmup as _linear_warmup

        linear_warmup = _linear_warmup
        pnn_cfg = config["pnn"]
        excluded_params = set()

        stabilizer = PNNStabilizer(
            model=model,
            beta=pnn_cfg["beta"],
            eps=pnn_cfg["eps"],
            kappa=pnn_cfg["kappa"],
            gamma=pnn_cfg["gamma"],
            lambda_max=pnn_cfg["lambda_max"],
            grouping=pnn_cfg["grouping"],
            device=device,
            excluded_params=excluded_params,
            adaptive_enabled=pnn_cfg.get("adaptive_enabled", False),
            adaptive_alpha=pnn_cfg.get("adaptive_alpha", 0.6),
            adaptive_beta_g=pnn_cfg.get("adaptive_beta_g", 0.95),
        )
        logger.log_text(f"Initialized PNNStabilizer with grouping={pnn_cfg['grouping']}")

    epochs_per_task = config["epochs_per_task"]
    task_accuracies: List[List[float]] = []

    max_tasks = config.get("max_tasks", None)

    for task_id, train_loader in enumerate(train_loaders):
        if max_tasks is not None and task_id >= max_tasks:
            break

        logger.log_text(f"\n=== Task {task_id} / {num_tasks - 1} ===")
        logger.log_text(f"Classes for this task: {task_classes[task_id]}")

        if stabilizer is not None:
            stabilizer.begin_task(reset_importance=True)

        task_step = 0
        warmup_epochs = config["pnn"]["warmup_epochs"] if use_pnn else 0
        warmup_steps = warmup_epochs * max(1, len(train_loader)) if use_pnn else 0

        for epoch in range(epochs_per_task):
            model.train()
            epoch_loss = 0.0
            epoch_reg_loss = 0.0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                task_loss = criterion(outputs, targets)

                reg_loss = torch.tensor(0.0, device=device)

                if stabilizer is not None:
                    named_params = [
                        (n, p) for n, p in model.named_parameters()
                        if p.requires_grad and n not in stabilizer.excluded_params
                    ]
                    if len(named_params) > 0:
                        grads = torch.autograd.grad(
                            task_loss,
                            [p for _, p in named_params],
                            retain_graph=False,
                            allow_unused=True,
                        )
                        stabilizer.accumulate_importance_from_grads(
                            list(zip([n for n, _ in named_params], grads))
                        )

                        mu = linear_warmup(
                            task_step,
                            warmup_steps,
                            config["pnn"]["mu_min"],
                            config["pnn"]["mu_max"],
                        )
                        reg_loss = stabilizer.reg_loss(mu)
                        epoch_reg_loss += reg_loss.item()

                total_loss = task_loss + reg_loss
                total_loss.backward()
                optimizer.step()

                epoch_loss += task_loss.item()
                task_step += 1

            avg_loss = epoch_loss / max(1, len(train_loader))
            avg_reg = epoch_reg_loss / max(1, len(train_loader)) if stabilizer is not None else 0.0
            logger.log_text(
                f"Task {task_id}, Epoch {epoch}: loss={avg_loss:.4f}, reg={avg_reg:.4f}"
            )
            if stabilizer is not None:
                logger.log_reg_loss(
                    task=task_id,
                    epoch=epoch,
                    reg_loss=avg_reg,
                    method=method,
                    seed=seed,
                )

        if stabilizer is not None:
            stabilizer.end_task()

        current_task_accs: List[float] = []
        with torch.no_grad():
            for eval_task_id in range(task_id + 1):
                acc = _evaluate(model, test_loaders[eval_task_id], device)
                current_task_accs.append(acc)
                logger.log_text(
                    f"Accuracy on task {eval_task_id} after training task {task_id}: {acc:.2f}%"
                )
                logger.log_metric(
                    task_train=task_id,
                    task_eval=eval_task_id,
                    accuracy=acc,
                    epoch=epochs_per_task,
                    method=method,
                    seed=seed,
                )

        row = [0.0 for _ in range(num_tasks)]
        for k, acc in enumerate(current_task_accs):
            row[k] = acc
        task_accuracies.append(row)

        if stabilizer is not None and hasattr(stabilizer, "lambda_"):
            for name, lam in stabilizer.lambda_.items():
                flat = lam.detach().view(-1)
                if flat.numel() == 0:
                    continue
                p5 = float(torch.quantile(flat, 0.05).item())
                p50 = float(torch.quantile(flat, 0.5).item())
                p95 = float(torch.quantile(flat, 0.95).item())
                logger.log_lambda_stats(
                    task=task_id,
                    layer=name,
                    p5=p5,
                    p50=p50,
                    p95=p95,
                    method=method,
                    seed=seed,
                )

    metrics = compute_metrics(task_accuracies)
    summary_path = os.path.join(config["run_dir"], "summary.json")
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.log_text(
        f"\nFinal Average Accuracy: {metrics['average_accuracy']:.2f}% | "
        f"Average Forgetting: {metrics['average_forgetting']:.2f}% | "
        f"BWT: {metrics['backward_transfer']:.2f}%"
    )
    logger.log_text("Training finished.")
