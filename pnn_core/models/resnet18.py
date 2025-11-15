import torch.nn as nn
from torchvision.models import resnet18

def build_resnet18(num_classes: int = 100, pretrained: bool = False) -> nn.Module:
    """
    ResNet-18 برای CIFAR-100.
    بعداً می‌تونیم تنظیمات بیشتری اضافه کنیم (pretrained روی ImageNet و غیره).
    """
    model = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# یادآوری: فعلاً mlp.py و cnn.py رو بعداً از ریپوی سوپروایزر کپی می‌کنیم.


%%writefile pnn_core/training/eval_metrics.py
"""
توابع متریک: accuracy matrix, forgetting, BWT و غیره.

در مرحله‌ی بعد محتوا را از eval.py سوپروایزر استخراج می‌کنیم و اینجا می‌ریزیم.
"""

import numpy as np
from typing import Dict, Any

def compute_metrics_from_matrix(acc_matrix: np.ndarray) -> Dict[str, Any]:
    """
    ورودی: ماتریس [num_tasks, num_tasks] که acc[t_train, t_eval] را دارد.
    خروجی: دیکشنری شامل avg_acc, avg_forgetting, bwt و غیره.
    """
    num_tasks = acc_matrix.shape[0]
    final_accs = acc_matrix[-1]  # آخرین ردیف
    avg_acc = final_accs.mean()

    # Forgetting: max accuracy for each task - final accuracy for that task
    max_acc_per_task = acc_matrix.max(axis=0)
    forgetting_per_task = max_acc_per_task - final_accs
    avg_forgetting = forgetting_per_task.mean()

    # BWT: میانگین تاثیر تسک‌های جدید روی تسک‌های قدیمی
    # (بعداً می‌تونیم دقیق‌تر مطابق eval.py سوپروایزر پیاده کنیم)
    bwt = 0.0  # placeholder

    return {
        "avg_acc": float(avg_acc),
        "avg_forgetting": float(avg_forgetting),
        "bwt": float(bwt),
        "final_accs": final_accs.tolist(),
        "forgetting_per_task": forgetting_per_task.tolist(),
    }


%%writefile pnn_core/utils/logging.py
"""
لاگِر مشترک برای همه‌ی اکسپریمنت‌ها.
"""

import os
import csv
import json
from datetime import datetime
from typing import Dict, Any, Optional

class ExperimentLogger:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)

        self.text_log_path = os.path.join(self.run_dir, "train_log.txt")
        self.metrics_path = os.path.join(self.run_dir, "metrics.csv")
        self.lambda_stats_path = os.path.join(self.run_dir, "lambda_stats.csv")
        self.reg_loss_path = os.path.join(self.run_dir, "reg_loss.csv")

        # فایل‌های CSV را اگر وجود نداشتن، هدرشان را بنویس
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["task_train", "task_eval", "accuracy", "epoch", "method", "seed"])

        if not os.path.exists(self.lambda_stats_path):
            with open(self.lambda_stats_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["task", "layer", "p5", "p50", "p95", "method", "seed"])

        if not os.path.exists(self.reg_loss_path):
            with open(self.reg_loss_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["task", "epoch", "reg_loss", "method", "seed"])

    def log_text(self, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(self.text_log_path, "a") as f:
            f.write(line + "\n")

    def log_metric(self, task_train: int, task_eval: int, accuracy: float,
                   epoch: int, method: str, seed: int):
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([task_train, task_eval, accuracy, epoch, method, seed])

    def log_lambda_stats(self, task: int, layer: str,
                         p5: float, p50: float, p95: float,
                         method: str, seed: int):
        with open(self.lambda_stats_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([task, layer, p5, p50, p95, method, seed])

    def log_reg_loss(self, task: int, epoch: int, reg_loss: float,
                     method: str, seed: int):
        with open(self.reg_loss_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([task, epoch, reg_loss, method, seed])

    def save_config(self, config: Dict[str, Any]):
        path = os.path.join(self.run_dir, "config.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
