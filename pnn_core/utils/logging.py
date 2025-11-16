import os
import csv
import json
from datetime import datetime
from typing import Dict, Any


class ExperimentLogger:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)

        self.text_log_path = os.path.join(self.run_dir, "train_log.txt")
        self.metrics_path = os.path.join(self.run_dir, "metrics.csv")
        self.lambda_stats_path = os.path.join(self.run_dir, "lambda_stats.csv")
        self.reg_loss_path = os.path.join(self.run_dir, "reg_loss.csv")

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
