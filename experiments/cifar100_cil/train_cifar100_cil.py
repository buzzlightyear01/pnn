"""
اسکریپت اصلی train برای CIFAR-100 class-incremental.
"""

import os
import argparse
import json
from datetime import datetime

import torch
from pnn_core.models.resnet18 import build_resnet18
from pnn_core.utils.logging import ExperimentLogger
from pnn_core.data.cifar100_cil import get_cifar100_cil_dataloaders

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, help="Path to config JSON/YAML (later).")
    parser.add_argument("--method", type=str, required=True,
                        choices=["baseline", "ewc", "pnn_param", "pnn_layer"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_root", type=str, default="experiments/cifar100_cil/runs")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.run_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.method}_seed{args.seed}_{ts}"
    else:
        run_name = args.run_name

    run_dir = os.path.join(args.output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # config ساده موقت
    config = {
        "method": args.method,
        "seed": args.seed,
        "data_root": args.data_root,
        "output_root": args.output_root,
        "run_dir": run_dir,
        "model": "resnet18",
    }

    logger = ExperimentLogger(run_dir)
    logger.save_config(config)
    logger.log_text(f"Starting run: {run_name}")

    # TODO: set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(num_classes=100).to(device)

    # TODO:
    # 1) ساخت optimizer و scheduler
    # 2) گرفتن train_loaders و eval_loaders از get_cifar100_cil_dataloaders
    # 3) حلقه‌ی over tasks و epochs
    # 4) اگر method = pnn_* → ساخت PNNStabilizer و log λها
    # 5) log_accuracy_matrix و غیره

    logger.log_text("Finished (stub).")

if __name__ == "__main__":
    main()
