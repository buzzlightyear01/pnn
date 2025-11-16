import os
import sys
import argparse
import json
from datetime import datetime


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pnn_core.utils.logging import ExperimentLogger
from pnn_core.training.loop_cifar100_cil import train_cifar100_cil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/cifar100_cil/configs/base.json",
        help="Path to base config JSON.",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["baseline", "ewc", "pnn_param", "pnn_layer"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument(
        "--output_root",
        type=str,
        default="experiments/cifar100_cil/runs",
    )
    return parser.parse_args()


def main():
    args = parse_args()

   
    with open(args.config, "r") as f:
        config = json.load(f)

    config["method"] = args.method
    config["seed"] = args.seed
    config["data_root"] = args.data_root

    if args.run_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.method}_seed{args.seed}_{ts}"
    else:
        run_name = args.run_name

    run_dir = os.path.join(args.output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    config["run_dir"] = run_dir

    logger = ExperimentLogger(run_dir)
    logger.save_config(config)
    logger.log_text(f"Starting CIFAR-100 CIL run: {run_name}")

    train_cifar100_cil(config, logger)


if __name__ == "__main__":
    main()
