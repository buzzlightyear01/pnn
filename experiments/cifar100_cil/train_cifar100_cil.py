import os
import json
import argparse
from pathlib import Path

from pnn_core.training.loop_cifar100_cil import train_cifar100_cil_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/cifar100_cil/configs/base.json",
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="pnn_layer",
        choices=["naive", "pnn_param", "pnn_layer"],
        help="Which method to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)

    data_root = config["data_root"]
    run_root = config["run_root"]
    os.makedirs(run_root, exist_ok=True)

    run_dir = Path(run_root) / f"{args.method}_seed{args.seed}"
    os.makedirs(run_dir, exist_ok=True)

    # Merge config + CLI info for logging
    full_config = {
        **config,
        "method": args.method,
        "seed": args.seed,
        "run_dir": str(run_dir)
    }
    # Save config in run_dir
    with open(run_dir / "config.json", "w") as f:
        json.dump(full_config, f, indent=2)

    # Run experiment
    train_cifar100_cil_experiment(
        data_root=data_root,
        run_dir=str(run_dir),
        method=args.method,
        seed=args.seed,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        momentum=config["momentum"],
        epochs_per_task=config["epochs_per_task"],
        beta=config["beta"],
        eps=config["eps"],
        kappa=config["kappa"],
        gamma=config["gamma"],
        lambda_max=config["lambda_max"],
        mu_min=config["mu_min"],
        mu_max=config["mu_max"],
        warmup_epochs=config["warmup_epochs"],
    )


if __name__ == "__main__":
    main()
