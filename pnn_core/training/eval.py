import argparse
import json
import yaml
import torch
from pathlib import Path
from typing import List, Dict

from models import MLP, CNN
from adapters import add_lora_adapters
from data import get_splitmnist_tasks
from pnn.utils import set_seed


def compute_metrics(task_accuracies: List[List[float]]) -> Dict[str, float]:
    num_tasks = len(task_accuracies)
    
    avg_acc = sum(task_accuracies[-1]) / len(task_accuracies[-1])
    
    forgetting = []
    for k in range(num_tasks - 1):
        max_acc = max(task_accuracies[i][k] for i in range(k, num_tasks))
        final_acc = task_accuracies[-1][k]
        forgetting.append(max_acc - final_acc)
    avg_forgetting = sum(forgetting) / len(forgetting) if forgetting else 0.0
    
    bwt = 0.0
    if num_tasks > 1:
        bwt_sum = 0.0
        for k in range(num_tasks - 1):
            acc_at_end_of_k = task_accuracies[k][k]
            final_acc = task_accuracies[-1][k]
            bwt_sum += (final_acc - acc_at_end_of_k)
        bwt = bwt_sum / (num_tasks - 1)
    
    return {
        'average_accuracy': avg_acc,
        'average_forgetting': avg_forgetting,
        'backward_transfer': bwt,
        'per_task_forgetting': forgetting,
        'final_accuracies': task_accuracies[-1],
    }


def evaluate_run(run_dir: Path):
    if not (run_dir / 'config.yaml').exists():
        print(f"Skipping {run_dir.name}: incomplete run (no config.yaml)")
        return None
    
    if not (run_dir / 'metrics.json').exists():
        print(f"Skipping {run_dir.name}: incomplete run (no metrics.json)")
        return None
    
    with open(run_dir / 'config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    with open(run_dir / 'metrics.json', 'r') as f:
        metrics = json.load(f)
    
    task_accuracies = metrics['task_accuracies']
    
    results = compute_metrics(task_accuracies)
    
    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results for: {run_dir.name}")
    print(f"{'='*60}")
    print(f"Average Accuracy: {results['average_accuracy']:.2f}%")
    print(f"Average Forgetting: {results['average_forgetting']:.2f}%")
    print(f"Backward Transfer: {results['backward_transfer']:.2f}%")
    print(f"\nFinal Accuracies per Task:")
    for i, acc in enumerate(results['final_accuracies']):
        print(f"  Task {i}: {acc:.2f}%")
    print(f"\nForgetting per Task:")
    for i, forg in enumerate(results['per_task_forgetting']):
        print(f"  Task {i}: {forg:.2f}%")
    print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True)
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    
    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist")
        return
    
    evaluate_run(run_dir)


if __name__ == '__main__':
    main()

