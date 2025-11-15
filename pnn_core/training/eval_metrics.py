"""
توابع متریک برای continual learning:
- average accuracy
- average forgetting
- backward transfer (BWT)
"""

from typing import List, Dict


def compute_metrics(task_accuracies: List[List[float]]) -> Dict[str, float]:
    """
    task_accuracies: لیستی با طول num_tasks
      task_accuracies[t][k] = دقت روی تسک k بعد از این‌که تسک t آموزش دیده.

    این دقیقاً همون ساختاریه که در eval.py سوپروایزر استفاده شده.
    """
    num_tasks = len(task_accuracies)
    if num_tasks == 0:
        return {
            "average_accuracy": 0.0,
            "average_forgetting": 0.0,
            "backward_transfer": 0.0,
            "final_accuracies": [],
            "per_task_forgetting": [],
        }

    # فرض: task_accuracies[-1] شامل دقت نهایی روی همه‌ی تسک‌هاست
    final_accuracies = task_accuracies[-1]
    avg_acc = sum(final_accuracies) / len(final_accuracies)

    # Forgetting
    forgetting = []
    for k in range(num_tasks - 1):
        # بیشترین دقت روی تسک k از زمانی که اولین بار آموزش داده شده تا انتها
        max_acc = max(task_accuracies[i][k] for i in range(k, num_tasks))
        final_acc = final_accuracies[k]
        forgetting.append(max_acc - final_acc)
    avg_forgetting = sum(forgetting) / len(forgetting) if forgetting else 0.0

    # Backward Transfer (BWT)
    bwt = 0.0
    if num_tasks > 1:
        bwt_sum = 0.0
        for k in range(num_tasks - 1):
            # تاثیر تسک‌های بعدی روی تسک k:
            # accuracy نهایی روی تسک k - accuracy بلافاصله بعد از یادگیری خود تسک k
            bwt_sum += final_accuracies[k] - task_accuracies[k][k]
        bwt = bwt_sum / (num_tasks - 1)

    return {
        "average_accuracy": float(avg_acc),
        "average_forgetting": float(avg_forgetting),
        "backward_transfer": float(bwt),
        "final_accuracies": [float(a) for a in final_accuracies],
        "per_task_forgetting": [float(f) for f in forgetting],
    }
