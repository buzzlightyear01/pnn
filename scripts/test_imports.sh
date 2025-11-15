#!/usr/bin/env bash
set -e  # اگر خطا خورد، اسکریپت رو متوقف کن

echo "[TEST] Python & imports"

python - << 'EOF'
import pnn_core
from pnn_core.data.cifar100_cil import get_cifar100_cil_dataloaders
from pnn_core.training.loop_cifar100_cil import train_cifar100_cil
from pnn_core.utils.logging import ExperimentLogger

print("OK: imports from pnn_core worked.")

EOF

echo "[TEST] Done."
