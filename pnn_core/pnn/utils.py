import random
import numpy as np
import torch
from typing import Dict


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_buffers(path: str, buffers: Dict[str, torch.Tensor]) -> None:
    torch.save(buffers, path)


def load_buffers(path: str) -> Dict[str, torch.Tensor]:
    return torch.load(path)

