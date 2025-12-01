import torch
import torch.nn as nn
from typing import Dict, Optional, Set
from .base import PNNBase


class PNNStabilizer(PNNBase):
    def __init__(
        self,
        model: nn.Module,
        beta: float,
        eps: float,
        kappa: float,
        gamma: float,
        lambda_max: float,
        grouping: str,
        device: torch.device,
        excluded_params: Optional[Set[str]] = None,
    ):
        super().__init__(
            model=model,
            beta=beta,
            eps=eps,
            kappa=kappa,
            gamma=gamma,
            lambda_max=lambda_max,
            grouping=grouping,
            device=device,
            excluded_params=excluded_params,
        )
