import torch
import torch.nn as nn
from typing import Dict, Optional, Set
from .grouping import group_indices


class PNNBase:
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
        self.model = model
        self.beta = beta
        self.eps = eps
        self.kappa = kappa
        self.gamma = gamma
        self.lambda_max = lambda_max
        self.grouping = grouping
        self.device = device
        self.excluded_params = excluded_params or set()
        
        self.groups = group_indices(model, grouping)
        
        self.theta_star: Dict[str, torch.Tensor] = {}
        self.lambda_: Dict[str, torch.Tensor] = {}
        self.importance: Dict[str, torch.Tensor] = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad and name not in self.excluded_params:
                self.theta_star[name] = param.data.clone().to(device)
                self.lambda_[name] = torch.zeros_like(param.data, device=device)
                self.importance[name] = torch.zeros_like(param.data, device=device)
    
    def begin_task(self, reset_importance: bool = True) -> None:
        if reset_importance:
            for name in self.importance:
                self.importance[name].zero_()
    
    def accumulate_importance(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and name not in self.excluded_params and param.grad is not None:
                grad_squared = param.grad.data ** 2
                self.importance[name] = (
                    self.beta * self.importance[name] + (1 - self.beta) * grad_squared
                )
    
    def accumulate_importance_from_grads(self, named_grads):
        for name, g in named_grads:
            if g is None:
                continue
            if name in self.importance:
                g_detached = g.detach()
                gsq = g_detached ** 2
                self.importance[name] = self.beta * self.importance[name] + (1 - self.beta) * gsq
    
    def reg_loss(self, mu: float) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            if param.requires_grad and name not in self.excluded_params:
                diff = param - self.theta_star[name]
                loss = loss + (self.lambda_[name] * diff ** 2).sum()
        return mu * loss
    
    def end_task(self) -> None:
        normalized_importance = self._normalize_importance()
        
        for name in self.lambda_:
            self.lambda_[name] = self.gamma * self.lambda_[name] + self.kappa * normalized_importance[name]
            self.lambda_[name] = torch.clamp(self.lambda_[name], 0.0, self.lambda_max)
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name not in self.excluded_params:
                self.theta_star[name] = param.data.clone()
        
        for name in self.importance:
            self.importance[name].zero_()
    
    def _normalize_importance(self) -> Dict[str, torch.Tensor]:
        normalized = {}
        
        if self.grouping == "param":
            for name in self.importance:
                mean_val = self.importance[name].mean()
                normalized[name] = self.importance[name] / (self.eps + mean_val)
        
        elif self.grouping == "layer":
            for group_id, params_list in self.groups.items():
                all_importance = []
                for item in params_list:
                    name = item[0]
                    if name in self.importance:
                        all_importance.append(self.importance[name].flatten())
                
                if all_importance:
                    combined = torch.cat(all_importance)
                    mean_val = combined.mean()
                    
                    for item in params_list:
                        name = item[0]
                        if name in self.importance:
                            normalized[name] = self.importance[name] / (self.eps + mean_val)
        
        elif self.grouping in ["channel", "head"]:
            for group_id, params_list in self.groups.items():
                group_importance = []
                for item in params_list:
                    name = item[0]
                    param = item[1]
                    if name in self.importance:
                        if len(item) > 2:
                            ch_idx = item[2]
                            if len(param.shape) == 2:
                                group_importance.append(self.importance[name][ch_idx, :].flatten())
                            elif len(param.shape) == 4:
                                group_importance.append(self.importance[name][ch_idx, :, :, :].flatten())
                        else:
                            group_importance.append(self.importance[name].flatten())
                
                if group_importance:
                    combined = torch.cat(group_importance)
                    mean_val = combined.mean()
                    
                    for item in params_list:
                        name = item[0]
                        param = item[1]
                        if name in self.importance:
                            if len(item) > 2:
                                ch_idx = item[2]
                                if name not in normalized:
                                    normalized[name] = self.importance[name].clone()
                                if len(param.shape) == 2:
                                    normalized[name][ch_idx, :] = (
                                        self.importance[name][ch_idx, :] / (self.eps + mean_val)
                                    )
                                elif len(param.shape) == 4:
                                    normalized[name][ch_idx, :, :, :] = (
                                        self.importance[name][ch_idx, :, :, :] / (self.eps + mean_val)
                                    )
                            else:
                                normalized[name] = self.importance[name] / (self.eps + mean_val)
        
        else:
            for name in self.importance:
                normalized[name] = self.importance[name].clone()
        
        return normalized
    
    def state_dict(self) -> dict:
        return {
            "theta_star": {k: v.cpu() for k, v in self.theta_star.items()},
            "lambda": {k: v.cpu() for k, v in self.lambda_.items()},
            "importance": {k: v.cpu() for k, v in self.importance.items()},
        }
    
    def load_state_dict(self, state: dict) -> None:
        self.theta_star = {k: v.to(self.device) for k, v in state["theta_star"].items()}
        self.lambda_ = {k: v.to(self.device) for k, v in state["lambda"].items()}
        self.importance = {k: v.to(self.device) for k, v in state["importance"].items()}

