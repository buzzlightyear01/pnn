import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Set
from .base import PNNBase


class AdaptivePNNStabilizer(PNNBase):
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
        adaptive_alpha: float = 0.6,
        adaptive_beta_g: float = 0.95,
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
        
        self.adaptive_alpha = adaptive_alpha
        self.adaptive_beta_g = adaptive_beta_g
        
        self.grad_ema: Dict[str, torch.Tensor] = {}
        self.conflict: Dict[str, torch.Tensor] = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad and name not in self.excluded_params:
                self.grad_ema[name] = torch.zeros_like(param.data, device=device)
                self.conflict[name] = torch.zeros_like(param.data, device=device)
    
    def begin_task(self, reset_importance: bool = True) -> None:
        super().begin_task(reset_importance)
        for name in self.conflict:
            self.conflict[name].zero_()
    
    def accumulate_importance_from_grads(self, named_grads):
        for name, g in named_grads:
            if g is None:
                continue
            if name in self.importance:
                g_detached = g.detach()
                gsq = g_detached ** 2
                self.importance[name] = self.beta * self.importance[name] + (1 - self.beta) * gsq
                
                if self.grad_ema[name].abs().sum() > 1e-12:
                    cos_sim = F.cosine_similarity(
                        g_detached.flatten().unsqueeze(0),
                        self.grad_ema[name].flatten().unsqueeze(0)
                    )
                    conflict_score = (1.0 - cos_sim.item()) / 2.0
                    conflict_score = max(0.0, min(1.0, conflict_score))
                    self.conflict[name] = (
                        self.beta * self.conflict[name] + 
                        (1 - self.beta) * conflict_score * torch.ones_like(self.conflict[name])
                    )
                
                self.grad_ema[name] = (
                    self.adaptive_beta_g * self.grad_ema[name] + 
                    (1 - self.adaptive_beta_g) * g_detached
                )
    
    def end_task(self) -> None:
        normalized_importance = self._normalize_importance()
        normalized_conflict = self._normalize_conflict()
        
        for name in self.lambda_:
            combined = (
                self.adaptive_alpha * normalized_importance[name] - 
                (1.0 - self.adaptive_alpha) * normalized_conflict[name]
            )
            self.lambda_[name] = self.gamma * self.lambda_[name] + self.kappa * combined
            self.lambda_[name] = torch.clamp(self.lambda_[name], 0.0, self.lambda_max)
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name not in self.excluded_params:
                self.theta_star[name] = param.data.clone()
        
        for name in self.importance:
            self.importance[name].zero_()
        for name in self.conflict:
            self.conflict[name].zero_()
    
    def _normalize_conflict(self) -> Dict[str, torch.Tensor]:
        normalized = {}
        
        if self.grouping == "param":
            for name in self.conflict:
                mean_val = self.conflict[name].mean()
                normalized[name] = self.conflict[name] / (self.eps + mean_val)
        
        elif self.grouping == "layer":
            for group_id, params_list in self.groups.items():
                all_conflict = []
                for item in params_list:
                    name = item[0]
                    if name in self.conflict:
                        all_conflict.append(self.conflict[name].flatten())
                
                if all_conflict:
                    combined = torch.cat(all_conflict)
                    mean_val = combined.mean()
                    
                    for item in params_list:
                        name = item[0]
                        if name in self.conflict:
                            normalized[name] = self.conflict[name] / (self.eps + mean_val)
        
        elif self.grouping in ["channel", "head"]:
            for group_id, params_list in self.groups.items():
                group_conflict = []
                for item in params_list:
                    name = item[0]
                    param = item[1]
                    if name in self.conflict:
                        if len(item) > 2:
                            ch_idx = item[2]
                            if len(param.shape) == 2:
                                group_conflict.append(self.conflict[name][ch_idx, :].flatten())
                            elif len(param.shape) == 4:
                                group_conflict.append(self.conflict[name][ch_idx, :, :, :].flatten())
                        else:
                            group_conflict.append(self.conflict[name].flatten())
                
                if group_conflict:
                    combined = torch.cat(group_conflict)
                    mean_val = combined.mean()
                    
                    for item in params_list:
                        name = item[0]
                        param = item[1]
                        if name in self.conflict:
                            if len(item) > 2:
                                ch_idx = item[2]
                                if name not in normalized:
                                    normalized[name] = self.conflict[name].clone()
                                if len(param.shape) == 2:
                                    normalized[name][ch_idx, :] = (
                                        self.conflict[name][ch_idx, :] / (self.eps + mean_val)
                                    )
                                elif len(param.shape) == 4:
                                    normalized[name][ch_idx, :, :, :] = (
                                        self.conflict[name][ch_idx, :, :, :] / (self.eps + mean_val)
                                    )
                            else:
                                normalized[name] = self.conflict[name] / (self.eps + mean_val)
        
        else:
            for name in self.conflict:
                normalized[name] = self.conflict[name].clone()
        
        return normalized
    
    def state_dict(self) -> dict:
        state = super().state_dict()
        state["grad_ema"] = {k: v.cpu() for k, v in self.grad_ema.items()}
        state["conflict"] = {k: v.cpu() for k, v in self.conflict.items()}
        return state
    
    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        if "grad_ema" in state:
            self.grad_ema = {k: v.to(self.device) for k, v in state["grad_ema"].items()}
            self.conflict = {k: v.to(self.device) for k, v in state["conflict"].items()}

