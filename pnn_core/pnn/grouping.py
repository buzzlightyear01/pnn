import torch
import torch.nn as nn
from typing import Dict, List, Tuple


def group_indices(
    model: nn.Module, mode: str
) -> Dict[int, List[Tuple[str, nn.Parameter]]]:
    groups = {}
    
    if mode == "param":
        for idx, (name, param) in enumerate(model.named_parameters()):
            if param.requires_grad:
                groups[idx] = [(name, param)]
    
    elif mode == "layer":
        group_id = 0
        for module_name, module in model.named_modules():
            params_in_module = []
            for param_name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    full_name = f"{module_name}.{param_name}" if module_name else param_name
                    params_in_module.append((full_name, param))
            if params_in_module:
                groups[group_id] = params_in_module
                group_id += 1
    
    elif mode == "channel":
        group_id = 0
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                for param_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad and param_name == "weight":
                        full_name = f"{module_name}.{param_name}" if module_name else param_name
                        out_features = param.shape[0]
                        for ch in range(out_features):
                            groups[group_id] = [(full_name, param, ch)]
                            group_id += 1
                    elif param.requires_grad and param_name == "bias":
                        full_name = f"{module_name}.{param_name}" if module_name else param_name
                        groups[group_id] = [(full_name, param)]
                        group_id += 1
            elif isinstance(module, nn.Conv2d):
                for param_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad and param_name == "weight":
                        full_name = f"{module_name}.{param_name}" if module_name else param_name
                        out_channels = param.shape[0]
                        for ch in range(out_channels):
                            groups[group_id] = [(full_name, param, ch)]
                            group_id += 1
                    elif param.requires_grad and param_name == "bias":
                        full_name = f"{module_name}.{param_name}" if module_name else param_name
                        groups[group_id] = [(full_name, param)]
                        group_id += 1
            else:
                for param_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        full_name = f"{module_name}.{param_name}" if module_name else param_name
                        groups[group_id] = [(full_name, param)]
                        group_id += 1
    
    elif mode == "head":
        group_id = 0
        for module_name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                for param_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad and param_name == "weight":
                        full_name = f"{module_name}.{param_name}" if module_name else param_name
                        if isinstance(module, nn.Linear):
                            out_features = param.shape[0]
                            for head in range(out_features):
                                groups[group_id] = [(full_name, param, head)]
                                group_id += 1
                        else:
                            out_channels = param.shape[0]
                            for head in range(out_channels):
                                groups[group_id] = [(full_name, param, head)]
                                group_id += 1
                    elif param.requires_grad and param_name == "bias":
                        full_name = f"{module_name}.{param_name}" if module_name else param_name
                        groups[group_id] = [(full_name, param)]
                        group_id += 1
            else:
                for param_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        full_name = f"{module_name}.{param_name}" if module_name else param_name
                        groups[group_id] = [(full_name, param)]
                        group_id += 1
    
    else:
        raise ValueError(f"Unknown grouping mode: {mode}")
    
    return groups

