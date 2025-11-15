def linear_warmup(step: int, warmup_steps: int, mu_min: float, mu_max: float) -> float:
    if warmup_steps <= 0:
        return mu_max
    progress = min(1.0, step / warmup_steps)
    return mu_min + (mu_max - mu_min) * progress

