"""
Cosine annealing with linear warmup — standard for ViT / MAE training.
Updates LR each step (not each epoch) for smooth warmup.
"""

import math
from torch.optim import Optimizer


class CosineWarmupScheduler:
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        base_lr: float,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self._get_lr()
        for group in self.optimizer.param_groups:
            # Respect per-group scale factors (used for layer-wise LR decay)
            scale = group.get('lr_scale', 1.0)
            group['lr'] = lr * scale

    def _get_lr(self) -> float:
        s = self.current_step
        if s < self.warmup_steps:
            return self.base_lr * s / max(1, self.warmup_steps)
        progress = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def get_last_lr(self) -> float:
        return self._get_lr()
