"""PGD / MI-FGSM optimization primitives for a *supplied* differentiable loss.

The objective is an explicit argument: this module does NOT claim that using
our Pull-Push loss reproduces the original classifier-targeted attacks.
The caller must provide a baseline-specific targeted loss and record its origin.
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import Literal

import torch

from primary_ml_cka.attack.optimization.projection import project_linf

AttackName = Literal["pgd", "mi_fgsm"]


@dataclass(frozen=True, slots=True)
class AttackConfig:
    method: AttackName
    steps: int
    epsilon: float
    step_size: float
    momentum: float = 1.0
    random_start: bool = False

    def __post_init__(self) -> None:
        if self.method not in {"pgd", "mi_fgsm"}:
            raise ValueError("Only pgd and mi_fgsm are implemented")
        if type(self.steps) is not int or self.steps <= 0:
            raise ValueError("steps must be positive")
        if not 0 < self.epsilon <= 1:
            raise ValueError("epsilon must be in (0, 1]")
        if not isfinite(self.step_size) or self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if not 0 <= self.momentum <= 1:
            raise ValueError("momentum must be in [0, 1]")


@dataclass(slots=True)
class GradientAttack:
    config: AttackConfig

    def run(
        self,
        clean: torch.Tensor,
        loss_function: Callable[[torch.Tensor], torch.Tensor],
        *,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Minimize a scalar loss, projected to the clean-image L-infinity ball.

        clean must be NCHW float tensor in [0, 1].
        No target-model calls or task-specific features occur here.
        """
        if clean.ndim != 4 or not clean.is_floating_point():
            raise ValueError("Expected floating NCHW input")
        if not torch.isfinite(clean).all() or clean.min() < 0 or clean.max() > 1:
            raise ValueError("clean image must be finite and in [0, 1]")
        cfg = self.config
        clean = clean.detach()
        if cfg.random_start:
            generator = torch.Generator(device=clean.device)
            if seed is not None:
                generator.manual_seed(seed)
            noise = torch.empty_like(clean).uniform_(-cfg.epsilon, cfg.epsilon, generator=generator)
            x = project_linf(clean + noise, clean, cfg.epsilon).detach()
        else:
            x = clean.clone()
        velocity = torch.zeros_like(clean)
        for index in range(cfg.steps):
            x = x.detach().requires_grad_(True)
            loss = loss_function(x)
            if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
                raise ValueError(f"Loss must be a finite scalar (step {index})")
            gradient = torch.autograd.grad(loss, x, only_inputs=True)[0]
            if not bool(torch.isfinite(gradient).all().item()):
                raise ValueError(f"Non-finite gradient at step {index}")
            if cfg.method == "mi_fgsm":
                normalized = gradient / gradient.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(
                    1e-12
                )
                velocity = cfg.momentum * velocity + normalized
                direction = velocity.sign()
            else:
                direction = gradient.sign()
            x = project_linf(x - cfg.step_size * direction, clean, cfg.epsilon)
        result = x.detach()
        if not torch.all((result - clean).abs() <= cfg.epsilon + 1e-6):
            raise AssertionError("Output violates perturbation budget")
        return result


def attack_image(
    clean: torch.Tensor,
    loss_function: Callable[[torch.Tensor], torch.Tensor],
    config: AttackConfig,
    *,
    seed: int | None = None,
) -> torch.Tensor:
    return GradientAttack(config).run(clean, loss_function, seed=seed)
