"""Neural surrogate models for the goal-net-xpbd dataset.

The first model is a simple MLP that learns the offset-frame mapping::

    input_state (13) = [ball_pos(3), ball_vel(3), ball_ang(3), radius, mass, t_norm, ?]
    -> ball_pos(3), ball_vel(3), particle_pos(N, 3)

In practice we feed the 13-D vector documented in ``train.py`` (which is the
classic 12-D ``[pos, vel, ang, t_norm]`` augmented with one extra slot for
``radius`` so the model sees ball mass/size). The model is intentionally
plain: it serves as a baseline against which to measure more elaborate
architectures.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class GoalNetMLP(nn.Module):
    """Plain MLP surrogate.

    Args:
        in_dim: input feature dimension (default 13: 9-D state + radius +
            mass + t_norm + spare slot; see ``train.py::build_input``).
        n_particles: number of net particles N (output net is N*3 floats).
        hidden: hidden layer widths.
        activation: name of the activation (``relu``, ``gelu``, ``silu``).
        predict_velocity: whether to also output ball velocity (3 floats).
            Default True; saves a head if the user wants pos-only later.
        dropout: optional dropout probability applied between hidden layers.
    """

    def __init__(
        self,
        in_dim: int = 13,
        n_particles: int = 0,
        hidden: Sequence[int] = (512, 512, 512, 512),
        activation: str = "gelu",
        predict_velocity: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_particles <= 0:
            raise ValueError("n_particles must be > 0")
        self.in_dim = int(in_dim)
        self.n_particles = int(n_particles)
        self.predict_velocity = bool(predict_velocity)

        act_cls = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[activation.lower()]

        layers = []
        prev = self.in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.trunk = nn.Sequential(*layers)

        self.head_ball_pos = nn.Linear(prev, 3)
        self.head_ball_vel = nn.Linear(prev, 3) if self.predict_velocity else None
        # net head: predict per-particle offset against rest position upstream
        self.head_net = nn.Linear(prev, self.n_particles * 3)

    def forward(self, x: torch.Tensor) -> dict:
        """Inputs: (B, in_dim). Returns dict with keys
        ``ball_pos`` (B, 3), ``ball_vel`` (B, 3) [if enabled],
        ``net`` (B, N, 3)."""
        h = self.trunk(x)
        out = {
            "ball_pos": self.head_ball_pos(h),
            "net": self.head_net(h).view(-1, self.n_particles, 3),
        }
        if self.head_ball_vel is not None:
            out["ball_vel"] = self.head_ball_vel(h)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["GoalNetMLP", "count_parameters"]
