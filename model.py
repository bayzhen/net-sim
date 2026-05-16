"""Neural surrogate models for the goal-net-xpbd dataset.

The first model is a simple MLP that learns the offset-frame mapping::

    input encoding (D_in) -> ball_pos(3) + ball_vel(3) + particle_pos(N, 3)

Input/output normalization is *built into the module* via non-trainable
buffers (``in_mean / in_std`` and the per-head ``*_mean / *_std``). This
way:

* The training loop hands raw physical-units tensors to the model; the
  model handles standardization internally and returns physical units.
* Checkpoints carry the statistics; inference / fine-tuning works
  without re-loading the original dataset.
* Loss is computed in the standardized space so heads with very
  different scales (ball position ~5 m vs. particle position ~0.7 m
  on y) get equal gradient pressure.

Use ``GoalNetMLP.set_norm_stats(...)`` to install statistics and
``GoalNetMLP.get_norm_stats()`` to round-trip them in checkpoints.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn


def _safe_std(std: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.clamp(std, min=eps)


class GoalNetMLP(nn.Module):
    """Plain MLP surrogate with built-in normalization.

    Args:
        in_dim: input feature dimension. With the default encoder this is
            ``3 (vel) + 3 (ang) + 2 (pos[x,y]) + 1 (radius) + 1 (mass) +
            n_time_freq*2`` = 10 + 2*n_time_freq. Default 18 with
            n_time_freq=4 sinusoidal embeddings.
        n_particles: number of net particles N (output net is N*3 floats).
        hidden: hidden layer widths.
        activation: ``relu`` / ``gelu`` / ``silu``.
        predict_velocity: include the ball velocity head (default True).
        dropout: optional dropout probability between hidden layers.
    """

    def __init__(
        self,
        in_dim: int = 18,
        n_particles: int = 0,
        hidden: Sequence[int] = (1024, 1024, 1024, 1024, 1024, 1024),
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
        self.head_net = nn.Linear(prev, self.n_particles * 3)

        # Normalization buffers — initialized to identity so an unconfigured
        # model is still numerically correct (just unscaled).
        self.register_buffer("in_mean",      torch.zeros(self.in_dim))
        self.register_buffer("in_std",       torch.ones(self.in_dim))
        self.register_buffer("ball_pos_mean", torch.zeros(3))
        self.register_buffer("ball_pos_std",  torch.ones(3))
        self.register_buffer("ball_vel_mean", torch.zeros(3))
        self.register_buffer("ball_vel_std",  torch.ones(3))
        # Per-particle statistics: net_mean (N, 3), net_std (N, 3).
        self.register_buffer("net_mean", torch.zeros(self.n_particles, 3))
        self.register_buffer("net_std",  torch.ones(self.n_particles, 3))

    # ------------------------------------------------------------------
    # Stats configuration
    # ------------------------------------------------------------------

    @torch.no_grad()
    def set_norm_stats(self, stats: Dict[str, torch.Tensor]) -> None:
        """Install normalization statistics computed elsewhere.

        Required keys:
            in_mean, in_std            — (in_dim,)
            ball_pos_mean, ball_pos_std — (3,)
            ball_vel_mean, ball_vel_std — (3,)
            net_mean, net_std           — (N, 3)
        """
        for name in ("in_mean", "in_std",
                     "ball_pos_mean", "ball_pos_std",
                     "ball_vel_mean", "ball_vel_std",
                     "net_mean", "net_std"):
            if name not in stats:
                raise KeyError(f"missing '{name}' in stats")
            target = getattr(self, name)
            t = stats[name].to(target.device, dtype=target.dtype)
            if t.shape != target.shape:
                raise ValueError(
                    f"shape mismatch for {name}: got {tuple(t.shape)}, "
                    f"expected {tuple(target.shape)}")
            target.copy_(t)

    def get_norm_stats(self) -> Dict[str, torch.Tensor]:
        return {
            "in_mean":       self.in_mean.detach().clone(),
            "in_std":        self.in_std.detach().clone(),
            "ball_pos_mean": self.ball_pos_mean.detach().clone(),
            "ball_pos_std":  self.ball_pos_std.detach().clone(),
            "ball_vel_mean": self.ball_vel_mean.detach().clone(),
            "ball_vel_std":  self.ball_vel_std.detach().clone(),
            "net_mean":      self.net_mean.detach().clone(),
            "net_std":       self.net_std.detach().clone(),
        }

    # ------------------------------------------------------------------
    # Forward / inverse pieces (exposed so trainer can compute loss in
    # standardized space without re-implementing the inverse).
    # ------------------------------------------------------------------

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.in_mean) / _safe_std(self.in_std)

    def standardize_targets(
        self,
        ball_pos: torch.Tensor,
        ball_vel: Optional[torch.Tensor],
        net: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        out = {
            "ball_pos": (ball_pos - self.ball_pos_mean) / _safe_std(self.ball_pos_std),
            "net":      (net - self.net_mean) / _safe_std(self.net_std),
        }
        if ball_vel is not None:
            out["ball_vel"] = (ball_vel - self.ball_vel_mean) / _safe_std(self.ball_vel_std)
        return out

    def forward(self, x: torch.Tensor, *, return_normalized: bool = False) -> dict:
        """Inputs: (B, in_dim) in raw physical units. Returns dict with
        ``ball_pos`` (B, 3), ``ball_vel`` (B, 3) [if enabled], ``net``
        (B, N, 3) — denormalized to physical units by default.

        If ``return_normalized=True`` the heads are returned in
        standardized space, which is what the trainer uses for loss.
        """
        h = self.trunk(self.normalize_input(x))
        out_norm = {
            "ball_pos": self.head_ball_pos(h),
            "net":      self.head_net(h).view(-1, self.n_particles, 3),
        }
        if self.head_ball_vel is not None:
            out_norm["ball_vel"] = self.head_ball_vel(h)
        if return_normalized:
            return out_norm

        # Denormalize for "give me physics-units" callers.
        out = {
            "ball_pos": out_norm["ball_pos"] * _safe_std(self.ball_pos_std) + self.ball_pos_mean,
            "net":      out_norm["net"] * _safe_std(self.net_std) + self.net_mean,
        }
        if "ball_vel" in out_norm:
            out["ball_vel"] = out_norm["ball_vel"] * _safe_std(self.ball_vel_std) + self.ball_vel_mean
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Input encoder — turn the raw (vel, ang, pos[x,y], radius, mass, t_norm)
# tuple into a (in_dim,) feature vector with sinusoidal time embedding.
# ---------------------------------------------------------------------------


def encode_input_features(
    pos_xy: torch.Tensor,    # (B, 2)
    vel: torch.Tensor,       # (B, 3)
    ang: torch.Tensor,       # (B, 3)
    radius: torch.Tensor,    # (B,)
    mass: torch.Tensor,      # (B,)
    t_norm: torch.Tensor,    # (B,) in [0, 1]
    n_time_freq: int = 4,
) -> torch.Tensor:
    """Concatenate inputs + sinusoidal time embedding.

    The input vector is::

        [pos_x, pos_y,
         vel_x, vel_y, vel_z,
         ang_x, ang_y, ang_z,
         radius, mass,
         sin(2pi*t), cos(2pi*t),
         sin(4pi*t), cos(4pi*t),
         ...]

    Total dim = 10 + 2 * n_time_freq.

    Note ``input_position[2]`` is dropped because the dataset has it
    fixed at 1.5 m (no information). ``radius`` and ``mass`` are also
    constants in the v2 dataset but kept in the input so future datasets
    with varied balls do not require schema changes.
    """
    B = pos_xy.shape[0]
    if t_norm.dim() == 1:
        t = t_norm.unsqueeze(-1)  # (B, 1)
    else:
        t = t_norm
    freqs = torch.arange(1, n_time_freq + 1, device=t.device, dtype=t.dtype)
    angles = 2.0 * torch.pi * t * freqs  # (B, n_time_freq)
    sinp = torch.sin(angles)
    cosp = torch.cos(angles)
    return torch.cat([
        pos_xy,
        vel,
        ang,
        radius.unsqueeze(-1) if radius.dim() == 1 else radius,
        mass.unsqueeze(-1) if mass.dim() == 1 else mass,
        sinp,
        cosp,
    ], dim=-1)


def input_dim_for(n_time_freq: int = 4) -> int:
    return 2 + 3 + 3 + 1 + 1 + 2 * n_time_freq  # = 10 + 2*F


__all__ = [
    "GoalNetMLP",
    "count_parameters",
    "encode_input_features",
    "input_dim_for",
]

