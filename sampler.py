"""Shot sampler — generates ShotInput instances per §6.

5 panel targets × 5 style profiles, with optional "corner-post" bias to
exercise crossbar/goalpost collisions (§10.3, bug C).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple

from params import BallState

Vec3 = Tuple[float, float, float]


@dataclass
class StyleProfile:
    elevation_bias_deg: float
    speed_range: Tuple[float, float]


STYLE_PROFILES = {
    "ground": StyleProfile(-2.0, (22.0, 28.0)),
    "low": StyleProfile(2.0, (24.0, 30.0)),
    "mid": StyleProfile(6.0, (26.0, 32.0)),
    "high": StyleProfile(12.0, (24.0, 30.0)),
    "lob": StyleProfile(18.0, (18.0, 24.0)),
}


@dataclass
class SamplerConfig:
    count: int = 5
    seed: int = 1
    panels: List[str] = field(
        default_factory=lambda: ["back", "left", "right", "top", "corner"]
    )
    styles: List[str] = field(
        default_factory=lambda: ["ground", "low", "mid", "high", "lob"]
    )
    speed_range: Tuple[float, float] = (22.0, 32.0)
    spin_range: Tuple[float, float] = (-18.0, 18.0)
    azimuth_range_deg: Tuple[float, float] = (-45.0, 45.0)
    elevation_range_deg: Tuple[float, float] = (-2.0, 25.0)
    distance_range: Tuple[float, float] = (6.0, 18.0)
    target_jitter: float = 0.25
    goal_width: float = 7.32
    goal_height: float = 2.44
    goal_depth: float = 2.0
    corner_post_ratio: float = 0.15  # bug C fix: ~15% target near posts/crossbar
    corner_post_distance: float = 0.3
    # Source-point envelope. Shots that would originate far outside this box
    # produce grazing-net trajectories that fail quality checks even though
    # the physics is correct.
    source_x_clamp: float = 9.0
    source_y_clamp: Tuple[float, float] = (0.2, 4.0)
    source_z_min: float = 1.5


@dataclass
class ShotInput:
    sample_id: str
    target_panel: str
    seed: int
    template: str
    ball: BallState


def _direction_from_angles(azimuth_deg: float, elevation_deg: float) -> Vec3:
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    cos_el = math.cos(el)
    # azimuth=0, elevation=0 → (0, 0, +1)
    return (math.sin(az) * cos_el, math.sin(el), math.cos(az) * cos_el)


def _solve_initial_velocity(source: Vec3, target: Vec3, speed: float) -> Vec3:
    """Return launch velocity of magnitude `speed` that hits `target` from
    `source` under gravity g=9.81, using one Newton step on the parabolic
    trajectory.
    """
    g = 9.81
    dx = target[0] - source[0]
    dy = target[1] - source[1]
    dz = target[2] - source[2]
    horiz2 = dx * dx + dz * dz
    horiz = math.sqrt(horiz2)
    if horiz < 1e-6:
        return (0.0, speed, 0.0)
    v2 = speed * speed
    # Quartic in tan(theta); solve via launch angle from low-arc solution:
    # tan(theta) = (v^2 ± sqrt(v^4 - g*(g*horiz^2 + 2*dy*v^2))) / (g*horiz)
    disc = v2 * v2 - g * (g * horiz2 + 2.0 * dy * v2)
    if disc < 0:
        # not enough speed — fall back to direct aim, scaled.
        norm = math.sqrt(horiz2 + dy * dy)
        return (dx / norm * speed, dy / norm * speed, dz / norm * speed)
    tan_theta = (v2 - math.sqrt(disc)) / (g * horiz)
    theta = math.atan(tan_theta)
    vy = speed * math.sin(theta)
    v_horiz = speed * math.cos(theta)
    vx = v_horiz * dx / horiz
    vz = v_horiz * dz / horiz
    return (vx, vy, vz)


def _sample_panel_target(rng: random.Random, cfg: SamplerConfig, panel: str) -> Vec3:
    W = cfg.goal_width
    H = cfg.goal_height
    D = cfg.goal_depth
    m = max(0.1, cfg.target_jitter)
    if panel == "back":
        x = rng.uniform(-W / 2 + m, W / 2 - m)
        y = rng.uniform(0.2, H - m)
        z = -D + 0.05
    elif panel == "left":
        x = -W / 2 + 0.05
        y = rng.uniform(0.3, H - m)
        z = rng.uniform(-D + m, -0.2)
    elif panel == "right":
        x = W / 2 - 0.05
        y = rng.uniform(0.3, H - m)
        z = rng.uniform(-D + m, -0.2)
    elif panel == "top":
        x = rng.uniform(-W / 2 + m, W / 2 - m)
        y = H - 0.05
        z = rng.uniform(-D + m, -0.1)
    elif panel == "corner":
        sign = rng.choice([-1.0, 1.0])
        x = sign * (W / 2 - 0.05)
        y = H - rng.uniform(0.05, 0.45)
        z = rng.uniform(-D + m, -0.2)
    else:
        raise ValueError(f"unknown panel: {panel}")
    jx = rng.uniform(-cfg.target_jitter, cfg.target_jitter)
    jy = rng.uniform(-cfg.target_jitter, cfg.target_jitter)
    jz = rng.uniform(-cfg.target_jitter, cfg.target_jitter) * 0.4
    return (x + jx, y + jy, z + jz)


def _sample_post_target(rng: random.Random, cfg: SamplerConfig) -> Tuple[Vec3, str]:
    """Bug C: sample a target near posts or crossbar."""
    W = cfg.goal_width
    H = cfg.goal_height
    r = cfg.corner_post_distance
    choice = rng.choice(["post_left", "post_right", "crossbar"])
    if choice == "post_left":
        x = -W / 2 + rng.uniform(-r, r)
        y = rng.uniform(0.3, H - 0.1)
        z = rng.uniform(-r, r)
    elif choice == "post_right":
        x = W / 2 + rng.uniform(-r, r)
        y = rng.uniform(0.3, H - 0.1)
        z = rng.uniform(-r, r)
    else:  # crossbar
        x = rng.uniform(-W / 2 + 0.3, W / 2 - 0.3)
        y = H + rng.uniform(-r, r)
        z = rng.uniform(-r, r)
    return (x, y, z), choice


def sample_shots(cfg: SamplerConfig) -> List[ShotInput]:
    rng = random.Random(cfg.seed)
    shots: List[ShotInput] = []
    for i in range(cfg.count):
        panel = cfg.panels[i % len(cfg.panels)]
        style_name = cfg.styles[i % len(cfg.styles)]
        style = STYLE_PROFILES[style_name]

        # bug C: with probability corner_post_ratio, replace target by a
        # corner/post target instead of panel-interior.
        force_post = rng.random() < cfg.corner_post_ratio
        if force_post:
            target, post_name = _sample_post_target(rng, cfg)
            template = f"post/{post_name}/{style_name}"
            target_panel = panel  # keep panel for reporting balance
        else:
            target = _sample_panel_target(rng, cfg, panel)
            template = f"{panel}/{style_name}"
            target_panel = panel

        az = rng.uniform(*cfg.azimuth_range_deg)
        elev = rng.uniform(*cfg.elevation_range_deg) + style.elevation_bias_deg
        elev = max(cfg.elevation_range_deg[0], min(cfg.elevation_range_deg[1], elev))

        distance = rng.uniform(*cfg.distance_range)
        speed = rng.uniform(*style.speed_range)
        direction = _direction_from_angles(az, elev)
        # source = target - direction * distance
        source = (
            target[0] - direction[0] * distance,
            target[1] - direction[1] * distance,
            target[2] - direction[2] * distance,
        )
        sx = max(-cfg.source_x_clamp, min(cfg.source_x_clamp, source[0]))
        sy = max(cfg.source_y_clamp[0], min(cfg.source_y_clamp[1], source[1]))
        sz = max(cfg.source_z_min, source[2])
        source = (sx, sy, sz)

        velocity = _solve_initial_velocity(source, target, speed)
        spin = (
            rng.uniform(*cfg.spin_range),
            rng.uniform(*cfg.spin_range),
            rng.uniform(*cfg.spin_range),
        )

        ball = BallState(
            position=source,
            velocity=velocity,
            angular_velocity=spin,
            radius=0.13,
            mass=1.0,
        )
        shots.append(
            ShotInput(
                sample_id=f"sample_{i:05d}",
                target_panel=target_panel,
                seed=cfg.seed + i,
                template=template,
                ball=ball,
            )
        )
    return shots


__all__ = [
    "StyleProfile",
    "STYLE_PROFILES",
    "SamplerConfig",
    "ShotInput",
    "sample_shots",
]
