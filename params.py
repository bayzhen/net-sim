"""Parameter dataclasses for the goal-net XPBD dataset tool.

All defaults mirror Section 2 of goal_net_warp_design.md (units: meters, seconds, kg).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Tuple, List


Vec3 = Tuple[float, float, float]


@dataclass
class GoalSizeParams:
    width: float = 7.32
    height: float = 2.44
    depth: float = 2.0


@dataclass
class GridParams:
    cell_size_x: float = 0.305
    cell_size_y: float = 0.305
    cell_size_z: float = 0.305


@dataclass
class RopeParams:
    radius: float = 0.018
    collision_radius: float = 0.16
    particle_mass: float = 0.035
    stretch_stiffness: float = 0.92
    bend_stiffness: float = 0.18
    damping: float = 0.035
    friction: float = 0.32
    restitution: float = 0.55
    panel_restitution_back: float = 0.10
    panel_restitution_top: float = 0.30
    panel_restitution_side: float = 0.20
    panel_friction_back_tangent: float = 0.80
    impulse_clamp: float = 6.0
    # Skip swept-impulse injection into the rope when the ball is barely
    # moving — repeated low-speed contact (e.g. ball rolling against the net
    # bottom after a ground bounce) otherwise pumps energy every substep and
    # makes the net oscillate forever.
    impulse_speed_threshold: float = 1.0

    def effective_collision_radius(self) -> float:
        return self.collision_radius if self.collision_radius > 0 else self.radius


@dataclass
class AnchorParams:
    stiffness: float = 1.0
    soft_stiffness: float = 0.65
    hard: bool = True


@dataclass
class ShapeParams:
    top_sag: float = 0.16
    side_slope: float = 0.15
    back_slope: float = 0.05
    # Pocket: middle of back panel bulges further into -z direction, like a
    # real soccer-goal net pulled out by support stays. Peak at panel centre,
    # zero at all four edges (so seams with side/top/floor stay welded).
    back_pocket_depth: float = 0.30
    # Support-stay ropes lift the back-top corners of the net UP-AND-BACK
    # toward an elevated anchor point (like the eyelet on a real soccer
    # goal's upper rear bar). The net corner is *not* anchored to its rest
    # position — it hangs from the stay; the anchor at the other end is what
    # holds the whole back of the net up.
    stay_count: int = 2  # 0 / 2 / 4
    stay_anchor_offset_x: float = 0.3   # outward from ±W/2
    stay_anchor_offset_y: float = 0.6   # above corner.y (= H by default)
    stay_anchor_offset_z: float = 0.4   # behind goal-back (= -depth)


@dataclass
class SolverParams:
    frame_dt: float = 1.0 / 60.0
    substeps: int = 12
    iterations: int = 8
    duration: float = 2.0
    gravity: Vec3 = (0.0, -9.81, 0.0)
    sample_every_frames: int = 1
    enable_bend_constraints: bool = True
    stuck_speed_threshold: float = 0.5
    stuck_duration_seconds: float = 0.3


@dataclass
class CollisionParams:
    severe_penetration_threshold: float = 0.15
    safety_back_z: float = -2.25
    max_ball_speed: float = 80.0
    max_particle_speed: float = 60.0
    max_net_displacement: float = 4.0
    stuck_speed_threshold: float = 0.05
    stuck_duration: float = 0.25


@dataclass
class GroundParams:
    enabled: bool = True
    y: float = 0.0
    bounce_restitution: float = 0.8
    bounce_speed_loss: float = 12.0
    bounce_to_roll_vertical_threshold: float = 2.0
    bounce_to_roll_total_threshold: float = 3.0
    roll_speed_loss: float = 2.0
    bounce_floor_velocity_offset: float = 0.1


@dataclass
class GoalpostParams:
    enabled: bool = True
    radius: float = 0.06
    speed_change_factor: float = 0.6
    crossbar_z_min_speed: float = 0.1


@dataclass
class BallState:
    position: Vec3 = (0.0, 1.0, 8.0)
    velocity: Vec3 = (0.0, 5.0, -25.0)
    angular_velocity: Vec3 = (0.0, 0.0, 0.0)
    radius: float = 0.13
    mass: float = 1.0


@dataclass
class GoalNetParams:
    """Top-level container aggregating every parameter group."""

    goal: GoalSizeParams = field(default_factory=GoalSizeParams)
    grid: GridParams = field(default_factory=GridParams)
    rope: RopeParams = field(default_factory=RopeParams)
    anchor: AnchorParams = field(default_factory=AnchorParams)
    shape: ShapeParams = field(default_factory=ShapeParams)
    solver: SolverParams = field(default_factory=SolverParams)
    collision: CollisionParams = field(default_factory=CollisionParams)
    ground: GroundParams = field(default_factory=GroundParams)
    goalpost: GoalpostParams = field(default_factory=GoalpostParams)

    schema_version: str = "goal_net_params.v1"

    def to_dict(self) -> dict:
        return asdict(self)


PANEL_BACK = 0
PANEL_LEFT = 1
PANEL_RIGHT = 2
PANEL_TOP = 3
PANEL_NAMES: List[str] = ["back", "left", "right", "top"]


def panel_id(name: str) -> int:
    return PANEL_NAMES.index(name)


def panel_name(idx: int) -> str:
    return PANEL_NAMES[idx]


__all__ = [
    "GoalSizeParams",
    "GridParams",
    "RopeParams",
    "AnchorParams",
    "ShapeParams",
    "SolverParams",
    "CollisionParams",
    "GroundParams",
    "GoalpostParams",
    "BallState",
    "GoalNetParams",
    "PANEL_BACK",
    "PANEL_LEFT",
    "PANEL_RIGHT",
    "PANEL_TOP",
    "PANEL_NAMES",
    "panel_id",
    "panel_name",
]
