"""Topology generation for the goal net.

Particles, distance constraints, anchor constraints, and the 3 goalpost capsule
segments. Corner particles that geometrically coincide are merged
(``bug B`` from goal_net_warp_design.md §10.2).

`to_warp_arrays` flattens the topology into the dense ndarray layout required
by the Warp solver (§3.4).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from params import (
    GoalNetParams,
    PANEL_BACK,
    PANEL_LEFT,
    PANEL_RIGHT,
    PANEL_TOP,
    PANEL_NAMES,
)

Vec3 = Tuple[float, float, float]

_POS_QUANTIZATION = 1000.0  # round to nearest mm for dedup


@dataclass
class Particle:
    index: int
    position: Vec3
    panel: int  # first panel that introduced it
    u: int
    v: int
    anchored: bool


@dataclass
class DistanceConstraint:
    index: int
    i0: int
    i1: int
    rest_length: float
    stiffness: float
    kind: int  # 0=stretch, 1=bend
    panel: int


@dataclass
class AnchorConstraint:
    index: int
    particle: int
    target: Vec3
    stiffness: float
    hard: bool


@dataclass
class GoalpostSegment:
    index: int
    name: str
    p0: Vec3
    p1: Vec3
    radius: float
    kind: int  # 0=post, 1=crossbar


@dataclass
class SupportStay:
    """Physical support-stay rope between a net corner particle and a ground
    stake. The stake itself is a normal Particle (anchored, inverse_mass=0).
    The rope is a normal stretch DistanceConstraint. This struct records the
    indices so the viewer can render the stay specially and the user/tests
    can introspect it.
    """
    index: int
    name: str
    corner_particle: int  # particle index on the net (un-anchored)
    stake_particle: int   # particle index of the ground stake (anchored)
    constraint: int       # distance-constraint index of the stay rope
    radius: float = 0.012


@dataclass
class Topology:
    particles: List[Particle] = field(default_factory=list)
    distance_constraints: List[DistanceConstraint] = field(default_factory=list)
    anchor_constraints: List[AnchorConstraint] = field(default_factory=list)
    goalpost_segments: List[GoalpostSegment] = field(default_factory=list)
    support_stays: List[SupportStay] = field(default_factory=list)
    panel_particle_indices: Dict[int, List[int]] = field(default_factory=dict)
    stake_particle_indices: List[int] = field(default_factory=list)

    @property
    def num_particles(self) -> int:
        return len(self.particles)

    @property
    def num_constraints(self) -> int:
        return len(self.distance_constraints)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _quantize(pos: Vec3) -> Tuple[int, int, int]:
    return (
        int(round(pos[0] * _POS_QUANTIZATION)),
        int(round(pos[1] * _POS_QUANTIZATION)),
        int(round(pos[2] * _POS_QUANTIZATION)),
    )


class _PanelGrid:
    """Maps (panel, u, v) → particle index (after dedup)."""

    def __init__(self) -> None:
        self.by_uv: Dict[Tuple[int, int, int], int] = {}
        self.by_pos: Dict[Tuple[int, int, int], int] = {}

    def lookup(self, panel: int, u: int, v: int) -> Optional[int]:
        return self.by_uv.get((panel, u, v))


def _back_position(p: GoalNetParams, u: int, v: int, nx: int, ny: int) -> Vec3:
    W = p.goal.width
    H = p.goal.height
    x = -W / 2 + (u / nx) * W if nx > 0 else 0.0
    y = (v / ny) * H if ny > 0 else 0.0
    # Two bulges in -z direction, both vanish at all four edges so back-panel
    # seams with side/top/floor stay welded:
    # - back_slope: small bottom-middle "tilt-back" near the floor
    # - back_pocket_depth: bigger 2D bulge representing the support stays
    #   pulling the net's centre outward into a pocket shape
    frac_top = (v / ny) if ny > 0 else 1.0
    frac_u = (u / nx) if nx > 0 else 0.5
    bulge_u = 4.0 * frac_u * (1.0 - frac_u)
    bulge_v = 4.0 * frac_top * (1.0 - frac_top)
    slope_bulge = p.shape.back_slope * (1.0 - frac_top) ** 2 * bulge_u
    pocket_bulge = p.shape.back_pocket_depth * bulge_u * bulge_v
    z = -p.goal.depth - slope_bulge - pocket_bulge
    return (x, y, z)


def _left_position(p: GoalNetParams, u: int, v: int, nz: int, ny: int) -> Vec3:
    W = p.goal.width
    H = p.goal.height
    D = p.goal.depth
    z = -(u / nz) * D if nz > 0 else 0.0
    y = (v / ny) * H if ny > 0 else 0.0
    # parabolic outward bulge: 0 at u=0 and u=nz so corners coincide with
    # the back panel / posts.
    t = (u / nz) if nz > 0 else 0.0
    bulge = 4.0 * t * (1.0 - t) * p.shape.side_slope
    x = -W / 2 - bulge
    return (x, y, z)


def _right_position(p: GoalNetParams, u: int, v: int, nz: int, ny: int) -> Vec3:
    W = p.goal.width
    H = p.goal.height
    D = p.goal.depth
    z = -(u / nz) * D if nz > 0 else 0.0
    y = (v / ny) * H if ny > 0 else 0.0
    t = (u / nz) if nz > 0 else 0.0
    bulge = 4.0 * t * (1.0 - t) * p.shape.side_slope
    x = W / 2 + bulge
    return (x, y, z)


def _top_position(p: GoalNetParams, u: int, v: int, nx: int, nz: int) -> Vec3:
    W = p.goal.width
    H = p.goal.height
    D = p.goal.depth
    x = -W / 2 + (u / nx) * W if nx > 0 else 0.0
    z = -(v / nz) * D if nz > 0 else 0.0
    t = (v / nz) if nz > 0 else 0.0
    sag = 4.0 * t * (1.0 - t) * p.shape.top_sag
    y = H - sag
    return (x, y, z)


def _is_anchored(panel: int, u: int, v: int, nx_or_nz: int, ny_or_nz: int) -> bool:
    """Per §3.2 + bottom-edge ground anchoring (fix A: net bottom stays put):

    | back       | iy == ny or ix in {0, nx} or iy == 0 |
    | left/right | iy == ny or iz == 0 or iy == 0       |
    | top        | iz == 0 or ix in {0, nx}             |
    """
    if panel == PANEL_BACK:
        nx, ny = nx_or_nz, ny_or_nz
        return v == ny or u == 0 or u == nx or v == 0
    if panel in (PANEL_LEFT, PANEL_RIGHT):
        nz, ny = nx_or_nz, ny_or_nz
        return v == ny or u == 0 or v == 0
    if panel == PANEL_TOP:
        nx, nz = nx_or_nz, ny_or_nz
        return v == 0 or u == 0 or u == nx
    raise ValueError(f"unknown panel: {panel}")


def _add_panel(
    topo: Topology,
    grid: _PanelGrid,
    panel: int,
    nu: int,
    nv: int,
    position_fn,
) -> None:
    panel_indices: List[int] = []
    for v in range(nv + 1):
        for u in range(nu + 1):
            pos = position_fn(u, v)
            anchored = _is_anchored(panel, u, v, nu, nv)
            key = _quantize(pos)
            existing = grid.by_pos.get(key)
            if existing is None:
                idx = len(topo.particles)
                topo.particles.append(
                    Particle(
                        index=idx,
                        position=pos,
                        panel=panel,
                        u=u,
                        v=v,
                        anchored=anchored,
                    )
                )
                grid.by_pos[key] = idx
            else:
                idx = existing
                # promote to anchored if any contributor was anchored
                if anchored and not topo.particles[idx].anchored:
                    p = topo.particles[idx]
                    topo.particles[idx] = Particle(
                        index=p.index,
                        position=p.position,
                        panel=p.panel,
                        u=p.u,
                        v=p.v,
                        anchored=True,
                    )
            grid.by_uv[(panel, u, v)] = idx
            panel_indices.append(idx)
    topo.panel_particle_indices[panel] = panel_indices


def _add_distance(
    topo: Topology,
    panel: int,
    i0: int,
    i1: int,
    stiffness: float,
    kind: int,
    seen: set,
) -> None:
    if i0 == i1:
        return
    a, b = (i0, i1) if i0 < i1 else (i1, i0)
    if (a, b, kind) in seen:
        return
    seen.add((a, b, kind))
    p0 = topo.particles[a].position
    p1 = topo.particles[b].position
    rest = math.sqrt(
        (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2 + (p1[2] - p0[2]) ** 2
    )
    topo.distance_constraints.append(
        DistanceConstraint(
            index=len(topo.distance_constraints),
            i0=a,
            i1=b,
            rest_length=rest,
            stiffness=stiffness,
            kind=kind,
            panel=panel,
        )
    )


def _add_panel_constraints(
    topo: Topology,
    grid: _PanelGrid,
    panel: int,
    nu: int,
    nv: int,
    stretch_stiff: float,
    bend_stiff: float,
    seen: set,
) -> None:
    # stretch: 4-neighbour
    for v in range(nv + 1):
        for u in range(nu + 1):
            i = grid.lookup(panel, u, v)
            if i is None:
                continue
            if u + 1 <= nu:
                j = grid.lookup(panel, u + 1, v)
                if j is not None:
                    _add_distance(topo, panel, i, j, stretch_stiff, 0, seen)
            if v + 1 <= nv:
                j = grid.lookup(panel, u, v + 1)
                if j is not None:
                    _add_distance(topo, panel, i, j, stretch_stiff, 0, seen)
    # bend: 2-step neighbours
    for v in range(nv + 1):
        for u in range(nu + 1):
            i = grid.lookup(panel, u, v)
            if i is None:
                continue
            if u + 2 <= nu:
                j = grid.lookup(panel, u + 2, v)
                if j is not None:
                    _add_distance(topo, panel, i, j, bend_stiff, 1, seen)
            if v + 2 <= nv:
                j = grid.lookup(panel, u, v + 2)
                if j is not None:
                    _add_distance(topo, panel, i, j, bend_stiff, 1, seen)


def _add_anchor_constraints(topo: Topology, params: GoalNetParams) -> None:
    anchor = params.anchor
    seen = set()
    for p in topo.particles:
        if not p.anchored or p.index in seen:
            continue
        seen.add(p.index)
        topo.anchor_constraints.append(
            AnchorConstraint(
                index=len(topo.anchor_constraints),
                particle=p.index,
                target=p.position,
                stiffness=anchor.stiffness,
                hard=anchor.hard,
            )
        )


def _make_goalpost_segments(params: GoalNetParams) -> List[GoalpostSegment]:
    W = params.goal.width
    H = params.goal.height
    r = params.goalpost.radius
    return [
        GoalpostSegment(
            index=0,
            name="post_left",
            p0=(W / 2 + r, 0.0, 0.0),
            p1=(W / 2 + r, H + r, 0.0),
            radius=r,
            kind=0,
        ),
        GoalpostSegment(
            index=1,
            name="post_right",
            p0=(-W / 2 - r, 0.0, 0.0),
            p1=(-W / 2 - r, H + r, 0.0),
            radius=r,
            kind=0,
        ),
        GoalpostSegment(
            index=2,
            name="crossbar",
            p0=(-W / 2, H + r, 0.0),
            p1=(W / 2, H + r, 0.0),
            radius=r,
            kind=1,
        ),
    ]


def _add_support_stays(
    topo: Topology, params: GoalNetParams, grid: _PanelGrid, nx: int, ny: int, nz: int
) -> None:
    """Make the back-top corners (and optionally back-bottom) physically hang
    from ground stakes via stretch ropes. Each stay:

      1. Picks a net corner particle (already created during _add_panel).
      2. Marks that corner as NOT anchored (so it can swing).
      3. Adds a new "stake" Particle at the ground stake position, with
         inverse_mass=0 (anchored hard).
      4. Adds a stretch DistanceConstraint between corner and stake. The rest
         length is the geometric distance, so the stay is taut at the rest
         configuration.

    If no stays are configured, the back-top corners stay anchored to their
    rest positions and the function is a no-op.
    """
    shape = params.shape
    if shape.stay_count <= 0:
        return
    W = params.goal.width
    D = params.goal.depth
    stretch_stiff = params.rope.stretch_stiffness

    # Corner descriptions: (name, panel_lookup_args, x_sign)
    corners: List[Tuple[str, int, int, int, float]] = [
        ("stay_back_top_left", PANEL_BACK, 0, ny, -1.0),
        ("stay_back_top_right", PANEL_BACK, nx, ny, +1.0),
    ]
    if shape.stay_count >= 4:
        corners += [
            ("stay_back_bottom_left", PANEL_BACK, 0, 0, -1.0),
            ("stay_back_bottom_right", PANEL_BACK, nx, 0, +1.0),
        ]

    for stay_idx, (name, panel, u, v, x_sign) in enumerate(corners):
        corner_idx = grid.lookup(panel, u, v)
        if corner_idx is None:
            continue
        corner = topo.particles[corner_idx]
        # 2) un-anchor the corner so the stay rope is what holds it up
        if corner.anchored:
            topo.particles[corner_idx] = Particle(
                index=corner.index,
                position=corner.position,
                panel=corner.panel,
                u=corner.u,
                v=corner.v,
                anchored=False,
            )

        # 3) add the elevated stake anchor (up-and-back from the corner, like
        # the eyelet on a real goal's upper rear bar)
        corner_pos_for_offset = corner.position
        stake_pos: Vec3 = (
            x_sign * (W / 2 + shape.stay_anchor_offset_x),
            corner_pos_for_offset[1] + shape.stay_anchor_offset_y,
            -D - shape.stay_anchor_offset_z,
        )
        stake_idx = len(topo.particles)
        topo.particles.append(
            Particle(
                index=stake_idx,
                position=stake_pos,
                panel=PANEL_BACK,  # any valid panel; stake doesn't really collide
                u=-1,
                v=-1,
                anchored=True,
            )
        )
        topo.stake_particle_indices.append(stake_idx)

        # 4) add the rope as a normal stretch distance constraint
        corner_pos = topo.particles[corner_idx].position
        rest_length = math.sqrt(
            (stake_pos[0] - corner_pos[0]) ** 2
            + (stake_pos[1] - corner_pos[1]) ** 2
            + (stake_pos[2] - corner_pos[2]) ** 2
        )
        constraint_idx = len(topo.distance_constraints)
        topo.distance_constraints.append(
            DistanceConstraint(
                index=constraint_idx,
                i0=corner_idx,
                i1=stake_idx,
                rest_length=rest_length,
                stiffness=stretch_stiff,
                kind=0,
                panel=PANEL_BACK,
            )
        )

        topo.support_stays.append(
            SupportStay(
                index=stay_idx,
                name=name,
                corner_particle=corner_idx,
                stake_particle=stake_idx,
                constraint=constraint_idx,
            )
        )


def generate_topology(params: GoalNetParams) -> Topology:
    nx = max(1, round(params.goal.width / params.grid.cell_size_x))
    ny = max(1, round(params.goal.height / params.grid.cell_size_y))
    nz = max(1, round(params.goal.depth / params.grid.cell_size_z))

    topo = Topology()
    grid = _PanelGrid()

    _add_panel(topo, grid, PANEL_BACK, nx, ny, lambda u, v: _back_position(params, u, v, nx, ny))
    _add_panel(topo, grid, PANEL_LEFT, nz, ny, lambda u, v: _left_position(params, u, v, nz, ny))
    _add_panel(topo, grid, PANEL_RIGHT, nz, ny, lambda u, v: _right_position(params, u, v, nz, ny))
    _add_panel(topo, grid, PANEL_TOP, nx, nz, lambda u, v: _top_position(params, u, v, nx, nz))

    seen: set = set()
    _add_panel_constraints(topo, grid, PANEL_BACK, nx, ny, params.rope.stretch_stiffness, params.rope.bend_stiffness, seen)
    _add_panel_constraints(topo, grid, PANEL_LEFT, nz, ny, params.rope.stretch_stiffness, params.rope.bend_stiffness, seen)
    _add_panel_constraints(topo, grid, PANEL_RIGHT, nz, ny, params.rope.stretch_stiffness, params.rope.bend_stiffness, seen)
    _add_panel_constraints(topo, grid, PANEL_TOP, nx, nz, params.rope.stretch_stiffness, params.rope.bend_stiffness, seen)

    # Stays must be added BEFORE anchor constraints — they un-anchor the
    # back-top corner particles and add the ground-stake particles that get
    # anchored instead.
    _add_support_stays(topo, params, grid, nx, ny, nz)
    _add_anchor_constraints(topo, params)
    topo.goalpost_segments = _make_goalpost_segments(params)
    return topo


# ---------------------------------------------------------------------------
# Stable signature (deterministic hash) for test_topology
# ---------------------------------------------------------------------------


def stable_signature(topo: Topology) -> str:
    hasher = hashlib.sha256()

    def upd(*items):
        hasher.update(("|".join(repr(x) for x in items) + "\n").encode())

    for p in topo.particles:
        upd("P", p.index, p.position, p.panel, p.u, p.v, p.anchored)
    for c in topo.distance_constraints:
        upd("D", c.index, c.i0, c.i1, c.rest_length, c.stiffness, c.kind, c.panel)
    for a in topo.anchor_constraints:
        upd("A", a.index, a.particle, a.target, a.stiffness, a.hard)
    for g in topo.goalpost_segments:
        upd("G", g.index, g.name, g.p0, g.p1, g.radius, g.kind)
    for s in topo.support_stays:
        upd("S", s.index, s.name, s.corner_particle, s.stake_particle, s.constraint)
    for panel, idxs in sorted(topo.panel_particle_indices.items()):
        upd("PI", panel, tuple(idxs))
    return hasher.hexdigest()


def summary(topo: Topology) -> Dict[str, int]:
    stretch = sum(1 for c in topo.distance_constraints if c.kind == 0)
    bend = sum(1 for c in topo.distance_constraints if c.kind == 1)
    anchored = sum(1 for p in topo.particles if p.anchored)
    return {
        "particles": len(topo.particles),
        "anchored_particles": anchored,
        "distance_constraints": len(topo.distance_constraints),
        "stretch_constraints": stretch,
        "bend_constraints": bend,
        "anchor_constraints": len(topo.anchor_constraints),
        "goalpost_segments": len(topo.goalpost_segments),
        "support_stays": len(topo.support_stays),
    }


# ---------------------------------------------------------------------------
# Warp-friendly ndarray export (§3.4)
# ---------------------------------------------------------------------------


def to_warp_arrays(topo: Topology, params: GoalNetParams) -> Dict[str, np.ndarray]:
    rope = params.rope
    inv_mass_value = (
        0.0 if rope.particle_mass <= 0 else 1.0 / rope.particle_mass
    )

    N = len(topo.particles)
    particle_pos_init = np.zeros((N, 3), dtype=np.float32)
    particle_inv_mass = np.zeros((N,), dtype=np.float32)
    particle_panel_id = np.zeros((N,), dtype=np.int32)

    for p in topo.particles:
        particle_pos_init[p.index] = p.position
        particle_inv_mass[p.index] = 0.0 if p.anchored else inv_mass_value
        particle_panel_id[p.index] = p.panel

    M = len(topo.distance_constraints)
    constraint_i0 = np.zeros((M,), dtype=np.int32)
    constraint_i1 = np.zeros((M,), dtype=np.int32)
    constraint_rest = np.zeros((M,), dtype=np.float32)
    constraint_stiffness = np.zeros((M,), dtype=np.float32)
    constraint_panel_id = np.zeros((M,), dtype=np.int32)
    constraint_kind = np.zeros((M,), dtype=np.int32)

    for c in topo.distance_constraints:
        constraint_i0[c.index] = c.i0
        constraint_i1[c.index] = c.i1
        constraint_rest[c.index] = c.rest_length
        constraint_stiffness[c.index] = c.stiffness
        constraint_panel_id[c.index] = c.panel
        constraint_kind[c.index] = c.kind

    A = len(topo.anchor_constraints)
    anchor_particle = np.zeros((A,), dtype=np.int32)
    anchor_target = np.zeros((A, 3), dtype=np.float32)
    anchor_stiffness = np.zeros((A,), dtype=np.float32)
    anchor_hard = np.zeros((A,), dtype=np.int32)

    for a in topo.anchor_constraints:
        anchor_particle[a.index] = a.particle
        anchor_target[a.index] = a.target
        anchor_stiffness[a.index] = a.stiffness
        anchor_hard[a.index] = 1 if a.hard else 0

    P = len(topo.goalpost_segments)
    post_p0 = np.zeros((P, 3), dtype=np.float32)
    post_p1 = np.zeros((P, 3), dtype=np.float32)
    post_radius = np.zeros((P,), dtype=np.float32)
    post_kind = np.zeros((P,), dtype=np.int32)

    for g in topo.goalpost_segments:
        post_p0[g.index] = g.p0
        post_p1[g.index] = g.p1
        post_radius[g.index] = g.radius
        post_kind[g.index] = g.kind

    panel_restitution = np.array(
        [
            rope.panel_restitution_back,
            rope.panel_restitution_side,
            rope.panel_restitution_side,
            rope.panel_restitution_top,
        ],
        dtype=np.float32,
    )
    # Only the back panel has the extra tangential friction multiplier; others
    # use the default friction.
    panel_friction = np.array(
        [
            rope.panel_friction_back_tangent,
            rope.friction,
            rope.friction,
            rope.friction,
        ],
        dtype=np.float32,
    )

    return {
        "particle_pos_init": particle_pos_init,
        "particle_inv_mass": particle_inv_mass,
        "particle_panel_id": particle_panel_id,
        "constraint_i0": constraint_i0,
        "constraint_i1": constraint_i1,
        "constraint_rest": constraint_rest,
        "constraint_stiffness": constraint_stiffness,
        "constraint_panel_id": constraint_panel_id,
        "constraint_kind": constraint_kind,
        "anchor_particle": anchor_particle,
        "anchor_target": anchor_target,
        "anchor_stiffness": anchor_stiffness,
        "anchor_hard": anchor_hard,
        "post_p0": post_p0,
        "post_p1": post_p1,
        "post_radius": post_radius,
        "post_kind": post_kind,
        "panel_restitution": panel_restitution,
        "panel_friction": panel_friction,
    }


__all__ = [
    "Particle",
    "DistanceConstraint",
    "AnchorConstraint",
    "GoalpostSegment",
    "Topology",
    "generate_topology",
    "stable_signature",
    "summary",
    "to_warp_arrays",
]
