"""NVIDIA Warp implementation of the XPBD goal-net solver.

Implements every kernel listed in goal_net_warp_design.md §8.4 and the main
loop order from §4.1. Bug A (swept early-out reflection) and Bug B (corner
particle dedup) are fixed; Bug B is handled at topology level.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import warp as wp

from params import (
    BallState,
    GoalNetParams,
    PANEL_NAMES,
)
from topology import Topology, to_warp_arrays

# ---------------------------------------------------------------------------
# Contact type identifiers (must match output.py and viewer_rerun.py)
# ---------------------------------------------------------------------------
CT_PARTICLE = 0
CT_SEGMENT = 1
CT_SEGMENT_SWEPT = 2
CT_GOALPOST = 3
CT_CROSSBAR = 4
CT_GROUND_BOUNCE = 5
CT_GROUND_ROLL = 6

CONTACT_TYPE_NAMES = {
    CT_PARTICLE: "particle",
    CT_SEGMENT: "segment",
    CT_SEGMENT_SWEPT: "segment_swept",
    CT_GOALPOST: "goalpost",
    CT_CROSSBAR: "crossbar",
    CT_GROUND_BOUNCE: "ground_bounce",
    CT_GROUND_ROLL: "ground_roll",
}


_INIT_DONE = False


def _ensure_warp_initialized() -> None:
    global _INIT_DONE
    if not _INIT_DONE:
        wp.init()
        _INIT_DONE = True


# ---------------------------------------------------------------------------
# Warp helper functions
# ---------------------------------------------------------------------------


@wp.func
def closest_point_on_segment(p: wp.vec3, a: wp.vec3, b: wp.vec3):
    seg = b - a
    seg_len2 = wp.dot(seg, seg)
    if seg_len2 < 1.0e-12:
        return a, float(0.0)
    t = wp.dot(p - a, seg) / seg_len2
    if t < 0.0:
        t = 0.0
    if t > 1.0:
        t = 1.0
    return a + seg * t, t


@wp.func
def ray_sphere_first_hit(origin: wp.vec3, direction: wp.vec3, centre: wp.vec3, radius: float) -> float:
    m = origin - centre
    b = wp.dot(m, direction)
    c = wp.dot(m, m) - radius * radius
    if c > 0.0 and b > 0.0:
        return float(2.0)
    a = wp.dot(direction, direction)
    if a < 1.0e-12:
        return float(2.0)
    disc = b * b - a * c
    if disc < 0.0:
        return float(2.0)
    sd = wp.sqrt(disc)
    t = (-b - sd) / a
    if t < 0.0:
        t = (-b + sd) / a
    if t < 0.0 or t > 1.0:
        return float(2.0)
    return t


@wp.func
def swept_sphere_vs_segment_toi(
    ball_a: wp.vec3,
    ball_b: wp.vec3,
    radius: float,
    seg_p0: wp.vec3,
    seg_p1: wp.vec3,
) -> float:
    """Return TOI in [0,1] of the moving sphere vs static segment, or 2.0 for
    no hit. A returned TOI of 0.0 means the start point is already in
    penetration.
    """
    # Step 1: early out if start already penetrating
    closest_a, _ = closest_point_on_segment(ball_a, seg_p0, seg_p1)
    d_a = wp.length(ball_a - closest_a)
    if d_a <= radius:
        return float(0.0)

    d = ball_b - ball_a
    seg = seg_p1 - seg_p0
    m = ball_a - seg_p0
    seg_len2 = wp.dot(seg, seg)
    best_t = float(2.0)

    # Infinite-cylinder test (only if segment has length)
    if seg_len2 >= 1.0e-12:
        dd = wp.dot(d, d)
        d_dot_seg = wp.dot(d, seg)
        m_dot_d = wp.dot(m, d)
        m_dot_seg = wp.dot(m, seg)
        a_coeff = seg_len2 * dd - d_dot_seg * d_dot_seg
        b_coeff = seg_len2 * m_dot_d - d_dot_seg * m_dot_seg
        c_coeff = seg_len2 * (wp.dot(m, m) - radius * radius) - m_dot_seg * m_dot_seg
        if wp.abs(a_coeff) > 1.0e-12:
            disc = b_coeff * b_coeff - a_coeff * c_coeff
            if disc >= 0.0:
                sd = wp.sqrt(disc)
                t0 = (-b_coeff - sd) / a_coeff
                t1 = (-b_coeff + sd) / a_coeff
                if t0 >= 0.0 and t0 <= 1.0:
                    s = (m_dot_seg + t0 * d_dot_seg) / seg_len2
                    if s >= 0.0 and s <= 1.0:
                        if t0 < best_t:
                            best_t = t0
                if t1 >= 0.0 and t1 <= 1.0:
                    s = (m_dot_seg + t1 * d_dot_seg) / seg_len2
                    if s >= 0.0 and s <= 1.0:
                        if t1 < best_t:
                            best_t = t1

    # Endcap spheres
    t_cap0 = ray_sphere_first_hit(ball_a, d, seg_p0, radius)
    if t_cap0 < best_t:
        best_t = t_cap0
    t_cap1 = ray_sphere_first_hit(ball_a, d, seg_p1, radius)
    if t_cap1 < best_t:
        best_t = t_cap1
    return best_t


@wp.func
def reflect_velocity(v: wp.vec3, n: wp.vec3, e: float, f: float) -> wp.vec3:
    vn = wp.dot(v, n)
    if vn >= 0.0:
        return v
    v_normal = n * vn
    v_tangent = v - v_normal
    return v_normal * (-e) + v_tangent * (1.0 - f)


# Sentinel "no hit" TOI value (anything > 1.0 is treated as miss).
TOI_INF = wp.constant(float(2.0))


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@wp.kernel
def k_save_previous_positions(
    pos: wp.array2d(dtype=wp.vec3),
    prev: wp.array2d(dtype=wp.vec3),
    frozen: wp.array(dtype=int),
):
    b, i = wp.tid()
    if frozen[b] != 0:
        return
    prev[b, i] = pos[b, i]


@wp.kernel
def k_integrate_particles(
    pos: wp.array2d(dtype=wp.vec3),
    vel: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    gravity: wp.vec3,
    dt: float,
    frozen: wp.array(dtype=int),
):
    b, i = wp.tid()
    if frozen[b] != 0:
        return
    if inv_mass[i] <= 0.0:
        return
    new_v = vel[b, i] + gravity * dt
    vel[b, i] = new_v
    pos[b, i] = pos[b, i] + new_v * dt


@wp.kernel
def k_resolve_particle_ground(
    pos: wp.array2d(dtype=wp.vec3),
    prev: wp.array2d(dtype=wp.vec3),
    vel: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    ground_y: float,
    frozen: wp.array(dtype=int),
):
    b, i = wp.tid()
    if frozen[b] != 0:
        return
    if inv_mass[i] <= 0.0:
        return
    p = pos[b, i]
    if p[1] < ground_y:
        pos[b, i] = wp.vec3(p[0], ground_y, p[2])
        pp = prev[b, i]
        if pp[1] < ground_y:
            prev[b, i] = wp.vec3(pp[0], ground_y, pp[2])
        v = vel[b, i]
        if v[1] < 0.0:
            vel[b, i] = wp.vec3(v[0], 0.0, v[2])


@wp.kernel
def k_integrate_ball(
    ball_pos: wp.array(dtype=wp.vec3),
    ball_vel: wp.array(dtype=wp.vec3),
    ball_prev: wp.array(dtype=wp.vec3),
    gravity: wp.vec3,
    dt: float,
    frozen: wp.array(dtype=int),
):
    b = wp.tid()
    if frozen[b] != 0:
        return
    new_v = ball_vel[b] + gravity * dt
    ball_vel[b] = new_v
    ball_prev[b] = ball_pos[b]
    ball_pos[b] = ball_pos[b] + new_v * dt


# -- ball vs goalposts ----------------------------------------------------


IDX_SENTINEL = wp.constant(int(1 << 30))


@wp.kernel
def k_reset_best_toi(best_toi: wp.array(dtype=float), best_idx: wp.array(dtype=int)):
    b = wp.tid()
    best_toi[b] = TOI_INF
    best_idx[b] = IDX_SENTINEL


@wp.kernel
def k_swept_ball_vs_posts(
    ball_prev: wp.array(dtype=wp.vec3),
    ball_pos: wp.array(dtype=wp.vec3),
    ball_radius: float,
    post_p0: wp.array(dtype=wp.vec3),
    post_p1: wp.array(dtype=wp.vec3),
    post_radius: wp.array(dtype=float),
    best_toi: wp.array(dtype=float),
    frozen: wp.array(dtype=int),
):
    b, c = wp.tid()
    if frozen[b] != 0:
        return
    a = ball_prev[b]
    p = ball_pos[b]
    if wp.length(p - a) < 1.0e-8:
        return
    radius = ball_radius + post_radius[c]
    toi = swept_sphere_vs_segment_toi(a, p, radius, post_p0[c], post_p1[c])
    if toi >= 0.0 and toi <= 1.0:
        wp.atomic_min(best_toi, b, toi)


@wp.kernel
def k_swept_ball_vs_posts_argmin(
    ball_prev: wp.array(dtype=wp.vec3),
    ball_pos: wp.array(dtype=wp.vec3),
    ball_radius: float,
    post_p0: wp.array(dtype=wp.vec3),
    post_p1: wp.array(dtype=wp.vec3),
    post_radius: wp.array(dtype=float),
    best_toi: wp.array(dtype=float),
    best_idx: wp.array(dtype=int),
    frozen: wp.array(dtype=int),
):
    b, c = wp.tid()
    if frozen[b] != 0:
        return
    target = best_toi[b]
    if target >= TOI_INF:
        return
    a = ball_prev[b]
    p = ball_pos[b]
    if wp.length(p - a) < 1.0e-8:
        return
    radius = ball_radius + post_radius[c]
    toi = swept_sphere_vs_segment_toi(a, p, radius, post_p0[c], post_p1[c])
    if toi == target:
        wp.atomic_min(best_idx, b, c)


@wp.kernel
def k_apply_post_response(
    ball_pos: wp.array(dtype=wp.vec3),
    ball_vel: wp.array(dtype=wp.vec3),
    ball_prev: wp.array(dtype=wp.vec3),
    ball_radius: float,
    post_p0: wp.array(dtype=wp.vec3),
    post_p1: wp.array(dtype=wp.vec3),
    post_radius: wp.array(dtype=float),
    post_kind: wp.array(dtype=int),
    best_toi: wp.array(dtype=float),
    best_idx: wp.array(dtype=int),
    speed_change: float,
    crossbar_z_min_speed: float,
    sub_time: float,
    # contact buffers
    contact_count: wp.array(dtype=int),
    contact_time: wp.array2d(dtype=float),
    contact_type: wp.array2d(dtype=int),
    contact_obj: wp.array2d(dtype=int),
    contact_pos: wp.array2d(dtype=wp.vec3),
    contact_normal: wp.array2d(dtype=wp.vec3),
    contact_strength: wp.array2d(dtype=float),
    max_contacts: int,
    frozen: wp.array(dtype=int),
):
    b = wp.tid()
    if frozen[b] != 0:
        return
    toi = best_toi[b]
    if toi > 1.0:
        return
    idx = best_idx[b]
    if idx >= IDX_SENTINEL:
        return

    a = ball_prev[b]
    p = ball_pos[b]
    delta = p - a
    p0 = post_p0[idx]
    p1 = post_p1[idx]
    total_radius = ball_radius + post_radius[idx]

    contact_p = a + delta * toi
    closest, _ = closest_point_on_segment(contact_p, p0, p1)
    offset = contact_p - closest
    olen = wp.length(offset)
    v = ball_vel[b]
    vlen = wp.length(v)
    if olen <= 1.0e-8:
        if vlen > 1.0e-8:
            n = -v / vlen
        else:
            n = wp.vec3(0.0, 0.0, 1.0)
    else:
        n = offset / olen
        # Bug A defence: enforce n opposes incoming velocity if available
        if vlen > 1.0e-8 and wp.dot(n, v) > 0.0:
            n = -v / vlen
    ball_pos[b] = closest + n * (total_radius + 0.001)

    vn = wp.dot(v, n)
    va = n * vn
    new_v = (v - va) - va * speed_change
    is_crossbar = post_kind[idx] == 1
    if is_crossbar:
        vz = new_v[2]
        if wp.abs(vz) < crossbar_z_min_speed:
            sign = float(1.0)
            if vz < 0.0:
                sign = -1.0
            new_v = wp.vec3(new_v[0], new_v[1], sign * crossbar_z_min_speed)
    ball_vel[b] = new_v

    # write contact
    slot = wp.atomic_add(contact_count, b, 1)
    if slot < max_contacts:
        ctype = CT_GOALPOST
        if is_crossbar:
            ctype = CT_CROSSBAR
        contact_time[b, slot] = sub_time
        contact_type[b, slot] = ctype
        contact_obj[b, slot] = idx
        contact_pos[b, slot] = contact_p
        contact_normal[b, slot] = n
        contact_strength[b, slot] = wp.length(new_v - v)


# -- ball-vs-net-segment swept --------------------------------------------


@wp.kernel
def k_swept_ball_vs_segments(
    ball_prev: wp.array(dtype=wp.vec3),
    ball_pos: wp.array(dtype=wp.vec3),
    ball_radius: float,
    rope_collision_radius: float,
    prev_positions: wp.array2d(dtype=wp.vec3),
    c_i0: wp.array(dtype=int),
    c_i1: wp.array(dtype=int),
    best_toi: wp.array(dtype=float),
    frozen: wp.array(dtype=int),
):
    b, c = wp.tid()
    if frozen[b] != 0:
        return
    a = ball_prev[b]
    p = ball_pos[b]
    if wp.length(p - a) < 1.0e-8:
        return
    radius = ball_radius + rope_collision_radius
    p0 = prev_positions[b, c_i0[c]]
    p1 = prev_positions[b, c_i1[c]]
    toi = swept_sphere_vs_segment_toi(a, p, radius, p0, p1)
    if toi >= 0.0 and toi <= 1.0:
        wp.atomic_min(best_toi, b, toi)


@wp.kernel
def k_swept_ball_vs_segments_argmin(
    ball_prev: wp.array(dtype=wp.vec3),
    ball_pos: wp.array(dtype=wp.vec3),
    ball_radius: float,
    rope_collision_radius: float,
    prev_positions: wp.array2d(dtype=wp.vec3),
    c_i0: wp.array(dtype=int),
    c_i1: wp.array(dtype=int),
    best_toi: wp.array(dtype=float),
    best_idx: wp.array(dtype=int),
    frozen: wp.array(dtype=int),
):
    b, c = wp.tid()
    if frozen[b] != 0:
        return
    target = best_toi[b]
    if target >= TOI_INF:
        return
    a = ball_prev[b]
    p = ball_pos[b]
    if wp.length(p - a) < 1.0e-8:
        return
    radius = ball_radius + rope_collision_radius
    p0 = prev_positions[b, c_i0[c]]
    p1 = prev_positions[b, c_i1[c]]
    toi = swept_sphere_vs_segment_toi(a, p, radius, p0, p1)
    if toi == target:
        wp.atomic_min(best_idx, b, c)


@wp.kernel
def k_apply_segment_response(
    ball_pos: wp.array(dtype=wp.vec3),
    ball_vel: wp.array(dtype=wp.vec3),
    ball_prev: wp.array(dtype=wp.vec3),
    ball_radius: float,
    ball_mass: float,
    rope_collision_radius: float,
    prev_positions: wp.array2d(dtype=wp.vec3),
    positions: wp.array2d(dtype=wp.vec3),
    velocities: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    c_i0: wp.array(dtype=int),
    c_i1: wp.array(dtype=int),
    c_panel: wp.array(dtype=int),
    panel_restitution: wp.array(dtype=float),
    panel_friction: wp.array(dtype=float),
    impulse_clamp: float,
    best_toi: wp.array(dtype=float),
    best_idx: wp.array(dtype=int),
    sub_dt: float,
    sub_time: float,
    contact_count: wp.array(dtype=int),
    contact_time: wp.array2d(dtype=float),
    contact_type: wp.array2d(dtype=int),
    contact_obj: wp.array2d(dtype=int),
    contact_pos: wp.array2d(dtype=wp.vec3),
    contact_normal: wp.array2d(dtype=wp.vec3),
    contact_strength: wp.array2d(dtype=float),
    max_contacts: int,
    frozen: wp.array(dtype=int),
):
    b = wp.tid()
    if frozen[b] != 0:
        return
    toi = best_toi[b]
    if toi > 1.0:
        return
    c_idx = best_idx[b]
    if c_idx >= IDX_SENTINEL:
        return

    a = ball_prev[b]
    p = ball_pos[b]
    delta = p - a
    total_radius = ball_radius + rope_collision_radius
    p0 = prev_positions[b, c_i0[c_idx]]
    p1 = prev_positions[b, c_i1[c_idx]]

    contact_p = a + delta * toi
    closest, t = closest_point_on_segment(contact_p, p0, p1)
    offset = contact_p - closest
    olen = wp.length(offset)
    v_before = ball_vel[b]
    vlen = wp.length(v_before)

    if olen <= 1.0e-8:
        # Bug A: degenerate offset; pick a normal that opposes velocity.
        if vlen > 1.0e-8:
            n = -v_before / vlen
        else:
            n = wp.vec3(0.0, 1.0, 0.0)
    else:
        n = offset / olen
        # Bug A: if the geometric normal points into the velocity direction,
        # the early-out / start-penetrated path may have flipped it; correct.
        if vlen > 1.0e-8 and wp.dot(n, v_before) > 0.0:
            n = -v_before / vlen

    ball_pos[b] = closest + n * (total_radius + 0.001)

    panel = c_panel[c_idx]
    e = panel_restitution[panel]
    f = panel_friction[panel]
    new_v = reflect_velocity(v_before, n, e, f)
    ball_vel[b] = new_v

    # impulse injection to two endpoints
    dv_ball = new_v - v_before
    impulse_n = ball_mass * wp.dot(dv_ball, n)  # >0 means ball is pushed in +n
    if impulse_n > 0.0:
        w0 = float(1.0 - t)
        w1 = float(t)
        i0 = c_i0[c_idx]
        i1 = c_i1[c_idx]
        inv0 = inv_mass[i0]
        inv1 = inv_mass[i1]
        if inv0 > 0.0 and w0 > 0.0:
            dv_mag = impulse_n * w0 * inv0
            dv = -n * dv_mag
            dv_len = wp.length(dv)
            if dv_len > impulse_clamp:
                dv = dv * (impulse_clamp / dv_len)
            shift = dv * sub_dt
            wp.atomic_add(velocities, b, i0, dv)
            wp.atomic_add(positions, b, i0, shift)
            wp.atomic_add(prev_positions, b, i0, -shift)
        if inv1 > 0.0 and w1 > 0.0:
            dv_mag = impulse_n * w1 * inv1
            dv = -n * dv_mag
            dv_len = wp.length(dv)
            if dv_len > impulse_clamp:
                dv = dv * (impulse_clamp / dv_len)
            shift = dv * sub_dt
            wp.atomic_add(velocities, b, i1, dv)
            wp.atomic_add(positions, b, i1, shift)
            wp.atomic_add(prev_positions, b, i1, -shift)

    slot = wp.atomic_add(contact_count, b, 1)
    if slot < max_contacts:
        contact_time[b, slot] = sub_time
        contact_type[b, slot] = CT_SEGMENT_SWEPT
        contact_obj[b, slot] = c_idx
        contact_pos[b, slot] = contact_p
        contact_normal[b, slot] = n
        contact_strength[b, slot] = wp.length(dv_ball)


# -- ball ground bounce ----------------------------------------------------


@wp.kernel
def k_resolve_ball_ground(
    ball_pos: wp.array(dtype=wp.vec3),
    ball_vel: wp.array(dtype=wp.vec3),
    ball_ang: wp.array(dtype=wp.vec3),
    ground_state: wp.array(dtype=int),
    ball_radius: float,
    ground_y: float,
    bounce_rest: float,
    bounce_speed_loss: float,
    bounce_to_roll_vy: float,
    bounce_to_roll_total: float,
    roll_speed_loss: float,
    bounce_floor_offset: float,
    sub_dt: float,
    sub_time: float,
    contact_count: wp.array(dtype=int),
    contact_time: wp.array2d(dtype=float),
    contact_type: wp.array2d(dtype=int),
    contact_obj: wp.array2d(dtype=int),
    contact_pos: wp.array2d(dtype=wp.vec3),
    contact_normal: wp.array2d(dtype=wp.vec3),
    contact_strength: wp.array2d(dtype=float),
    max_contacts: int,
    frozen: wp.array(dtype=int),
):
    b = wp.tid()
    if frozen[b] != 0:
        return
    p = ball_pos[b]
    if p[1] - ball_radius >= ground_y:
        ground_state[b] = -1
        return
    old_v = ball_vel[b]
    old_speed = wp.length(old_v)
    old_state = ground_state[b]

    # 1) lift ball
    ball_pos[b] = wp.vec3(p[0], ground_y + ball_radius, p[2])

    # If the ball is already moving upward, just lift it and skip the bounce/
    # roll response — the formula `new_vy = max(0, -vy*e - offset)` assumes a
    # downward-moving ball and would erroneously zero out genuine lob shots
    # whose source is below the ground threshold.
    if old_v[1] >= 0.0:
        ground_state[b] = -1
        return

    # 2) vertical bounce
    new_vy = -old_v[1] * bounce_rest - bounce_floor_offset
    if new_vy < 0.0:
        new_vy = 0.0

    # 3) bounce vs roll
    is_bounce = new_vy > bounce_to_roll_vy and old_speed >= bounce_to_roll_total
    if is_bounce:
        new_v = wp.vec3(old_v[0], new_vy, old_v[2])
        nvl = wp.length(new_v)
        if nvl > 1.0e-8:
            scale = 1.0 - bounce_speed_loss * 9.8 * sub_dt / nvl
            if scale < 0.0:
                scale = 0.0
            new_v = new_v * scale
        ball_vel[b] = new_v
        ctype = CT_GROUND_BOUNCE
        ground_state[b] = 0
    else:
        new_v = wp.vec3(old_v[0], 0.0, old_v[2])
        nvl = wp.length(new_v)
        if nvl >= 0.01:
            scale = 1.0 - roll_speed_loss * sub_dt / nvl
            if scale < 0.0:
                scale = 0.0
            new_v = new_v * scale
        else:
            new_v = wp.vec3(0.0, 0.0, 0.0)
        ball_vel[b] = new_v
        ctype = CT_GROUND_ROLL
        ground_state[b] = 1

    # 4) angular damp
    if old_speed >= 0.01:
        ball_ang[b] = ball_ang[b] * (wp.length(ball_vel[b]) / old_speed)
    else:
        ball_ang[b] = wp.vec3(0.0, 0.0, 0.0)

    # 5) emit contact only on transition (every bounce; first roll only)
    emit = int(1)
    if not is_bounce and old_state == 1:
        emit = 0
    if emit != 0:
        slot = wp.atomic_add(contact_count, b, 1)
        if slot < max_contacts:
            contact_time[b, slot] = sub_time
            contact_type[b, slot] = ctype
            contact_obj[b, slot] = 0
            contact_pos[b, slot] = wp.vec3(ball_pos[b][0], ground_y, ball_pos[b][2])
            contact_normal[b, slot] = wp.vec3(0.0, 1.0, 0.0)
            contact_strength[b, slot] = wp.length(ball_vel[b] - old_v)


# -- anchors and constraints ----------------------------------------------


@wp.kernel
def k_solve_anchors(
    positions: wp.array2d(dtype=wp.vec3),
    velocities: wp.array2d(dtype=wp.vec3),
    anchor_particle: wp.array(dtype=int),
    anchor_target: wp.array(dtype=wp.vec3),
    anchor_stiffness: wp.array(dtype=float),
    anchor_hard: wp.array(dtype=int),
    frozen: wp.array(dtype=int),
):
    b, a = wp.tid()
    if frozen[b] != 0:
        return
    idx = anchor_particle[a]
    target = anchor_target[a]
    if anchor_hard[a] != 0:
        positions[b, idx] = target
        velocities[b, idx] = wp.vec3(0.0, 0.0, 0.0)
    else:
        s = anchor_stiffness[a]
        positions[b, idx] = positions[b, idx] + (target - positions[b, idx]) * s


@wp.kernel
def k_solve_distance_constraints(
    positions: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    c_i0: wp.array(dtype=int),
    c_i1: wp.array(dtype=int),
    c_rest: wp.array(dtype=float),
    c_stiff: wp.array(dtype=float),
    c_kind: wp.array(dtype=int),
    target_kind: int,
    frozen: wp.array(dtype=int),
):
    b, c = wp.tid()
    if frozen[b] != 0:
        return
    if c_kind[c] != target_kind:
        return
    i0 = c_i0[c]
    i1 = c_i1[c]
    p0 = positions[b, i0]
    p1 = positions[b, i1]
    delta = p1 - p0
    L = wp.length(delta)
    if L < 1.0e-8:
        return
    error = L - c_rest[c]
    w0 = inv_mass[i0]
    w1 = inv_mass[i1]
    ws = w0 + w1
    if ws <= 0.0:
        return
    direction = delta / L
    correction = error * c_stiff[c] / ws
    if w0 > 0.0:
        wp.atomic_add(positions, b, i0, direction * (correction * w0))
    if w1 > 0.0:
        wp.atomic_add(positions, b, i1, -direction * (correction * w1))


@wp.kernel
def k_solve_ball_particle_collisions(
    positions: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    particle_panel: wp.array(dtype=int),
    panel_restitution: wp.array(dtype=float),
    panel_friction: wp.array(dtype=float),
    ball_pos: wp.array(dtype=wp.vec3),
    ball_vel: wp.array(dtype=wp.vec3),
    ball_radius: float,
    ball_mass: float,
    rope_collision_radius: float,
    sub_time: float,
    contact_count: wp.array(dtype=int),
    contact_time: wp.array2d(dtype=float),
    contact_type: wp.array2d(dtype=int),
    contact_obj: wp.array2d(dtype=int),
    contact_pos: wp.array2d(dtype=wp.vec3),
    contact_normal: wp.array2d(dtype=wp.vec3),
    contact_strength: wp.array2d(dtype=float),
    max_contacts: int,
    frozen: wp.array(dtype=int),
):
    b, i = wp.tid()
    if frozen[b] != 0:
        return
    inv_p = inv_mass[i]
    p = positions[b, i]
    bp = ball_pos[b]
    delta = p - bp
    d = wp.length(delta)
    total_r = ball_radius + rope_collision_radius
    if d <= 1.0e-8 or d >= total_r:
        return
    n = delta / d
    pen = total_r - d
    inv_b = 1.0 / ball_mass
    denom = inv_p + inv_b
    if denom <= 0.0:
        return
    # Update particle position via atomic add (multiple particles could shift ball,
    # so accumulate ball shifts atomically too).
    if inv_p > 0.0:
        wp.atomic_add(positions, b, i, n * (pen * inv_p / denom))
    # ball shift back
    wp.atomic_add(ball_pos, b, -n * (pen * inv_b / denom))

    # velocity reflect (only one collision wins per iteration in practice; this
    # is racey, but matches CPU pattern of last-writer)
    e = panel_restitution[particle_panel[i]]
    f = panel_friction[particle_panel[i]]
    v_before = ball_vel[b]
    new_v = reflect_velocity(v_before, n, e, f)
    ball_vel[b] = new_v

    # Only record discrete-iteration contacts that represent meaningful events
    # (penetration > 1 cm). Sub-cm fixes happen every iteration as XPBD residual
    # cleanup and would bloat the contact log.
    if pen > 0.01:
        slot = wp.atomic_add(contact_count, b, 1)
        if slot < max_contacts:
            contact_time[b, slot] = sub_time
            contact_type[b, slot] = CT_PARTICLE
            contact_obj[b, slot] = i
            contact_pos[b, slot] = p
            contact_normal[b, slot] = n
            contact_strength[b, slot] = wp.length(new_v - v_before)


@wp.kernel
def k_solve_ball_segment_collisions(
    positions: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    c_i0: wp.array(dtype=int),
    c_i1: wp.array(dtype=int),
    c_panel: wp.array(dtype=int),
    panel_restitution: wp.array(dtype=float),
    panel_friction: wp.array(dtype=float),
    ball_pos: wp.array(dtype=wp.vec3),
    ball_vel: wp.array(dtype=wp.vec3),
    ball_radius: float,
    ball_mass: float,
    rope_collision_radius: float,
    sub_time: float,
    contact_count: wp.array(dtype=int),
    contact_time: wp.array2d(dtype=float),
    contact_type: wp.array2d(dtype=int),
    contact_obj: wp.array2d(dtype=int),
    contact_pos: wp.array2d(dtype=wp.vec3),
    contact_normal: wp.array2d(dtype=wp.vec3),
    contact_strength: wp.array2d(dtype=float),
    max_contacts: int,
    frozen: wp.array(dtype=int),
):
    b, c = wp.tid()
    if frozen[b] != 0:
        return
    i0 = c_i0[c]
    i1 = c_i1[c]
    p0 = positions[b, i0]
    p1 = positions[b, i1]
    bp = ball_pos[b]
    closest, t = closest_point_on_segment(bp, p0, p1)
    delta = closest - bp
    d = wp.length(delta)
    total_r = ball_radius + rope_collision_radius
    if d >= total_r or d <= 1.0e-8:
        return
    n = delta / d
    pen = total_r - d
    w0 = inv_mass[i0] * (1.0 - t)
    w1 = inv_mass[i1] * t
    wb = 1.0 / ball_mass
    total = w0 + w1 + wb
    if total <= 0.0:
        return
    if w0 > 0.0:
        wp.atomic_add(positions, b, i0, n * (pen * w0 / total))
    if w1 > 0.0:
        wp.atomic_add(positions, b, i1, n * (pen * w1 / total))
    wp.atomic_add(ball_pos, b, -n * (pen * wb / total))

    e = panel_restitution[c_panel[c]]
    f = panel_friction[c_panel[c]]
    v_before = ball_vel[b]
    new_v = reflect_velocity(v_before, n, e, f)
    ball_vel[b] = new_v

    if pen > 0.01:
        slot = wp.atomic_add(contact_count, b, 1)
        if slot < max_contacts:
            contact_time[b, slot] = sub_time
            contact_type[b, slot] = CT_SEGMENT
            contact_obj[b, slot] = c
            contact_pos[b, slot] = closest
            contact_normal[b, slot] = -n  # outward-from-rope into ball
            contact_strength[b, slot] = wp.length(new_v - v_before)


@wp.kernel
def k_update_velocities_and_damp(
    positions: wp.array2d(dtype=wp.vec3),
    prev_positions: wp.array2d(dtype=wp.vec3),
    velocities: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    sub_dt: float,
    damping: float,
    frozen: wp.array(dtype=int),
):
    b, i = wp.tid()
    if frozen[b] != 0:
        return
    if inv_mass[i] <= 0.0:
        return
    v = (positions[b, i] - prev_positions[b, i]) / sub_dt
    factor = 1.0 - damping
    if factor < 0.0:
        factor = 0.0
    velocities[b, i] = v * factor


@wp.kernel
def k_record_frame(
    ball_pos: wp.array(dtype=wp.vec3),
    ball_vel: wp.array(dtype=wp.vec3),
    positions: wp.array2d(dtype=wp.vec3),
    frame_ball_pos: wp.array2d(dtype=wp.vec3),
    frame_ball_vel: wp.array2d(dtype=wp.vec3),
    frame_particles: wp.array3d(dtype=wp.vec3),
    f: int,
    record_particles: int,
):
    b, i = wp.tid()
    if i == 0:
        frame_ball_pos[b, f] = ball_pos[b]
        frame_ball_vel[b, f] = ball_vel[b]
    if record_particles != 0:
        frame_particles[b, f, i] = positions[b, i]


@wp.kernel
def k_update_substep_stats(
    ball_pos: wp.array(dtype=wp.vec3),
    ball_vel: wp.array(dtype=wp.vec3),
    safety_back_z: float,
    safety_threshold: float,
    sub_dt: float,
    sub_time: float,
    contact_count: wp.array(dtype=int),
    slow_speed_threshold: float,
    slow_duration: float,
    stuck_speed_threshold: float,
    max_pen: wp.array(dtype=float),
    max_pen_time: wp.array(dtype=float),
    slow_time: wp.array(dtype=float),
    frozen: wp.array(dtype=int),
    contact_started: wp.array(dtype=int),
    stuck_time: wp.array(dtype=float),
):
    b = wp.tid()
    if frozen[b] != 0:
        return
    pen = safety_back_z - ball_pos[b][2]
    if pen > max_pen[b]:
        max_pen[b] = pen
        max_pen_time[b] = sub_time

    speed = wp.length(ball_vel[b])
    if speed <= slow_speed_threshold:
        slow_time[b] = slow_time[b] + sub_dt
        if slow_time[b] >= slow_duration:
            frozen[b] = 1
    else:
        slow_time[b] = 0.0

    if contact_count[b] > 0:
        contact_started[b] = 1
    if contact_started[b] != 0:
        if speed < stuck_speed_threshold:
            stuck_time[b] = stuck_time[b] + sub_dt
        else:
            stuck_time[b] = 0.0


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ContactEvent:
    time: float
    object_type: str
    object_index: int
    position: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    strength: float


@dataclass
class FrameSample:
    time: float
    ball_position: Tuple[float, float, float]
    ball_velocity: Tuple[float, float, float]
    particle_positions: Optional[np.ndarray]  # (N, 3) or None


@dataclass
class QualityReport:
    clean: bool
    issues: List[str]
    target_hit: bool
    max_penetration_depth: float
    max_penetration_time: float


@dataclass
class StatsReport:
    frame_dt: float
    substeps: int
    iterations: int
    duration: float
    frame_count: int
    contact_count: int
    max_constraint_error: float
    max_net_displacement: float
    ball_came_to_rest: bool


@dataclass
class SimulationResult:
    sample_id: str
    frames: List[FrameSample]
    contacts: List[ContactEvent]
    quality: QualityReport
    stats: StatsReport


# ---------------------------------------------------------------------------
# Driver class
# ---------------------------------------------------------------------------


class XpbdWarpSolver:
    def __init__(
        self,
        params: GoalNetParams,
        topology: Topology,
        batch_size: int,
        device: str = "cuda",
        record_particles: bool = False,
        max_contacts: int = 16384,
    ) -> None:
        _ensure_warp_initialized()
        self.params = params
        self.topology = topology
        self.B = batch_size
        self.device = device
        self.record_particles = record_particles
        self.max_contacts = max_contacts

        arrs = to_warp_arrays(topology, params)
        self.N = arrs["particle_pos_init"].shape[0]
        self.M = arrs["constraint_i0"].shape[0]
        self.A = arrs["anchor_particle"].shape[0]
        self.P = arrs["post_p0"].shape[0]

        d = device

        # static topology arrays (1 copy, shared across batches)
        self.inv_mass_wp = wp.array(arrs["particle_inv_mass"], dtype=float, device=d)
        self.particle_panel_wp = wp.array(arrs["particle_panel_id"], dtype=int, device=d)
        self.rest_pos_wp = wp.array(arrs["particle_pos_init"], dtype=wp.vec3, device=d)

        self.c_i0_wp = wp.array(arrs["constraint_i0"], dtype=int, device=d)
        self.c_i1_wp = wp.array(arrs["constraint_i1"], dtype=int, device=d)
        self.c_rest_wp = wp.array(arrs["constraint_rest"], dtype=float, device=d)
        self.c_stiff_wp = wp.array(arrs["constraint_stiffness"], dtype=float, device=d)
        self.c_panel_wp = wp.array(arrs["constraint_panel_id"], dtype=int, device=d)
        self.c_kind_wp = wp.array(arrs["constraint_kind"], dtype=int, device=d)

        self.anchor_particle_wp = wp.array(arrs["anchor_particle"], dtype=int, device=d)
        self.anchor_target_wp = wp.array(arrs["anchor_target"], dtype=wp.vec3, device=d)
        self.anchor_stiff_wp = wp.array(arrs["anchor_stiffness"], dtype=float, device=d)
        self.anchor_hard_wp = wp.array(arrs["anchor_hard"], dtype=int, device=d)

        self.post_p0_wp = wp.array(arrs["post_p0"], dtype=wp.vec3, device=d)
        self.post_p1_wp = wp.array(arrs["post_p1"], dtype=wp.vec3, device=d)
        self.post_radius_wp = wp.array(arrs["post_radius"], dtype=float, device=d)
        self.post_kind_wp = wp.array(arrs["post_kind"], dtype=int, device=d)

        self.panel_restitution_wp = wp.array(arrs["panel_restitution"], dtype=float, device=d)
        self.panel_friction_wp = wp.array(arrs["panel_friction"], dtype=float, device=d)

        # frame layout
        solver = params.solver
        self.total_frames = int(round(solver.duration / solver.frame_dt))
        self.frame_count = self.total_frames + 1  # inclusive end

        # per-batch dynamic state allocated lazily in simulate()
        self.pos_wp: Optional[wp.array] = None
        self.prev_wp: Optional[wp.array] = None
        self.vel_wp: Optional[wp.array] = None
        self.ball_pos_wp: Optional[wp.array] = None
        self.ball_vel_wp: Optional[wp.array] = None
        self.ball_prev_wp: Optional[wp.array] = None
        self.ball_ang_wp: Optional[wp.array] = None

    # -----------------------------------------------------------------
    # public API
    # -----------------------------------------------------------------

    def simulate(self, balls: List[BallState], sample_ids: List[str]) -> List[SimulationResult]:
        B = len(balls)
        if B != self.B:
            raise ValueError(f"expected {self.B} balls, got {B}")

        d = self.device
        # allocate per-batch arrays
        init_pos = np.broadcast_to(self.rest_pos_wp.numpy(), (B, self.N, 3)).copy()
        zero_vec = np.zeros((B, self.N, 3), dtype=np.float32)
        self.pos_wp = wp.array(init_pos, dtype=wp.vec3, device=d)
        self.prev_wp = wp.array(init_pos, dtype=wp.vec3, device=d)
        self.vel_wp = wp.array(zero_vec, dtype=wp.vec3, device=d)

        ball_pos_np = np.array([b.position for b in balls], dtype=np.float32)
        ball_vel_np = np.array([b.velocity for b in balls], dtype=np.float32)
        ball_ang_np = np.array([b.angular_velocity for b in balls], dtype=np.float32)
        ball_radius_np = np.array([b.radius for b in balls], dtype=np.float32)
        ball_mass_np = np.array([b.mass for b in balls], dtype=np.float32)

        # In this design batches share a single ball radius/mass; if they differ
        # we'd need per-batch parametrisation. We pick the first sample's value.
        ball_radius = float(ball_radius_np[0])
        ball_mass = float(ball_mass_np[0])
        if not np.allclose(ball_radius_np, ball_radius) or not np.allclose(ball_mass_np, ball_mass):
            raise NotImplementedError("per-batch ball radius/mass not yet supported")

        self.ball_pos_wp = wp.array(ball_pos_np, dtype=wp.vec3, device=d)
        self.ball_vel_wp = wp.array(ball_vel_np, dtype=wp.vec3, device=d)
        self.ball_prev_wp = wp.array(ball_pos_np, dtype=wp.vec3, device=d)
        self.ball_ang_wp = wp.array(ball_ang_np, dtype=wp.vec3, device=d)

        frozen_wp = wp.zeros(B, dtype=int, device=d)
        slow_time_wp = wp.zeros(B, dtype=float, device=d)
        stuck_time_wp = wp.zeros(B, dtype=float, device=d)
        contact_started_wp = wp.zeros(B, dtype=int, device=d)
        max_pen_wp = wp.zeros(B, dtype=float, device=d)
        max_pen_time_wp = wp.zeros(B, dtype=float, device=d)

        contact_count_wp = wp.zeros(B, dtype=int, device=d)
        contact_time_wp = wp.zeros((B, self.max_contacts), dtype=float, device=d)
        contact_type_wp = wp.zeros((B, self.max_contacts), dtype=int, device=d)
        contact_obj_wp = wp.zeros((B, self.max_contacts), dtype=int, device=d)
        contact_pos_wp = wp.zeros((B, self.max_contacts), dtype=wp.vec3, device=d)
        contact_normal_wp = wp.zeros((B, self.max_contacts), dtype=wp.vec3, device=d)
        contact_strength_wp = wp.zeros((B, self.max_contacts), dtype=float, device=d)

        post_best_toi_wp = wp.zeros(B, dtype=float, device=d)
        post_best_idx_wp = wp.zeros(B, dtype=int, device=d)
        seg_best_toi_wp = wp.zeros(B, dtype=float, device=d)
        seg_best_idx_wp = wp.zeros(B, dtype=int, device=d)
        ground_state_wp = wp.full(shape=(B,), value=-1, dtype=int, device=d)

        F = self.frame_count
        frame_ball_pos_wp = wp.zeros((B, F), dtype=wp.vec3, device=d)
        frame_ball_vel_wp = wp.zeros((B, F), dtype=wp.vec3, device=d)
        if self.record_particles:
            frame_particles_wp = wp.zeros((B, F, self.N), dtype=wp.vec3, device=d)
        else:
            # dummy 1-slot array — kernel skips writes when record_particles==0
            frame_particles_wp = wp.zeros((B, 1, 1), dtype=wp.vec3, device=d)

        solver = self.params.solver
        gravity = wp.vec3(*solver.gravity)
        sub_dt = solver.frame_dt / solver.substeps
        damping = self.params.rope.damping
        ground_y = self.params.ground.y
        rope_collision = self.params.rope.effective_collision_radius()
        ground = self.params.ground
        post = self.params.goalpost
        collision = self.params.collision

        record_particles_int = 1 if self.record_particles else 0

        def launch_post_reset():
            wp.launch(k_reset_best_toi, dim=B, inputs=[post_best_toi_wp, post_best_idx_wp], device=d)

        def launch_seg_reset():
            wp.launch(k_reset_best_toi, dim=B, inputs=[seg_best_toi_wp, seg_best_idx_wp], device=d)

        # main loop
        for frame_index in range(F):
            if frame_index % solver.sample_every_frames == 0:
                wp.launch(
                    k_record_frame,
                    dim=(B, max(1, self.N)),
                    inputs=[
                        self.ball_pos_wp,
                        self.ball_vel_wp,
                        self.pos_wp,
                        frame_ball_pos_wp,
                        frame_ball_vel_wp,
                        frame_particles_wp,
                        frame_index,
                        record_particles_int,
                    ],
                    device=d,
                )
            if frame_index == self.total_frames:
                break

            for s in range(solver.substeps):
                sub_time = (frame_index * solver.substeps + s + 1) * sub_dt

                # save previous positions
                wp.launch(
                    k_save_previous_positions,
                    dim=(B, self.N),
                    inputs=[self.pos_wp, self.prev_wp, frozen_wp],
                    device=d,
                )
                # integrate particles
                wp.launch(
                    k_integrate_particles,
                    dim=(B, self.N),
                    inputs=[self.pos_wp, self.vel_wp, self.inv_mass_wp, gravity, sub_dt, frozen_wp],
                    device=d,
                )
                # ground for particles
                wp.launch(
                    k_resolve_particle_ground,
                    dim=(B, self.N),
                    inputs=[self.pos_wp, self.prev_wp, self.vel_wp, self.inv_mass_wp, ground_y, frozen_wp],
                    device=d,
                )
                # ball integrate
                wp.launch(
                    k_integrate_ball,
                    dim=B,
                    inputs=[self.ball_pos_wp, self.ball_vel_wp, self.ball_prev_wp, gravity, sub_dt, frozen_wp],
                    device=d,
                )
                # ball vs posts (must run before swept segments)
                launch_post_reset()
                wp.launch(
                    k_swept_ball_vs_posts,
                    dim=(B, self.P),
                    inputs=[
                        self.ball_prev_wp,
                        self.ball_pos_wp,
                        ball_radius,
                        self.post_p0_wp,
                        self.post_p1_wp,
                        self.post_radius_wp,
                        post_best_toi_wp,
                        frozen_wp,
                    ],
                    device=d,
                )
                wp.launch(
                    k_swept_ball_vs_posts_argmin,
                    dim=(B, self.P),
                    inputs=[
                        self.ball_prev_wp,
                        self.ball_pos_wp,
                        ball_radius,
                        self.post_p0_wp,
                        self.post_p1_wp,
                        self.post_radius_wp,
                        post_best_toi_wp,
                        post_best_idx_wp,
                        frozen_wp,
                    ],
                    device=d,
                )
                wp.launch(
                    k_apply_post_response,
                    dim=B,
                    inputs=[
                        self.ball_pos_wp,
                        self.ball_vel_wp,
                        self.ball_prev_wp,
                        ball_radius,
                        self.post_p0_wp,
                        self.post_p1_wp,
                        self.post_radius_wp,
                        self.post_kind_wp,
                        post_best_toi_wp,
                        post_best_idx_wp,
                        post.speed_change_factor,
                        post.crossbar_z_min_speed,
                        sub_time,
                        contact_count_wp,
                        contact_time_wp,
                        contact_type_wp,
                        contact_obj_wp,
                        contact_pos_wp,
                        contact_normal_wp,
                        contact_strength_wp,
                        self.max_contacts,
                        frozen_wp,
                    ],
                    device=d,
                )
                # ball-vs-net swept
                launch_seg_reset()
                wp.launch(
                    k_swept_ball_vs_segments,
                    dim=(B, self.M),
                    inputs=[
                        self.ball_prev_wp,
                        self.ball_pos_wp,
                        ball_radius,
                        rope_collision,
                        self.prev_wp,
                        self.c_i0_wp,
                        self.c_i1_wp,
                        seg_best_toi_wp,
                        frozen_wp,
                    ],
                    device=d,
                )
                wp.launch(
                    k_swept_ball_vs_segments_argmin,
                    dim=(B, self.M),
                    inputs=[
                        self.ball_prev_wp,
                        self.ball_pos_wp,
                        ball_radius,
                        rope_collision,
                        self.prev_wp,
                        self.c_i0_wp,
                        self.c_i1_wp,
                        seg_best_toi_wp,
                        seg_best_idx_wp,
                        frozen_wp,
                    ],
                    device=d,
                )
                wp.launch(
                    k_apply_segment_response,
                    dim=B,
                    inputs=[
                        self.ball_pos_wp,
                        self.ball_vel_wp,
                        self.ball_prev_wp,
                        ball_radius,
                        ball_mass,
                        rope_collision,
                        self.prev_wp,
                        self.pos_wp,
                        self.vel_wp,
                        self.inv_mass_wp,
                        self.c_i0_wp,
                        self.c_i1_wp,
                        self.c_panel_wp,
                        self.panel_restitution_wp,
                        self.panel_friction_wp,
                        self.params.rope.impulse_clamp,
                        seg_best_toi_wp,
                        seg_best_idx_wp,
                        sub_dt,
                        sub_time,
                        contact_count_wp,
                        contact_time_wp,
                        contact_type_wp,
                        contact_obj_wp,
                        contact_pos_wp,
                        contact_normal_wp,
                        contact_strength_wp,
                        self.max_contacts,
                        frozen_wp,
                    ],
                    device=d,
                )
                # ball ground
                wp.launch(
                    k_resolve_ball_ground,
                    dim=B,
                    inputs=[
                        self.ball_pos_wp,
                        self.ball_vel_wp,
                        self.ball_ang_wp,
                        ground_state_wp,
                        ball_radius,
                        ground.y,
                        ground.bounce_restitution,
                        ground.bounce_speed_loss,
                        ground.bounce_to_roll_vertical_threshold,
                        ground.bounce_to_roll_total_threshold,
                        ground.roll_speed_loss,
                        ground.bounce_floor_velocity_offset,
                        sub_dt,
                        sub_time,
                        contact_count_wp,
                        contact_time_wp,
                        contact_type_wp,
                        contact_obj_wp,
                        contact_pos_wp,
                        contact_normal_wp,
                        contact_strength_wp,
                        self.max_contacts,
                        frozen_wp,
                    ],
                    device=d,
                )

                # XPBD iterations
                for it in range(solver.iterations):
                    if self.A > 0:
                        wp.launch(
                            k_solve_anchors,
                            dim=(B, self.A),
                            inputs=[
                                self.pos_wp,
                                self.vel_wp,
                                self.anchor_particle_wp,
                                self.anchor_target_wp,
                                self.anchor_stiff_wp,
                                self.anchor_hard_wp,
                                frozen_wp,
                            ],
                            device=d,
                        )
                    wp.launch(
                        k_solve_distance_constraints,
                        dim=(B, self.M),
                        inputs=[
                            self.pos_wp,
                            self.inv_mass_wp,
                            self.c_i0_wp,
                            self.c_i1_wp,
                            self.c_rest_wp,
                            self.c_stiff_wp,
                            self.c_kind_wp,
                            0,  # stretch
                            frozen_wp,
                        ],
                        device=d,
                    )
                    if solver.enable_bend_constraints:
                        wp.launch(
                            k_solve_distance_constraints,
                            dim=(B, self.M),
                            inputs=[
                                self.pos_wp,
                                self.inv_mass_wp,
                                self.c_i0_wp,
                                self.c_i1_wp,
                                self.c_rest_wp,
                                self.c_stiff_wp,
                                self.c_kind_wp,
                                1,  # bend
                                frozen_wp,
                            ],
                            device=d,
                        )
                    wp.launch(
                        k_solve_ball_particle_collisions,
                        dim=(B, self.N),
                        inputs=[
                            self.pos_wp,
                            self.inv_mass_wp,
                            self.particle_panel_wp,
                            self.panel_restitution_wp,
                            self.panel_friction_wp,
                            self.ball_pos_wp,
                            self.ball_vel_wp,
                            ball_radius,
                            ball_mass,
                            rope_collision,
                            sub_time,
                            contact_count_wp,
                            contact_time_wp,
                            contact_type_wp,
                            contact_obj_wp,
                            contact_pos_wp,
                            contact_normal_wp,
                            contact_strength_wp,
                            self.max_contacts,
                            frozen_wp,
                        ],
                        device=d,
                    )
                    wp.launch(
                        k_solve_ball_segment_collisions,
                        dim=(B, self.M),
                        inputs=[
                            self.pos_wp,
                            self.inv_mass_wp,
                            self.c_i0_wp,
                            self.c_i1_wp,
                            self.c_panel_wp,
                            self.panel_restitution_wp,
                            self.panel_friction_wp,
                            self.ball_pos_wp,
                            self.ball_vel_wp,
                            ball_radius,
                            ball_mass,
                            rope_collision,
                            sub_time,
                            contact_count_wp,
                            contact_time_wp,
                            contact_type_wp,
                            contact_obj_wp,
                            contact_pos_wp,
                            contact_normal_wp,
                            contact_strength_wp,
                            self.max_contacts,
                            frozen_wp,
                        ],
                        device=d,
                    )

                # velocity update + damp
                wp.launch(
                    k_update_velocities_and_damp,
                    dim=(B, self.N),
                    inputs=[
                        self.pos_wp,
                        self.prev_wp,
                        self.vel_wp,
                        self.inv_mass_wp,
                        sub_dt,
                        damping,
                        frozen_wp,
                    ],
                    device=d,
                )

                wp.launch(
                    k_update_substep_stats,
                    dim=B,
                    inputs=[
                        self.ball_pos_wp,
                        self.ball_vel_wp,
                        collision.safety_back_z,
                        collision.severe_penetration_threshold,
                        sub_dt,
                        sub_time,
                        contact_count_wp,
                        solver.stuck_speed_threshold,
                        solver.stuck_duration_seconds,
                        collision.stuck_speed_threshold,
                        max_pen_wp,
                        max_pen_time_wp,
                        slow_time_wp,
                        frozen_wp,
                        contact_started_wp,
                        stuck_time_wp,
                    ],
                    device=d,
                )

        wp.synchronize_device(d)

        # ---- pull results back to host ----
        frame_ball_pos = frame_ball_pos_wp.numpy()
        frame_ball_vel = frame_ball_vel_wp.numpy()
        if self.record_particles:
            frame_particles = frame_particles_wp.numpy()
        else:
            frame_particles = None
        contact_counts = contact_count_wp.numpy()
        contact_times = contact_time_wp.numpy()
        contact_types = contact_type_wp.numpy()
        contact_objs = contact_obj_wp.numpy()
        contact_positions = contact_pos_wp.numpy()
        contact_normals = contact_normal_wp.numpy()
        contact_strengths = contact_strength_wp.numpy()
        max_pen = max_pen_wp.numpy()
        max_pen_time = max_pen_time_wp.numpy()
        stuck_time = stuck_time_wp.numpy()
        contact_started = contact_started_wp.numpy()
        final_particle_vel = self.vel_wp.numpy()
        final_positions = self.pos_wp.numpy()
        rest_positions = self.rest_pos_wp.numpy()

        return self._assemble_results(
            sample_ids=sample_ids,
            frame_ball_pos=frame_ball_pos,
            frame_ball_vel=frame_ball_vel,
            frame_particles=frame_particles,
            contact_counts=contact_counts,
            contact_times=contact_times,
            contact_types=contact_types,
            contact_objs=contact_objs,
            contact_positions=contact_positions,
            contact_normals=contact_normals,
            contact_strengths=contact_strengths,
            max_pen=max_pen,
            max_pen_time=max_pen_time,
            stuck_time=stuck_time,
            contact_started=contact_started,
            final_particle_vel=final_particle_vel,
            final_positions=final_positions,
            rest_positions=rest_positions,
        )

    # -----------------------------------------------------------------
    # post-processing
    # -----------------------------------------------------------------

    def _assemble_results(self, **k) -> List[SimulationResult]:
        params = self.params
        solver = params.solver
        collision = params.collision

        B = self.B
        F = self.frame_count
        results: List[SimulationResult] = []
        for b in range(B):
            frames: List[FrameSample] = []
            for f in range(F):
                t = f * solver.frame_dt
                particles = None
                if k["frame_particles"] is not None:
                    particles = k["frame_particles"][b, f]
                frames.append(
                    FrameSample(
                        time=t,
                        ball_position=tuple(k["frame_ball_pos"][b, f].tolist()),
                        ball_velocity=tuple(k["frame_ball_vel"][b, f].tolist()),
                        particle_positions=particles,
                    )
                )
            n_contacts = min(int(k["contact_counts"][b]), self.max_contacts)
            contacts: List[ContactEvent] = []
            for i in range(n_contacts):
                ct = int(k["contact_types"][b, i])
                contacts.append(
                    ContactEvent(
                        time=float(k["contact_times"][b, i]),
                        object_type=CONTACT_TYPE_NAMES.get(ct, f"unknown_{ct}"),
                        object_index=int(k["contact_objs"][b, i]),
                        position=tuple(k["contact_positions"][b, i].tolist()),
                        normal=tuple(k["contact_normals"][b, i].tolist()),
                        strength=float(k["contact_strengths"][b, i]),
                    )
                )
            # sort contacts by time for nicer output
            contacts.sort(key=lambda c: c.time)

            # quality checks
            issues: List[str] = []
            max_pen_val = float(k["max_pen"][b])
            ball_pos = k["frame_ball_pos"][b]
            ball_vel = k["frame_ball_vel"][b]
            if max_pen_val > collision.severe_penetration_threshold:
                issues.append("severe_penetration")
            if not np.all(np.isfinite(ball_pos)) or not np.all(np.isfinite(ball_vel)):
                issues.append("nan_or_inf")
            ball_speed = np.linalg.norm(ball_vel, axis=1)
            if np.any(ball_speed > collision.max_ball_speed):
                issues.append("velocity_explosion")
            part_speed = np.linalg.norm(k["final_particle_vel"][b], axis=1)
            if np.any(part_speed > collision.max_particle_speed):
                issues.append("particle_velocity_explosion")
            disp = np.linalg.norm(
                k["final_positions"][b] - k["rest_positions"], axis=1
            )
            if np.any(disp > collision.max_net_displacement):
                issues.append("constraint_divergence")
            if (
                int(k["contact_started"][b]) != 0
                and float(k["stuck_time"][b]) > collision.stuck_duration
            ):
                issues.append("stuck")
            if n_contacts == 0:
                issues.append("target_panel_missed")

            quality = QualityReport(
                clean=len(issues) == 0,
                issues=issues,
                target_hit=n_contacts > 0,
                max_penetration_depth=max(0.0, max_pen_val),
                max_penetration_time=float(k["max_pen_time"][b]),
            )

            # stats
            ball_came_to_rest = ball_speed[-1] < 0.5
            stats = StatsReport(
                frame_dt=solver.frame_dt,
                substeps=solver.substeps,
                iterations=solver.iterations,
                duration=solver.duration,
                frame_count=F,
                contact_count=n_contacts,
                max_constraint_error=float(np.max(disp)) if disp.size > 0 else 0.0,
                max_net_displacement=float(np.max(disp)) if disp.size > 0 else 0.0,
                ball_came_to_rest=bool(ball_came_to_rest),
            )

            results.append(
                SimulationResult(
                    sample_id=k["sample_ids"][b],
                    frames=frames,
                    contacts=contacts,
                    quality=quality,
                    stats=stats,
                )
            )
        return results


__all__ = [
    "XpbdWarpSolver",
    "SimulationResult",
    "FrameSample",
    "ContactEvent",
    "QualityReport",
    "StatsReport",
    "CONTACT_TYPE_NAMES",
]
