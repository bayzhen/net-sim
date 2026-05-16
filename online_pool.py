"""Sliding-window frame pool for the online training pipeline.

The pool holds the most recent K simulation batches of clean samples in
host RAM, providing O(1) random ``(sample_idx, frame_idx)`` sampling for
the trainer. As new batches come in from the Warp solver, the oldest
slot is overwritten — like a ring buffer at the *batch* granularity.

Lifecycle::

    pool = OnlineFramePool(capacity_batches=4, max_samples_per_batch=512,
                           frame_count=601, particle_count=514,
                           drop_last_frames=5)

    # Warm-up
    for _ in range(capacity_batches):
        arrs = solver.simulate_arrays(balls, sample_ids)
        pool.push(arrs)         # filters by quality_clean

    # Steady state
    while training:
        if step % refill_every == 0:
            arrs = solver.simulate_arrays(...)
            pool.push(arrs)
        batch_dict = pool.sample(B_train, rng)
        ...

The pool stores everything as float32 numpy arrays:

    ball_pos:  (cap_batches, B_max, F, 3)
    ball_vel:  (cap_batches, B_max, F, 3)
    net_pos:   (cap_batches, B_max, F, N, 3)   <- biggest array (~7.5 GB at K=4)
    valid_n:   (cap_batches,) int32            <- # valid samples in each slot
    input_pos_xy / input_vel / input_ang / input_radius / input_mass

Notes
-----
* Capacity is in **simulation batches**, not samples. After a sim batch
  produces ``n_clean ≤ B_sim`` clean samples we copy that prefix into
  one slot. The slot's ``valid_n`` records the actual count for sampling.
* Frame-offset sampling honors ``drop_last_frames`` (default 5) the same
  way ``OffsetFrameDataset`` does in ``train.py``: random frame ∈
  ``[0, F - drop_last_frames)``.
* All arrays are pre-allocated and reused; no per-step allocation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class PoolStats:
    """Lightweight accumulators surfaced to the trainer for logging."""
    total_pushed: int = 0          # total clean samples ever ingested
    total_dropped_dirty: int = 0   # samples filtered out by quality_clean
    total_pushed_batches: int = 0  # number of sim batches push'd
    total_sampled_frames: int = 0  # frames yielded to trainer


class OnlineFramePool:
    """Ring buffer of recent simulation batches, sample-able by frame.

    Memory footprint (float32, default config)::

        net_pos:   K * B * F * N * 3 * 4
            K=4, B=512, F=601, N=514  →  ~7.5 GB
        ball_pos+vel: 2 * K * B * F * 3 * 4
            K=4, B=512, F=601         →  ~30 MB
        inputs:    K * B * (2+3+3+1+1) * 4 = ~80 KB / slot
    """

    def __init__(
        self,
        capacity_batches: int,
        max_samples_per_batch: int,
        frame_count: int,
        particle_count: int,
        drop_last_frames: int = 5,
    ) -> None:
        if capacity_batches < 1:
            raise ValueError("capacity_batches must be >= 1")
        if max_samples_per_batch < 1:
            raise ValueError("max_samples_per_batch must be >= 1")
        if frame_count < 1:
            raise ValueError("frame_count must be >= 1")
        if particle_count < 1:
            raise ValueError("particle_count must be >= 1")

        self.K = int(capacity_batches)
        self.B_max = int(max_samples_per_batch)
        self.F = int(frame_count)
        self.N = int(particle_count)
        self.drop_last_frames = max(0, int(drop_last_frames))

        K, B, F, N = self.K, self.B_max, self.F, self.N

        # Per-frame tensors
        self._ball_pos = np.zeros((K, B, F, 3), dtype=np.float32)
        self._ball_vel = np.zeros((K, B, F, 3), dtype=np.float32)
        self._net_pos = np.zeros((K, B, F, N, 3), dtype=np.float32)

        # Per-sample static inputs
        self._pos_xy = np.zeros((K, B, 2), dtype=np.float32)
        self._vel = np.zeros((K, B, 3), dtype=np.float32)
        self._ang = np.zeros((K, B, 3), dtype=np.float32)
        self._radius = np.zeros((K, B), dtype=np.float32)
        self._mass = np.zeros((K, B), dtype=np.float32)

        # Slot bookkeeping: how many of the B_max samples in slot k are valid.
        # A slot with valid_n == 0 is effectively empty.
        self._valid_n = np.zeros((K,), dtype=np.int32)
        self._next_slot = 0  # ring write pointer

        self.stats = PoolStats()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def n_filled_slots(self) -> int:
        return int((self._valid_n > 0).sum())

    @property
    def total_valid_samples(self) -> int:
        return int(self._valid_n.sum())

    def is_warm(self, min_slots: int = 1) -> bool:
        """Return True once at least ``min_slots`` slots have been filled."""
        return self.n_filled_slots >= min_slots

    # ------------------------------------------------------------------
    # Ingest: pull a sim-batch result dict from solver_warp.simulate_arrays
    # ------------------------------------------------------------------

    def push(self, arrs: Dict, balls: Sequence) -> int:
        """Filter by quality_clean and copy clean samples into the next slot.

        Args:
            arrs: dict returned by ``XpbdWarpSolver.simulate_arrays``.
                  Required keys: ``frame_ball_pos`` (B, F, 3), ``frame_ball_vel``
                  (B, F, 3), ``frame_particles`` (B, F, N, 3), ``per_sample_quality``
                  (list of QualityReport with ``.clean``).
            balls: the same ``List[BallState]`` passed to ``simulate_arrays``;
                   we read ``position[:2]`` / ``velocity`` / ``angular_velocity``
                   / ``radius`` / ``mass`` to populate the input cache.

        Returns:
            number of clean samples copied into the slot.
        """
        ball_pos = arrs["frame_ball_pos"]   # (B, F, 3) float32
        ball_vel = arrs["frame_ball_vel"]
        net_pos = arrs.get("frame_particles")
        if net_pos is None:
            raise ValueError("OnlineFramePool requires record_particles=True "
                             "(arrs['frame_particles'] is None)")
        quals = arrs["per_sample_quality"]

        B = ball_pos.shape[0]
        if ball_pos.shape != (B, self.F, 3):
            raise ValueError(
                f"frame_ball_pos shape {ball_pos.shape} does not match "
                f"expected (B, {self.F}, 3)")
        if net_pos.shape != (B, self.F, self.N, 3):
            raise ValueError(
                f"frame_particles shape {net_pos.shape} does not match "
                f"expected (B, {self.F}, {self.N}, 3)")
        if len(quals) != B:
            raise ValueError(f"per_sample_quality has {len(quals)} entries "
                             f"but batch has {B}")
        if len(balls) != B:
            raise ValueError(f"balls has {len(balls)} entries but batch has {B}")

        clean_mask = np.fromiter((q.clean for q in quals),
                                 dtype=np.bool_, count=B)
        clean_idx = np.flatnonzero(clean_mask)
        n_clean = int(clean_idx.size)
        n_dirty = B - n_clean
        self.stats.total_dropped_dirty += n_dirty
        self.stats.total_pushed_batches += 1

        # Ring write into the next slot (overwrite even if it has data).
        slot = self._next_slot
        self._next_slot = (slot + 1) % self.K

        if n_clean == 0:
            # Nothing useful — mark slot empty so we don't sample stale data.
            self._valid_n[slot] = 0
            return 0

        # Cap to B_max in case sim batch was larger than promised.
        n = min(n_clean, self.B_max)
        idx = clean_idx[:n]

        self._ball_pos[slot, :n] = ball_pos[idx]
        self._ball_vel[slot, :n] = ball_vel[idx]
        self._net_pos[slot, :n] = net_pos[idx]

        # Copy static inputs from the BallState list. Iterating Python objects
        # is fine — at most B_max=512, microseconds.
        for j, src_i in enumerate(idx.tolist()):
            b = balls[src_i]
            self._pos_xy[slot, j, 0] = b.position[0]
            self._pos_xy[slot, j, 1] = b.position[1]
            self._vel[slot, j, 0] = b.velocity[0]
            self._vel[slot, j, 1] = b.velocity[1]
            self._vel[slot, j, 2] = b.velocity[2]
            self._ang[slot, j, 0] = b.angular_velocity[0]
            self._ang[slot, j, 1] = b.angular_velocity[1]
            self._ang[slot, j, 2] = b.angular_velocity[2]
            self._radius[slot, j] = b.radius
            self._mass[slot, j] = b.mass

        self._valid_n[slot] = n
        self.stats.total_pushed += n
        return n

    # ------------------------------------------------------------------
    # Sample: produce a training mini-batch dict (numpy, ready for collate)
    # ------------------------------------------------------------------

    def sample(self, batch_size: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        """Random ``(slot, sample_in_slot, frame)`` triples → flat arrays.

        Returns the same dict schema as ``OffsetFrameDataset.__getitem__``
        (already batched along axis 0). Caller passes the result through
        ``train.collate``-equivalent code (here we directly emit batched
        tensors so the trainer can ``torch.from_numpy`` per key).
        """
        if self.total_valid_samples == 0:
            raise RuntimeError("pool is empty; push at least one batch before sampling")

        # Build a flat catalogue of valid (slot, j) pairs, weighted uniformly.
        # We do it via a pre-summed CDF so sampling is O(B + K), not O(K * B).
        valid = self._valid_n.astype(np.int64)  # (K,)
        cum = np.cumsum(valid)                    # (K,)
        total = int(cum[-1])

        flat_idx = rng.integers(0, total, size=batch_size, dtype=np.int64)
        # For each flat_idx, find which slot it belongs to.
        slots = np.searchsorted(cum, flat_idx, side="right")  # (B_train,)
        prev = np.concatenate(([0], cum[:-1]))
        in_slot = (flat_idx - prev[slots]).astype(np.int64)   # (B_train,)

        F_eff = max(self.F - self.drop_last_frames, 1)
        frames = rng.integers(0, F_eff, size=batch_size, dtype=np.int64)

        # Fancy indexing for per-frame arrays.
        ball_pos = self._ball_pos[slots, in_slot, frames]    # (B_train, 3)
        ball_vel = self._ball_vel[slots, in_slot, frames]    # (B_train, 3)
        net_pos = self._net_pos[slots, in_slot, frames]      # (B_train, N, 3)

        pos_xy = self._pos_xy[slots, in_slot]                # (B_train, 2)
        vel = self._vel[slots, in_slot]                      # (B_train, 3)
        ang = self._ang[slots, in_slot]                      # (B_train, 3)
        radius = self._radius[slots, in_slot]                # (B_train,)
        mass = self._mass[slots, in_slot]                    # (B_train,)

        t_norm = (frames.astype(np.float32) / max(self.F - 1, 1))

        self.stats.total_sampled_frames += batch_size

        return {
            "pos_xy": pos_xy.astype(np.float32, copy=False),
            "vel": vel.astype(np.float32, copy=False),
            "ang": ang.astype(np.float32, copy=False),
            "radius": radius.astype(np.float32, copy=False),
            "mass": mass.astype(np.float32, copy=False),
            "t_norm": t_norm,
            "target_ball": ball_pos.astype(np.float32, copy=False),
            "target_ball_v": ball_vel.astype(np.float32, copy=False),
            "target_net": net_pos.astype(np.float32, copy=False),
            "frame": frames,
        }

    # ------------------------------------------------------------------
    # Iterate ALL valid (slot, j, frame) for stats / validation passes.
    # ------------------------------------------------------------------

    def iter_all_frames(self):
        """Yield per-sample views for streaming statistics computation.

        Each yielded item is a dict with::

            ball_pos: (F, 3)  ball_vel: (F, 3)  net_pos: (F, N, 3)
            pos_xy: (2,)  vel: (3,)  ang: (3,)  radius: ()  mass: ()

        This mirrors what ``train.compute_norm_stats`` consumes from the
        preload caches sample-by-sample.
        """
        for slot in range(self.K):
            n = int(self._valid_n[slot])
            for j in range(n):
                yield {
                    "ball_pos": self._ball_pos[slot, j],
                    "ball_vel": self._ball_vel[slot, j],
                    "net_pos": self._net_pos[slot, j],
                    "pos_xy": self._pos_xy[slot, j],
                    "vel": self._vel[slot, j],
                    "ang": self._ang[slot, j],
                    "radius": float(self._radius[slot, j]),
                    "mass": float(self._mass[slot, j]),
                }


__all__ = ["OnlineFramePool", "PoolStats"]
