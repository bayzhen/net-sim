"""Train a neural surrogate on a goal-net-xpbd HDF5 dataset.

Default task: offset-frame supervised learning. Given a ball's initial
state plus a normalized time ``t_norm = f / (F-1)``, predict the ball's
position + velocity and the net particle positions at frame ``f``.

Usage::

    python cli.py train \\
        --dataset E:/dataset_v2/dataset.h5 \\
        --epochs 30 --batch 256 --device cuda \\
        --output runs/mlp_v1

What it produces under ``--output``:
    config.json        # hyperparams + dataset stats
    metrics.jsonl      # one line per epoch (train/val losses)
    best.pt            # checkpoint (val loss minimum)
    last.pt            # final-epoch checkpoint

Implementation notes
--------------------
* The dataset has ``F`` frames per sample and ``N`` particles. To avoid
  reading the full 3.5 MB ``particle_position[i]`` slab just to pick one
  frame, this trainer uses a custom ``OffsetFrameDataset`` that reads a
  single ``(N, 3)`` slice via h5py's hyperslab.
* Small per-sample tensors (initial state, mass, radius) are loaded once
  into RAM and kept as numpy arrays; index lookups are O(1).
* We rely on ``num_workers=0`` by default — h5py file handles do not
  pickle cleanly across forked workers on Windows, and the bottleneck
  is the GPU step anyway. Increase via ``--num-workers`` only on Linux
  and verify with ``--smoke`` first.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# torch is imported lazily so that ``python cli.py --help`` works even on
# environments without torch installed.
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    dataset: str
    output: str
    epochs: int = 30
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-5
    hidden: Tuple[int, ...] = (512, 512, 512, 512)
    activation: str = "gelu"
    dropout: float = 0.0
    seed: int = 0
    val_frac: float = 0.05
    test_frac: float = 0.05
    clean_only: bool = True
    device: str = "cuda"
    num_workers: int = 0
    log_every: int = 50
    grad_clip: float = 1.0
    w_ball_pos: float = 1.0
    w_ball_vel: float = 1.0
    w_net: float = 1.0
    n_time_freq: int = 4
    preload: bool = True             # default for train/val
    preload_test: bool = False       # test runs once at the end; stream is fine
    smoke: bool = False  # tiny run for sanity
    # v3 normalization-scale strategy.
    #   "global":  legacy behavior — ball_{pos,vel}_std = stddev over all frames.
    #              Suffers from "stats leak" because most frames are end-game where
    #              the ball is nearly static, dragging std way below the dynamic
    #              range. With v2 dataset this pushes init-frame targets to ~8.5
    #              sigma and the network regresses to mean (see design.md §17.4).
    #   "init":    ball_pos_std uses frame-0 std (mostly initial-position scale);
    #              ball_vel_std uses input_velocity.std (true initial-shot scale).
    #              Simple and physically meaningful for the *initial condition*.
    #   "robust":  ball_pos / ball_vel std = (max - min) / 4 over all frames
    #              (i.e. the half-range of the full trajectory). Covers both
    #              initial spread and end-game positions without being dragged
    #              down by static frames. RECOMMENDED default.
    norm_scale_mode: str = "robust"
    # Drop the last K frames from the per-epoch random offset table. Boundary
    # artifact: t_norm=1 makes the sin/cos embedding collide with t_norm=0,
    # and the very last frames are over-represented in end-game stats. Set to
    # 0 to disable.
    drop_last_frames: int = 5
    # Append a binary `is_settled` indicator (ball_speed < threshold) to the
    # input encoding. Helps the network discriminate "ball is at rest" from
    # "ball is moving slowly through the net" — both produce small velocities
    # but require very different position predictions.
    use_settled_flag: bool = False
    # CosineAnnealingLR final lr = ``lr * lr_min_frac``. v3 baseline used
    # 0.05 (= 5% of initial); on long runs this floors training too early.
    lr_min_frac: float = 0.01


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class OffsetFrameDataset:
    """torch.utils.data.Dataset compatible.

    Two access modes:

    * ``preload=False`` (h5 streaming): each ``__getitem__`` reads one
      ``(F, N, 3)`` chunk from disk and slices a single frame. Cheap on
      memory but I/O bound — typically ~30s/epoch on NVMe.
    * ``preload=True`` (RAM cache): loads ALL particle / ball / input
      arrays for the chosen sample indices into RAM up-front, then
      ``__getitem__`` is a pure numpy index. Required for I/O-free
      training. For 14k clean samples × (601 frames × 514 particles) the
      cache is ~52 GB, so make sure your machine has the RAM.

    Empirical (E:/dataset_v2/dataset.h5):
        h5 streaming  ~3 ms/sample, GPU util ~0%
        preload RAM   ~3 us/sample, GPU util saturated
    """

    def __init__(
        self,
        h5_path: str,
        sample_indices: np.ndarray,
        rng_seed: int = 0,
        load_particles: bool = True,
        preload: bool = False,
        preload_label: str = "",
        drop_last_frames: int = 0,
    ) -> None:
        import h5py

        self._h5py = h5py
        self.path = h5_path
        self.indices = np.asarray(sample_indices, dtype=np.int64)
        self.load_particles = load_particles
        self.preload = bool(preload)
        # Sampleable frame range is [0, frame_count - drop_last_frames). We
        # store ``drop_last_frames`` so set_epoch can clip the random table.
        self.drop_last_frames = max(0, int(drop_last_frames))
        self._file: Optional["h5py.File"] = None

        # Read tiny metadata + per-sample inputs once into RAM.
        with h5py.File(self.path, "r") as f:
            self.frame_count = int(f.attrs["frame_count"])
            self.particle_count = int(f.attrs["particle_count"])
            self.frame_dt = float(f.attrs["frame_dt"])
            self.input_position = f["input_position"][:].astype(np.float32)
            self.input_velocity = f["input_velocity"][:].astype(np.float32)
            self.input_angular  = f["input_angular"][:].astype(np.float32)
            self.input_radius   = f["input_radius"][:].astype(np.float32)
            self.input_mass     = f["input_mass"][:].astype(np.float32)

            if self.preload:
                self._preload_arrays(f, label=preload_label)

        # Reproducible per-sample random offset: each item gets a frame
        # picked from this fixed table so two epochs see different frames
        # but a single (epoch, idx) is deterministic.
        self._epoch_seed = rng_seed
        self._epoch_offsets: Optional[np.ndarray] = None
        self.set_epoch(0)

    # ------------------------------------------------------------------
    # Preload path
    # ------------------------------------------------------------------

    def _preload_arrays(self, f, label: str = "") -> None:
        """Load ball_position/velocity and particle_position for the
        chosen ``self.indices`` into contiguous RAM arrays.

        We read sample-by-sample to keep peak memory bounded at one
        sample's slab (~3.5 MB) above the destination array, and so we
        can show progress for what is otherwise a multi-second wait.
        """
        n = int(self.indices.shape[0])
        F = self.frame_count
        N = self.particle_count

        ball_bytes = n * F * 3 * 4
        net_bytes = n * F * N * 3 * 4 if self.load_particles else 0
        total_bytes = ball_bytes * 2 + net_bytes
        total_gb = total_bytes / (1024 ** 3)
        tag = f" [{label}]" if label else ""

        # Sanity-check available RAM. We need not just `total_gb` for the
        # cache itself, but also a comfortable buffer for the rest of the
        # training pipeline (Welford accumulators, h5 chunk decompression,
        # PyTorch staging, page cache for OS, etc.). 1.4x is a soft margin
        # that has worked in practice; surface a clear error otherwise.
        try:
            import psutil  # type: ignore
            avail = psutil.virtual_memory().available
            need = int(total_bytes * 1.4)
            if avail < need:
                raise MemoryError(
                    f"preload{tag}: would need ~{need/(1024**3):.1f} GB free "
                    f"(cache itself is {total_gb:.1f} GB; we leave a 1.4x margin) "
                    f"but only {avail/(1024**3):.1f} GB available. "
                    f"Pass --no-preload to fall back to HDF5 streaming."
                )
        except ModuleNotFoundError:
            pass  # psutil is optional; user takes their chances

        print(f"  preload{tag}: allocating {total_gb:.2f} GB RAM "
              f"({n} samples × F={F} × N={N})", flush=True)
        t0 = time.time()

        self._ball_pos_cache = np.empty((n, F, 3), dtype=np.float32)
        self._ball_vel_cache = np.empty((n, F, 3), dtype=np.float32)
        if self.load_particles:
            self._net_cache = np.empty((n, F, N, 3), dtype=np.float32)
        else:
            self._net_cache = None

        # Sort h5 read indices ascending for sequential file access — much
        # friendlier to the page cache and reduces seek overhead even on
        # NVMe. We then write into the destination at the original
        # position (so __getitem__(i) keeps mapping to indices[i]).
        order = np.argsort(self.indices)
        bp = f["ball_position"]
        bv = f["ball_velocity"]
        pp = f["particle_position"] if self.load_particles else None

        log_every = max(n // 20, 1)
        for k, j in enumerate(order):
            si = int(self.indices[int(j)])
            self._ball_pos_cache[int(j)] = bp[si]
            self._ball_vel_cache[int(j)] = bv[si]
            if pp is not None:
                self._net_cache[int(j)] = pp[si]
            if (k + 1) % log_every == 0 or k + 1 == n:
                pct = (k + 1) / n * 100
                dt = time.time() - t0
                rate = (k + 1) / max(dt, 1e-6)
                eta = (n - k - 1) / max(rate, 1e-6)
                print(f"    [{k+1:5d}/{n}] {pct:5.1f}%  "
                      f"{rate:.0f} samp/s  ETA {eta:.1f}s", flush=True)
        print(f"  preload{tag}: done in {time.time()-t0:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Common
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Reseed the per-sample frame-offset table for this epoch.

        Honors ``drop_last_frames``: the last K frames are excluded from
        the random sampling pool to avoid the t_norm=1 boundary artifact
        (sin/cos embedding collides with t_norm=0) and the over-static
        end-game frames that drag the std down.
        """
        rng = np.random.default_rng(self._epoch_seed + epoch)
        F = self.frame_count
        F_eff = max(F - self.drop_last_frames, 1)
        self._epoch_offsets = rng.integers(0, F_eff, size=int(self.indices.shape[0]),
                                           dtype=np.int64)

    def _open(self) -> "h5py.File":
        if self._file is None:
            self._file = self._h5py.File(self.path, "r", swmr=True)
        return self._file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        si = int(self.indices[i])
        frame = int(self._epoch_offsets[i])  # type: ignore[index]
        F = self.frame_count
        t_norm = np.float32(frame / max(F - 1, 1))

        if self.preload:
            # In-memory path — ~free.
            ball_pos = self._ball_pos_cache[i, frame]
            ball_vel = self._ball_vel_cache[i, frame]
            if self._net_cache is not None:
                net = self._net_cache[i, frame]
            else:
                net = np.empty((self.particle_count, 3), dtype=np.float32)
        else:
            f = self._open()
            ball_pos = f["ball_position"][si][frame].astype(np.float32, copy=False)
            ball_vel = f["ball_velocity"][si][frame].astype(np.float32, copy=False)
            if self.load_particles:
                net = f["particle_position"][si][frame].astype(np.float32, copy=False)
            else:
                net = np.empty((self.particle_count, 3), dtype=np.float32)

        # Return raw pieces; the encoder runs on the GPU side after batching.
        # input_position[2] is fixed at 1.5 m in the v2 dataset, so we drop
        # it; the encoder consumes pos_xy only.
        return {
            "pos_xy":      self.input_position[si, :2].astype(np.float32),
            "vel":         self.input_velocity[si].astype(np.float32),
            "ang":         self.input_angular[si].astype(np.float32),
            "radius":      np.float32(self.input_radius[si]),
            "mass":        np.float32(self.input_mass[si]),
            "t_norm":      t_norm,
            "target_ball": ball_pos,
            "target_ball_v": ball_vel,
            "target_net":  net,
            "frame":       np.int64(frame),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def split_indices(n: int, val_frac: float, test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    n_val = int(round(n * val_frac))
    n_test = int(round(n * test_frac))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError(f"split too aggressive: n={n} val={n_val} test={n_test}")
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def load_clean_indices(h5_path: str, clean_only: bool) -> np.ndarray:
    import h5py
    with h5py.File(h5_path, "r") as f:
        n_total = int(f["sample_id"].shape[0])
        if clean_only:
            mask = f["quality_clean"][:].astype(bool)
            return np.flatnonzero(mask)
        return np.arange(n_total, dtype=np.int64)


def collate(batch: List[Dict[str, np.ndarray]]):
    import torch
    out = {}
    for k in ("pos_xy", "vel", "ang", "target_ball", "target_ball_v", "target_net"):
        out[k] = torch.from_numpy(np.stack([b[k] for b in batch], axis=0))
    for k in ("radius", "mass", "t_norm"):
        out[k] = torch.from_numpy(np.asarray([b[k] for b in batch], dtype=np.float32))
    out["frame"] = torch.from_numpy(np.stack([b["frame"] for b in batch], axis=0))
    return out


def encode_batch_input(batch: dict, n_time_freq: int):
    """Run the input encoder on a batched dict produced by ``collate``.

    Returns the (B, in_dim) raw-units feature tensor — the model's
    ``forward`` will normalize it internally.
    """
    from model import encode_input_features
    return encode_input_features(
        pos_xy=batch["pos_xy"],
        vel=batch["vel"],
        ang=batch["ang"],
        radius=batch["radius"],
        mass=batch["mass"],
        t_norm=batch["t_norm"],
        n_time_freq=n_time_freq,
    )


def compute_loss(model, pred_norm: dict, batch: dict, cfg: TrainConfig):
    """Compute MSE in **standardized** space.

    ``pred_norm`` is the model output with ``return_normalized=True`` so
    each head has roughly unit variance, eliminating the scale imbalance
    between ball and net targets.
    """
    import torch.nn.functional as F
    tgt = model.standardize_targets(
        batch["target_ball"],
        batch.get("target_ball_v"),
        batch["target_net"],
    )

    l_pos = F.mse_loss(pred_norm["ball_pos"], tgt["ball_pos"])
    l_net = F.mse_loss(pred_norm["net"],      tgt["net"])
    l_vel = (F.mse_loss(pred_norm["ball_vel"], tgt["ball_vel"])
             if "ball_vel" in pred_norm and "ball_vel" in tgt else None)

    total = cfg.w_ball_pos * l_pos + cfg.w_net * l_net
    parts = {"ball_pos": float(l_pos.detach()),
             "net":      float(l_net.detach())}
    if l_vel is not None:
        total = total + cfg.w_ball_vel * l_vel
        parts["ball_vel"] = float(l_vel.detach())
    return total, parts


def compute_phys_metrics(pred_phys: dict, batch: dict) -> Dict[str, float]:
    """Compute per-head MSE in **physical units** for monitoring.

    These metrics are what users care about (mean squared meters), and
    are *not* the loss — see ``compute_loss``.
    """
    import torch.nn.functional as F
    out = {
        "ball_pos_phys": float(F.mse_loss(pred_phys["ball_pos"],
                                          batch["target_ball"]).detach()),
        "net_phys": float(F.mse_loss(pred_phys["net"],
                                     batch["target_net"]).detach()),
    }
    if "ball_vel" in pred_phys:
        out["ball_vel_phys"] = float(F.mse_loss(pred_phys["ball_vel"],
                                                batch["target_ball_v"]).detach())
    return out


def evaluate(model, loader, device, cfg) -> Dict[str, float]:
    import torch
    model.eval()
    sums = {"ball_pos": 0.0, "ball_vel": 0.0, "net": 0.0, "total": 0.0,
            "ball_pos_phys": 0.0, "ball_vel_phys": 0.0, "net_phys": 0.0}
    n_samples = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            x = encode_batch_input(batch, cfg.n_time_freq)
            pred_norm = model(x, return_normalized=True)
            total, parts = compute_loss(model, pred_norm, batch, cfg)
            pred_phys = model(x, return_normalized=False)
            phys = compute_phys_metrics(pred_phys, batch)

            B = x.shape[0]
            sums["total"] += float(total.detach()) * B
            for k, v in parts.items():
                sums[k] += v * B
            for k, v in phys.items():
                sums[k] += v * B
            n_samples += B
    for k in sums:
        sums[k] /= max(n_samples, 1)
    return sums


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def compute_norm_stats(train_ds: "OffsetFrameDataset", n_time_freq: int,
                       max_samples_for_input: int = 8192,
                       norm_scale_mode: str = "robust") -> Dict[str, "torch.Tensor"]:
    """Compute mean/std for input features and targets from the train split.

    ``norm_scale_mode`` controls how ``ball_pos_std`` / ``ball_vel_std`` are
    derived. See ``TrainConfig`` for the three options ("global" / "init" /
    "robust"). ``net_*`` and ``in_*`` always use the global / sampled-mean
    formulation since they don't suffer from the same leak.

    Memory note
    -----------
    The naive ``net_cache.mean(axis=(0, 1))`` allocates a temporary buffer
    that is comparable in size to ``net_cache`` itself (~50 GB for our
    14k-sample subset). On a machine where the cache already eats 50 GB
    that triggers OOM. We instead accumulate sums and squared-sums in a
    streaming fashion (one sample at a time = 1.2 MB peak temp).

    Requires ``train_ds.preload == True`` so we can iterate the cached
    arrays without re-reading HDF5.
    """
    import torch

    if not train_ds.preload:
        raise RuntimeError("compute_norm_stats currently requires preload=True")
    if norm_scale_mode not in ("global", "init", "robust"):
        raise ValueError(
            f"unknown norm_scale_mode={norm_scale_mode!r}; "
            f"expected 'global' / 'init' / 'robust'")

    n = len(train_ds)
    F = train_ds.frame_count
    N = train_ds.particle_count

    # ------- input statistics (sample many random (idx, frame) pairs) -------
    rng = np.random.default_rng(123456)
    m = min(max_samples_for_input, n)
    pick = rng.choice(n, size=m, replace=False)
    frames = rng.integers(0, F, size=m, dtype=np.int64)

    pos_xy = train_ds.input_position[train_ds.indices[pick], :2].astype(np.float32)
    vel = train_ds.input_velocity[train_ds.indices[pick]].astype(np.float32)
    ang = train_ds.input_angular[train_ds.indices[pick]].astype(np.float32)
    rad = train_ds.input_radius[train_ds.indices[pick]].astype(np.float32)
    mas = train_ds.input_mass[train_ds.indices[pick]].astype(np.float32)
    tnorm = (frames.astype(np.float32) / max(F - 1, 1))

    from model import encode_input_features
    feats = encode_input_features(
        pos_xy=torch.from_numpy(pos_xy),
        vel=torch.from_numpy(vel),
        ang=torch.from_numpy(ang),
        radius=torch.from_numpy(rad),
        mass=torch.from_numpy(mas),
        t_norm=torch.from_numpy(tnorm),
        n_time_freq=n_time_freq,
    )
    in_mean = feats.mean(0)
    in_std = torch.clamp(feats.std(0), min=1e-3)

    # ------- target statistics: streaming accumulate over samples -------
    # We keep float64 accumulators (small: 3 + 3 + N*3 = 1551 doubles).
    sum_bp = np.zeros(3, dtype=np.float64)
    sumsq_bp = np.zeros(3, dtype=np.float64)
    sum_bv = np.zeros(3, dtype=np.float64)
    sumsq_bv = np.zeros(3, dtype=np.float64)
    # Min/max trackers for "robust" scale.
    min_bp = np.full(3, np.inf, dtype=np.float64)
    max_bp = np.full(3, -np.inf, dtype=np.float64)
    min_bv = np.full(3, np.inf, dtype=np.float64)
    max_bv = np.full(3, -np.inf, dtype=np.float64)
    # Frame-0-only trackers for "init" scale.
    sum_bp_f0 = np.zeros(3, dtype=np.float64)
    sumsq_bp_f0 = np.zeros(3, dtype=np.float64)
    sum_bv_f0 = np.zeros(3, dtype=np.float64)
    sumsq_bv_f0 = np.zeros(3, dtype=np.float64)
    sum_net = np.zeros((N, 3), dtype=np.float64)
    sumsq_net = np.zeros((N, 3), dtype=np.float64)
    total_frames = 0

    bp_cache = train_ds._ball_pos_cache
    bv_cache = train_ds._ball_vel_cache
    net_cache = train_ds._net_cache

    # Process one sample at a time — peak temporary is one (F, N, 3) ~3.5 MB.
    log_every = max(n // 20, 1)
    t0 = time.time()
    for i in range(n):
        bp = bp_cache[i]                       # (F, 3) view, no copy
        bv = bv_cache[i]
        net = net_cache[i]                     # (F, N, 3) view, no copy
        bp64 = bp.astype(np.float64)
        bv64 = bv.astype(np.float64)
        sum_bp += bp64.sum(axis=0)
        sumsq_bp += (bp64 ** 2).sum(axis=0)
        sum_bv += bv64.sum(axis=0)
        sumsq_bv += (bv64 ** 2).sum(axis=0)
        # min/max per component over the F frames of this sample.
        np.minimum(min_bp, bp64.min(axis=0), out=min_bp)
        np.maximum(max_bp, bp64.max(axis=0), out=max_bp)
        np.minimum(min_bv, bv64.min(axis=0), out=min_bv)
        np.maximum(max_bv, bv64.max(axis=0), out=max_bv)
        # frame-0 accumulators.
        sum_bp_f0 += bp64[0]
        sumsq_bp_f0 += bp64[0] ** 2
        sum_bv_f0 += bv64[0]
        sumsq_bv_f0 += bv64[0] ** 2
        # For the (F, N, 3) slab we cast to float64 in-place for the squaring;
        # the temp here is one sample = ~7 MB float64, OK.
        net64 = net.astype(np.float64, copy=False)
        sum_net += net64.sum(axis=0)
        sumsq_net += (net64 ** 2).sum(axis=0)
        total_frames += F
        if (i + 1) % log_every == 0 or i + 1 == n:
            print(f"    stats [{i+1:5d}/{n}]", flush=True)

    M = float(total_frames)
    n_f = float(n)  # number of frame-0 observations = number of samples
    ball_pos_mean_np = sum_bp / M
    ball_vel_mean_np = sum_bv / M
    net_mean_np = sum_net / M
    # Global var (legacy "global" mode std).
    ball_pos_var_global = np.maximum(sumsq_bp / M - ball_pos_mean_np ** 2, 0.0)
    ball_vel_var_global = np.maximum(sumsq_bv / M - ball_vel_mean_np ** 2, 0.0)
    net_var_np = np.maximum(sumsq_net / M - net_mean_np ** 2, 0.0)
    # Frame-0 var ("init" mode std).
    bp_f0_mean = sum_bp_f0 / n_f
    bv_f0_mean = sum_bv_f0 / n_f
    ball_pos_var_init = np.maximum(sumsq_bp_f0 / n_f - bp_f0_mean ** 2, 0.0)
    ball_vel_var_init = np.maximum(sumsq_bv_f0 / n_f - bv_f0_mean ** 2, 0.0)

    std_global_bp = np.sqrt(ball_pos_var_global)
    std_global_bv = np.sqrt(ball_vel_var_global)
    std_init_bp = np.sqrt(ball_pos_var_init)
    std_init_bv = np.sqrt(ball_vel_var_init)
    # "robust" mode = half-range of full trajectory (covers ±2 sigma of a
    # uniform distribution); for unimodal smooth distributions it is a
    # somewhat looser scale than 1 sigma but is immune to the static-tail
    # collapse. We multiply by 1/2 (not 1/4) because we want the *half*
    # range; experimentally this gives unit-variance-ish targets without
    # over-shrinking large-amplitude signals.
    std_robust_bp = (max_bp - min_bp) / 2.0
    std_robust_bv = (max_bv - min_bv) / 2.0

    if norm_scale_mode == "global":
        ball_pos_std_np = std_global_bp
        ball_vel_std_np = std_global_bv
    elif norm_scale_mode == "init":
        ball_pos_std_np = std_init_bp
        ball_vel_std_np = std_init_bv
    else:  # "robust"
        ball_pos_std_np = std_robust_bp
        ball_vel_std_np = std_robust_bv

    print(f"    norm_scale_mode={norm_scale_mode}", flush=True)
    print(f"    ball_pos_std: global={std_global_bp.tolist()} "
          f"init={std_init_bp.tolist()} robust={std_robust_bp.tolist()} "
          f"chosen={ball_pos_std_np.tolist()}", flush=True)
    print(f"    ball_vel_std: global={std_global_bv.tolist()} "
          f"init={std_init_bv.tolist()} robust={std_robust_bv.tolist()} "
          f"chosen={ball_vel_std_np.tolist()}", flush=True)

    ball_pos_mean = torch.from_numpy(ball_pos_mean_np.astype(np.float32))
    ball_vel_mean = torch.from_numpy(ball_vel_mean_np.astype(np.float32))
    net_mean = torch.from_numpy(net_mean_np.astype(np.float32))
    ball_pos_std = torch.clamp(torch.from_numpy(ball_pos_std_np.astype(np.float32)), min=1e-3)
    ball_vel_std = torch.clamp(torch.from_numpy(ball_vel_std_np.astype(np.float32)), min=1e-3)
    net_std = torch.clamp(torch.from_numpy(np.sqrt(net_var_np).astype(np.float32)), min=1e-3)

    print(f"    streaming stats: {time.time()-t0:.1f}s "
          f"({total_frames} frames)", flush=True)

    return {
        "in_mean": in_mean,
        "in_std": in_std,
        "ball_pos_mean": ball_pos_mean,
        "ball_pos_std": ball_pos_std,
        "ball_vel_mean": ball_vel_mean,
        "ball_vel_std": ball_vel_std,
        "net_mean": net_mean,
        "net_std": net_std,
    }


def run_training(cfg: TrainConfig) -> Dict[str, str]:
    import torch
    from torch.utils.data import DataLoader

    out = Path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)

    # --- splits ---
    all_idx = load_clean_indices(cfg.dataset, cfg.clean_only)
    n = int(all_idx.shape[0])
    if cfg.smoke:
        # 1024 train + 128 val + 128 test
        all_idx = all_idx[:1280]
        n = 1280
    tr, va, te = split_indices(n, cfg.val_frac, cfg.test_frac, cfg.seed)
    tr_idx, va_idx, te_idx = all_idx[tr], all_idx[va], all_idx[te]
    print(f"split: train={tr_idx.size} val={va_idx.size} test={te_idx.size} "
          f"(of {n} clean samples)", flush=True)

    train_ds = OffsetFrameDataset(cfg.dataset, tr_idx, rng_seed=cfg.seed,
                                   preload=cfg.preload, preload_label="train",
                                   drop_last_frames=cfg.drop_last_frames)
    val_ds   = OffsetFrameDataset(cfg.dataset, va_idx, rng_seed=cfg.seed + 1,
                                   preload=cfg.preload, preload_label="val",
                                   drop_last_frames=cfg.drop_last_frames)
    test_ds  = OffsetFrameDataset(cfg.dataset, te_idx, rng_seed=cfg.seed + 2,
                                   preload=cfg.preload_test, preload_label="test",
                                   drop_last_frames=cfg.drop_last_frames)

    # When data is in RAM, multi-worker DataLoader is pure overhead (each
    # fork would COW-touch our 50+ GB cache). Force single-process.
    effective_workers = 0 if cfg.preload else cfg.num_workers
    if effective_workers != cfg.num_workers:
        print(f"[note] preload=True; forcing num_workers=0 "
              f"(was {cfg.num_workers})", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=effective_workers, collate_fn=collate,
        persistent_workers=effective_workers > 0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=effective_workers, collate_fn=collate,
        persistent_workers=effective_workers > 0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=effective_workers, collate_fn=collate,
        persistent_workers=effective_workers > 0, pin_memory=True,
    )

    # --- model ---
    from model import GoalNetMLP, count_parameters, input_dim_for
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[warn] --device cuda requested but torch.cuda.is_available() is "
              "False; falling back to CPU. Reinstall PyTorch with CUDA support "
              "(e.g. `pip install torch --index-url "
              "https://download.pytorch.org/whl/cu121`) for GPU training.",
              flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)

    in_dim = input_dim_for(cfg.n_time_freq)
    model = GoalNetMLP(
        in_dim=in_dim,
        n_particles=train_ds.particle_count,
        hidden=tuple(cfg.hidden),
        activation=cfg.activation,
        predict_velocity=True,
        dropout=cfg.dropout,
    ).to(device)
    print(f"model: GoalNetMLP, params={count_parameters(model):,}, "
          f"in_dim={in_dim}, device={device}, "
          f"N={train_ds.particle_count}, F={train_ds.frame_count}",
          flush=True)

    # Compute and install normalization statistics from the train split.
    print("computing normalization statistics from train split...", flush=True)
    t_stats0 = time.time()
    stats = compute_norm_stats(train_ds, cfg.n_time_freq,
                               norm_scale_mode=cfg.norm_scale_mode)
    model.set_norm_stats({k: v.to(device) for k, v in stats.items()})
    print(f"  done in {time.time()-t_stats0:.1f}s "
          f"(ball_pos_std={stats['ball_pos_std'].tolist()}, "
          f"net_std mean={stats['net_std'].mean().item():.4f})", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    total_steps = max(cfg.epochs * len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=cfg.lr * cfg.lr_min_frac
    )

    # --- save config ---
    cfg_dump = asdict(cfg)
    cfg_dump["hidden"] = list(cfg.hidden)
    cfg_dump["particle_count"] = train_ds.particle_count
    cfg_dump["frame_count"] = train_ds.frame_count
    cfg_dump["frame_dt"] = train_ds.frame_dt
    cfg_dump["n_train"] = int(tr_idx.size)
    cfg_dump["n_val"] = int(va_idx.size)
    cfg_dump["n_test"] = int(te_idx.size)
    (out / "config.json").write_text(json.dumps(cfg_dump, indent=2))

    metrics_path = out / "metrics.jsonl"
    metrics_fp = metrics_path.open("w")

    best_val = math.inf
    best_path = out / "best.pt"
    last_path = out / "last.pt"

    global_step = 0
    t_start = time.time()
    for epoch in range(cfg.epochs):
        train_ds.set_epoch(epoch)
        val_ds.set_epoch(epoch)

        model.train()
        epoch_t0 = time.time()
        running = {"ball_pos": 0.0, "ball_vel": 0.0, "net": 0.0, "total": 0.0,
                   "ball_pos_phys": 0.0, "ball_vel_phys": 0.0, "net_phys": 0.0}
        n_seen = 0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            x = encode_batch_input(batch, cfg.n_time_freq)
            pred_norm = model(x, return_normalized=True)
            total, parts = compute_loss(model, pred_norm, batch, cfg)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            # Bookkeeping in physical units (cheap; uses the same forward).
            with torch.no_grad():
                pred_phys = model(x, return_normalized=False)
                phys = compute_phys_metrics(pred_phys, batch)

            B = x.shape[0]
            running["total"] += float(total.detach()) * B
            for k, v in parts.items():
                running[k] += v * B
            for k, v in phys.items():
                running[k] += v * B
            n_seen += B
            global_step += 1

            if (step + 1) % cfg.log_every == 0:
                avg = {k: v / max(n_seen, 1) for k, v in running.items()}
                lr = scheduler.get_last_lr()[0]
                print(f"  ep {epoch:03d} step {step+1:05d} "
                      f"loss={avg['total']:.4e}  norm[pos={avg['ball_pos']:.3e} "
                      f"vel={avg.get('ball_vel', 0):.3e} net={avg['net']:.3e}]  "
                      f"phys[pos={avg['ball_pos_phys']:.3f} "
                      f"vel={avg.get('ball_vel_phys', 0):.3f} "
                      f"net={avg['net_phys']:.3f}]  lr={lr:.2e}", flush=True)

        train_loss = {k: v / max(n_seen, 1) for k, v in running.items()}
        val_loss = evaluate(model, val_loader, device, cfg)
        epoch_dt = time.time() - epoch_t0
        elapsed = time.time() - t_start
        eta = elapsed / max(epoch + 1, 1) * (cfg.epochs - epoch - 1)
        print(f"epoch {epoch:03d} done in {epoch_dt:.1f}s  "
              f"train_norm={train_loss['total']:.4e} "
              f"val_norm={val_loss['total']:.4e}  "
              f"phys[pos={val_loss['ball_pos_phys']:.3f} "
              f"vel={val_loss['ball_vel_phys']:.3f} "
              f"net={val_loss['net_phys']:.4f}]  ETA={eta/60:.1f}min", flush=True)

        rec = {
            "epoch": epoch,
            "train": train_loss,
            "val": val_loss,
            "lr": scheduler.get_last_lr()[0],
            "time_s": epoch_dt,
        }
        metrics_fp.write(json.dumps(rec) + "\n")
        metrics_fp.flush()

        # checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "config": cfg_dump,
            "val_loss": val_loss,
        }
        torch.save(ckpt, last_path)
        if val_loss["total"] < best_val:
            best_val = val_loss["total"]
            torch.save(ckpt, best_path)

    metrics_fp.close()

    # final test
    print("evaluating on test split (using best checkpoint)...", flush=True)
    state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    test_loss = evaluate(model, test_loader, device, cfg)
    print(f"test: norm_total={test_loss['total']:.4e}  "
          f"phys[pos={test_loss['ball_pos_phys']:.3f} "
          f"vel={test_loss['ball_vel_phys']:.3f} "
          f"net={test_loss['net_phys']:.4f}]", flush=True)
    (out / "test_metrics.json").write_text(json.dumps(test_loss, indent=2))

    train_ds.close(); val_ds.close(); test_ds.close()

    return {
        "config": str(out / "config.json"),
        "metrics": str(metrics_path),
        "best_ckpt": str(best_path),
        "last_ckpt": str(last_path),
        "test_metrics": str(out / "test_metrics.json"),
    }


# ---------------------------------------------------------------------------
# CLI entry helpers
# ---------------------------------------------------------------------------


def add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", required=True, help="path to dataset.h5")
    p.add_argument("--output", required=True, help="output directory for checkpoints/logs")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", dest="batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-5)
    p.add_argument("--hidden", type=int, nargs="+",
                   default=[1024, 1024, 1024, 1024, 1024, 1024])
    p.add_argument("--n-time-freq", dest="n_time_freq", type=int, default=4,
                   help="number of sinusoidal time-embedding frequencies "
                   "(input_dim = 10 + 2*n)")
    p.add_argument("--activation", choices=["relu", "gelu", "silu"], default="gelu")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", dest="val_frac", type=float, default=0.05)
    p.add_argument("--test-frac", dest="test_frac", type=float, default=0.05)
    p.add_argument("--no-clean-only", dest="clean_only", action="store_false",
                   help="train on all samples (default: clean_only)")
    p.set_defaults(clean_only=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--num-workers", dest="num_workers", type=int, default=0)
    p.add_argument("--log-every", dest="log_every", type=int, default=50)
    p.add_argument("--grad-clip", dest="grad_clip", type=float, default=1.0)
    p.add_argument("--w-ball-pos", dest="w_ball_pos", type=float, default=1.0)
    p.add_argument("--w-ball-vel", dest="w_ball_vel", type=float, default=1.0)
    p.add_argument("--w-net", dest="w_net", type=float, default=1.0)
    p.add_argument(
        "--no-preload", dest="preload", action="store_false",
        help="disable preloading the train+val arrays into RAM. "
        "Default: preload train+val (uses ~52 GB RAM for a 14k-clean-sample "
        "subset, but makes training ~20x faster by removing HDF5 I/O from "
        "the loop).",
    )
    p.set_defaults(preload=True)
    p.add_argument(
        "--preload-test", dest="preload_test", action="store_true",
        help="also preload the test split (default: stream from HDF5 since "
        "test runs only once at end-of-training).",
    )
    p.set_defaults(preload_test=False)
    p.add_argument("--smoke", action="store_true",
                   help="tiny run (1280 samples, 2 epochs) for sanity")
    p.add_argument("--lr-min-frac", dest="lr_min_frac", type=float, default=0.01,
                   help="CosineAnnealingLR final lr as a fraction of --lr "
                   "(default: 0.01). v3 baseline used 0.05 which floored "
                   "training too early on 100-epoch runs.")
    p.add_argument("--norm-scale-mode", dest="norm_scale_mode",
                   choices=["global", "init", "robust"], default="robust",
                   help="how to derive ball_pos_std/ball_vel_std for "
                   "target normalization. 'global'=legacy stddev over all "
                   "frames (suffers from end-game leak); 'init'=use frame-0 "
                   "or input_velocity stddev; 'robust'=half-range of full "
                   "trajectory (RECOMMENDED, fixes regress-to-mean).")
    p.add_argument("--drop-last-frames", dest="drop_last_frames",
                   type=int, default=5,
                   help="exclude the last K frames from the per-epoch random "
                   "offset table (default: 5). Avoids the t_norm=1 boundary "
                   "artifact in the sin/cos time embedding. Set to 0 to "
                   "disable.")


def cfg_from_args(args: argparse.Namespace) -> TrainConfig:
    epochs = 2 if args.smoke else args.epochs
    return TrainConfig(
        dataset=args.dataset,
        output=args.output,
        epochs=epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden=tuple(args.hidden),
        activation=args.activation,
        dropout=args.dropout,
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        clean_only=args.clean_only,
        device=args.device,
        num_workers=args.num_workers,
        log_every=args.log_every,
        grad_clip=args.grad_clip,
        w_ball_pos=args.w_ball_pos,
        w_ball_vel=args.w_ball_vel,
        w_net=args.w_net,
        n_time_freq=args.n_time_freq,
        preload=args.preload,
        preload_test=args.preload_test,
        smoke=args.smoke,
        norm_scale_mode=args.norm_scale_mode,
        drop_last_frames=args.drop_last_frames,
        lr_min_frac=args.lr_min_frac,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train MLP surrogate on goal-net dataset")
    add_train_args(parser)
    args = parser.parse_args(argv)
    cfg = cfg_from_args(args)
    paths = run_training(cfg)
    print("wrote:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return 0


# ===========================================================================
# Online training pipeline (sim → pool → train, design.md §19.2)
# ===========================================================================
#
# The offline trainer above reads a fixed dataset.h5 from disk; once a clean
# sample is selected it is replayed every epoch. Empirically the 29M MLP in
# mlp_v6 saturates after ~16k clean samples (mlp_v7 already overfits at 1k
# epochs). The online pipeline removes the dataset entirely: a Warp solver
# generates a fresh batch every ~12 s and feeds it directly into a sliding
# pool sampled by the trainer. There is no concept of "epoch" — only a step
# counter — and the only on-disk artifact is the checkpoint.
#
# Architectural notes
# -------------------
# * Single GPU is assumed (4090 has plenty of headroom: ~2.5 GB sim + ~5 GB
#   train ≈ 7.5 GB). PyTorch and Warp share the default CUDA stream; in
#   practice Warp synchronizes its kernel graph at simulate_arrays' end so
#   the trainer simply waits its turn each refill.
# * Validation uses a *fixed* hold-out pool generated once with a different
#   seed range and never refreshed. Val RMSE is the optimization target for
#   `best.pt` selection.
# * Normalization stats are computed once during warm-up and frozen — the
#   trajectories from a fixed sampler distribution are statistically
#   homogeneous, so re-estimating during training adds noise without value.


@dataclass
class OnlineTrainConfig:
    """Hyperparameters for online (sim+train) training.

    Note we deliberately reuse a small subset of the offline TrainConfig
    knobs so users can read existing docs and translate intuitions.
    """
    output: str

    # ---- model + optimizer (same semantics as offline) ----
    hidden: Tuple[int, ...] = (1024, 1024, 1024, 1024, 1024, 1024)
    activation: str = "gelu"
    dropout: float = 0.0
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    w_ball_pos: float = 1.0
    w_ball_vel: float = 1.0
    w_net: float = 1.0
    n_time_freq: int = 4
    drop_last_frames: int = 5
    lr_min_frac: float = 0.01
    norm_scale_mode: str = "robust"

    # ---- online training schedule ----
    total_steps: int = 50000
    refill_every: int = 50            # train_steps between two sim refills
    warmup_refills: int = 4           # how many initial sim batches before training starts
    val_every_refills: int = 10       # validate every N refills
    log_every: int = 50               # train-step log cadence
    seed: int = 0                     # base seed; sampler seeds derive from this
    device: str = "cuda"

    # ---- simulator ----
    sim_batch: int = 512              # samples per Warp simulate_arrays call
    train_batch: int = 4096           # mini-batch the trainer pulls from the pool
    pool_batches: int = 4             # K = ring-buffer depth (in sim batches)
    max_contacts: int = 16384         # passed to XpbdWarpSolver

    # ---- validation pool ----
    val_shots: int = 1024             # total clean samples we *aim* to hold
    val_seed: int = 999_001           # seed for val sampler; disjoint from train

    # Optional cap on warmup_refills if pool capacity is smaller.
    # (Effectively warmup_refills = min(warmup_refills, pool_batches).)


# ---------------------------------------------------------------------------
# Helpers shared between online warmup and validation
# ---------------------------------------------------------------------------


def _shots_to_balls(shots) -> List:
    """Extract BallState list + sample_id list from sampler.ShotInput list."""
    balls = [s.ball for s in shots]
    sample_ids = [s.sample_id for s in shots]
    return balls, sample_ids


def _stream_compute_norm_stats_from_pool(
    pool, n_time_freq: int, norm_scale_mode: str
) -> Dict[str, "torch.Tensor"]:
    """Same statistic formula as ``compute_norm_stats`` but consumes the
    online pool's iterator instead of an OffsetFrameDataset.

    Pool must contain at least one filled slot.
    """
    import torch

    if pool.total_valid_samples == 0:
        raise RuntimeError("pool is empty; cannot compute stats")
    if norm_scale_mode not in ("global", "init", "robust"):
        raise ValueError(f"unknown norm_scale_mode={norm_scale_mode!r}")

    F = pool.F
    N = pool.N

    # ---- input statistics: sample (sample, frame) pairs from the pool ----
    rng = np.random.default_rng(424242)
    n_samples_for_input = min(8192, pool.total_valid_samples * F)
    pos_xy_list = []
    vel_list = []
    ang_list = []
    rad_list = []
    mas_list = []
    t_list = []
    items = list(pool.iter_all_frames())
    n_items = len(items)
    for _ in range(n_samples_for_input):
        s = items[rng.integers(0, n_items)]
        f = int(rng.integers(0, F))
        pos_xy_list.append(s["pos_xy"])
        vel_list.append(s["vel"])
        ang_list.append(s["ang"])
        rad_list.append(s["radius"])
        mas_list.append(s["mass"])
        t_list.append(f / max(F - 1, 1))

    pos_xy = np.asarray(pos_xy_list, dtype=np.float32)
    vel = np.asarray(vel_list, dtype=np.float32)
    ang = np.asarray(ang_list, dtype=np.float32)
    rad = np.asarray(rad_list, dtype=np.float32)
    mas = np.asarray(mas_list, dtype=np.float32)
    tnorm = np.asarray(t_list, dtype=np.float32)

    from model import encode_input_features
    feats = encode_input_features(
        pos_xy=torch.from_numpy(pos_xy),
        vel=torch.from_numpy(vel),
        ang=torch.from_numpy(ang),
        radius=torch.from_numpy(rad),
        mass=torch.from_numpy(mas),
        t_norm=torch.from_numpy(tnorm),
        n_time_freq=n_time_freq,
    )
    in_mean = feats.mean(0)
    in_std = torch.clamp(feats.std(0), min=1e-3)

    # ---- target statistics: streaming over all valid samples ----
    sum_bp = np.zeros(3, dtype=np.float64)
    sumsq_bp = np.zeros(3, dtype=np.float64)
    sum_bv = np.zeros(3, dtype=np.float64)
    sumsq_bv = np.zeros(3, dtype=np.float64)
    min_bp = np.full(3, np.inf, dtype=np.float64)
    max_bp = np.full(3, -np.inf, dtype=np.float64)
    min_bv = np.full(3, np.inf, dtype=np.float64)
    max_bv = np.full(3, -np.inf, dtype=np.float64)
    sum_bp_f0 = np.zeros(3, dtype=np.float64)
    sumsq_bp_f0 = np.zeros(3, dtype=np.float64)
    sum_bv_f0 = np.zeros(3, dtype=np.float64)
    sumsq_bv_f0 = np.zeros(3, dtype=np.float64)
    sum_net = np.zeros((N, 3), dtype=np.float64)
    sumsq_net = np.zeros((N, 3), dtype=np.float64)
    total_frames = 0
    n_samples = 0

    for s in pool.iter_all_frames():
        bp64 = s["ball_pos"].astype(np.float64)
        bv64 = s["ball_vel"].astype(np.float64)
        net64 = s["net_pos"].astype(np.float64)
        sum_bp += bp64.sum(axis=0)
        sumsq_bp += (bp64 ** 2).sum(axis=0)
        sum_bv += bv64.sum(axis=0)
        sumsq_bv += (bv64 ** 2).sum(axis=0)
        np.minimum(min_bp, bp64.min(axis=0), out=min_bp)
        np.maximum(max_bp, bp64.max(axis=0), out=max_bp)
        np.minimum(min_bv, bv64.min(axis=0), out=min_bv)
        np.maximum(max_bv, bv64.max(axis=0), out=max_bv)
        sum_bp_f0 += bp64[0]
        sumsq_bp_f0 += bp64[0] ** 2
        sum_bv_f0 += bv64[0]
        sumsq_bv_f0 += bv64[0] ** 2
        sum_net += net64.sum(axis=0)
        sumsq_net += (net64 ** 2).sum(axis=0)
        total_frames += F
        n_samples += 1

    M = float(total_frames)
    n_f = float(n_samples)
    ball_pos_mean_np = sum_bp / M
    ball_vel_mean_np = sum_bv / M
    net_mean_np = sum_net / M
    ball_pos_var_global = np.maximum(sumsq_bp / M - ball_pos_mean_np ** 2, 0.0)
    ball_vel_var_global = np.maximum(sumsq_bv / M - ball_vel_mean_np ** 2, 0.0)
    net_var_np = np.maximum(sumsq_net / M - net_mean_np ** 2, 0.0)
    bp_f0_mean = sum_bp_f0 / n_f
    bv_f0_mean = sum_bv_f0 / n_f
    ball_pos_var_init = np.maximum(sumsq_bp_f0 / n_f - bp_f0_mean ** 2, 0.0)
    ball_vel_var_init = np.maximum(sumsq_bv_f0 / n_f - bv_f0_mean ** 2, 0.0)

    std_global_bp = np.sqrt(ball_pos_var_global)
    std_global_bv = np.sqrt(ball_vel_var_global)
    std_init_bp = np.sqrt(ball_pos_var_init)
    std_init_bv = np.sqrt(ball_vel_var_init)
    std_robust_bp = (max_bp - min_bp) / 2.0
    std_robust_bv = (max_bv - min_bv) / 2.0

    if norm_scale_mode == "global":
        ball_pos_std_np = std_global_bp
        ball_vel_std_np = std_global_bv
    elif norm_scale_mode == "init":
        ball_pos_std_np = std_init_bp
        ball_vel_std_np = std_init_bv
    else:
        ball_pos_std_np = std_robust_bp
        ball_vel_std_np = std_robust_bv

    print(f"    norm_scale_mode={norm_scale_mode}", flush=True)
    print(f"    ball_pos_std chosen={ball_pos_std_np.tolist()}", flush=True)
    print(f"    ball_vel_std chosen={ball_vel_std_np.tolist()}", flush=True)

    ball_pos_mean = torch.from_numpy(ball_pos_mean_np.astype(np.float32))
    ball_vel_mean = torch.from_numpy(ball_vel_mean_np.astype(np.float32))
    net_mean = torch.from_numpy(net_mean_np.astype(np.float32))
    ball_pos_std = torch.clamp(torch.from_numpy(ball_pos_std_np.astype(np.float32)), min=1e-3)
    ball_vel_std = torch.clamp(torch.from_numpy(ball_vel_std_np.astype(np.float32)), min=1e-3)
    net_std = torch.clamp(torch.from_numpy(np.sqrt(net_var_np).astype(np.float32)), min=1e-3)

    return {
        "in_mean": in_mean,
        "in_std": in_std,
        "ball_pos_mean": ball_pos_mean,
        "ball_pos_std": ball_pos_std,
        "ball_vel_mean": ball_vel_mean,
        "ball_vel_std": ball_vel_std,
        "net_mean": net_mean,
        "net_std": net_std,
    }


def _np_batch_to_torch(batch_np: Dict[str, np.ndarray], device) -> Dict[str, "torch.Tensor"]:
    """Convert a pool.sample() dict (numpy) into a torch dict on device."""
    import torch
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch_np.items():
        t = torch.from_numpy(v) if not isinstance(v, np.ndarray) or True else None  # noqa
        out[k] = torch.from_numpy(v).to(device, non_blocking=True)
    return out


def _evaluate_on_pool(model, val_pool, device, cfg: "OnlineTrainConfig",
                       eval_batch: int = 4096,
                       n_eval_batches: int = 8) -> Dict[str, float]:
    """Compute val metrics by sampling from the held-out pool.

    We do n_eval_batches * eval_batch random (sample, frame) pairs which
    is enough for stable RMSE numbers (val_pool has thousands of clean
    samples × 601 frames = millions of (sample, frame) cells).
    """
    import torch
    model.eval()
    sums = {"ball_pos": 0.0, "ball_vel": 0.0, "net": 0.0, "total": 0.0,
            "ball_pos_phys": 0.0, "ball_vel_phys": 0.0, "net_phys": 0.0}
    n_seen = 0
    rng = np.random.default_rng(20260516)
    with torch.no_grad():
        for _ in range(n_eval_batches):
            np_batch = val_pool.sample(eval_batch, rng)
            batch = {k: torch.from_numpy(v).to(device, non_blocking=True)
                     for k, v in np_batch.items()}
            x = encode_batch_input(batch, cfg.n_time_freq)
            pred_norm = model(x, return_normalized=True)
            total, parts = compute_loss(model, pred_norm, batch, cfg)
            pred_phys = model(x, return_normalized=False)
            phys = compute_phys_metrics(pred_phys, batch)
            B = x.shape[0]
            sums["total"] += float(total.detach()) * B
            for k_, v_ in parts.items():
                sums[k_] += v_ * B
            for k_, v_ in phys.items():
                sums[k_] += v_ * B
            n_seen += B
    for k_ in sums:
        sums[k_] /= max(n_seen, 1)
    return sums


def run_online_training(cfg: OnlineTrainConfig) -> Dict[str, str]:
    """End-to-end online sim→train pipeline.

    Side-effects:
        Writes ``cfg.output / {config.json, metrics.jsonl, best.pt, last.pt}``.

    Returns:
        Map of artifact label → path.
    """
    import torch

    out = Path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1) Build sim infrastructure ----
    print("== online training: bringing up Warp solver ==", flush=True)
    from params import GoalNetParams
    from topology import generate_topology
    from sampler import SamplerConfig, sample_shots
    from solver_warp import XpbdWarpSolver
    from online_pool import OnlineFramePool

    params = GoalNetParams()
    topo = generate_topology(params)
    print(f"  topology: {len(topo.particles)} particles, "
          f"{len(topo.distance_constraints)} constraints", flush=True)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[warn] --device cuda requested but torch CUDA unavailable; "
              "falling back to CPU. Sim will also use CPU.", flush=True)
        sim_device = "cpu"
        torch_device = torch.device("cpu")
    else:
        sim_device = cfg.device
        torch_device = torch.device(cfg.device)

    solver = XpbdWarpSolver(
        params=params,
        topology=topo,
        batch_size=cfg.sim_batch,
        device=sim_device,
        record_particles=True,
        max_contacts=cfg.max_contacts,
    )

    # Validation solver uses its own batch size (= val_shots, capped at
    # sim_batch) so we can run it once and amortize.
    val_sim_batch = min(cfg.val_shots, cfg.sim_batch)
    val_solver = XpbdWarpSolver(
        params=params,
        topology=topo,
        batch_size=val_sim_batch,
        device=sim_device,
        record_particles=True,
        max_contacts=cfg.max_contacts,
    )

    F = solver.frame_count
    N = solver.N
    print(f"  solver: B_sim={cfg.sim_batch} F={F} N={N} device={sim_device}",
          flush=True)

    # ---- 2) Validation pool: generate ONCE with disjoint seed range ----
    print("== generating fixed validation pool ==", flush=True)
    val_pool = OnlineFramePool(
        capacity_batches=max(1, (cfg.val_shots + val_sim_batch - 1) // val_sim_batch),
        max_samples_per_batch=val_sim_batch,
        frame_count=F,
        particle_count=N,
        drop_last_frames=cfg.drop_last_frames,
    )
    val_total = 0
    val_target = cfg.val_shots
    val_seed_iter = cfg.val_seed
    while val_total < val_target and val_pool.n_filled_slots < val_pool.K:
        scfg = SamplerConfig(count=val_sim_batch, seed=val_seed_iter)
        shots = sample_shots(scfg)
        balls, sample_ids = _shots_to_balls(shots)
        arrs = val_solver.simulate_arrays(balls, sample_ids)
        n = val_pool.push(arrs, balls)
        val_total += n
        val_seed_iter += 1
        print(f"  val refill seed={val_seed_iter-1} → +{n} clean "
              f"(total {val_total}/{val_target})", flush=True)
    print(f"  val pool: {val_pool.total_valid_samples} clean samples "
          f"({val_pool.n_filled_slots}/{val_pool.K} slots)", flush=True)

    # ---- 3) Train pool: warm up ----
    print("== warming up training pool ==", flush=True)
    train_pool = OnlineFramePool(
        capacity_batches=cfg.pool_batches,
        max_samples_per_batch=cfg.sim_batch,
        frame_count=F,
        particle_count=N,
        drop_last_frames=cfg.drop_last_frames,
    )
    warmup_refills = min(max(cfg.warmup_refills, 1), cfg.pool_batches)
    train_seed_iter = cfg.seed
    sim_step = 0  # how many sim batches have been pushed so far
    sim_time_total = 0.0
    for _ in range(warmup_refills):
        scfg = SamplerConfig(count=cfg.sim_batch, seed=train_seed_iter)
        shots = sample_shots(scfg)
        balls, sample_ids = _shots_to_balls(shots)
        t_sim0 = time.time()
        arrs = solver.simulate_arrays(balls, sample_ids)
        sim_dt = time.time() - t_sim0
        sim_time_total += sim_dt
        n = train_pool.push(arrs, balls)
        sim_step += 1
        train_seed_iter += 1
        print(f"  train refill {sim_step} seed={train_seed_iter-1}: "
              f"+{n} clean / {cfg.sim_batch} ({sim_dt:.1f}s)", flush=True)
    print(f"  train pool warmed: {train_pool.total_valid_samples} clean samples "
          f"({train_pool.n_filled_slots}/{train_pool.K} slots, "
          f"avg {sim_time_total/warmup_refills:.1f}s/refill)", flush=True)

    if train_pool.total_valid_samples == 0:
        raise RuntimeError("no clean samples after warmup; sampler/solver "
                           "is broken or all samples failed quality check")

    # ---- 4) Build model + stats from warmup data ----
    print("== building model + computing normalization stats ==", flush=True)
    from model import GoalNetMLP, count_parameters, input_dim_for
    in_dim = input_dim_for(cfg.n_time_freq)
    model = GoalNetMLP(
        in_dim=in_dim,
        n_particles=N,
        hidden=tuple(cfg.hidden),
        activation=cfg.activation,
        predict_velocity=True,
        dropout=cfg.dropout,
    ).to(torch_device)
    print(f"  model: GoalNetMLP, params={count_parameters(model):,}, "
          f"in_dim={in_dim}, device={torch_device}", flush=True)

    stats = _stream_compute_norm_stats_from_pool(
        train_pool, cfg.n_time_freq, cfg.norm_scale_mode
    )
    model.set_norm_stats({k: v.to(torch_device) for k, v in stats.items()})
    print(f"  norm: ball_pos_std={stats['ball_pos_std'].tolist()}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.total_steps, eta_min=cfg.lr * cfg.lr_min_frac
    )

    # ---- 5) Save config ----
    cfg_dump = asdict(cfg)
    cfg_dump["hidden"] = list(cfg.hidden)
    cfg_dump["particle_count"] = N
    cfg_dump["frame_count"] = F
    cfg_dump["frame_dt"] = float(params.solver.frame_dt)
    cfg_dump["mode"] = "online"
    (out / "config.json").write_text(json.dumps(cfg_dump, indent=2))

    metrics_path = out / "metrics.jsonl"
    metrics_fp = metrics_path.open("w")
    best_val = math.inf
    best_path = out / "best.pt"
    last_path = out / "last.pt"

    # ---- 6) Main loop ----
    print(f"== training: {cfg.total_steps} steps, "
          f"refill every {cfg.refill_every} steps ==", flush=True)
    t_start = time.time()
    rng = np.random.default_rng(cfg.seed + 1)
    n_refills = 0
    sim_time_total = 0.0
    train_time_total = 0.0
    running = {"ball_pos": 0.0, "ball_vel": 0.0, "net": 0.0, "total": 0.0,
               "ball_pos_phys": 0.0, "ball_vel_phys": 0.0, "net_phys": 0.0}
    n_seen_run = 0

    for step in range(cfg.total_steps):
        # Refill cadence: at boundary step (and not step 0 which is post-warmup).
        if step > 0 and step % cfg.refill_every == 0:
            scfg = SamplerConfig(count=cfg.sim_batch, seed=train_seed_iter)
            shots = sample_shots(scfg)
            balls, sample_ids = _shots_to_balls(shots)
            t_sim0 = time.time()
            arrs = solver.simulate_arrays(balls, sample_ids)
            sim_dt = time.time() - t_sim0
            sim_time_total += sim_dt
            n_clean = train_pool.push(arrs, balls)
            n_refills += 1
            train_seed_iter += 1
            print(f"  [step {step}] refill {n_refills}: "
                  f"+{n_clean}/{cfg.sim_batch} clean ({sim_dt:.1f}s) "
                  f"pool={train_pool.total_valid_samples}", flush=True)

            # Validation cadence aligned with refills.
            if n_refills % cfg.val_every_refills == 0:
                val_metrics = _evaluate_on_pool(model, val_pool, torch_device, cfg)
                running_avg = {k: v / max(n_seen_run, 1) for k, v in running.items()}
                rec = {
                    "step": step,
                    "refill": n_refills,
                    "train_running": running_avg,
                    "val": val_metrics,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_s": time.time() - t_start,
                    "sim_time_s_total": sim_time_total,
                    "train_time_s_total": train_time_total,
                    "pool_total": train_pool.total_valid_samples,
                    "pool_dropped_dirty": train_pool.stats.total_dropped_dirty,
                }
                metrics_fp.write(json.dumps(rec) + "\n")
                metrics_fp.flush()
                print(f"    val: pos={val_metrics['ball_pos_phys']:.3f} "
                      f"vel={val_metrics['ball_vel_phys']:.3f} "
                      f"net={val_metrics['net_phys']:.4f} "
                      f"total_norm={val_metrics['total']:.4e}", flush=True)

                if val_metrics["total"] < best_val:
                    best_val = val_metrics["total"]
                    torch.save({
                        "step": step,
                        "refill": n_refills,
                        "model_state": model.state_dict(),
                        "optim_state": optimizer.state_dict(),
                        "config": cfg_dump,
                        "val_loss": val_metrics,
                    }, best_path)

                # Reset running averages so they reflect post-validation activity.
                running = {k: 0.0 for k in running}
                n_seen_run = 0
                model.train()

        # Train step.
        np_batch = train_pool.sample(cfg.train_batch, rng)
        batch = {k: torch.from_numpy(v).to(torch_device, non_blocking=True)
                 for k, v in np_batch.items()}
        t_train0 = time.time()
        model.train()
        x = encode_batch_input(batch, cfg.n_time_freq)
        pred_norm = model(x, return_normalized=True)
        total, parts = compute_loss(model, pred_norm, batch, cfg)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            pred_phys = model(x, return_normalized=False)
            phys = compute_phys_metrics(pred_phys, batch)

        train_dt = time.time() - t_train0
        train_time_total += train_dt
        B = x.shape[0]
        running["total"] += float(total.detach()) * B
        for k_, v_ in parts.items():
            running[k_] += v_ * B
        for k_, v_ in phys.items():
            running[k_] += v_ * B
        n_seen_run += B

        if (step + 1) % cfg.log_every == 0:
            avg = {k: v / max(n_seen_run, 1) for k, v in running.items()}
            elapsed = time.time() - t_start
            lr = scheduler.get_last_lr()[0]
            steps_per_s = (step + 1) / max(elapsed, 1e-6)
            eta = (cfg.total_steps - step - 1) / max(steps_per_s, 1e-6)
            print(f"  step {step+1:06d}/{cfg.total_steps} "
                  f"loss={avg['total']:.4e} "
                  f"phys[pos={avg['ball_pos_phys']:.3f} "
                  f"vel={avg.get('ball_vel_phys', 0):.3f} "
                  f"net={avg['net_phys']:.4f}] "
                  f"lr={lr:.2e} elapsed={elapsed/60:.1f}min "
                  f"eta={eta/60:.1f}min", flush=True)

    # ---- 7) Final save ----
    torch.save({
        "step": cfg.total_steps,
        "refill": n_refills,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "config": cfg_dump,
        "val_loss": None,
    }, last_path)

    # Final validation pass for completeness.
    final_val = _evaluate_on_pool(model, val_pool, torch_device, cfg,
                                    n_eval_batches=16)
    (out / "final_val_metrics.json").write_text(json.dumps(final_val, indent=2))

    metrics_fp.close()

    print("== online training done ==", flush=True)
    print(f"  total time: {(time.time()-t_start)/60:.1f} min "
          f"({sim_time_total:.1f}s sim + {train_time_total:.1f}s train)",
          flush=True)
    print(f"  final val: pos={final_val['ball_pos_phys']:.3f} "
          f"vel={final_val['ball_vel_phys']:.3f} "
          f"net={final_val['net_phys']:.4f}", flush=True)

    return {
        "config": str(out / "config.json"),
        "metrics": str(metrics_path),
        "best_ckpt": str(best_path),
        "last_ckpt": str(last_path),
        "final_val_metrics": str(out / "final_val_metrics.json"),
    }


def add_train_online_args(p: argparse.ArgumentParser) -> None:
    """CLI for `cli.py train-online`."""
    p.add_argument("--output", required=True, help="output directory")
    # Model / optimizer
    p.add_argument("--hidden", type=int, nargs="+",
                   default=[1024, 1024, 1024, 1024, 1024, 1024])
    p.add_argument("--activation", choices=["relu", "gelu", "silu"], default="gelu")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", dest="grad_clip", type=float, default=1.0)
    p.add_argument("--w-ball-pos", dest="w_ball_pos", type=float, default=1.0)
    p.add_argument("--w-ball-vel", dest="w_ball_vel", type=float, default=1.0)
    p.add_argument("--w-net", dest="w_net", type=float, default=1.0)
    p.add_argument("--n-time-freq", dest="n_time_freq", type=int, default=4)
    p.add_argument("--drop-last-frames", dest="drop_last_frames", type=int, default=5)
    p.add_argument("--lr-min-frac", dest="lr_min_frac", type=float, default=0.01)
    p.add_argument("--norm-scale-mode", dest="norm_scale_mode",
                   choices=["global", "init", "robust"], default="robust")
    # Online schedule
    p.add_argument("--total-steps", dest="total_steps", type=int, default=50000,
                   help="total optimizer steps (default 50000 ≈ "
                        "1000 sim refills at refill_every=50)")
    p.add_argument("--refill-every", dest="refill_every", type=int, default=50,
                   help="train steps between two sim refills")
    p.add_argument("--warmup-refills", dest="warmup_refills", type=int, default=4,
                   help="initial sim batches before training starts (capped "
                        "at --pool-batches)")
    p.add_argument("--val-every-refills", dest="val_every_refills", type=int, default=10)
    p.add_argument("--log-every", dest="log_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    # Simulator
    p.add_argument("--sim-batch", dest="sim_batch", type=int, default=512)
    p.add_argument("--train-batch", dest="train_batch", type=int, default=4096)
    p.add_argument("--pool-batches", dest="pool_batches", type=int, default=4)
    p.add_argument("--max-contacts", dest="max_contacts", type=int, default=16384)
    # Validation pool
    p.add_argument("--val-shots", dest="val_shots", type=int, default=1024)
    p.add_argument("--val-seed", dest="val_seed", type=int, default=999_001)
    # Smoke
    p.add_argument("--smoke", action="store_true",
                   help="tiny run for sanity (overrides several knobs)")


def online_cfg_from_args(args: argparse.Namespace) -> OnlineTrainConfig:
    if args.smoke:
        # Tiny config that completes in < 2 minutes on CPU device (Warp CPU
        # backend is much slower than CUDA but works on any machine).
        return OnlineTrainConfig(
            output=args.output,
            hidden=(64, 64),
            total_steps=50,
            refill_every=20,
            warmup_refills=2,
            val_every_refills=1,
            log_every=10,
            sim_batch=8,
            train_batch=64,
            pool_batches=2,
            val_shots=16,
            device=args.device,
            seed=args.seed,
            max_contacts=2048,
            n_time_freq=args.n_time_freq,
            drop_last_frames=args.drop_last_frames,
        )
    return OnlineTrainConfig(
        output=args.output,
        hidden=tuple(args.hidden),
        activation=args.activation,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        w_ball_pos=args.w_ball_pos,
        w_ball_vel=args.w_ball_vel,
        w_net=args.w_net,
        n_time_freq=args.n_time_freq,
        drop_last_frames=args.drop_last_frames,
        lr_min_frac=args.lr_min_frac,
        norm_scale_mode=args.norm_scale_mode,
        total_steps=args.total_steps,
        refill_every=args.refill_every,
        warmup_refills=args.warmup_refills,
        val_every_refills=args.val_every_refills,
        log_every=args.log_every,
        seed=args.seed,
        device=args.device,
        sim_batch=args.sim_batch,
        train_batch=args.train_batch,
        pool_batches=args.pool_batches,
        max_contacts=args.max_contacts,
        val_shots=args.val_shots,
        val_seed=args.val_seed,
    )


if __name__ == "__main__":
    sys.exit(main())
