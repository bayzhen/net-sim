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
    preload: bool = True
    smoke: bool = False  # tiny run for sanity


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
    ) -> None:
        import h5py

        self._h5py = h5py
        self.path = h5_path
        self.indices = np.asarray(sample_indices, dtype=np.int64)
        self.load_particles = load_particles
        self.preload = bool(preload)
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
        total_gb = (ball_bytes * 2 + net_bytes) / (1024 ** 3)
        tag = f" [{label}]" if label else ""
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
        """Reseed the per-sample frame-offset table for this epoch."""
        rng = np.random.default_rng(self._epoch_seed + epoch)
        F = self.frame_count
        self._epoch_offsets = rng.integers(0, F, size=int(self.indices.shape[0]),
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
                       max_samples_for_input: int = 8192) -> Dict[str, "torch.Tensor"]:
    """Compute mean/std for input features and targets from the train split.

    For target stats we use **all frames of all train samples** (so the
    estimator is very tight); for input stats we just need ``n`` samples
    × ``n_time_freq*2 + 10`` so 8192 is plenty.

    Requires ``train_ds.preload == True`` so we can iterate the cached
    arrays without re-reading HDF5.
    """
    import torch

    if not train_ds.preload:
        raise RuntimeError("compute_norm_stats currently requires preload=True")

    n = len(train_ds)
    F = train_ds.frame_count

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
    in_std = feats.std(0)
    # Avoid std=0 on constant features (radius/mass/sin(2pi*0)=0 etc.).
    in_std = torch.clamp(in_std, min=1e-3)

    # ------- target statistics (use all frames) -------
    bp = train_ds._ball_pos_cache.reshape(-1, 3)        # (n*F, 3)
    bv = train_ds._ball_vel_cache.reshape(-1, 3)
    ball_pos_mean = torch.from_numpy(bp.mean(axis=0))
    ball_pos_std = torch.from_numpy(bp.std(axis=0))
    ball_vel_mean = torch.from_numpy(bv.mean(axis=0))
    ball_vel_std = torch.from_numpy(bv.std(axis=0))
    ball_pos_std = torch.clamp(ball_pos_std, min=1e-3)
    ball_vel_std = torch.clamp(ball_vel_std, min=1e-3)

    # Per-particle mean/std: (N, 3). Computing on CPU with float32 avoids
    # the (n*F, N, 3) reshape blowup; just reduce along axes 0,1.
    net_cache = train_ds._net_cache  # (n, F, N, 3)
    net_mean = torch.from_numpy(net_cache.mean(axis=(0, 1)).astype(np.float32))
    net_std = torch.from_numpy(net_cache.std(axis=(0, 1)).astype(np.float32))
    net_std = torch.clamp(net_std, min=1e-3)

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
                                   preload=cfg.preload, preload_label="train")
    val_ds   = OffsetFrameDataset(cfg.dataset, va_idx, rng_seed=cfg.seed + 1,
                                   preload=cfg.preload, preload_label="val")
    test_ds  = OffsetFrameDataset(cfg.dataset, te_idx, rng_seed=cfg.seed + 2,
                                   preload=cfg.preload, preload_label="test")

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
    stats = compute_norm_stats(train_ds, cfg.n_time_freq)
    model.set_norm_stats({k: v.to(device) for k, v in stats.items()})
    print(f"  done in {time.time()-t_stats0:.1f}s "
          f"(ball_pos_std={stats['ball_pos_std'].tolist()}, "
          f"net_std mean={stats['net_std'].mean().item():.4f})", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    total_steps = max(cfg.epochs * len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=cfg.lr * 0.05
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
        help="disable preloading the train/val/test arrays into RAM. "
        "Default: preload (uses ~52 GB RAM for a 14k-clean-sample subset, "
        "but makes training ~20x faster by removing HDF5 I/O from the loop).",
    )
    p.set_defaults(preload=True)
    p.add_argument("--smoke", action="store_true",
                   help="tiny run (1280 samples, 2 epochs) for sanity")


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
        smoke=args.smoke,
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


if __name__ == "__main__":
    sys.exit(main())
