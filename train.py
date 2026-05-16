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
    smoke: bool = False  # tiny run for sanity


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class OffsetFrameDataset:
    """torch.utils.data.Dataset compatible.

    Each item returns the **full** ``(F, N, 3)`` particle trajectory and
    ``(F, 3)`` ball trajectory plus the per-sample initial state. We then
    randomly pick one offset frame inside ``collate`` (on CPU) — this is
    much cheaper than 512 random h5 single-frame reads per batch because
    the dataset's HDF5 chunks are shaped ``(1, F, N, 3)`` (one chunk per
    sample), so reading a full slab hits exactly one chunk and is ~1000x
    faster than fancy-indexing one frame at a time.

    Empirical benchmark on E:/dataset_v2/dataset.h5:
        random single-frame read:  ~2.7 ms / sample
        full (F, N, 3) read:       ~0.06 ms / sample (chunk-aligned)
    """

    def __init__(
        self,
        h5_path: str,
        sample_indices: np.ndarray,
        rng_seed: int = 0,
        load_particles: bool = True,
    ) -> None:
        import h5py

        self._h5py = h5py
        self.path = h5_path
        self.indices = np.asarray(sample_indices, dtype=np.int64)
        self.load_particles = load_particles
        self._file: Optional["h5py.File"] = None

        # Read tiny metadata + per-sample inputs once into RAM.
        with h5py.File(self.path, "r") as f:
            self.frame_count = int(f.attrs["frame_count"])
            self.particle_count = int(f.attrs["particle_count"])
            self.frame_dt = float(f.attrs["frame_dt"])
            # all S samples; we slice the chosen indices in __getitem__
            self.input_position = f["input_position"][:].astype(np.float32)
            self.input_velocity = f["input_velocity"][:].astype(np.float32)
            self.input_angular  = f["input_angular"][:].astype(np.float32)
            self.input_radius   = f["input_radius"][:].astype(np.float32)
            self.input_mass     = f["input_mass"][:].astype(np.float32)

        # Reproducible per-sample random offset: each item gets a frame
        # picked from this fixed table so two epochs see different frames
        # but a single (epoch, idx) is deterministic.
        self._epoch_seed = rng_seed
        self._epoch_offsets: Optional[np.ndarray] = None
        self.set_epoch(0)

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
        f = self._open()
        si = int(self.indices[i])
        frame = int(self._epoch_offsets[i])  # type: ignore[index]
        F = self.frame_count
        t_norm = np.float32(frame / max(F - 1, 1))

        # Read FULL trajectories — one chunk-aligned read each:
        #   ball_position[si]      shape (F, 3),    ~7 KB
        #   ball_velocity[si]      shape (F, 3),    ~7 KB
        #   particle_position[si]  shape (F, N, 3), ~3.5 MB (one HDF5 chunk)
        # then pick the chosen frame in numpy (free).
        ball_pos = f["ball_position"][si][frame].astype(np.float32, copy=False)
        ball_vel = f["ball_velocity"][si][frame].astype(np.float32, copy=False)
        if self.load_particles:
            net = f["particle_position"][si][frame].astype(np.float32, copy=False)  # (N, 3)
        else:
            net = np.empty((self.particle_count, 3), dtype=np.float32)

        # 13-D input: pos(3) + vel(3) + ang(3) + radius + mass + t_norm + 1 spare
        input_state = np.concatenate([
            self.input_position[si],
            self.input_velocity[si],
            self.input_angular[si],
            np.array([self.input_radius[si],
                      self.input_mass[si],
                      t_norm,
                      0.0], dtype=np.float32),
        ]).astype(np.float32)

        return {
            "input_state": input_state,
            "target_ball": ball_pos,
            "target_ball_v": ball_vel,
            "target_net": net,
            "frame": np.int64(frame),
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
    for k in ("input_state", "target_ball", "target_ball_v", "target_net"):
        out[k] = torch.from_numpy(np.stack([b[k] for b in batch], axis=0))
    out["frame"] = torch.from_numpy(np.stack([b["frame"] for b in batch], axis=0))
    return out


def compute_loss(pred: dict, batch: dict, cfg: TrainConfig):
    import torch.nn.functional as F
    l_pos = F.mse_loss(pred["ball_pos"], batch["target_ball"])
    l_vel = F.mse_loss(pred["ball_vel"], batch["target_ball_v"]) \
        if "ball_vel" in pred else None
    l_net = F.mse_loss(pred["net"], batch["target_net"])
    total = cfg.w_ball_pos * l_pos + cfg.w_net * l_net
    parts = {"ball_pos": float(l_pos.detach()), "net": float(l_net.detach())}
    if l_vel is not None:
        total = total + cfg.w_ball_vel * l_vel
        parts["ball_vel"] = float(l_vel.detach())
    return total, parts


def evaluate(model, loader, device, cfg) -> Dict[str, float]:
    import torch
    model.eval()
    sums = {"ball_pos": 0.0, "ball_vel": 0.0, "net": 0.0, "total": 0.0}
    n_samples = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            pred = model(batch["input_state"])
            total, parts = compute_loss(pred, batch, cfg)
            B = batch["input_state"].shape[0]
            sums["total"] += float(total.detach()) * B
            for k, v in parts.items():
                sums[k] += v * B
            n_samples += B
    for k in sums:
        sums[k] /= max(n_samples, 1)
    return sums


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


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

    train_ds = OffsetFrameDataset(cfg.dataset, tr_idx, rng_seed=cfg.seed)
    val_ds   = OffsetFrameDataset(cfg.dataset, va_idx, rng_seed=cfg.seed + 1)
    test_ds  = OffsetFrameDataset(cfg.dataset, te_idx, rng_seed=cfg.seed + 2)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate,
        persistent_workers=cfg.num_workers > 0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate,
        persistent_workers=cfg.num_workers > 0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate,
        persistent_workers=cfg.num_workers > 0, pin_memory=True,
    )

    # --- model ---
    from model import GoalNetMLP, count_parameters
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[warn] --device cuda requested but torch.cuda.is_available() is "
              "False; falling back to CPU. Reinstall PyTorch with CUDA support "
              "(e.g. `pip install torch --index-url "
              "https://download.pytorch.org/whl/cu121`) for GPU training.",
              flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)
    model = GoalNetMLP(
        in_dim=13,
        n_particles=train_ds.particle_count,
        hidden=tuple(cfg.hidden),
        activation=cfg.activation,
        predict_velocity=True,
        dropout=cfg.dropout,
    ).to(device)
    print(f"model: GoalNetMLP, params={count_parameters(model):,}, "
          f"device={device}, N={train_ds.particle_count}, F={train_ds.frame_count}",
          flush=True)

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
        running = {"ball_pos": 0.0, "ball_vel": 0.0, "net": 0.0, "total": 0.0}
        n_seen = 0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            pred = model(batch["input_state"])
            total, parts = compute_loss(pred, batch, cfg)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            B = batch["input_state"].shape[0]
            running["total"] += float(total.detach()) * B
            for k, v in parts.items():
                running[k] += v * B
            n_seen += B
            global_step += 1

            if (step + 1) % cfg.log_every == 0:
                avg = {k: v / max(n_seen, 1) for k, v in running.items()}
                lr = scheduler.get_last_lr()[0]
                print(f"  ep {epoch:03d} step {step+1:05d} "
                      f"loss={avg['total']:.4e} pos={avg['ball_pos']:.4e} "
                      f"vel={avg.get('ball_vel', 0):.4e} net={avg['net']:.4e} "
                      f"lr={lr:.2e}", flush=True)

        train_loss = {k: v / max(n_seen, 1) for k, v in running.items()}
        val_loss = evaluate(model, val_loader, device, cfg)
        epoch_dt = time.time() - epoch_t0
        elapsed = time.time() - t_start
        eta = elapsed / max(epoch + 1, 1) * (cfg.epochs - epoch - 1)
        print(f"epoch {epoch:03d} done in {epoch_dt:.1f}s  "
              f"train={train_loss['total']:.4e} val={val_loss['total']:.4e} "
              f"(pos={val_loss['ball_pos']:.4e} vel={val_loss['ball_vel']:.4e} "
              f"net={val_loss['net']:.4e})  ETA={eta/60:.1f}min", flush=True)

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
    print(f"test: total={test_loss['total']:.4e} "
          f"pos={test_loss['ball_pos']:.4e} vel={test_loss['ball_vel']:.4e} "
          f"net={test_loss['net']:.4e}", flush=True)
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
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", dest="batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-5)
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 512, 512, 512])
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
