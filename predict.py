"""Evaluate / predict with a trained surrogate checkpoint.

Two modes:

* ``per-frame``  — for each test sample, run all F frames through the
  model and report RMSE-vs-frame averaged across the test split. Tells
  us *when* the model fails (e.g. only after the ball hits the net).
* ``worst-K``    — rank test samples by their ball-position RMSE
  (averaged over frames) and print the K worst sample IDs together
  with their per-head errors, so we know which scenarios to inspect
  visually next.

Both are run by default; toggle via ``--mode``.

Example::

    python cli.py predict \\
        --ckpt /data3/netsim/runs/mlp_v2/best.pt \\
        --dataset /data3/netsim/dataset_v2/dataset.h5 \\
        --output /data3/netsim/runs/mlp_v2/eval \\
        --device cuda --batch 1024 --worst-k 16

Outputs (in ``--output`` dir):
    per_frame_rmse.json         # arrays of (F,) for ball_pos / ball_vel / net
    per_sample_summary.json     # list of {sample_id, ball_pos_rmse, ...}
    worst_k.json                # top-K worst by ball_pos_rmse
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _load_ckpt(path: str, device: "torch.device"):
    """Re-instantiate the model from a checkpoint and return
    ``(model, cfg_dump)``."""
    import torch
    from model import GoalNetMLP, input_dim_for

    state = torch.load(path, map_location=device, weights_only=False)
    cfg = state["config"]
    n_time_freq = int(cfg.get("n_time_freq", 4))
    in_dim = input_dim_for(n_time_freq)
    model = GoalNetMLP(
        in_dim=in_dim,
        n_particles=int(cfg["particle_count"]),
        hidden=tuple(cfg["hidden"]),
        activation=str(cfg.get("activation", "gelu")),
        predict_velocity=True,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, cfg


def _build_test_indices(h5_path: str, cfg: dict) -> np.ndarray:
    """Replay the train/val/test split using the seeds and fractions
    stored in the checkpoint config; return the test-split indices.
    """
    from train import load_clean_indices, split_indices

    clean = load_clean_indices(h5_path, bool(cfg.get("clean_only", True)))
    n = int(clean.size)
    if cfg.get("smoke", False):
        clean = clean[:1280]
        n = 1280
    _, _, te = split_indices(
        n,
        float(cfg.get("val_frac", 0.05)),
        float(cfg.get("test_frac", 0.05)),
        int(cfg.get("seed", 0)),
    )
    return clean[te]


# ---------------------------------------------------------------------------
# Per-frame evaluation: streams the test split frame-by-frame.
# ---------------------------------------------------------------------------


def evaluate_per_frame(
    h5_path: str,
    ckpt_path: str,
    output_dir: str,
    device_name: str = "cuda",
    sample_batch: int = 8,
    worst_k: int = 16,
) -> Dict[str, str]:
    """Evaluate the checkpoint on the test split, all frames.

    For each test sample we reconstruct an (F,) sequence of inputs, run
    the model in chunks of ``sample_batch * F`` frames at a time, and
    accumulate squared errors:

        per_frame_rmse[head][f]  = sqrt(mean over samples & axes of
                                        (pred[s, f] - tgt[s, f])^2)

    ``worst_k`` ranks samples by their *time-averaged* ball-position
    RMSE — useful to find pathological inputs.
    """
    import h5py
    import torch
    from model import encode_input_features

    device = torch.device(device_name if (device_name == "cpu"
                                          or torch.cuda.is_available()) else "cpu")
    if device.type != device_name:
        print(f"[warn] requested {device_name}, falling back to {device.type}",
              flush=True)

    print(f"loading checkpoint: {ckpt_path}", flush=True)
    model, cfg = _load_ckpt(ckpt_path, device)
    n_time_freq = int(cfg.get("n_time_freq", 4))
    F = int(cfg["frame_count"])
    N = int(cfg["particle_count"])

    test_idx = _build_test_indices(h5_path, cfg)
    n_test = int(test_idx.size)
    print(f"test samples: {n_test} (F={F}, N={N})", flush=True)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    f = h5py.File(h5_path, "r", swmr=True)

    # Frame indices and t_norm vector are constant across samples.
    t_norm_F = (np.arange(F, dtype=np.float32) / max(F - 1, 1))
    t_norm_F_t = torch.from_numpy(t_norm_F).to(device)

    # Per-frame accumulators, all (F,).
    sumsq_pos = np.zeros(F, dtype=np.float64)   # m^2 (mean over xyz)
    sumsq_vel = np.zeros(F, dtype=np.float64)
    sumsq_net = np.zeros(F, dtype=np.float64)   # m^2 (mean over N*3)
    count_per_frame = np.zeros(F, dtype=np.int64)

    # Per-sample summaries.
    per_sample: List[dict] = []

    t0 = time.time()
    for s_start in range(0, n_test, sample_batch):
        s_end = min(s_start + sample_batch, n_test)
        s_chunk = test_idx[s_start:s_end]            # (B,)
        B = int(s_chunk.size)

        # Read this chunk of samples — small (B,) arrays then big slabs.
        pos_xy = f["input_position"][s_chunk, :2].astype(np.float32)   # (B, 2)
        vel = f["input_velocity"][s_chunk].astype(np.float32)
        ang = f["input_angular"][s_chunk].astype(np.float32)
        rad = f["input_radius"][s_chunk].astype(np.float32)
        mas = f["input_mass"][s_chunk].astype(np.float32)
        sample_ids = [bytes(x).decode("utf-8", errors="replace")
                      if isinstance(x, (bytes, np.bytes_))
                      else str(x) for x in f["sample_id"][s_chunk]]

        # Big targets: (B, F, 3), (B, F, 3), (B, F, N, 3)
        bp_tgt = f["ball_position"][s_chunk]
        bv_tgt = f["ball_velocity"][s_chunk]
        net_tgt = f["particle_position"][s_chunk]

        # Build (B*F, ...) input batch.
        pos_xy_t = torch.from_numpy(pos_xy).to(device)
        vel_t = torch.from_numpy(vel).to(device)
        ang_t = torch.from_numpy(ang).to(device)
        rad_t = torch.from_numpy(rad).to(device)
        mas_t = torch.from_numpy(mas).to(device)

        # Each sample contributes F frames; broadcast inputs.
        pos_xy_BF = pos_xy_t.unsqueeze(1).expand(B, F, 2).reshape(B * F, 2)
        vel_BF = vel_t.unsqueeze(1).expand(B, F, 3).reshape(B * F, 3)
        ang_BF = ang_t.unsqueeze(1).expand(B, F, 3).reshape(B * F, 3)
        rad_BF = rad_t.unsqueeze(1).expand(B, F).reshape(B * F)
        mas_BF = mas_t.unsqueeze(1).expand(B, F).reshape(B * F)
        t_BF = t_norm_F_t.unsqueeze(0).expand(B, F).reshape(B * F)

        x = encode_input_features(
            pos_xy=pos_xy_BF, vel=vel_BF, ang=ang_BF,
            radius=rad_BF, mass=mas_BF, t_norm=t_BF,
            n_time_freq=n_time_freq,
        )

        with torch.no_grad():
            pred = model(x, return_normalized=False)

        pred_bp = pred["ball_pos"].cpu().numpy().reshape(B, F, 3)
        pred_bv = pred.get("ball_vel")
        pred_bv = (pred_bv.cpu().numpy().reshape(B, F, 3)
                   if pred_bv is not None else None)
        pred_net = pred["net"].cpu().numpy().reshape(B, F, N, 3)

        # Per-frame squared error, per sample:
        # mean over xyz so the unit is m^2 (RMSE = sqrt of that).
        err_pos2 = ((pred_bp - bp_tgt) ** 2).mean(axis=2)            # (B, F)
        err_net2 = ((pred_net - net_tgt) ** 2).mean(axis=(2, 3))     # (B, F)
        if pred_bv is not None:
            err_vel2 = ((pred_bv - bv_tgt) ** 2).mean(axis=2)         # (B, F)
        else:
            err_vel2 = None

        sumsq_pos += err_pos2.sum(axis=0).astype(np.float64)
        sumsq_net += err_net2.sum(axis=0).astype(np.float64)
        if err_vel2 is not None:
            sumsq_vel += err_vel2.sum(axis=0).astype(np.float64)
        count_per_frame += B

        # Per-sample summaries (time-averaged).
        for bi in range(B):
            entry = {
                "sample_id": sample_ids[bi],
                "ball_pos_rmse": float(np.sqrt(err_pos2[bi].mean())),
                "net_rmse":      float(np.sqrt(err_net2[bi].mean())),
            }
            if err_vel2 is not None:
                entry["ball_vel_rmse"] = float(np.sqrt(err_vel2[bi].mean()))
            # Worst single frame for this sample
            worst_f = int(err_pos2[bi].argmax())
            entry["worst_frame"] = worst_f
            entry["worst_frame_pos_rmse"] = float(np.sqrt(err_pos2[bi, worst_f]))
            per_sample.append(entry)

        if (s_start // sample_batch + 1) % 20 == 0 or s_end == n_test:
            dt = time.time() - t0
            done = s_end
            rate = done / max(dt, 1e-6)
            print(f"  [{done}/{n_test}] {rate:.1f} samp/s  "
                  f"elapsed {dt:.1f}s", flush=True)

    f.close()

    # Aggregate.
    rmse_pos = np.sqrt(sumsq_pos / np.maximum(count_per_frame, 1))
    rmse_vel = np.sqrt(sumsq_vel / np.maximum(count_per_frame, 1))
    rmse_net = np.sqrt(sumsq_net / np.maximum(count_per_frame, 1))

    # Sort per_sample by ball_pos_rmse desc.
    per_sample.sort(key=lambda r: r["ball_pos_rmse"], reverse=True)

    summary = {
        "per_frame": {
            "frame_dt": float(cfg.get("frame_dt", 1.0 / 60.0)),
            "ball_pos_rmse": rmse_pos.tolist(),
            "ball_vel_rmse": rmse_vel.tolist(),
            "net_rmse":      rmse_net.tolist(),
        },
        "per_sample_count": len(per_sample),
        "summary": {
            "ball_pos_rmse_mean":  float(np.mean([r["ball_pos_rmse"] for r in per_sample])),
            "ball_pos_rmse_med":   float(np.median([r["ball_pos_rmse"] for r in per_sample])),
            "ball_pos_rmse_p95":   float(np.percentile([r["ball_pos_rmse"] for r in per_sample], 95)),
            "net_rmse_mean":       float(np.mean([r["net_rmse"] for r in per_sample])),
            "net_rmse_med":        float(np.median([r["net_rmse"] for r in per_sample])),
            "net_rmse_p95":        float(np.percentile([r["net_rmse"] for r in per_sample], 95)),
        },
        "worst_k": per_sample[:max(worst_k, 0)],
    }

    # ------- write outputs -------
    pf_path = out_dir / "per_frame_rmse.json"
    pf_path.write_text(json.dumps(summary["per_frame"], indent=2))
    sample_path = out_dir / "per_sample_summary.json"
    sample_path.write_text(json.dumps(per_sample, indent=2))
    worst_path = out_dir / "worst_k.json"
    worst_path.write_text(json.dumps(summary["worst_k"], indent=2))
    overall_path = out_dir / "summary.json"
    overall_path.write_text(json.dumps(summary["summary"], indent=2))

    # ------- console summary -------
    print("\n=== per-frame RMSE summary ===")
    s = summary["summary"]
    print(f"  ball_pos_rmse  mean={s['ball_pos_rmse_mean']:.3f} m  "
          f"median={s['ball_pos_rmse_med']:.3f} m  "
          f"p95={s['ball_pos_rmse_p95']:.3f} m")
    print(f"  net_rmse       mean={s['net_rmse_mean']*1000:.2f} mm  "
          f"median={s['net_rmse_med']*1000:.2f} mm  "
          f"p95={s['net_rmse_p95']*1000:.2f} mm")
    # frame trend (sample 5 frames evenly)
    pick = np.linspace(0, F - 1, 5).astype(int)
    print("  frame: " + "  ".join(f"f={int(p):3d}" for p in pick))
    print("  pos  : " + "  ".join(f"{rmse_pos[p]:5.2f}m" for p in pick))
    print("  net  : " + "  ".join(f"{rmse_net[p]*1000:5.1f}mm" for p in pick))
    if rmse_vel.any():
        print("  vel  : " + "  ".join(f"{rmse_vel[p]:5.2f}" for p in pick))
    print(f"\nworst {min(worst_k, len(per_sample))} samples by ball_pos_rmse:")
    for r in summary["worst_k"][:min(worst_k, 10)]:
        print(f"  {r['sample_id']:>40s}  ball_pos={r['ball_pos_rmse']:5.2f}m  "
              f"net={r['net_rmse']*1000:5.1f}mm  "
              f"worst@f{r['worst_frame']}={r['worst_frame_pos_rmse']:.2f}m")

    return {
        "per_frame_rmse": str(pf_path),
        "per_sample_summary": str(sample_path),
        "worst_k": str(worst_path),
        "summary": str(overall_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_predict_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ckpt", required=True, help="path to a .pt checkpoint")
    p.add_argument("--dataset", required=True, help="path to dataset.h5")
    p.add_argument("--output", required=True, help="output directory for eval json")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--batch", dest="sample_batch", type=int, default=8,
                   help="number of test samples per forward (each contributes F frames)")
    p.add_argument("--worst-k", dest="worst_k", type=int, default=16)


def cfg_from_args(args: argparse.Namespace) -> dict:
    return {
        "ckpt": args.ckpt,
        "dataset": args.dataset,
        "output": args.output,
        "device": args.device,
        "sample_batch": args.sample_batch,
        "worst_k": args.worst_k,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a goal-net surrogate checkpoint")
    add_predict_args(parser)
    args = parser.parse_args(argv)
    paths = evaluate_per_frame(
        h5_path=args.dataset,
        ckpt_path=args.ckpt,
        output_dir=args.output,
        device_name=args.device,
        sample_batch=args.sample_batch,
        worst_k=args.worst_k,
    )
    print("\nwrote:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
