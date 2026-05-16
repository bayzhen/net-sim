"""Benchmark utility: probe the 4090 utilisation envelope for `train-online`.

Goal
----
Find the largest ``sim_batch`` and ``train_batch`` (× ``hidden``) that the
single-GPU online pipeline can sustain on this machine, before committing
to a multi-hour training run.

It does *not* train a model end-to-end — it isolates the two GPU-bound
phases of ``run_online_training`` and times them in isolation:

  Probe A (sim):
    For each candidate ``sim_batch``, build an ``XpbdWarpSolver`` with
    ``record_particles=True`` (matches online training) and call
    ``simulate_arrays`` twice. Record wall time, per-sample seconds and
    peak host RAM growth attributable to the result arrays.

  Probe B (train):
    Build a ``GoalNetMLP`` with the requested ``hidden`` config and run
    a tiny ``OnlineFramePool`` so we can sample real (B, F, N) tensors.
    Run a few warm-up steps then time ``train_batch`` mini-batches, and
    record peak ``torch.cuda.max_memory_allocated``.

OOMs are caught and reported as hard upper bounds.

Usage::

    python bench_online.py --device cuda --output bench_results.json

Output is a JSON dump of all observations plus a small summary table on
stdout. We deliberately do NOT pick the "best" config automatically —
the user reads the table and decides.
"""
from __future__ import annotations

import argparse
import json
import time
import gc
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# Defer torch / warp imports until after argparse so --help is fast and
# CPU-only smoke runs do not need CUDA.


# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------

@dataclass
class BenchConfig:
    device: str = "cuda"
    output: str = "bench_results.json"

    # Probe A (sim) sweep. Each value triggers one solver build + 2 refills.
    sim_batch_sweep: Tuple[int, ...] = (256, 512, 768, 1024, 1536, 2048)
    sim_warmup_refills: int = 1   # discard first refill (kernel jit warmup)
    sim_timed_refills: int = 2

    # Probe B (train) sweep. Each entry is (train_batch, hidden_layers).
    # We keep particle_count fixed to whatever topology produces.
    train_batch_sweep: Tuple[int, ...] = (2048, 4096, 8192, 16384, 32768)
    hidden_sweep: Tuple[Tuple[int, ...], ...] = (
        (2048,) * 8,             # default in README (29M)
        (1024,) * 8,             # ~7.5M
        (768,) * 8,              # ~4.5M  (matches P2 student candidate)
    )
    train_warmup_steps: int = 5
    train_timed_steps: int = 25

    # Pool settings used only inside Probe B (we want real (B,F,N,3) tensors).
    bench_sim_batch_for_pool: int = 256   # cheap; just need data to sample
    bench_pool_batches: int = 2

    # Other knobs forwarded to existing code paths
    n_time_freq: int = 4
    drop_last_frames: int = 5
    seed: int = 1


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _bytes_to_gib(n: int) -> float:
    return n / (1024 ** 3)


def _now() -> float:
    return time.perf_counter()


def _clear_gpu_caches(device_str: str) -> None:
    import torch
    gc.collect()
    if device_str.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _wp_sync(device_str: str) -> None:
    import warp as wp
    if device_str.startswith("cuda"):
        wp.synchronize_device(device_str)


def _torch_sync(device_str: str) -> None:
    import torch
    if device_str.startswith("cuda"):
        torch.cuda.synchronize()


# -------------------------------------------------------------------------
# Probe A: sim_batch sweep
# -------------------------------------------------------------------------

@dataclass
class SimProbe:
    sim_batch: int
    ok: bool
    error: Optional[str] = None
    sim_time_per_refill_s: float = 0.0
    sim_time_per_sample_ms: float = 0.0
    clean_ratio: float = 0.0
    frame_count: int = 0
    particle_count: int = 0
    arrays_size_gib: float = 0.0  # B * F * N * 3 * 4  for net_pos


def probe_sim(cfg: BenchConfig) -> List[SimProbe]:
    from params import GoalNetParams
    from topology import generate_topology
    from sampler import SamplerConfig, sample_shots
    from solver_warp import XpbdWarpSolver
    # _shots_to_balls is a private helper inside train.py; re-implement locally
    # to avoid pulling its heavy import surface.

    def shots_to_balls(shots):
        balls = [s.ball for s in shots]
        sample_ids = [s.sample_id for s in shots]
        return balls, sample_ids

    params = GoalNetParams()
    topo = generate_topology(params)
    F = int(round(params.solver.duration / params.solver.frame_dt)) + 1
    N = topo.num_particles

    out: List[SimProbe] = []

    seed_iter = cfg.seed * 100_003

    for sb in cfg.sim_batch_sweep:
        rec = SimProbe(sim_batch=sb, ok=False, frame_count=F, particle_count=N)
        rec.arrays_size_gib = _bytes_to_gib(sb * F * N * 3 * 4)
        print(f"\n[sim] sim_batch={sb}  (net_pos arr ~{rec.arrays_size_gib:.2f} GiB host)",
              flush=True)
        solver = None
        try:
            solver = XpbdWarpSolver(
                params=params,
                topology=topo,
                batch_size=sb,
                device=cfg.device,
                record_particles=True,
                max_contacts=16384,
            )

            total_clean = 0
            total_seen = 0
            t_total = 0.0
            for r in range(cfg.sim_warmup_refills + cfg.sim_timed_refills):
                seed_iter += 1
                scfg = SamplerConfig(count=sb, seed=seed_iter)
                shots = sample_shots(scfg)
                balls, ids = shots_to_balls(shots)
                _wp_sync(cfg.device)
                t0 = _now()
                arrs = solver.simulate_arrays(balls, ids)
                _wp_sync(cfg.device)
                dt = _now() - t0
                clean = sum(1 for q in arrs["per_sample_quality"] if q.clean)
                if r >= cfg.sim_warmup_refills:
                    t_total += dt
                    total_clean += clean
                    total_seen += sb
                print(f"  refill {r}: {dt:.2f}s  clean {clean}/{sb}"
                      f"{'  (warmup, not counted)' if r < cfg.sim_warmup_refills else ''}",
                      flush=True)

            n_timed = cfg.sim_timed_refills
            rec.sim_time_per_refill_s = t_total / max(n_timed, 1)
            rec.sim_time_per_sample_ms = (t_total * 1000.0) / max(total_seen, 1)
            rec.clean_ratio = total_clean / max(total_seen, 1)
            rec.ok = True
            print(f"  -> avg {rec.sim_time_per_refill_s:.2f}s/refill, "
                  f"{rec.sim_time_per_sample_ms:.2f} ms/sample, "
                  f"clean {rec.clean_ratio*100:.1f}%", flush=True)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            rec.error = err[:240]
            print(f"  FAILED: {err}", flush=True)
        finally:
            del solver
            _clear_gpu_caches(cfg.device)
            out.append(rec)

    return out


# -------------------------------------------------------------------------
# Probe B: train_batch x hidden sweep
# -------------------------------------------------------------------------

@dataclass
class TrainProbe:
    train_batch: int
    hidden: Tuple[int, ...]
    ok: bool
    error: Optional[str] = None
    n_params: int = 0
    step_time_ms: float = 0.0
    sample_time_us: float = 0.0  # step_time / B
    peak_vram_gib: float = 0.0


def _populate_pool(cfg: BenchConfig):
    """Build a small pool of real sim data so train probes index real tensors."""
    from params import GoalNetParams
    from topology import generate_topology
    from sampler import SamplerConfig, sample_shots
    from solver_warp import XpbdWarpSolver
    from online_pool import OnlineFramePool

    params = GoalNetParams()
    topo = generate_topology(params)
    F = int(round(params.solver.duration / params.solver.frame_dt)) + 1
    N = topo.num_particles

    sb = cfg.bench_sim_batch_for_pool
    solver = XpbdWarpSolver(
        params=params, topology=topo, batch_size=sb,
        device=cfg.device, record_particles=True,
    )

    pool = OnlineFramePool(
        capacity_batches=cfg.bench_pool_batches,
        max_samples_per_batch=sb,
        frame_count=F,
        particle_count=N,
        drop_last_frames=cfg.drop_last_frames,
    )

    seed = cfg.seed * 100_007
    for k in range(cfg.bench_pool_batches):
        scfg = SamplerConfig(count=sb, seed=seed + k)
        shots = sample_shots(scfg)
        balls = [s.ball for s in shots]
        ids = [s.sample_id for s in shots]
        arrs = solver.simulate_arrays(balls, ids)
        pool.push(arrs, balls)
        print(f"  pool refill {k}: clean {pool._valid_n[k]}/{sb}", flush=True)

    if pool.total_valid_samples == 0:
        raise RuntimeError("pool is empty after warm-up, cannot bench training")

    del solver
    _clear_gpu_caches(cfg.device)
    return pool, F, N


def probe_train(cfg: BenchConfig) -> List[TrainProbe]:
    import torch
    from model import GoalNetMLP, count_parameters, input_dim_for
    from train import encode_batch_input, compute_loss, TrainConfig

    print("\n[train] populating shared pool ...", flush=True)
    pool, F, N = _populate_pool(cfg)
    print(f"  pool ready: total_valid_samples={pool.total_valid_samples}, "
          f"F={F}, N={N}", flush=True)

    tcfg = TrainConfig(
        dataset="", output="",
        n_time_freq=cfg.n_time_freq,
        drop_last_frames=cfg.drop_last_frames,
    )
    rng = np.random.default_rng(cfg.seed + 1)
    device = torch.device(cfg.device)

    out: List[TrainProbe] = []

    for hidden in cfg.hidden_sweep:
        for tb in cfg.train_batch_sweep:
            rec = TrainProbe(train_batch=tb, hidden=tuple(hidden), ok=False)
            label = f"train_batch={tb} hidden={'x'.join(map(str,hidden))}"
            print(f"\n[train] {label}", flush=True)
            try:
                in_dim = input_dim_for(cfg.n_time_freq)
                model = GoalNetMLP(
                    in_dim=in_dim, n_particles=N,
                    hidden=tuple(hidden),
                    activation="gelu",   # matches OnlineTrainConfig default
                    predict_velocity=True,
                    dropout=0.0,
                ).to(device)
                rec.n_params = count_parameters(model)
                # Use ones for norm stats (we only care about timing/memory).
                from model import _safe_std
                stats = {
                    "in_mean": torch.zeros(in_dim),
                    "in_std": torch.ones(in_dim),
                    "ball_pos_mean": torch.zeros(3),
                    "ball_pos_std": torch.ones(3),
                    "ball_vel_mean": torch.zeros(3),
                    "ball_vel_std": torch.ones(3),
                    "net_mean": torch.zeros(N, 3),
                    "net_std": torch.ones(N, 3),
                }
                model.set_norm_stats({k: v.to(device) for k, v in stats.items()})
                opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
                _torch_sync(cfg.device)
                if cfg.device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()

                # Warm-up
                for _ in range(cfg.train_warmup_steps):
                    np_b = pool.sample(tb, rng)
                    b = {k: torch.from_numpy(v).to(device, non_blocking=True)
                         for k, v in np_b.items()}
                    x = encode_batch_input(b, cfg.n_time_freq)
                    pred = model(x, return_normalized=True)
                    total, _ = compute_loss(model, pred, b, tcfg)
                    opt.zero_grad(set_to_none=True)
                    total.backward()
                    opt.step()
                _torch_sync(cfg.device)

                t0 = _now()
                for _ in range(cfg.train_timed_steps):
                    np_b = pool.sample(tb, rng)
                    b = {k: torch.from_numpy(v).to(device, non_blocking=True)
                         for k, v in np_b.items()}
                    x = encode_batch_input(b, cfg.n_time_freq)
                    pred = model(x, return_normalized=True)
                    total, _ = compute_loss(model, pred, b, tcfg)
                    opt.zero_grad(set_to_none=True)
                    total.backward()
                    opt.step()
                _torch_sync(cfg.device)
                dt = _now() - t0

                rec.step_time_ms = (dt * 1000.0) / cfg.train_timed_steps
                rec.sample_time_us = (dt * 1e6) / (cfg.train_timed_steps * tb)
                if cfg.device.startswith("cuda"):
                    rec.peak_vram_gib = _bytes_to_gib(torch.cuda.max_memory_allocated())
                rec.ok = True
                print(f"  params={rec.n_params:,}  "
                      f"step={rec.step_time_ms:.2f} ms  "
                      f"sample={rec.sample_time_us:.2f} us  "
                      f"peak_vram={rec.peak_vram_gib:.2f} GiB", flush=True)
            except Exception as e:
                rec.error = f"{type(e).__name__}: {str(e)[:240]}"
                print(f"  FAILED: {rec.error}", flush=True)
            finally:
                try:
                    del model, opt
                except Exception:
                    pass
                _clear_gpu_caches(cfg.device)
                out.append(rec)

    return out


# -------------------------------------------------------------------------
# Reporting
# -------------------------------------------------------------------------

def print_sim_table(rs: List[SimProbe]) -> None:
    print("\n=== Probe A: sim_batch sweep ===")
    print(f"{'sim_batch':>9} {'time/refill':>12} {'time/sample':>13} {'clean%':>8} {'arr GiB':>9} {'status':>8}")
    for r in rs:
        if r.ok:
            print(f"{r.sim_batch:>9} {r.sim_time_per_refill_s:>11.2f}s "
                  f"{r.sim_time_per_sample_ms:>11.2f}ms "
                  f"{r.clean_ratio*100:>7.1f}% "
                  f"{r.arrays_size_gib:>8.2f} {'OK':>8}")
        else:
            print(f"{r.sim_batch:>9} {'-':>12} {'-':>13} {'-':>8} "
                  f"{r.arrays_size_gib:>8.2f} {'FAIL':>8}    {r.error}")


def print_train_table(rs: List[TrainProbe]) -> None:
    print("\n=== Probe B: train_batch x hidden sweep ===")
    print(f"{'hidden':>20} {'train_batch':>11} {'params':>12} "
          f"{'step ms':>9} {'sample us':>10} {'peak GiB':>9} {'status':>8}")
    for r in rs:
        h = "x".join(map(str, r.hidden))
        if r.ok:
            print(f"{h:>20} {r.train_batch:>11} {r.n_params:>12,} "
                  f"{r.step_time_ms:>8.2f}  {r.sample_time_us:>9.2f}  "
                  f"{r.peak_vram_gib:>8.2f}  {'OK':>8}")
        else:
            print(f"{h:>20} {r.train_batch:>11} {'-':>12} "
                  f"{'-':>9} {'-':>10} {'-':>9} {'FAIL':>8}    {r.error}")


# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------

def parse_args(argv=None) -> BenchConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="bench_results.json")
    p.add_argument("--sim-batches", type=int, nargs="+",
                   default=list(BenchConfig.sim_batch_sweep))
    p.add_argument("--train-batches", type=int, nargs="+",
                   default=list(BenchConfig.train_batch_sweep))
    p.add_argument("--hiddens", type=str, nargs="+",
                   default=None,
                   help="each hidden config as a string like '2048x8' or "
                        "'1024,1024,1024,1024,1024,1024'.")
    p.add_argument("--skip-sim", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args(argv)

    cfg = BenchConfig(
        device=a.device,
        output=a.output,
        sim_batch_sweep=tuple(a.sim_batches),
        train_batch_sweep=tuple(a.train_batches),
        seed=a.seed,
    )
    if a.hiddens:
        parsed = []
        for h in a.hiddens:
            if "x" in h:
                w, n = h.split("x")
                parsed.append(tuple([int(w)] * int(n)))
            else:
                parsed.append(tuple(int(x) for x in h.split(",")))
        cfg.hidden_sweep = tuple(parsed)
    cfg._skip_sim = a.skip_sim     # type: ignore
    cfg._skip_train = a.skip_train  # type: ignore
    return cfg


def main(argv=None) -> int:
    cfg = parse_args(argv)
    print("== bench_online ==")
    print(f"device={cfg.device}  seed={cfg.seed}")
    print(f"sim_batch_sweep={list(cfg.sim_batch_sweep)}")
    print(f"train_batch_sweep={list(cfg.train_batch_sweep)}")
    print(f"hidden_sweep={[list(h) for h in cfg.hidden_sweep]}")

    sim_results: List[SimProbe] = []
    train_results: List[TrainProbe] = []

    if not getattr(cfg, "_skip_sim", False):
        sim_results = probe_sim(cfg)
        print_sim_table(sim_results)
    else:
        print("[sim] skipped")

    if not getattr(cfg, "_skip_train", False):
        train_results = probe_train(cfg)
        print_train_table(train_results)
    else:
        print("[train] skipped")

    out = {
        "config": {
            "device": cfg.device,
            "seed": cfg.seed,
            "sim_batch_sweep": list(cfg.sim_batch_sweep),
            "train_batch_sweep": list(cfg.train_batch_sweep),
            "hidden_sweep": [list(h) for h in cfg.hidden_sweep],
            "n_time_freq": cfg.n_time_freq,
            "drop_last_frames": cfg.drop_last_frames,
        },
        "sim": [asdict(r) for r in sim_results],
        "train": [asdict(r) for r in train_results],
    }
    with open(cfg.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nresults written to {cfg.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
