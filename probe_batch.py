"""Probe the maximum/optimal batch size on the current GPU.

Tries a descending list of candidate batch sizes; for each one warms up
once (kernel JIT) then times one batch. Prints results line-by-line so
progress is visible during long runs.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
import traceback
from typing import List, Tuple

from params import GoalNetParams
from sampler import SamplerConfig, sample_shots
from solver_warp import XpbdWarpSolver
from topology import generate_topology, summary as topology_summary


def _flush(msg: str) -> None:
    print(msg, flush=True)


def _probe_one(params, topo, batch: int, seed: int, raw: bool) -> Tuple[float, str]:
    """Warm-up once, then time one batch. Return (samples_per_sec, note)."""
    import warp as wp

    cfg = SamplerConfig(count=max(batch * 2, 32), seed=seed)
    shots = sample_shots(cfg)
    if len(shots) < batch * 2:
        shots = (shots * ((batch * 2 // len(shots)) + 1))[: batch * 2]

    _flush(f"  [batch={batch}] allocating solver (record_particles={raw})...")
    t_alloc = time.time()
    solver = XpbdWarpSolver(
        params, topo,
        batch_size=batch,
        device="cuda",
        record_particles=raw,
        max_contacts=16384,
    )
    _flush(f"  [batch={batch}] alloc ok ({time.time()-t_alloc:.2f}s); warm-up...")

    warm = shots[:batch]
    t_warm = time.time()
    solver.simulate([s.ball for s in warm], [s.sample_id for s in warm])
    wp.synchronize()
    _flush(f"  [batch={batch}] warm-up done ({time.time()-t_warm:.2f}s); timed run...")

    timed = shots[batch : batch * 2]
    t0 = time.time()
    solver.simulate([s.ball for s in timed], [s.sample_id for s in timed])
    wp.synchronize()
    elapsed = time.time() - t0
    sps = batch / elapsed
    return sps, f"warm={time.time()-t_warm:.2f}s timed={elapsed:.2f}s"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidates",
        default="128,96,64,48,32,16",
        help="comma-separated batch sizes to try, descending",
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--raw", action="store_true", help="enable record_particles (matches generate --raw)")
    args = ap.parse_args(argv)

    params = GoalNetParams()
    topo = generate_topology(params)
    _flush(f"topology: {topology_summary(topo)}")

    candidates: List[int] = [int(x) for x in args.candidates.split(",") if x.strip()]
    _flush(f"probing batches: {candidates}")

    rows = []
    for b in candidates:
        gc.collect()
        try:
            sps, note = _probe_one(params, topo, b, args.seed, args.raw)
            _flush(f"  batch={b:>4}  ->  {sps:7.2f} samples/s   ({sps*3600:.0f}/h)   {note}")
            rows.append((b, sps, note))
        except Exception as e:  # noqa: BLE001
            err = type(e).__name__
            msg = str(e).splitlines()[0][:120]
            _flush(f"  batch={b:>4}  ->  FAILED   {err}: {msg}")
            rows.append((b, 0.0, f"fail:{err}"))
            traceback.print_exc(limit=1, file=sys.stderr)
            continue

    _flush("")
    _flush("summary:")
    _flush(f"  {'batch':>6} {'samples/s':>12} {'samples/hour':>14}   note")
    for b, sps, note in rows:
        sph = sps * 3600
        _flush(f"  {b:>6} {sps:>12.2f} {sph:>14.0f}   {note}")

    ok = [r for r in rows if r[1] > 0]
    if ok:
        best = max(ok, key=lambda r: r[1])
        _flush("")
        _flush(f"best: batch={best[0]} -> {best[1]:.2f} samples/s ({best[1]*3600:.0f} samples/hour)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
