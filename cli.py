"""Command-line entry point.

Subcommands:
    topology       — print topology summary for current params
    generate       — run sampler + warp solver + write outputs
    view-rerun     — open one or more raw samples in rerun (web or save)
    train          — train the MLP surrogate on a generated HDF5 dataset
    predict        — evaluate a trained checkpoint on the test split
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import List, Optional

from params import GoalNetParams
from sampler import SamplerConfig, sample_shots
from solver_warp import XpbdWarpSolver, SimulationResult
from topology import generate_topology, stable_signature, summary as topology_summary
from output import write_outputs, make_incremental_writer, make_h5_writer


DEFAULT_OUTPUT = "Agent/Temp/goal_net_xpbd_dataset"


def _params_from_json(path: str) -> GoalNetParams:
    with open(path) as f:
        data = json.load(f)

    def _build(cls, payload):
        if payload is None:
            return cls()
        return cls(**payload)

    from params import (
        GoalSizeParams,
        GridParams,
        RopeParams,
        AnchorParams,
        ShapeParams,
        SolverParams,
        CollisionParams,
        GroundParams,
        GoalpostParams,
    )

    mapping = {
        "goal": GoalSizeParams,
        "grid": GridParams,
        "rope": RopeParams,
        "anchor": AnchorParams,
        "shape": ShapeParams,
        "solver": SolverParams,
        "collision": CollisionParams,
        "ground": GroundParams,
        "goalpost": GoalpostParams,
    }
    p = GoalNetParams()
    for key, cls in mapping.items():
        if key in data:
            setattr(p, key, _build(cls, data[key]))
    if "schema_version" in data:
        p.schema_version = data["schema_version"]
    return p


def cmd_topology(args: argparse.Namespace) -> int:
    params = GoalNetParams() if args.params is None else _params_from_json(args.params)
    topo = generate_topology(params)
    summary = topology_summary(topo)
    signature = stable_signature(topo)
    print(json.dumps({"summary": summary, "stable_signature": signature}, indent=2))
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    params = GoalNetParams() if args.params is None else _params_from_json(args.params)
    topo = generate_topology(params)

    cfg = SamplerConfig(count=args.count, seed=args.seed)
    shots = sample_shots(cfg)
    if not shots:
        print("no shots to simulate", file=sys.stderr)
        return 2

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch if args.batch > 0 else len(shots)
    if batch_size > len(shots):
        batch_size = len(shots)

    print(f"topology: {topology_summary(topo)}", flush=True)
    print(
        f"running {len(shots)} samples on '{args.device}' with batch={batch_size} "
        f"(incremental={'on' if args.incremental else 'off'})",
        flush=True,
    )

    # Reuse a single solver across batches (saves alloc + warm-up time).
    solver = XpbdWarpSolver(
        params,
        topo,
        batch_size=batch_size,
        device=args.device,
        record_particles=args.raw,
        max_contacts=args.max_contacts,
    )

    append_chunk = None
    finalize = None
    append_chunk_arrays = None
    fast_path = False
    use_h5 = False
    results: List[SimulationResult] = []
    writer_pool: Optional[ThreadPoolExecutor] = None
    pending_write: Optional[Future] = None
    if args.incremental:
        if args.raw_format == "h5":
            if not args.raw:
                # h5 stores raw frames; without --raw it would be near-empty,
                # so make this combination explicit for the user.
                print(
                    "[warn] --raw-format h5 without --raw: no per-frame data will be stored",
                    flush=True,
                )
            append_chunk_arrays, finalize = make_h5_writer(
                str(out_dir), params, topo,
                include_raw=args.raw,
                params_source_path=args.params,
                chunk_size=batch_size,
            )
            fast_path = True
            use_h5 = True
        else:
            append_chunk, finalize, append_chunk_arrays = make_incremental_writer(
                str(out_dir), params, topo,
                include_raw=args.raw,
                params_source_path=args.params,
                raw_format=args.raw_format,
            )
            # array fast-path skips the B*F SimulationResult construction loop;
            # legacy JSON path still uses the dataclass shape.
            fast_path = args.raw and args.raw_format == "npz"
        # Single-threaded executor: keeps writes ordered while overlapping
        # I/O with the next GPU batch.
        writer_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="writer")

    backend_t0 = time.time()
    i = 0
    clean_total = 0
    while i < len(shots):
        chunk = shots[i : i + batch_size]
        actual_b = len(chunk)
        pad = batch_size - actual_b
        if pad > 0:
            chunk = chunk + [chunk[-1]] * pad
        t0 = time.time()
        if fast_path:
            chunk_arrs = solver.simulate_arrays(
                [s.ball for s in chunk], [s.sample_id for s in chunk]
            )
            elapsed = time.time() - t0
            # Per-sample quality is precomputed; count clean directly without
            # building SimulationResult objects.
            quals = chunk_arrs["per_sample_quality"][:actual_b]
            clean_total += sum(1 for q in quals if q.clean)
            chunk_shots = chunk[:actual_b]
        else:
            chunk_results = solver.simulate(
                [s.ball for s in chunk], [s.sample_id for s in chunk]
            )
            chunk_results = chunk_results[:actual_b]
            chunk_shots = chunk[:actual_b]
            elapsed = time.time() - t0
            clean_total += sum(1 for r in chunk_results if r.quality.clean)

        write_wait = 0.0
        if args.incremental:
            # Wait for previous batch's write to finish (back-pressure so we
            # don't pile up unbounded result buffers in RAM).
            if pending_write is not None:
                tw = time.time()
                pending_write.result()
                write_wait = time.time() - tw
            # Submit this batch's write asynchronously.
            if fast_path:
                pending_write = writer_pool.submit(
                    append_chunk_arrays, chunk_shots, chunk_arrs
                )
            else:
                pending_write = writer_pool.submit(
                    append_chunk, chunk_shots, chunk_results
                )
        else:
            results.extend(chunk_results)

        done = i + actual_b
        wall = time.time() - backend_t0
        rate = done / wall if wall > 0 else 0.0
        eta = (len(shots) - done) / rate if rate > 0 else 0.0
        print(
            f"  batch [{i}..{done}) sim={elapsed:.2f}s wait_write={write_wait:.2f}s "
            f"({elapsed/actual_b:.3f}s/sample)  rate={rate:.2f}/s "
            f"clean={clean_total}/{done}  ETA={eta/60:.1f}min",
            flush=True,
        )
        i += actual_b

    if args.incremental and pending_write is not None:
        # Flush the last in-flight write.
        tw = time.time()
        pending_write.result()
        print(f"  final write flush: {time.time()-tw:.2f}s", flush=True)
        writer_pool.shutdown(wait=True)

    total = time.time() - backend_t0
    print(f"sim total: {total:.2f}s, {total/len(shots):.3f}s/sample", flush=True)

    if args.incremental:
        paths = finalize()
    else:
        paths = write_outputs(
            str(out_dir),
            params,
            topo,
            shots,
            results,
            include_raw=args.raw,
            params_source_path=args.params,
        )
    print("wrote:")
    for k, v in paths.items():
        if v:
            print(f"  {k}: {v}")

    print(f"clean ratio: {clean_total}/{len(shots)}")
    return 0


def cmd_view_rerun(args: argparse.Namespace) -> int:
    from viewer_rerun import view_rerun

    target = Path(args.path)
    if target.is_dir():
        sample_files = sorted(target.glob("sample_*.json")) + sorted(
            target.glob("sample_*.npz")
        )
    else:
        sample_files = [target]
    if not sample_files:
        print(f"no sample json/npz files found under {target}", file=sys.stderr)
        return 2
    view_rerun(
        sample_files,
        serve=args.serve,
        bind=args.bind,
        save_path=args.save,
        spawn=args.spawn,
        public_host=args.public_host,
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from train import cfg_from_args, run_training

    cfg = cfg_from_args(args)
    paths = run_training(cfg)
    print("wrote:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from predict import evaluate_per_frame

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="goal_net_xpbd")
    parser.add_argument("--params", help="JSON file overriding default params")
    sub = parser.add_subparsers(dest="command", required=True)

    p_topo = sub.add_parser("topology", help="dump topology summary")
    p_topo.set_defaults(func=cmd_topology)

    p_gen = sub.add_parser("generate", help="generate dataset samples")
    p_gen.add_argument("--count", type=int, default=5)
    p_gen.add_argument("--seed", type=int, default=1)
    p_gen.add_argument("--raw", action="store_true")
    p_gen.add_argument("--batch", type=int, default=0, help="0 = all samples in one batch")
    p_gen.add_argument(
        "--backend", choices=["warp"], default="warp", help="reserved; only warp supported"
    )
    p_gen.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p_gen.add_argument("--output", default=DEFAULT_OUTPUT)
    p_gen.add_argument("--max-contacts", dest="max_contacts", type=int, default=16384)
    p_gen.add_argument(
        "--incremental",
        action="store_true",
        help="flush each batch to disk immediately (required for very large datasets)",
    )
    p_gen.add_argument(
        "--raw-format",
        dest="raw_format",
        choices=["npz", "json", "h5"],
        default="npz",
        help="incremental raw writer format: 'h5' (single chunked HDF5 file, "
        "fastest for large datasets), 'npz' (one binary file per sample), "
        "or 'json' (legacy, every sample self-contained including topology)",
    )
    p_gen.set_defaults(func=cmd_generate)

    p_view = sub.add_parser("view-rerun", help="rerun.io viewer (web/serve/save)")
    p_view.add_argument("path", help="raw/sample_*.json or directory of them")
    p_view.add_argument("--serve", action="store_true", help="start rerun web server")
    p_view.add_argument("--spawn", action="store_true", help="open local rerun GUI")
    p_view.add_argument("--bind", default="0.0.0.0:9090", help="host:port for --serve")
    p_view.add_argument(
        "--public-host",
        dest="public_host",
        default=None,
        help="host the browser will use to reach the gRPC port "
        "(default: 127.0.0.1; set to your tunnel/public hostname so the page "
        "can reach the gRPC backend)",
    )
    p_view.add_argument("--save", help="path to write .rrd")
    p_view.set_defaults(func=cmd_view_rerun)

    p_train = sub.add_parser("train", help="train MLP surrogate on a dataset.h5")
    from train import add_train_args
    add_train_args(p_train)
    p_train.set_defaults(func=cmd_train)

    p_predict = sub.add_parser("predict", help="evaluate a trained checkpoint")
    from predict import add_predict_args
    add_predict_args(p_predict)
    p_predict.set_defaults(func=cmd_predict)
    return parser


def main(argv: List[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
