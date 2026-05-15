"""Command-line entry point.

Subcommands:
    topology       — print topology summary for current params
    generate       — run sampler + warp solver + write outputs
    view-rerun     — open one or more raw samples in rerun (web or save)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

from params import GoalNetParams
from sampler import SamplerConfig, sample_shots
from solver_warp import XpbdWarpSolver, SimulationResult
from topology import generate_topology, stable_signature, summary as topology_summary
from output import write_outputs


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

    print(f"topology: {topology_summary(topo)}")
    print(f"running {len(shots)} samples on '{args.device}' with batch={batch_size}")

    results: List[SimulationResult] = []
    backend_t0 = time.time()
    i = 0
    while i < len(shots):
        chunk = shots[i : i + batch_size]
        # final chunk may be smaller; pad with last shot, drop after sim
        actual_b = len(chunk)
        pad = batch_size - actual_b
        if pad > 0:
            chunk = chunk + [chunk[-1]] * pad
        solver = XpbdWarpSolver(
            params,
            topo,
            batch_size=batch_size,
            device=args.device,
            record_particles=args.raw,
            max_contacts=args.max_contacts,
        )
        t0 = time.time()
        chunk_results = solver.simulate(
            [s.ball for s in chunk], [s.sample_id for s in chunk]
        )
        print(
            f"  batch [{i}..{i+actual_b}) {time.time()-t0:.2f}s "
            f"({(time.time()-t0)/actual_b:.3f}s/sample)"
        )
        results.extend(chunk_results[:actual_b])
        i += actual_b

    total = time.time() - backend_t0
    print(f"sim total: {total:.2f}s, {total/len(shots):.3f}s/sample")

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

    clean = sum(1 for r in results if r.quality.clean)
    print(f"clean ratio: {clean}/{len(results)}")
    return 0


def cmd_view_rerun(args: argparse.Namespace) -> int:
    from viewer_rerun import view_rerun

    target = Path(args.path)
    if target.is_dir():
        sample_files = sorted(target.glob("sample_*.json"))
    else:
        sample_files = [target]
    if not sample_files:
        print(f"no sample json files found under {target}", file=sys.stderr)
        return 2
    view_rerun(
        sample_files,
        serve=args.serve,
        bind=args.bind,
        save_path=args.save,
        spawn=args.spawn,
    )
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
    p_gen.set_defaults(func=cmd_generate)

    p_view = sub.add_parser("view-rerun", help="rerun.io viewer (web/serve/save)")
    p_view.add_argument("path", help="raw/sample_*.json or directory of them")
    p_view.add_argument("--serve", action="store_true", help="start rerun web server")
    p_view.add_argument("--spawn", action="store_true", help="open local rerun GUI")
    p_view.add_argument("--bind", default="0.0.0.0:9090", help="host:port for --serve")
    p_view.add_argument("--save", help="path to write .rrd")
    p_view.set_defaults(func=cmd_view_rerun)
    return parser


def main(argv: List[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
