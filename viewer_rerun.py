"""Web/offline viewer for goal-net XPBD samples using rerun.io.

Reads raw/sample_*.json (full per-frame net state + contacts) and logs them
to a rerun recording — either serve as a web viewer, save to .rrd, or open in
the local GUI.
"""
from __future__ import annotations

import json
import signal
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def _contact_color(c_type: str) -> Tuple[int, int, int]:
    return {
        "particle": (240, 200, 0),
        "segment": (255, 160, 0),
        "segment_swept": (255, 80, 0),
        "goalpost": (200, 220, 255),
        "crossbar": (180, 220, 255),
        "ground_bounce": (120, 200, 120),
        "ground_roll": (60, 160, 80),
    }.get(c_type, (255, 255, 255))


def _build_static(rr, sample: dict) -> None:
    topo = sample["topology"]
    # goalposts as 3 capsules-ish (we use LineStrips3D for clarity)
    posts = topo.get("goalpost_segments", [])
    for g in posts:
        rr.log(
            f"goal/posts/{g['name']}",
            rr.LineStrips3D(
                [[g["p0"], g["p1"]]],
                colors=[(220, 220, 220)],
                radii=[g["radius"]],
            ),
            static=True,
        )
    # ground plane (1 large flat box)
    rr.log(
        "ground",
        rr.Boxes3D(
            centers=[[0.0, -0.01, -1.5]],
            half_sizes=[[15.0, 0.01, 15.0]],
            colors=[(80, 130, 80)],
        ),
        static=True,
    )
    # anchors
    anchor_positions = [
        topo["particles"][a["particle"]]["position"]
        for a in topo.get("anchor_constraints", [])
    ]
    if anchor_positions:
        rr.log(
            "net/anchors",
            rr.Points3D(
                anchor_positions,
                colors=[(220, 80, 80)] * len(anchor_positions),
                radii=[0.02] * len(anchor_positions),
            ),
            static=True,
        )


def _build_segment_pairs(particles_positions: List[List[float]], topology: dict):
    pairs = []
    for c in topology["distance_constraints"]:
        if c["kind"] != 0:  # only render stretch ropes
            continue
        i0 = c["i0"]
        i1 = c["i1"]
        if i0 < len(particles_positions) and i1 < len(particles_positions):
            pairs.append([particles_positions[i0], particles_positions[i1]])
    return pairs


def view_rerun(
    sample_paths: Iterable[Path],
    serve: bool = False,
    bind: str = "0.0.0.0:9090",
    save_path: str = None,
    spawn: bool = False,
) -> None:
    try:
        import rerun as rr  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "rerun-sdk is not installed; run `pip install rerun-sdk`"
        ) from e

    rr.init("goal_net_xpbd_dataset", spawn=spawn)
    web_port = None
    grpc_port = None
    if serve:
        host, _, port_str = bind.partition(":")
        web_port = int(port_str) if port_str else 9090
        grpc_port = web_port + 1
        # serve_grpc returns the URI that serve_web_viewer needs to connect to.
        server_uri = rr.serve_grpc(grpc_port=grpc_port)
        rr.serve_web_viewer(
            web_port=web_port, open_browser=False, connect_to=server_uri
        )
    if save_path:
        rr.save(save_path)

    for path in sample_paths:
        sample = json.loads(Path(path).read_text())
        sample_id = sample["shot"]["sample_id"]
        prefix = sample_id
        _log_sample(rr, sample, prefix)

    if serve and web_port is not None:
        print(
            f"rerun web viewer listening on 0.0.0.0:{web_port} "
            f"(grpc :{grpc_port})",
            flush=True,
        )
        print("press Ctrl+C to stop.", flush=True)
        try:
            signal.pause()
        except KeyboardInterrupt:
            return


def _log_sample(rr, sample: dict, prefix: str) -> None:
    topology = sample["topology"]
    frames = sample["frames"]
    contacts = sample.get("contacts", [])

    # static
    posts = topology.get("goalpost_segments", [])
    for g in posts:
        rr.log(
            f"{prefix}/goal/posts/{g['name']}",
            rr.LineStrips3D(
                [[g["p0"], g["p1"]]],
                colors=[(220, 220, 220)],
                radii=[g["radius"]],
            ),
            static=True,
        )
    rr.log(
        f"{prefix}/ground",
        rr.Boxes3D(
            centers=[[0.0, -0.01, -1.5]],
            half_sizes=[[15.0, 0.01, 15.0]],
            colors=[(80, 130, 80)],
        ),
        static=True,
    )
    anchor_positions = [
        topology["particles"][a["particle"]]["position"]
        for a in topology.get("anchor_constraints", [])
    ]
    if anchor_positions:
        rr.log(
            f"{prefix}/net/anchors",
            rr.Points3D(
                anchor_positions,
                colors=[(220, 80, 80)] * len(anchor_positions),
                radii=[0.02] * len(anchor_positions),
            ),
            static=True,
        )

    ball_radius = sample["shot"].get("radius", 0.13)
    trajectory_points: List[List[float]] = []
    for f in frames:
        t = float(f["time"])
        rr.set_time("sim_time", duration=t)
        ball_pos = f["ball_position"]
        trajectory_points.append(ball_pos)
        rr.log(
            f"{prefix}/ball",
            rr.Points3D(
                [ball_pos], colors=[(255, 100, 0)], radii=[ball_radius]
            ),
        )
        if len(trajectory_points) >= 2:
            rr.log(
                f"{prefix}/ball/trajectory",
                rr.LineStrips3D(
                    [trajectory_points], colors=[(255, 200, 100)]
                ),
            )
        if "particle_positions" in f:
            ps = f["particle_positions"]
            rr.log(
                f"{prefix}/net/particles",
                rr.Points3D(
                    ps, colors=[(80, 120, 255)] * len(ps), radii=[0.012] * len(ps)
                ),
            )
            pairs = _build_segment_pairs(ps, topology)
            if pairs:
                rr.log(
                    f"{prefix}/net/ropes",
                    rr.LineStrips3D(pairs, colors=[(140, 160, 220)]),
                )

    for c in contacts:
        rr.set_time("sim_time", duration=float(c["time"]))
        rr.log(
            f"{prefix}/contacts/{c['object_type']}",
            rr.Points3D(
                [c["position"]],
                colors=[_contact_color(c["object_type"])],
                radii=[0.04],
            ),
        )


__all__ = ["view_rerun"]
