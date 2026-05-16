"""Web/offline viewer for goal-net XPBD samples using rerun.io.

Reads raw/sample_*.json (full per-frame net state + contacts) and logs them
to a rerun recording — either serve as a web viewer, save to .rrd, or open in
the local GUI.
"""
from __future__ import annotations

import json
import signal
import time
import urllib.parse
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


def _build_segment_pairs(
    particles_positions: List[List[float]],
    topology: dict,
    skip_constraints: set,
):
    pairs = []
    for c in topology["distance_constraints"]:
        if c["kind"] != 0:  # only render stretch ropes
            continue
        if c["index"] in skip_constraints:
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
    public_host: str = None,
) -> None:
    try:
        import rerun as rr  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "rerun-sdk is not installed; run `pip install rerun-sdk`"
        ) from e

    # The default recording is what serve_grpc attaches to. Each sample gets
    # its own RecordingStream (same application_id, unique recording_id) so
    # the rerun viewer lists them as separate, switchable recordings.
    rr.init("goal_net_xpbd_dataset", spawn=spawn)
    web_port = None
    grpc_port = None
    server_uri = None
    if serve:
        _, _, port_str = bind.partition(":")
        web_port = int(port_str) if port_str else 9090
        grpc_port = web_port + 1
        server_uri = rr.serve_grpc(grpc_port=grpc_port)
        rr.serve_web_viewer(
            web_port=web_port, open_browser=False, connect_to=server_uri
        )
    if save_path:
        rr.save(save_path)

    for path in sample_paths:
        sample = json.loads(Path(path).read_text())
        sample_id = sample["shot"]["sample_id"]
        rec = rr.RecordingStream(
            application_id="goal_net_xpbd_dataset",
            recording_id=sample_id,
        )
        if server_uri is not None:
            rr.connect_grpc(url=server_uri, recording=rec)
        if save_path:
            # Append each per-sample recording to the same .rrd file.
            rr.save(save_path, recording=rec)
        _log_sample(rr, sample, prefix="world", recording=rec)

    if serve and web_port is not None:
        # The rerun 0.32 web viewer reads its gRPC backend URL from the page's
        # `?url=` query parameter — `serve_web_viewer(connect_to=...)` only
        # affects the auto-opened browser, not the served HTML itself. So we
        # print a ready-to-click URL with the encoded gRPC endpoint baked in.
        host = public_host or "localhost"
        grpc_uri = f"rerun+http://{host}:{grpc_port}/proxy"
        full_url = (
            f"http://{host}:{web_port}/?url="
            + urllib.parse.quote(grpc_uri, safe="")
        )
        print(
            f"rerun web viewer listening on 0.0.0.0:{web_port} "
            f"(grpc :{grpc_port})",
            flush=True,
        )
        print(f"open in browser: {full_url}", flush=True)
        print("press Ctrl+C to stop.", flush=True)
        try:
            signal.pause()
        except KeyboardInterrupt:
            return


def _log_sample(rr, sample: dict, prefix: str, recording=None) -> None:
    topology = sample["topology"]
    frames = sample["frames"]
    contacts = sample.get("contacts", [])

    # Helpers that always thread the recording through.
    def _log(path, ent, **kw):
        rr.log(path, ent, recording=recording, **kw)

    def _set_time(t):
        rr.set_time("sim_time", duration=t, recording=recording)

    # Declare world axes: y is up, right-handed (matches goal_net_warp_design
    # §1.1). Without this rerun defaults to its own convention and the scene
    # appears tipped on its side.
    _log(prefix, rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # static
    posts = topology.get("goalpost_segments", [])
    for g in posts:
        _log(
            f"{prefix}/goal/posts/{g['name']}",
            rr.LineStrips3D(
                [[g["p0"], g["p1"]]],
                colors=[(220, 220, 220)],
                radii=[g["radius"]],
            ),
            static=True,
        )
    # Stake markers are static (anchored), but the stay rope endpoints on
    # the net side can swing — so the stays themselves are logged per-frame
    # below, not here.
    stays = topology.get("support_stays", [])
    stake_idx_set = set(topology.get("stake_particle_indices", []))
    stay_constraint_set = {s["constraint"] for s in stays}
    if stays:
        # Each stay's far end is an elevated anchor (treated as a fixed
        # eyelet, somewhere above-and-behind the back-top corner of the net).
        anchor_positions = [
            topology["particles"][s["stake_particle"]]["position"] for s in stays
        ]
        _log(
            f"{prefix}/goal/stay_anchors",
            rr.Points3D(
                anchor_positions,
                colors=[(160, 110, 60)] * len(anchor_positions),
                radii=[0.04] * len(anchor_positions),
            ),
            static=True,
        )
    _log(
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
        _log(
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
        _set_time(t)
        ball_pos = f["ball_position"]
        trajectory_points.append(ball_pos)
        _log(
            f"{prefix}/ball",
            rr.Points3D([ball_pos], colors=[(255, 100, 0)], radii=[ball_radius]),
        )
        if len(trajectory_points) >= 2:
            _log(
                f"{prefix}/ball/trajectory",
                rr.LineStrips3D([trajectory_points], colors=[(255, 200, 100)]),
            )
        if "particle_positions" in f:
            ps = f["particle_positions"]
            # Net particles: exclude stake particles (they live in particle
            # array but are off-mesh anchors).
            net_ps = [p for i, p in enumerate(ps) if i not in stake_idx_set]
            _log(
                f"{prefix}/net/particles",
                rr.Points3D(
                    net_ps,
                    colors=[(80, 120, 255)] * len(net_ps),
                    radii=[0.012] * len(net_ps),
                ),
            )
            pairs = _build_segment_pairs(ps, topology, stay_constraint_set)
            if pairs:
                _log(
                    f"{prefix}/net/ropes",
                    rr.LineStrips3D(pairs, colors=[(140, 160, 220)]),
                )
            # Per-frame stays: corner end may swing, stake end is fixed.
            if stays:
                stay_pairs = []
                for s in stays:
                    ci = s["corner_particle"]
                    si = s["stake_particle"]
                    if ci < len(ps) and si < len(ps):
                        stay_pairs.append([ps[ci], ps[si]])
                if stay_pairs:
                    _log(
                        f"{prefix}/goal/stays",
                        rr.LineStrips3D(
                            stay_pairs,
                            colors=[(220, 200, 80)] * len(stay_pairs),
                            radii=[0.015] * len(stay_pairs),
                        ),
                    )

    for c in contacts:
        _set_time(float(c["time"]))
        _log(
            f"{prefix}/contacts/{c['object_type']}",
            rr.Points3D(
                [c["position"]],
                colors=[_contact_color(c["object_type"])],
                radii=[0.04],
            ),
        )


__all__ = ["view_rerun"]
