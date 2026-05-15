"""Output writer — features/raw/summary/batch_report per §7 of the design doc."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from params import GoalNetParams
from sampler import ShotInput
from solver_warp import SimulationResult, FrameSample, ContactEvent
from topology import Topology, summary as topology_summary


SCHEMA_VERSION = "goal_net_params.v1"
_CONTROL_POINT_COUNT = 16


def _list3(v) -> List[float]:
    return [float(v[0]), float(v[1]), float(v[2])]


def _input_features(shot: ShotInput) -> Dict:
    b = shot.ball
    return {
        "sample_id": shot.sample_id,
        "target_panel": shot.target_panel,
        "template": shot.template,
        "seed": shot.seed,
        "position": _list3(b.position),
        "velocity": _list3(b.velocity),
        "angular_velocity": _list3(b.angular_velocity),
        "radius": float(b.radius),
        "mass": float(b.mass),
    }


def _control_point_indices(num_particles: int) -> List[int]:
    if num_particles == 0:
        return []
    stride = max(1, num_particles // _CONTROL_POINT_COUNT)
    return list(range(0, num_particles, stride))[:_CONTROL_POINT_COUNT]


def _feature_doc(
    shot: ShotInput,
    result: SimulationResult,
    cp_indices: List[int],
) -> Dict:
    ball_traj = [
        {
            "time": f.time,
            "position": _list3(f.ball_position),
            "velocity": _list3(f.ball_velocity),
        }
        for f in result.frames
    ]
    cp_frames: List[Dict] = []
    for f in result.frames:
        if f.particle_positions is None:
            positions: List[List[float]] = []
        else:
            arr = np.asarray(f.particle_positions)
            positions = [arr[i].tolist() for i in cp_indices if i < arr.shape[0]]
        cp_frames.append({"time": f.time, "positions": positions})
    return {
        "sample_id": shot.sample_id,
        "schema_version": SCHEMA_VERSION,
        "input_features": _input_features(shot),
        "ball_trajectory": ball_traj,
        "net_control_points": cp_frames,
        "quality": {
            "clean": result.quality.clean,
            "issues": list(result.quality.issues),
            "target_hit": result.quality.target_hit,
            "max_penetration_depth": result.quality.max_penetration_depth,
            "max_penetration_time": result.quality.max_penetration_time,
        },
    }


def _raw_doc(
    shot: ShotInput,
    result: SimulationResult,
    params: GoalNetParams,
    topology: Topology,
    params_source_path: Optional[str],
) -> Dict:
    frames: List[Dict] = []
    for f in result.frames:
        entry = {
            "time": f.time,
            "ball_position": _list3(f.ball_position),
            "ball_velocity": _list3(f.ball_velocity),
        }
        if f.particle_positions is not None:
            entry["particle_positions"] = np.asarray(f.particle_positions).tolist()
        frames.append(entry)
    contacts = [
        {
            "time": c.time,
            "object_type": c.object_type,
            "object_index": c.object_index,
            "position": list(c.position),
            "normal": list(c.normal),
            "strength": c.strength,
        }
        for c in result.contacts
    ]
    panel_particle_indices = {
        int(panel): list(idxs)
        for panel, idxs in topology.panel_particle_indices.items()
    }
    topo_doc = {
        "particles": [
            {
                "index": p.index,
                "position": list(p.position),
                "panel": p.panel,
                "u": p.u,
                "v": p.v,
                "anchored": p.anchored,
            }
            for p in topology.particles
        ],
        "distance_constraints": [
            {
                "index": c.index,
                "i0": c.i0,
                "i1": c.i1,
                "rest_length": c.rest_length,
                "stiffness": c.stiffness,
                "kind": c.kind,
                "panel": c.panel,
            }
            for c in topology.distance_constraints
        ],
        "anchor_constraints": [
            {
                "index": a.index,
                "particle": a.particle,
                "target": list(a.target),
                "stiffness": a.stiffness,
                "hard": a.hard,
            }
            for a in topology.anchor_constraints
        ],
        "bend_constraints": [
            c.index for c in topology.distance_constraints if c.kind == 1
        ],
        "panel_particle_indices": panel_particle_indices,
        "goalpost_segments": [
            {
                "index": g.index,
                "name": g.name,
                "p0": list(g.p0),
                "p1": list(g.p1),
                "radius": g.radius,
                "kind": g.kind,
            }
            for g in topology.goalpost_segments
        ],
    }
    return {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "params_source_path": params_source_path,
            "params_snapshot": params.to_dict(),
            "seed": shot.seed,
        },
        "shot": _input_features(shot),
        "topology_summary": topology_summary(topology),
        "topology": topo_doc,
        "frames": frames,
        "contacts": contacts,
        "quality": {
            "clean": result.quality.clean,
            "issues": list(result.quality.issues),
            "target_hit": result.quality.target_hit,
            "max_penetration_depth": result.quality.max_penetration_depth,
            "max_penetration_time": result.quality.max_penetration_time,
        },
        "stats": asdict(result.stats),
    }


def _summary_doc(
    shot: ShotInput,
    result: SimulationResult,
    feature_path: str,
    raw_path: Optional[str],
    topology: Topology,
) -> Dict:
    return {
        "sample_id": shot.sample_id,
        "paths": {"feature": feature_path, "raw": raw_path},
        "metadata": {"schema_version": SCHEMA_VERSION, "seed": shot.seed},
        "input_features": _input_features(shot),
        "quality": {
            "clean": result.quality.clean,
            "issues": list(result.quality.issues),
            "target_hit": result.quality.target_hit,
            "max_penetration_depth": result.quality.max_penetration_depth,
            "max_penetration_time": result.quality.max_penetration_time,
        },
        "stats": asdict(result.stats),
        "contact_count": result.stats.contact_count,
        "topology_summary": topology_summary(topology),
    }


def write_outputs(
    out_dir: str,
    params: GoalNetParams,
    topology: Topology,
    shots: List[ShotInput],
    results: List[SimulationResult],
    include_raw: bool,
    params_source_path: Optional[str] = None,
) -> Dict[str, str]:
    """Write features, optionally raw, summary.jsonl, batch_report.json.

    Returns a dict with the output paths.
    """
    out = Path(out_dir)
    features_dir = out / "features"
    raw_dir = out / "raw"
    features_dir.mkdir(parents=True, exist_ok=True)
    if include_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    cp_indices = _control_point_indices(topology.num_particles)

    summary_lines: List[str] = []
    batch = {
        "sample_count": 0,
        "clean_count": 0,
        "abnormal_count": 0,
        "abnormal_types": {},
        "panel_stats": {},
    }

    for shot, result in zip(shots, results):
        feature_path = features_dir / f"{shot.sample_id}.json"
        feature_doc = _feature_doc(shot, result, cp_indices)
        with feature_path.open("w") as f:
            json.dump(feature_doc, f, separators=(",", ":"))
        raw_rel: Optional[str] = None
        if include_raw:
            raw_path = raw_dir / f"{shot.sample_id}.json"
            raw_doc = _raw_doc(shot, result, params, topology, params_source_path)
            with raw_path.open("w") as f:
                json.dump(raw_doc, f, separators=(",", ":"))
            raw_rel = str(raw_path.relative_to(out))
        summary = _summary_doc(
            shot,
            result,
            feature_path=str(feature_path.relative_to(out)),
            raw_path=raw_rel,
            topology=topology,
        )
        summary_lines.append(json.dumps(summary, separators=(",", ":")))

        batch["sample_count"] += 1
        if result.quality.clean:
            batch["clean_count"] += 1
        else:
            batch["abnormal_count"] += 1
            for issue in result.quality.issues:
                batch["abnormal_types"][issue] = batch["abnormal_types"].get(issue, 0) + 1
        panel = shot.target_panel
        ps = batch["panel_stats"].setdefault(
            panel, {"samples": 0, "contacts": 0, "abnormal": 0}
        )
        ps["samples"] += 1
        ps["contacts"] += result.stats.contact_count
        if not result.quality.clean:
            ps["abnormal"] += 1

    summary_path = out / "summary.jsonl"
    with summary_path.open("w") as f:
        f.write("\n".join(summary_lines) + ("\n" if summary_lines else ""))
    batch_path = out / "batch_report.json"
    with batch_path.open("w") as f:
        json.dump(batch, f, indent=2)

    return {
        "features_dir": str(features_dir),
        "raw_dir": str(raw_dir) if include_raw else "",
        "summary_path": str(summary_path),
        "batch_report_path": str(batch_path),
    }


__all__ = ["write_outputs", "SCHEMA_VERSION"]
