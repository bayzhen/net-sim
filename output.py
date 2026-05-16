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
from solver_warp import SimulationResult, FrameSample, ContactEvent, QualityReport, StatsReport
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


def _feature_doc_from_arrays(
    shot: "ShotInput",
    ball_pos: np.ndarray,        # (F, 3)
    ball_vel: np.ndarray,        # (F, 3)
    particle_positions: Optional[np.ndarray],  # (F, N, 3) or None
    cp_indices: List[int],
    contacts_time: np.ndarray,
    contacts_type: np.ndarray,
    contacts_index: np.ndarray,
    contacts_position: np.ndarray,
    contacts_normal: np.ndarray,
    contacts_strength: np.ndarray,
    quality,
    stats,
    frame_dt: float,
) -> Dict:
    """Same shape as ``_feature_doc`` but built from raw numpy arrays. Used by
    the array-fast-path writer.
    """
    F = ball_pos.shape[0]
    times = (np.arange(F) * frame_dt).tolist()
    ball_traj = [
        {"time": float(times[i]), "position": ball_pos[i].tolist(), "velocity": ball_vel[i].tolist()}
        for i in range(F)
    ]
    cp_frames: List[Dict] = []
    if particle_positions is None:
        for i in range(F):
            cp_frames.append({"time": float(times[i]), "positions": []})
    else:
        idx = np.asarray([i for i in cp_indices if i < particle_positions.shape[1]], dtype=np.int64)
        if idx.size:
            sliced = particle_positions[:, idx, :]  # (F, K, 3)
            for i in range(F):
                cp_frames.append({
                    "time": float(times[i]),
                    "positions": sliced[i].tolist(),
                })
        else:
            for i in range(F):
                cp_frames.append({"time": float(times[i]), "positions": []})

    return {
        "sample_id": shot.sample_id,
        "schema_version": SCHEMA_VERSION,
        "input_features": _input_features(shot),
        "ball_trajectory": ball_traj,
        "net_control_points": cp_frames,
        "quality": {
            "clean": quality.clean,
            "issues": list(quality.issues),
            "target_hit": quality.target_hit,
            "max_penetration_depth": quality.max_penetration_depth,
            "max_penetration_time": quality.max_penetration_time,
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
        "support_stays": [
            {
                "index": s.index,
                "name": s.name,
                "corner_particle": s.corner_particle,
                "stake_particle": s.stake_particle,
                "constraint": s.constraint,
                "radius": s.radius,
            }
            for s in topology.support_stays
        ],
        "stake_particle_indices": list(topology.stake_particle_indices),
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


# ---------------------------------------------------------------------------
# Incremental (per-batch) writer for very large datasets.
# Keeps the same on-disk format as ``write_outputs`` but flushes after every
# batch so the host RAM never holds all results at once.
# ---------------------------------------------------------------------------


def _topology_doc(topology: Topology) -> Dict:
    """Serializable topology doc — exactly the structure that used to be
    embedded inside every raw sample. Now written once per dataset."""
    panel_particle_indices = {
        int(panel): list(idxs)
        for panel, idxs in topology.panel_particle_indices.items()
    }
    return {
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
        "support_stays": [
            {
                "index": s.index,
                "name": s.name,
                "corner_particle": s.corner_particle,
                "stake_particle": s.stake_particle,
                "constraint": s.constraint,
                "radius": s.radius,
            }
            for s in topology.support_stays
        ],
        "stake_particle_indices": list(topology.stake_particle_indices),
    }


def _raw_npz_arrays(shot: ShotInput, result: SimulationResult) -> Dict[str, np.ndarray]:
    """Pack a single sample's per-frame + contact data into numpy arrays
    suitable for ``np.savez_compressed``. Compact float32 layout.
    """
    from solver_warp import CONTACT_TYPE_NAMES
    name_to_id = {v: k for k, v in CONTACT_TYPE_NAMES.items()}

    F = len(result.frames)
    time_arr = np.empty(F, dtype=np.float32)
    ball_pos = np.empty((F, 3), dtype=np.float32)
    ball_vel = np.empty((F, 3), dtype=np.float32)

    has_particles = F > 0 and result.frames[0].particle_positions is not None
    if has_particles:
        N = np.asarray(result.frames[0].particle_positions).shape[0]
        particle_pos = np.empty((F, N, 3), dtype=np.float32)
    else:
        particle_pos = None

    for i, f in enumerate(result.frames):
        time_arr[i] = f.time
        ball_pos[i] = f.ball_position
        ball_vel[i] = f.ball_velocity
        if has_particles:
            particle_pos[i] = np.asarray(f.particle_positions, dtype=np.float32)

    C = len(result.contacts)
    c_time = np.empty(C, dtype=np.float32)
    c_obj_type = np.empty(C, dtype=np.int32)
    c_obj_index = np.empty(C, dtype=np.int32)
    c_pos = np.empty((C, 3), dtype=np.float32)
    c_normal = np.empty((C, 3), dtype=np.float32)
    c_strength = np.empty(C, dtype=np.float32)
    for i, c in enumerate(result.contacts):
        c_time[i] = c.time
        # object_type is a string name (e.g. "segment_swept"); map back to int.
        ot = c.object_type
        c_obj_type[i] = name_to_id.get(ot, -1) if isinstance(ot, str) else int(ot)
        c_obj_index[i] = c.object_index
        c_pos[i] = c.position
        c_normal[i] = c.normal
        c_strength[i] = c.strength

    arrays: Dict[str, np.ndarray] = {
        "frame_time": time_arr,
        "ball_position": ball_pos,
        "ball_velocity": ball_vel,
        "contact_time": c_time,
        "contact_object_type": c_obj_type,
        "contact_object_index": c_obj_index,
        "contact_position": c_pos,
        "contact_normal": c_normal,
        "contact_strength": c_strength,
    }
    if particle_pos is not None:
        arrays["particle_position"] = particle_pos
    return arrays


def _raw_npz_meta(shot: ShotInput, result: SimulationResult) -> Dict:
    """Per-sample scalar metadata embedded inside the npz file as a json blob.

    Kept as JSON-encoded bytes (saved under key ``meta_json``) so we don't have
    to pickle dataclasses — readers can ``json.loads(npz['meta_json'].item())``.
    """
    return {
        "shot": {
            "sample_id": shot.sample_id,
            "target_panel": shot.target_panel,
            "template": shot.template,
            "seed": shot.seed,
            "ball": {
                "position": _list3(shot.ball.position),
                "velocity": _list3(shot.ball.velocity),
                "angular_velocity": _list3(shot.ball.angular_velocity),
                "radius": float(shot.ball.radius),
                "mass": float(shot.ball.mass),
            },
        },
        "quality": {
            "clean": result.quality.clean,
            "issues": list(result.quality.issues),
            "target_hit": result.quality.target_hit,
            "max_penetration_depth": result.quality.max_penetration_depth,
            "max_penetration_time": result.quality.max_penetration_time,
        },
        "stats": asdict(result.stats),
    }


def _write_dataset_metadata(
    out: Path,
    params: GoalNetParams,
    topology: Topology,
    params_source_path: Optional[str],
) -> None:
    """Write topology.json + metadata.json (both written once per dataset)."""
    with (out / "topology.json").open("w") as f:
        json.dump(_topology_doc(topology), f, separators=(",", ":"))
    with (out / "metadata.json").open("w") as f:
        json.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "params_source_path": params_source_path,
                "params_snapshot": params.to_dict(),
                "topology_summary": topology_summary(topology),
            },
            f,
            indent=2,
        )


def make_incremental_writer(
    out_dir: str,
    params: GoalNetParams,
    topology: Topology,
    include_raw: bool,
    params_source_path: Optional[str] = None,
    raw_format: str = "npz",
):
    """Create an incremental writer.

    Args:
        raw_format: ``"npz"`` (compact float32 binary, default — fast for huge
            datasets) or ``"json"`` (legacy per-sample JSON, every sample
            self-contained including a copy of topology).

    Returns a tuple ``(append_chunk, finalize)`` where:
        append_chunk(shots, results) -> None  flushes immediately to disk
        finalize() -> dict[str,str]           writes batch_report.json and
                                              returns the same paths dict as
                                              ``write_outputs``.
    """
    if raw_format not in ("npz", "json"):
        raise ValueError(f"raw_format must be 'npz' or 'json', got {raw_format!r}")

    out = Path(out_dir)
    features_dir = out / "features"
    raw_dir = out / "raw"
    features_dir.mkdir(parents=True, exist_ok=True)
    if include_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    # Write dataset-level metadata once (topology.json + metadata.json).
    _write_dataset_metadata(out, params, topology, params_source_path)

    cp_indices = _control_point_indices(topology.num_particles)
    summary_path = out / "summary.jsonl"
    # truncate any prior run
    summary_fp = summary_path.open("w", buffering=1)  # line-buffered

    batch = {
        "sample_count": 0,
        "clean_count": 0,
        "abnormal_count": 0,
        "abnormal_types": {},
        "panel_stats": {},
    }

    def append_chunk(shots: List[ShotInput], results: List[SimulationResult]) -> None:
        for shot, result in zip(shots, results):
            feature_path = features_dir / f"{shot.sample_id}.json"
            feature_doc = _feature_doc(shot, result, cp_indices)
            with feature_path.open("w") as f:
                json.dump(feature_doc, f, separators=(",", ":"))
            raw_rel: Optional[str] = None
            if include_raw:
                if raw_format == "npz":
                    raw_path = raw_dir / f"{shot.sample_id}.npz"
                    arrays = _raw_npz_arrays(shot, result)
                    meta = _raw_npz_meta(shot, result)
                    arrays["meta_json"] = np.array(
                        json.dumps(meta, separators=(",", ":"))
                    )
                    # ``savez`` (uncompressed) is far faster than
                    # ``savez_compressed``: float arrays are already mostly
                    # incompressible, and JSON-thick meta is tiny.
                    np.savez(raw_path, **arrays)
                else:
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
            summary_fp.write(json.dumps(summary, separators=(",", ":")) + "\n")

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
        summary_fp.flush()

    def append_chunk_arrays(shots: List[ShotInput], arrs: Dict) -> None:
        """Fast-path writer that takes the raw array dict from
        ``XpbdWarpSolver.simulate_arrays`` and slices it directly into per-
        sample npz files. Avoids constructing B*F FrameSample objects.

        Falls back to JSON raw format only on explicit request — JSON path is
        still legacy through ``append_chunk``.
        """
        if raw_format != "npz":
            raise ValueError(
                "append_chunk_arrays only supports raw_format='npz'; for json "
                "use the regular append_chunk pathway."
            )

        # Slice numpy views once per sample (cheap)
        bp = arrs["frame_ball_pos"]            # (B, F, 3) float32
        bv = arrs["frame_ball_vel"]            # (B, F, 3) float32
        pp = arrs.get("frame_particles")       # (B, F, N, 3) float32 or None
        c_count = arrs["contact_counts"]       # (B,) int
        c_time = arrs["contact_times"]         # (B, max_contacts)
        c_type = arrs["contact_types"]
        c_obj = arrs["contact_objs"]
        c_pos = arrs["contact_positions"]
        c_norm = arrs["contact_normals"]
        c_str = arrs["contact_strengths"]
        quals: List[QualityReport] = arrs["per_sample_quality"]
        stats_list: List[StatsReport] = arrs["per_sample_stats"]

        F = bp.shape[1]
        # frame_time is identical for every sample — precompute once
        frame_dt = stats_list[0].frame_dt if stats_list else 1.0 / 60.0
        time_arr = (np.arange(F, dtype=np.float32) * frame_dt)

        max_c = c_time.shape[1] if c_time.ndim == 2 else 0

        for b, shot in enumerate(shots):
            quality = quals[b]
            stats = stats_list[b]
            n_c = min(int(c_count[b]), max_c)

            # Slice contact arrays to actual count + sort by time for nicer
            # downstream consumption (matches legacy behavior).
            if n_c > 0:
                ct = c_time[b, :n_c]
                order = np.argsort(ct)
                contact_time = ct[order].astype(np.float32, copy=False)
                contact_object_type = c_type[b, :n_c][order].astype(np.int32, copy=False)
                contact_object_index = c_obj[b, :n_c][order].astype(np.int32, copy=False)
                contact_position = c_pos[b, :n_c][order].astype(np.float32, copy=False)
                contact_normal = c_norm[b, :n_c][order].astype(np.float32, copy=False)
                contact_strength = c_str[b, :n_c][order].astype(np.float32, copy=False)
            else:
                contact_time = np.empty(0, dtype=np.float32)
                contact_object_type = np.empty(0, dtype=np.int32)
                contact_object_index = np.empty(0, dtype=np.int32)
                contact_position = np.empty((0, 3), dtype=np.float32)
                contact_normal = np.empty((0, 3), dtype=np.float32)
                contact_strength = np.empty(0, dtype=np.float32)

            # Build npz contents — slices are zero-copy views into the big
            # batch arrays, so no per-sample copy.
            arr_dict = {
                "frame_time": time_arr,
                "ball_position": bp[b],
                "ball_velocity": bv[b],
                "contact_time": contact_time,
                "contact_object_type": contact_object_type,
                "contact_object_index": contact_object_index,
                "contact_position": contact_position,
                "contact_normal": contact_normal,
                "contact_strength": contact_strength,
            }
            if pp is not None:
                arr_dict["particle_position"] = pp[b]

            meta = {
                "shot": {
                    "sample_id": shot.sample_id,
                    "target_panel": shot.target_panel,
                    "template": shot.template,
                    "seed": shot.seed,
                    "ball": {
                        "position": _list3(shot.ball.position),
                        "velocity": _list3(shot.ball.velocity),
                        "angular_velocity": _list3(shot.ball.angular_velocity),
                        "radius": float(shot.ball.radius),
                        "mass": float(shot.ball.mass),
                    },
                },
                "quality": {
                    "clean": quality.clean,
                    "issues": list(quality.issues),
                    "target_hit": quality.target_hit,
                    "max_penetration_depth": quality.max_penetration_depth,
                    "max_penetration_time": quality.max_penetration_time,
                },
                "stats": asdict(stats),
            }
            arr_dict["meta_json"] = np.array(json.dumps(meta, separators=(",", ":")))

            raw_path = raw_dir / f"{shot.sample_id}.npz"
            np.savez(raw_path, **arr_dict)

            # features.json (sparse net control points + ball trajectory)
            feature_doc = _feature_doc_from_arrays(
                shot, bp[b], bv[b],
                particle_positions=pp[b] if pp is not None else None,
                cp_indices=cp_indices,
                contacts_time=contact_time,
                contacts_type=contact_object_type,
                contacts_index=contact_object_index,
                contacts_position=contact_position,
                contacts_normal=contact_normal,
                contacts_strength=contact_strength,
                quality=quality,
                stats=stats,
                frame_dt=frame_dt,
            )
            feature_path = features_dir / f"{shot.sample_id}.json"
            with feature_path.open("w") as fp:
                json.dump(feature_doc, fp, separators=(",", ":"))

            # summary line
            summary = {
                "sample_id": shot.sample_id,
                "paths": {
                    "feature": str(feature_path.relative_to(out)),
                    "raw": str(raw_path.relative_to(out)),
                },
                "metadata": {"schema_version": SCHEMA_VERSION, "seed": shot.seed},
                "input_features": {
                    "sample_id": shot.sample_id,
                    "target_panel": shot.target_panel,
                    "template": shot.template,
                    "seed": shot.seed,
                    "position": _list3(shot.ball.position),
                    "velocity": _list3(shot.ball.velocity),
                    "angular_velocity": _list3(shot.ball.angular_velocity),
                    "radius": float(shot.ball.radius),
                    "mass": float(shot.ball.mass),
                },
                "quality": {
                    "clean": quality.clean,
                    "issues": list(quality.issues),
                    "target_hit": quality.target_hit,
                    "max_penetration_depth": quality.max_penetration_depth,
                    "max_penetration_time": quality.max_penetration_time,
                },
                "stats": asdict(stats),
                "contact_count": stats.contact_count,
                "topology_summary": topology_summary(topology),
            }
            summary_fp.write(json.dumps(summary, separators=(",", ":")) + "\n")

            # batch report stats
            batch["sample_count"] += 1
            if quality.clean:
                batch["clean_count"] += 1
            else:
                batch["abnormal_count"] += 1
                for issue in quality.issues:
                    batch["abnormal_types"][issue] = batch["abnormal_types"].get(issue, 0) + 1
            panel = shot.target_panel
            ps = batch["panel_stats"].setdefault(
                panel, {"samples": 0, "contacts": 0, "abnormal": 0}
            )
            ps["samples"] += 1
            ps["contacts"] += stats.contact_count
            if not quality.clean:
                ps["abnormal"] += 1
        summary_fp.flush()

    def finalize() -> Dict[str, str]:
        summary_fp.close()
        batch_path = out / "batch_report.json"
        with batch_path.open("w") as f:
            json.dump(batch, f, indent=2)
        return {
            "features_dir": str(features_dir),
            "raw_dir": str(raw_dir) if include_raw else "",
            "summary_path": str(summary_path),
            "batch_report_path": str(batch_path),
        }

    return append_chunk, finalize, append_chunk_arrays


# ---------------------------------------------------------------------------
# HDF5 dataset writer (single-file, chunked, extendable). Designed for the
# "GPU-saturated" use case: each completed batch from
# ``XpbdWarpSolver.simulate_arrays`` slots into preallocated chunks with a
# single numpy assignment — no Python encoding, no per-sample syscalls.
#
# File layout (root group):
#   attrs:
#     schema_version, topology_json, metadata_json, frame_dt, frame_count,
#     particle_count, max_contacts, issue_names
#   datasets (S = sample count, F = frame_count, N = particle_count):
#     sample_id            (S,) vlen str
#     target_panel         (S,) vlen str
#     template             (S,) vlen str
#     seed                 (S,) int64
#     input_position       (S, 3) f32   ball initial pos
#     input_velocity       (S, 3) f32   ball initial vel
#     input_angular        (S, 3) f32
#     input_radius         (S,)   f32
#     input_mass           (S,)   f32
#     ball_position        (S, F, 3)    f32
#     ball_velocity        (S, F, 3)    f32
#     particle_position    (S, F, N, 3) f32   (only if record_particles)
#     contact_offset       (S+1,) i64   CSR index into flat contact arrays
#     contact_time         (Total,) f32
#     contact_object_type  (Total,) i32
#     contact_object_index (Total,) i32
#     contact_position     (Total, 3) f32
#     contact_normal       (Total, 3) f32
#     contact_strength     (Total,) f32
#     quality_clean        (S,) bool
#     quality_target_hit   (S,) bool
#     quality_issue_mask   (S, K) bool   K = len(issue_names)
#     quality_max_pen      (S,) f32
#     quality_max_pen_time (S,) f32
#     stats_contact_count  (S,) i32
#     stats_max_disp       (S,) f32
#     stats_came_to_rest   (S,) bool
# ---------------------------------------------------------------------------


# Fixed list of issue names that the solver can emit. Order = bit position in
# ``quality_issue_mask``. Keep in sync with ``solver_warp._compute_quality_*``.
_ISSUE_NAMES: List[str] = [
    "severe_penetration",
    "nan_or_inf",
    "velocity_explosion",
    "particle_velocity_explosion",
    "constraint_divergence",
    "stuck",
    "target_panel_missed",
]
_ISSUE_INDEX = {name: i for i, name in enumerate(_ISSUE_NAMES)}


def make_h5_writer(
    out_dir: str,
    params: GoalNetParams,
    topology: Topology,
    include_raw: bool,
    params_source_path: Optional[str] = None,
    chunk_size: int = 512,
):
    """Create an HDF5 dataset writer.

    Args:
        out_dir: directory to create. The actual file is ``<out_dir>/dataset.h5``.
            ``topology.json`` and ``metadata.json`` are still written alongside
            for human inspection / cross-tool compat.
        chunk_size: hint for HDF5 chunk shape along the sample dim. Should be
            close to your batch size for fastest writes.

    Returns ``(append_chunk_arrays, finalize)``:
        append_chunk_arrays(shots, arrs)  -> None
        finalize()                        -> dict[str, str]
    """
    try:
        import h5py
    except ImportError as e:
        raise SystemExit(
            "h5py is not installed; run `pip install h5py`"
        ) from e

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # also drop legacy auxiliary files for tooling compat
    _write_dataset_metadata(out, params, topology, params_source_path)

    h5_path = out / "dataset.h5"
    f = h5py.File(h5_path, "w", libver="latest")

    # Pre-compute frame layout from solver params
    frame_dt = params.solver.frame_dt
    F = int(round(params.solver.duration / frame_dt)) + 1
    N = topology.num_particles
    K = len(_ISSUE_NAMES)
    str_dtype = h5py.string_dtype(encoding="utf-8")

    # Root attrs
    f.attrs["schema_version"] = SCHEMA_VERSION
    f.attrs["topology_json"] = json.dumps(_topology_doc(topology))
    f.attrs["metadata_json"] = json.dumps({
        "params_source_path": params_source_path,
        "params_snapshot": params.to_dict(),
        "topology_summary": topology_summary(topology),
    })
    f.attrs["frame_dt"] = float(frame_dt)
    f.attrs["frame_count"] = F
    f.attrs["particle_count"] = N
    f.attrs["issue_names"] = np.array(_ISSUE_NAMES, dtype=object)
    f.attrs["include_raw"] = bool(include_raw)

    # Helper: create extendable dataset
    def _ds(name: str, shape, dtype, chunks=None):
        maxshape = (None,) + tuple(shape[1:])
        return f.create_dataset(
            name, shape=shape, maxshape=maxshape, dtype=dtype,
            chunks=chunks if chunks is not None else (chunk_size,) + tuple(shape[1:]),
        )

    # All preallocated as size-0 along the sample axis; we resize each chunk.
    _ds("sample_id", (0,), str_dtype, chunks=(chunk_size,))
    _ds("target_panel", (0,), str_dtype, chunks=(chunk_size,))
    _ds("template", (0,), str_dtype, chunks=(chunk_size,))
    _ds("seed", (0,), np.int64, chunks=(chunk_size,))
    _ds("input_position", (0, 3), np.float32, chunks=(chunk_size, 3))
    _ds("input_velocity", (0, 3), np.float32, chunks=(chunk_size, 3))
    _ds("input_angular", (0, 3), np.float32, chunks=(chunk_size, 3))
    _ds("input_radius", (0,), np.float32, chunks=(chunk_size,))
    _ds("input_mass", (0,), np.float32, chunks=(chunk_size,))

    if include_raw:
        # The big arrays. chunked along sample axis only (full F,N kept
        # together so each sample is a contiguous slab — fastest for both
        # batch writes and per-sample reads).
        _ds("ball_position", (0, F, 3), np.float32, chunks=(chunk_size, F, 3))
        _ds("ball_velocity", (0, F, 3), np.float32, chunks=(chunk_size, F, 3))
        # particle_position: ~24 GB for 80k samples; chunk one sample per row
        # so reads can mmap individual samples without pulling neighbours.
        _ds("particle_position", (0, F, N, 3), np.float32, chunks=(1, F, N, 3))

    # Contacts in CSR form. contact_offset has S+1 entries; start with [0].
    co = f.create_dataset(
        "contact_offset", shape=(1,), maxshape=(None,), dtype=np.int64,
        chunks=(chunk_size + 1,),
    )
    co[0] = 0
    _ds("contact_time", (0,), np.float32, chunks=(8192,))
    _ds("contact_object_type", (0,), np.int32, chunks=(8192,))
    _ds("contact_object_index", (0,), np.int32, chunks=(8192,))
    _ds("contact_position", (0, 3), np.float32, chunks=(8192, 3))
    _ds("contact_normal", (0, 3), np.float32, chunks=(8192, 3))
    _ds("contact_strength", (0,), np.float32, chunks=(8192,))

    _ds("quality_clean", (0,), np.bool_, chunks=(chunk_size,))
    _ds("quality_target_hit", (0,), np.bool_, chunks=(chunk_size,))
    _ds("quality_issue_mask", (0, K), np.bool_, chunks=(chunk_size, K))
    _ds("quality_max_pen", (0,), np.float32, chunks=(chunk_size,))
    _ds("quality_max_pen_time", (0,), np.float32, chunks=(chunk_size,))
    _ds("stats_contact_count", (0,), np.int32, chunks=(chunk_size,))
    _ds("stats_max_disp", (0,), np.float32, chunks=(chunk_size,))
    _ds("stats_came_to_rest", (0,), np.bool_, chunks=(chunk_size,))

    f.swmr_mode = True  # allow concurrent readers while writing

    # Mutable state captured by the closure
    state = {"n": 0, "n_contacts": 0}
    batch = {
        "sample_count": 0,
        "clean_count": 0,
        "abnormal_count": 0,
        "abnormal_types": {},
        "panel_stats": {},
    }

    def _ext(name: str, new_size_first_dim: int) -> None:
        ds = f[name]
        new_shape = (new_size_first_dim,) + ds.shape[1:]
        ds.resize(new_shape)

    def append_chunk_arrays(shots: List[ShotInput], arrs: Dict) -> None:
        """Append one solver chunk of B samples to the HDF5 file."""
        B = len(shots)
        i0 = state["n"]
        i1 = i0 + B

        # ---- per-sample scalars ----
        for name in (
            "sample_id", "target_panel", "template", "seed",
            "input_position", "input_velocity", "input_angular",
            "input_radius", "input_mass",
            "quality_clean", "quality_target_hit", "quality_issue_mask",
            "quality_max_pen", "quality_max_pen_time",
            "stats_contact_count", "stats_max_disp", "stats_came_to_rest",
        ):
            _ext(name, i1)
        f["sample_id"][i0:i1] = np.array([s.sample_id for s in shots], dtype=object)
        f["target_panel"][i0:i1] = np.array([s.target_panel for s in shots], dtype=object)
        f["template"][i0:i1] = np.array([s.template for s in shots], dtype=object)
        f["seed"][i0:i1] = np.array([s.seed for s in shots], dtype=np.int64)
        f["input_position"][i0:i1] = np.array([s.ball.position for s in shots], dtype=np.float32)
        f["input_velocity"][i0:i1] = np.array([s.ball.velocity for s in shots], dtype=np.float32)
        f["input_angular"][i0:i1] = np.array([s.ball.angular_velocity for s in shots], dtype=np.float32)
        f["input_radius"][i0:i1] = np.array([s.ball.radius for s in shots], dtype=np.float32)
        f["input_mass"][i0:i1] = np.array([s.ball.mass for s in shots], dtype=np.float32)

        quals: List[QualityReport] = arrs["per_sample_quality"]
        stats_list: List[StatsReport] = arrs["per_sample_stats"]
        clean_arr = np.fromiter((q.clean for q in quals), dtype=np.bool_, count=B)
        target_hit_arr = np.fromiter((q.target_hit for q in quals), dtype=np.bool_, count=B)
        max_pen_arr = np.fromiter((q.max_penetration_depth for q in quals), dtype=np.float32, count=B)
        max_pen_time_arr = np.fromiter((q.max_penetration_time for q in quals), dtype=np.float32, count=B)
        f["quality_clean"][i0:i1] = clean_arr
        f["quality_target_hit"][i0:i1] = target_hit_arr
        f["quality_max_pen"][i0:i1] = max_pen_arr
        f["quality_max_pen_time"][i0:i1] = max_pen_time_arr

        # issue mask
        issue_mask = np.zeros((B, K), dtype=np.bool_)
        for b, q in enumerate(quals):
            for issue in q.issues:
                idx = _ISSUE_INDEX.get(issue)
                if idx is not None:
                    issue_mask[b, idx] = True
        f["quality_issue_mask"][i0:i1] = issue_mask

        f["stats_contact_count"][i0:i1] = np.fromiter(
            (s.contact_count for s in stats_list), dtype=np.int32, count=B)
        f["stats_max_disp"][i0:i1] = np.fromiter(
            (s.max_net_displacement for s in stats_list), dtype=np.float32, count=B)
        f["stats_came_to_rest"][i0:i1] = np.fromiter(
            (s.ball_came_to_rest for s in stats_list), dtype=np.bool_, count=B)

        # ---- big arrays ----
        if include_raw:
            _ext("ball_position", i1)
            _ext("ball_velocity", i1)
            _ext("particle_position", i1)
            f["ball_position"][i0:i1] = arrs["frame_ball_pos"][:B]
            f["ball_velocity"][i0:i1] = arrs["frame_ball_vel"][:B]
            if arrs.get("frame_particles") is not None:
                f["particle_position"][i0:i1] = arrs["frame_particles"][:B]

        # ---- contacts (CSR) ----
        c_count = arrs["contact_counts"][:B]
        max_c = arrs["contact_times"].shape[1]
        per_sample_n = np.minimum(c_count, max_c).astype(np.int64)
        total_new = int(per_sample_n.sum())

        # build flat arrays for this chunk's contacts
        if total_new > 0:
            ct_chunks = []
            cot_chunks = []
            coi_chunks = []
            cp_chunks = []
            cn_chunks = []
            cs_chunks = []
            for b in range(B):
                n = int(per_sample_n[b])
                if n == 0:
                    continue
                ct = arrs["contact_times"][b, :n]
                # sort by time for downstream convenience
                order = np.argsort(ct)
                ct_chunks.append(ct[order].astype(np.float32, copy=False))
                cot_chunks.append(arrs["contact_types"][b, :n][order].astype(np.int32, copy=False))
                coi_chunks.append(arrs["contact_objs"][b, :n][order].astype(np.int32, copy=False))
                cp_chunks.append(arrs["contact_positions"][b, :n][order].astype(np.float32, copy=False))
                cn_chunks.append(arrs["contact_normals"][b, :n][order].astype(np.float32, copy=False))
                cs_chunks.append(arrs["contact_strengths"][b, :n][order].astype(np.float32, copy=False))
            ct_flat = np.concatenate(ct_chunks)
            cot_flat = np.concatenate(cot_chunks)
            coi_flat = np.concatenate(coi_chunks)
            cp_flat = np.concatenate(cp_chunks)
            cn_flat = np.concatenate(cn_chunks)
            cs_flat = np.concatenate(cs_chunks)

            j0 = state["n_contacts"]
            j1 = j0 + total_new
            _ext("contact_time", j1)
            _ext("contact_object_type", j1)
            _ext("contact_object_index", j1)
            _ext("contact_position", j1)
            _ext("contact_normal", j1)
            _ext("contact_strength", j1)
            f["contact_time"][j0:j1] = ct_flat
            f["contact_object_type"][j0:j1] = cot_flat
            f["contact_object_index"][j0:j1] = coi_flat
            f["contact_position"][j0:j1] = cp_flat
            f["contact_normal"][j0:j1] = cn_flat
            f["contact_strength"][j0:j1] = cs_flat
            state["n_contacts"] = j1

        # extend contact_offset by B entries (running cumulative count)
        co_ds = f["contact_offset"]
        co_ds.resize((i1 + 1,))
        prev_tail = int(co_ds[i0])
        cs = prev_tail + per_sample_n.cumsum().astype(np.int64)
        co_ds[i0 + 1 : i1 + 1] = cs

        # ---- batch report stats ----
        for b in range(B):
            batch["sample_count"] += 1
            if quals[b].clean:
                batch["clean_count"] += 1
            else:
                batch["abnormal_count"] += 1
                for issue in quals[b].issues:
                    batch["abnormal_types"][issue] = batch["abnormal_types"].get(issue, 0) + 1
            panel = shots[b].target_panel
            ps = batch["panel_stats"].setdefault(
                panel, {"samples": 0, "contacts": 0, "abnormal": 0}
            )
            ps["samples"] += 1
            ps["contacts"] += stats_list[b].contact_count
            if not quals[b].clean:
                ps["abnormal"] += 1

        state["n"] = i1
        f.flush()

    def finalize() -> Dict[str, str]:
        f.close()
        batch_path = out / "batch_report.json"
        with batch_path.open("w") as fp:
            json.dump(batch, fp, indent=2)
        return {
            "h5_path": str(h5_path),
            "topology_path": str(out / "topology.json"),
            "metadata_path": str(out / "metadata.json"),
            "batch_report_path": str(batch_path),
        }

    return append_chunk_arrays, finalize


__all__ = ["write_outputs", "SCHEMA_VERSION"]
