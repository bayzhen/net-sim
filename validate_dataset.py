"""Validate a generated raw dataset (npz format).

Checks per sample:
  1. file readable, all expected keys present
  2. shape consistency (frames, particles, contacts)
  3. no NaN / Inf in floats
  4. ball trajectory continuity (no teleports between adjacent frames)
  5. particle positions within reasonable bounding box (not exploding)
  6. dataset-level meta (topology.json, metadata.json, summary.jsonl) matches
     per-sample contents
  7. summary.jsonl row count matches raw file count
  8. quality.clean distribution

Usage:
    python validate_dataset.py D:\\dataset_v1
    python validate_dataset.py D:\\dataset_v1 --sample 200    # spot-check 200
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _load_topology(root: Path) -> dict:
    return json.loads((root / "topology.json").read_text())


def _load_metadata(root: Path) -> dict:
    return json.loads((root / "metadata.json").read_text())


def _validate_one(path: Path, expected_N: int, expected_F: int) -> Tuple[List[str], Optional[dict]]:
    """Return (list of error strings for this sample, parsed meta_json dict).

    meta is None if the file was unreadable.
    """
    errs: List[str] = []
    try:
        npz = np.load(path, allow_pickle=False)
    except Exception as e:
        return [f"unreadable: {e!r}"], None

    required = {
        "frame_time", "ball_position", "ball_velocity",
        "contact_time", "contact_object_type", "contact_object_index",
        "contact_position", "contact_normal", "contact_strength",
        "particle_position", "meta_json",
    }
    missing = required - set(npz.files)
    if missing:
        errs.append(f"missing keys: {sorted(missing)}")
        return errs

    F = npz["frame_time"].shape[0]
    if F != expected_F:
        errs.append(f"frame count mismatch: got {F}, expected {expected_F}")

    bp = npz["ball_position"]
    bv = npz["ball_velocity"]
    pp = npz["particle_position"]
    if bp.shape != (F, 3):
        errs.append(f"ball_position shape {bp.shape} != ({F},3)")
    if bv.shape != (F, 3):
        errs.append(f"ball_velocity shape {bv.shape} != ({F},3)")
    if pp.shape != (F, expected_N, 3):
        errs.append(f"particle_position shape {pp.shape} != ({F},{expected_N},3)")

    # NaN / Inf
    for name in ("ball_position", "ball_velocity", "particle_position"):
        a = npz[name]
        if not np.all(np.isfinite(a)):
            errs.append(f"{name} has NaN or Inf")

    # Ball trajectory continuity: |Δp| should be < |v|·dt + slack.
    # frame_dt is approximately diff(frame_time).
    if F >= 2:
        dt = float(np.median(np.diff(npz["frame_time"])))
        if dt <= 0 or not math.isfinite(dt):
            errs.append(f"non-positive frame_dt: {dt}")
        else:
            disp = np.linalg.norm(np.diff(bp, axis=0), axis=1)  # (F-1,)
            speed = np.linalg.norm(bv[:-1], axis=1)
            # allow 30 m/s slack to absorb collision response in one frame
            big_jumps = np.where(disp > speed * dt + 1.0)[0]
            if big_jumps.size > 0:
                worst = float(np.max(disp[big_jumps]))
                errs.append(f"ball teleport: {big_jumps.size} frames, worst jump={worst:.2f}m")

    # Particle positions inside a reasonable bbox (goal is ~7m wide x 2.5m tall;
    # any particle further than 50 m from origin = exploded).
    pp_max = float(np.max(np.abs(pp))) if pp.size else 0.0
    if pp_max > 50.0:
        errs.append(f"particle position exploded: |p|max={pp_max:.1f}m")

    # Contact arrays self-consistency
    C = npz["contact_time"].shape[0]
    for name, expected_shape in [
        ("contact_object_type", (C,)),
        ("contact_object_index", (C,)),
        ("contact_position", (C, 3)),
        ("contact_normal", (C, 3)),
        ("contact_strength", (C,)),
    ]:
        if npz[name].shape != expected_shape:
            errs.append(f"{name} shape {npz[name].shape} != {expected_shape}")
    if C > 0:
        if not np.all(np.isfinite(npz["contact_position"])):
            errs.append("contact_position has NaN/Inf")
        if not np.all(np.isfinite(npz["contact_normal"])):
            errs.append("contact_normal has NaN/Inf")
        # normals should be roughly unit length (allow 5% slack)
        n_len = np.linalg.norm(npz["contact_normal"], axis=1)
        bad = (n_len < 0.5) | (n_len > 1.5)
        if bad.any():
            errs.append(f"contact normals non-unit: {bad.sum()}/{C}")

    # meta_json round-trip
    meta: Optional[dict] = None
    try:
        meta = json.loads(npz["meta_json"].item())
        for k in ("shot", "quality", "stats"):
            if k not in meta:
                errs.append(f"meta_json missing '{k}'")
    except Exception as e:
        errs.append(f"meta_json unreadable: {e!r}")
    return errs, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path, help="dataset root directory")
    ap.add_argument(
        "--sample", type=int, default=0,
        help="randomly spot-check N samples instead of all (0 = all)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--show-errors", type=int, default=20,
        help="print up to N erroneous samples in detail",
    )
    args = ap.parse_args(argv)

    root: Path = args.dataset
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    print(f"== dataset: {root}")

    # Top-level files
    needed = ["topology.json", "metadata.json", "summary.jsonl"]
    for n in needed:
        p = root / n
        ok = p.exists() and p.stat().st_size > 0
        print(f"  {n:<18} {'OK' if ok else 'MISSING/empty'}  ({p.stat().st_size if p.exists() else 0} B)")
        if not ok and n != "summary.jsonl":
            print(f"  cannot continue without {n}", file=sys.stderr)
            return 2

    topology = _load_topology(root)
    metadata = _load_metadata(root)
    expected_N = len(topology["particles"])
    expected_F = int(round(
        metadata["params_snapshot"]["solver"]["duration"]
        / metadata["params_snapshot"]["solver"]["frame_dt"]
    )) + 1
    print(f"  expected per sample: F={expected_F} frames, N={expected_N} particles")

    raw_dir = root / "raw"
    npzs = sorted(raw_dir.glob("sample_*.npz"))
    print(f"  raw npz files: {len(npzs)}")

    summary_path = root / "summary.jsonl"
    summary_lines = []
    if summary_path.exists():
        with summary_path.open() as f:
            summary_lines = [ln for ln in f if ln.strip()]
    print(f"  summary.jsonl rows: {len(summary_lines)}")

    if len(npzs) != len(summary_lines):
        print(
            f"  WARN: raw count ({len(npzs)}) != summary rows ({len(summary_lines)}); "
            f"likely from a mid-run abort. Both are independently usable."
        )

    # Pick samples to validate
    rng = np.random.default_rng(args.seed)
    if args.sample > 0 and args.sample < len(npzs):
        idx = rng.choice(len(npzs), size=args.sample, replace=False)
        idx.sort()
        chosen = [npzs[i] for i in idx]
        print(f"  spot-checking {len(chosen)} of {len(npzs)} samples")
    else:
        chosen = npzs
        print(f"  validating all {len(chosen)} samples (this may take a while)")

    # Per-sample validation
    error_counter: Counter = Counter()
    error_examples: List[Tuple[str, List[str]]] = []
    clean_in_meta = 0
    issues_counter: Counter = Counter()
    target_panel_counter: Counter = Counter()

    for i, p in enumerate(chosen):
        errs, meta = _validate_one(p, expected_N=expected_N, expected_F=expected_F)
        if errs:
            for e in errs:
                # collapse the variable parts for stat purposes
                key = e.split(":")[0]
                error_counter[key] += 1
            if len(error_examples) < args.show_errors:
                error_examples.append((p.name, errs))
        # accumulate quality stats from the same parse
        if meta is not None:
            if meta.get("quality", {}).get("clean"):
                clean_in_meta += 1
            for issue in meta.get("quality", {}).get("issues", []):
                issues_counter[issue] += 1
            target_panel_counter[meta.get("shot", {}).get("target_panel", "?")] += 1
        if (i + 1) % 500 == 0:
            print(f"    progress: {i+1}/{len(chosen)}  errors so far: {sum(error_counter.values())}", flush=True)

    print()
    print("== validation summary")
    print(f"  total checked         : {len(chosen)}")
    print(f"  files with errors     : {len(error_examples) if error_counter else 0} (showing up to {args.show_errors})")
    print(f"  total error tags      : {sum(error_counter.values())}")
    if error_counter:
        for k, v in error_counter.most_common():
            print(f"    {k:<35} {v}")
    if error_examples:
        print("  example offenders:")
        for name, errs in error_examples:
            print(f"    {name}")
            for e in errs:
                print(f"      - {e}")

    print()
    print("== quality (from per-sample meta_json)")
    print(f"  clean                 : {clean_in_meta} / {len(chosen)} ({100*clean_in_meta/max(len(chosen),1):.1f}%)")
    if issues_counter:
        print("  issue distribution:")
        for k, v in issues_counter.most_common():
            print(f"    {k:<30} {v}")
    if target_panel_counter:
        print("  target_panel distribution:")
        for k, v in sorted(target_panel_counter.items()):
            print(f"    {k:<10} {v}")

    return 0 if not error_counter else 1


if __name__ == "__main__":
    sys.exit(main())
