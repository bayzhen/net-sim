"""PyTorch ``Dataset`` for the HDF5-format goal-net-xpbd dataset.

Use this as a starting point — adjust ``__getitem__`` to return only the
fields your model actually needs (loading the full ``particle_position``
slab per item is the expensive part).

Quick start::

    from torch.utils.data import DataLoader
    from dataset_loader import GoalNetH5Dataset

    ds = GoalNetH5Dataset("E:/dataset_v2/dataset.h5", clean_only=True)
    print(len(ds), ds[0]["particle_position"].shape)

    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)
    for batch in loader:
        ball = batch["ball_position"]            # (B, F, 3)
        net  = batch["particle_position"]        # (B, F, N, 3)
        cond = batch["input_state"]              # (B, 9)  pos+vel+ang
        ...

Notes
-----
* ``num_workers > 0`` requires ``persistent_workers=True`` if you want to
  keep h5py file handles open across epochs (otherwise each batch reopens
  the file). On Windows the file is reopened lazily inside each worker.
* HDF5 reads are ~free for ``ball_*`` and ``input_*`` (small), but
  ``particle_position[i]`` reads ~3.5 MB; if you batch 32 items this is
  ~110 MB/iter. Make sure your training loop is on a fast disk (SSD).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


class GoalNetH5Dataset:
    """A minimal ``torch.utils.data.Dataset``-compatible reader for
    ``dataset.h5`` produced by ``cli.py generate --raw-format h5``.

    Doesn't import torch at module level so the file is usable even from
    plain numpy scripts.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        clean_only: bool = False,
        sample_indices: Optional[Sequence[int]] = None,
        load_particles: bool = True,
    ) -> None:
        import h5py  # imported lazily so the module imports without h5py

        self.path = Path(path)
        self._h5py = h5py
        self._file: Optional["h5py.File"] = None  # opened lazily per worker
        self.load_particles = load_particles

        # Read static metadata + the indices we need with a short-lived handle
        with h5py.File(self.path, "r") as f:
            self.frame_count = int(f.attrs["frame_count"])
            self.particle_count = int(f.attrs["particle_count"])
            self.frame_dt = float(f.attrs["frame_dt"])
            self.schema_version = str(f.attrs["schema_version"])
            self.issue_names: List[str] = list(f.attrs["issue_names"])
            self.topology = json.loads(str(f.attrs["topology_json"]))
            self.metadata = json.loads(str(f.attrs["metadata_json"]))
            n_total = int(f["sample_id"].shape[0])
            clean = f["quality_clean"][:]
        self.n_total = n_total

        idx = np.arange(n_total, dtype=np.int64)
        if clean_only:
            idx = idx[clean]
        if sample_indices is not None:
            sel = np.asarray(sample_indices, dtype=np.int64)
            idx = idx[np.isin(idx, sel)]
        self.indices = idx

    # ------------------------------------------------------------------
    # File handle management
    # ------------------------------------------------------------------
    def _open(self) -> "h5py.File":
        if self._file is None:
            # SWMR-read: many workers can map the file simultaneously
            self._file = self._h5py.File(self.path, "r", swmr=True)
        return self._file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:  # best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # PyTorch Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        f = self._open()
        si = int(self.indices[i])
        sample: Dict[str, np.ndarray] = {
            "sample_id": str(f["sample_id"][si]),
            "input_position": f["input_position"][si].astype(np.float32),
            "input_velocity": f["input_velocity"][si].astype(np.float32),
            "input_angular":  f["input_angular"][si].astype(np.float32),
            "input_radius":   np.float32(f["input_radius"][si]),
            "input_mass":     np.float32(f["input_mass"][si]),
            "ball_position":  f["ball_position"][si].astype(np.float32),
            "ball_velocity":  f["ball_velocity"][si].astype(np.float32),
            "quality_clean":  bool(f["quality_clean"][si]),
            "quality_target_hit": bool(f["quality_target_hit"][si]),
        }
        # convenience: 9-D input state vector
        sample["input_state"] = np.concatenate([
            sample["input_position"],
            sample["input_velocity"],
            sample["input_angular"],
        ]).astype(np.float32)

        if self.load_particles:
            # (F, N, 3)
            sample["particle_position"] = f["particle_position"][si].astype(np.float32)

        # contacts via CSR offsets
        c0 = int(f["contact_offset"][si])
        c1 = int(f["contact_offset"][si + 1])
        if c1 > c0:
            sample["contact_time"]   = f["contact_time"][c0:c1].astype(np.float32)
            sample["contact_position"] = f["contact_position"][c0:c1].astype(np.float32)
            sample["contact_normal"]   = f["contact_normal"][c0:c1].astype(np.float32)
            sample["contact_object_type"] = f["contact_object_type"][c0:c1].astype(np.int32)
        else:
            sample["contact_time"]   = np.empty(0, dtype=np.float32)
            sample["contact_position"] = np.empty((0, 3), dtype=np.float32)
            sample["contact_normal"]   = np.empty((0, 3), dtype=np.float32)
            sample["contact_object_type"] = np.empty(0, dtype=np.int32)

        return sample


# ---------------------------------------------------------------------------
# Convenience: random offset-frame sampling for the supervised training task
# described in the project README:
#   input  = (ball state at t=0,  offset_frame f)
#   target = (ball state at f,    net particle positions at f)
# ---------------------------------------------------------------------------


class GoalNetOffsetSampler:
    """Wraps a :class:`GoalNetH5Dataset` and on each call returns a single
    ``(input, target)`` training example with a random offset frame.

    Returned dict::

        {
          "input_state":   (12,)        ball_pos0,vel0,ang0, offset_norm  *
          "target_ball":   (3,)         ball position at frame f
          "target_ball_v": (3,)         ball velocity at frame f
          "target_net":    (N, 3)       particle positions at frame f
        }

    * offset_norm = f / (F-1), i.e. normalized to [0, 1].
    """

    def __init__(self, ds: GoalNetH5Dataset, *, rng_seed: int = 0) -> None:
        self.ds = ds
        self.rng = np.random.default_rng(rng_seed)
        self.F = ds.frame_count

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        sample = self.ds[i]
        f = int(self.rng.integers(0, self.F))  # 0..F-1 inclusive
        offset_norm = np.float32(f / max(self.F - 1, 1))
        return {
            "input_state": np.concatenate([
                sample["input_position"], sample["input_velocity"],
                sample["input_angular"], np.array([offset_norm], dtype=np.float32),
            ]).astype(np.float32),
            "target_ball":   sample["ball_position"][f],
            "target_ball_v": sample["ball_velocity"][f],
            "target_net":    sample["particle_position"][f] if "particle_position" in sample else np.empty((0, 3), dtype=np.float32),
            "offset_frame":  np.int64(f),
        }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Dump a few samples from a goal-net HDF5 dataset.")
    ap.add_argument("path", help="path to dataset.h5")
    ap.add_argument("-n", type=int, default=3, help="how many samples to show")
    ap.add_argument("--clean-only", action="store_true")
    args = ap.parse_args()

    ds = GoalNetH5Dataset(args.path, clean_only=args.clean_only)
    print(f"dataset {args.path}")
    print(f"  schema {ds.schema_version}  frame_dt={ds.frame_dt}  F={ds.frame_count}  N={ds.particle_count}")
    print(f"  total {ds.n_total}, selected {len(ds)} (clean_only={args.clean_only})")
    for i in range(min(args.n, len(ds))):
        s = ds[i]
        print(f"  [{i}] id={s['sample_id']}  clean={s['quality_clean']}  "
              f"ball@0={s['ball_position'][0]}  ball@end={s['ball_position'][-1]}  "
              f"contacts={s['contact_time'].shape[0]}")
