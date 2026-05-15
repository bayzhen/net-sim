"""End-to-end smoke test (§11.1 last row).

Runs 1 shot on Warp CPU device and verifies physics sanity.
"""
from __future__ import annotations

import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from params import GoalNetParams, BallState  # noqa: E402
from topology import generate_topology  # noqa: E402
from solver_warp import XpbdWarpSolver  # noqa: E402


def test_smoke_single_sample_cpu() -> None:
    p = GoalNetParams()
    t = generate_topology(p)
    ball = BallState(
        position=(0.0, 1.2, 8.0),
        velocity=(0.0, 4.0, -24.0),
        angular_velocity=(0.0, 0.0, 0.0),
    )
    solver = XpbdWarpSolver(p, t, batch_size=1, device="cpu", record_particles=False)
    results = solver.simulate([ball], ["sample_00000"])
    r = results[0]
    assert len(r.frames) == int(round(p.solver.duration / p.solver.frame_dt)) + 1
    assert r.stats.contact_count > 0, "expected at least one contact"
    # First contact should be net-related (not ground)
    first = r.contacts[0]
    assert first.object_type in {
        "segment_swept",
        "segment",
        "particle",
        "goalpost",
        "crossbar",
    }, f"first contact type: {first.object_type}"
    # Ball should not escape past safety_back_z by more than the threshold
    assert r.quality.max_penetration_depth <= p.collision.severe_penetration_threshold
    # All recorded ball positions should be finite numbers
    for f in r.frames:
        for v in list(f.ball_position) + list(f.ball_velocity):
            assert math.isfinite(v), f"non-finite at t={f.time}: {v}"
    # Energy bounded above by initial KE + plausible PE injection.
    # initial KE ≈ 0.5*1*24^2 = 288 J; speed ceiling 60 m/s gives KE 1800 J;
    # we just assert clean=True for default params.
    assert r.quality.clean, f"quality not clean: {r.quality.issues}"


if __name__ == "__main__":
    test_smoke_single_sample_cpu()
    print("smoke test OK")
