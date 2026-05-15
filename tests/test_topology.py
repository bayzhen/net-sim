"""Topology stability tests (§11.1).

Run with:  PYTHONPATH=. python tests/test_topology.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from params import GoalNetParams  # noqa: E402
from topology import generate_topology, stable_signature, summary  # noqa: E402


def test_topology_is_stable() -> None:
    p = GoalNetParams()
    a = generate_topology(p)
    b = generate_topology(p)
    sig_a = stable_signature(a)
    sig_b = stable_signature(b)
    assert sig_a == sig_b, f"topology signatures differ: {sig_a} vs {sig_b}"
    sum_a = summary(a)
    sum_b = summary(b)
    assert sum_a == sum_b, f"summaries differ:\n  {sum_a}\n  {sum_b}"


def test_topology_has_expected_counts() -> None:
    p = GoalNetParams()
    t = generate_topology(p)
    s = summary(t)
    # Defensive numbers — these may shift if cell_size or dedup logic changes,
    # but they should remain in the ballpark.
    assert 400 <= s["particles"] <= 700, s
    assert 0 < s["anchored_particles"] < s["particles"], s
    assert s["stretch_constraints"] > s["bend_constraints"] // 2, s
    assert s["goalpost_segments"] == 3, s


def test_dedup_merges_corners() -> None:
    """Top corner particles where back/top/side meet must be welded (bug B)."""
    p = GoalNetParams()
    t = generate_topology(p)
    # Find any particle belonging to back panel at y == height and z == -depth
    height = p.goal.height
    depth = p.goal.depth
    width = p.goal.width
    top_corners = [
        pp
        for pp in t.particles
        if abs(pp.position[1] - height) < 1e-3
        and abs(pp.position[2] + depth) < 1e-3
        and abs(abs(pp.position[0]) - width / 2) < 1e-3
    ]
    # Without dedup we'd see at least 3 panels meeting at each top corner (=6
    # particles for the 2 corners); with dedup the count should match exactly 2
    # (one per ±x corner).
    assert len(top_corners) <= 2, f"top corners not deduplicated: {top_corners}"


if __name__ == "__main__":
    test_topology_is_stable()
    test_topology_has_expected_counts()
    test_dedup_merges_corners()
    print("topology tests OK")
