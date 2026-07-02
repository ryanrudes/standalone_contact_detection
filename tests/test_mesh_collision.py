"""Permanent validation of the convex-mesh collision kernel (DESIGN.md PART II.D / Phase 3).

These promote the session's GJK/EPA + mesh-resolver checks into self-contained tests. The
collision primitives (:func:`contact.mesh_collision.gjk_distance`) are exercised on *hand-built*
vertex clouds against the analytic answer, and the two mesh resolvers
(:class:`~contact.geometry_resolvers.MeshPlane` / :class:`~contact.geometry_resolvers.MeshConvex`)
are pinned to the closed-form primitives they generalize:

* GJK on two unit cubes 3 m apart recovers the separation 2.0 exactly, with the normal pointing
  ``support -> moving``; the diagonal-offset case recovers the corner-to-corner distance.
* EPA on two overlapping icosphere clouds recovers a NEGATIVE penetration depth ~ the analytic
  -0.02 m.
* ``MeshPlane`` fed a box's 8 corners reproduces ``BoxPlane`` bit-for-bit (gap AND normal), and
  ``MeshConvex`` fed two icosphere clouds matches ``SphereSphere`` on separated frames (within the
  tessellation error of the sampled cloud).
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("mujoco")  # the mesh-vs-primitive tests resolve over MuJoCo trajectories

import oracle  # noqa: E402
from contact.geometry_resolvers import (  # noqa: E402
    BoxPlane,
    MeshConvex,
    MeshPlane,
    SphereSphere,
)
from contact.mesh_collision import gjk_distance  # noqa: E402

# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _unit_cube(offset) -> np.ndarray:
    """The 8 corners of the unit cube ``[0,1]^3`` translated by ``offset`` (a convex cloud)."""
    corners = np.array(
        [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)], dtype=float
    )
    return corners + np.asarray(offset, dtype=float)


def _fib_sphere(n: int, r: float, center=(0.0, 0.0, 0.0)) -> np.ndarray:
    """A Fibonacci-lattice icosphere: ``n`` ~evenly-spread surface points of radius ``r``."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    th = np.pi * (1.0 + 5.0 ** 0.5) * i
    pts = r * np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]
    return pts + np.asarray(center, dtype=float)


def _min_gap_rep(frames):
    """The min-gap (gap, normal) representative per frame of a resolver's ContactFrame list."""
    gap = np.array([min(p.gap for p in fr) for fr in frames])
    normal = np.array([min(fr, key=lambda p: p.gap).normal for fr in frames])
    return gap, normal


# ======================================================================================
# gjk_distance / EPA -- the separation & penetration query (DESIGN.md PART II.D Mesh rung)
# ======================================================================================


class TestGjkDistance:
    def test_separated_axis_aligned_cubes(self) -> None:
        """Two unit cubes 3 m apart on x: GJK distance == 2.0, normal points support->moving (-x)."""
        moving = _unit_cube((0.0, 0.0, 0.0))     # hull A (moving): x in [0, 1]
        support = _unit_cube((3.0, 0.0, 0.0))    # hull B (support): x in [3, 4]
        gap, _point, normal = gjk_distance(moving, support)
        assert np.isclose(gap, 2.0, atol=1e-6), (
            f"the separation between the x=1 and x=3 faces must be 2.0; got {gap}"
        )
        # The normal is support -> moving: from B (high x) toward A (low x) => -x.
        assert np.allclose(normal, np.array([-1.0, 0.0, 0.0]), atol=1e-6), (
            f"the GJK normal must point support->moving (-x here); got {normal}"
        )

    def test_separated_diagonal_offset(self) -> None:
        """Diagonal offset (3,3,3): closest corners (1,1,1)-(3,3,3) => distance 2*sqrt(3)."""
        moving = _unit_cube((0.0, 0.0, 0.0))
        support = _unit_cube((3.0, 3.0, 3.0))
        gap, _point, normal = gjk_distance(moving, support)
        assert np.isclose(gap, 2.0 * np.sqrt(3.0), atol=1e-6), (
            f"the corner-to-corner diagonal separation must be 2*sqrt(3); got {gap}"
        )
        assert np.allclose(normal, -np.ones(3) / np.sqrt(3.0), atol=1e-6), (
            "the diagonal normal must point support->moving along -(1,1,1)/sqrt(3)"
        )

    def test_penetration_recovers_negative_gap_via_epa(self) -> None:
        """Two icosphere clouds (r=0.05) with centres 0.08 m apart penetrate by ~0.02 m (EPA).

        Sum of radii 0.10 m, centre distance 0.08 m => analytic penetration -0.02 m; EPA must
        return a NEGATIVE gap within ~2 mm of that."""
        moving = _fib_sphere(200, 0.05, center=(0.0, 0.0, 0.0))
        support = _fib_sphere(200, 0.05, center=(0.08, 0.0, 0.0))
        gap, _point, _normal = gjk_distance(moving, support)
        assert gap < 0.0, f"overlapping hulls must report a NEGATIVE (penetration) gap; got {gap}"
        assert abs(gap - (-0.02)) < 0.002, (
            f"the penetration depth must be ~-0.02 m (within ~2 mm); got {gap:.5f}"
        )

    def test_coplanar_box_penetration_is_exact(self) -> None:
        """The watertight close-out (DESIGN.md III.5): coplanar box-vs-box penetration is EXACT.

        Two unit cubes overlapping FACE-ON-FACE -- the degenerate case where a GJK simplex
        stalls on a lower-dimensional feature and EPA's forced tetra mis-converges. The
        Separating-Axis close-out recovers the exact overlap depth, the penetration normal is
        the contacting face normal (support->moving), and a merely-FLUSH contact reads gap ~0
        (not the spurious deep penetration the naive EPA produced)."""
        moving = _unit_cube((0.0, 0.0, 0.0))          # hull A: x,y,z in [0, 1]
        support = _unit_cube((0.7, 0.0, 0.0))         # hull B: x in [0.7, 1.7]; y,z full overlap
        gap, _point, normal = gjk_distance(moving, support)
        assert gap < 0.0, f"overlapping boxes must report a NEGATIVE gap; got {gap}"
        assert abs(gap - (-0.3)) < 1e-6, (
            f"the face-on-face penetration depth must be EXACTLY the 0.3 m overlap; got {gap}"
        )
        # Least-overlap (penetration) axis is x; the normal points support->moving (-x).
        assert np.allclose(normal, np.array([-1.0, 0.0, 0.0]), atol=1e-6), (
            f"the penetration normal must be the x face normal (support->moving = -x); got {normal}"
        )
        # Merely flush (faces meet at x=1, zero overlap) must read gap ~0, NOT deep penetration.
        flush = _unit_cube((1.0, 0.0, 0.0))
        gt, _p, _n = gjk_distance(moving, flush)
        assert abs(gt) < 1e-6, f"flush-touching boxes must read gap ~0; got {gt}"


# ======================================================================================
# Mesh resolvers vs the closed-form primitives they generalize (DESIGN.md III.5 Phase 3)
# ======================================================================================


class TestMeshResolvers:
    def test_meshplane_equals_boxplane_on_box_corners(self) -> None:
        """MeshPlane(box 8 corners) reproduces BoxPlane(half_extents): identical gap AND normal (<1e-9).

        ``MeshPlane`` is exactly ``BoxPlane``'s per-corner signed-distance arithmetic lifted to an
        arbitrary cloud, so a box's own corners must reproduce the primitive bit-for-bit over the
        whole tumbling trajectory."""
        raw = oracle.generate("tumbling_box")
        he = np.array([0.1, 0.1, 0.1])
        corners = np.array([np.array(s) * he for s in itertools.product((-1.0, 1.0), repeat=3)])
        g_mesh, n_mesh = _min_gap_rep(MeshPlane(corners, raw.surface).resolve(raw.moving, raw.support))
        g_box, n_box = _min_gap_rep(BoxPlane(he, raw.surface).resolve(raw.moving, raw.support))
        assert float(np.max(np.abs(g_mesh - g_box))) < 1e-9, (
            "a box mesh must reproduce BoxPlane's per-frame gap bit-for-bit"
        )
        assert float(np.max(np.abs(n_mesh - n_box))) < 1e-9, (
            "a box mesh must reproduce BoxPlane's per-frame normal bit-for-bit"
        )

    def test_meshconvex_matches_spheresphere_on_separated_frames(self) -> None:
        """MeshConvex(two icospheres) ~ SphereSphere gap within 5 mm on separated ball-ball frames.

        ``MeshConvex`` is the GJK/EPA generalization of ``SphereSphere``; on the separated frames
        of the ball-ball trajectory the two must agree to within the icosphere tessellation error
        (~few mm at this sampling)."""
        scene = oracle.generate_scene("two_balls_collide")
        ball_a = scene.bodies["ballA"]
        ball_b = scene.bodies["ballB"]
        mesh_a = _fib_sphere(200, 0.05)
        mesh_b = _fib_sphere(200, 0.05)
        g_mesh, _ = _min_gap_rep(MeshConvex(mesh_a, mesh_b).resolve(ball_a, ball_b))
        g_sphere, _ = _min_gap_rep(SphereSphere(0.05, 0.05).resolve(ball_a, ball_b))
        separated = g_sphere > 0.003  # the GJK distance regime (away from contact/penetration)
        assert separated.any(), "the ball-ball trajectory must contain clearly-separated frames"
        err = float(np.max(np.abs(g_mesh[separated] - g_sphere[separated])))
        assert err < 0.005, (
            f"icosphere MeshConvex must match SphereSphere within 5 mm on separated frames; "
            f"got {err * 1000:.2f} mm"
        )
