"""Tests for contact-implicit inverse dynamics — the THEORY.md s.8 "north star".

This suite stresses :mod:`contact.dynamics_id`, the final rung beyond s.10's rung 5:
the dual, dynamics-first question of s.8 — *given a rigid body's observed motion and
its mass/inertia, what physically-valid contact forces (Signorini normal >= 0 + Coulomb
friction cone ||f_t|| <= mu*f_n) supply the Newton-Euler net wrench?* The contacts that
come out carrying load ARE the active set.

Everything here is SYNTHETIC: poses are built in closed form so the right answer is
known exactly, and the tests check the three pillars of the module against it:

* **Newton-Euler (s.8).** :func:`required_wrench` on a body at rest must demand a wrench
  that exactly balances gravity (``F = [0,0,m*g]``, ``tau ~ 0``), and on a body in
  free-fall (``a_com = g``) must demand ~zero contact wrench.

* **Kinematics from pose (s.4/s.8).** :func:`body_accelerations` must recover a known
  constant linear acceleration and a known constant angular velocity from a synthetic
  pose trajectory, through the noisy double-differentiation path.

* **The contact-implicit solve (s.8) + the s.7 indeterminacy.**
  :func:`solve_contact_implicit` on a stationary 4-corner box must recover non-negative
  per-corner normal forces summing to ``m*g`` with ~zero residual and all four corners
  active; in flight (large gaps) it must recover ~zero force and an empty active set;
  it must respect the friction cone; and on the symmetric indeterminate load it must
  return the minimum-norm (near-equal, finite) split that the Tikhonov regularizer
  selects — the honest resolution of the s.7 load-split indeterminacy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the repo root importable regardless of how pytest is invoked.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from contact.config import InverseDynamicsParams
from contact.dynamics_id import (
    InertialParams,
    body_accelerations,
    contact_wrench_map,
    required_wrench,
    solve_contact_implicit,
)
from contact.types import PoseTrajectory

GRAVITY = 9.81
_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


# --------------------------------------------------------------------------------------
# Helpers: synthetic poses and a synthetic 4-corner box rig.
# --------------------------------------------------------------------------------------


def _static_pose(T: int = 50, dt: float = 0.01, position=(0.0, 0.0, 0.0)) -> PoseTrajectory:
    """A perfectly stationary pose: constant position, identity orientation."""
    t = np.arange(T) * dt
    pos = np.tile(np.asarray(position, dtype=float), (T, 1))
    quat = np.tile(_IDENTITY_QUAT, (T, 1))
    return PoseTrajectory(t=t, position=pos, quat=quat)


def _accel_pose(a: np.ndarray, T: int = 60, dt: float = 0.01) -> PoseTrajectory:
    """A pose undergoing constant linear acceleration ``a`` from rest (identity orient.)."""
    t = np.arange(T) * dt
    a = np.asarray(a, dtype=float)
    pos = 0.5 * a[None, :] * (t[:, None] ** 2)  # p(t) = 1/2 a t^2
    quat = np.tile(_IDENTITY_QUAT, (T, 1))
    return PoseTrajectory(t=t, position=pos, quat=quat)


def _spin_pose(omega_z: float, T: int = 80, dt: float = 0.01) -> PoseTrajectory:
    """A pose spinning at constant angular velocity ``omega_z`` about the world z-axis."""
    t = np.arange(T) * dt
    half = 0.5 * omega_z * t
    quat = np.stack([np.cos(half), np.zeros(T), np.zeros(T), np.sin(half)], axis=1)
    pos = np.zeros((T, 3))
    return PoseTrajectory(t=t, position=pos, quat=quat)


# A unit-ish box: half-extents in x,y, the contact face at body-local -z half-height.
_HALF = 0.1
_BOX_CORNERS = np.array(
    [
        [+_HALF, +_HALF, -_HALF],
        [+_HALF, -_HALF, -_HALF],
        [-_HALF, +_HALF, -_HALF],
        [-_HALF, -_HALF, -_HALF],
    ]
)
# The floor pushes UP on the box, so the candidate normal (direction the contact can
# push the body) is body-local +z for a bottom-face support.
_BOX_NORMALS = np.tile(np.array([0.0, 0.0, 1.0]), (4, 1))


def _box_wrench_map(pose: PoseTrajectory) -> np.ndarray:
    return contact_wrench_map(pose, _BOX_CORNERS, _BOX_NORMALS)


# --------------------------------------------------------------------------------------
# 1. required_wrench: Newton-Euler balance (s.8).
# --------------------------------------------------------------------------------------


def test_required_wrench_stationary_balances_gravity():
    """A body at rest needs a contact wrench of exactly [0,0,m*g; 0,0,0]."""
    mass = 2.0
    pose = _static_pose()
    inert = InertialParams(mass=mass, inertia=np.eye(3) * 0.01)
    w = required_wrench(pose, inert, gravity=GRAVITY)

    assert w.shape == (pose.t.shape[0], 6)
    # Use interior frames (the local-poly derivative is exact on a constant signal even
    # at the ends, but we keep a margin to be robust to any boundary handling).
    interior = w[3:-3]
    expected_F = np.array([0.0, 0.0, mass * GRAVITY])
    assert np.allclose(interior[:, 0:3], expected_F[None, :], atol=1e-6)
    assert np.allclose(interior[:, 3:6], 0.0, atol=1e-6)


def test_required_wrench_freefall_is_zero():
    """A body in free-fall (a_com = g) needs ~zero contact wrench: gravity does it all."""
    mass = 1.7
    g_vec = np.array([0.0, 0.0, -GRAVITY])
    pose = _accel_pose(g_vec)
    inert = InertialParams(mass=mass, inertia=np.eye(3) * 0.02)
    w = required_wrench(pose, inert, gravity=GRAVITY)

    interior = w[3:-3]
    assert np.allclose(interior, 0.0, atol=1e-5)


# --------------------------------------------------------------------------------------
# 2. body_accelerations: recover known accel + angular velocity from a pose (s.4/s.8).
# --------------------------------------------------------------------------------------


def test_body_accelerations_recovers_constant_acceleration():
    """Double-differentiating p(t)=1/2 a t^2 recovers the constant acceleration a."""
    a_true = np.array([0.3, -0.7, 1.1])
    pose = _accel_pose(a_true)
    a_com, alpha, omega = body_accelerations(pose, gravity=GRAVITY)

    interior = slice(5, -5)
    assert np.allclose(a_com[interior], a_true[None, :], atol=1e-4)
    # No rotation: angular velocity and acceleration are ~0.
    assert np.allclose(omega[interior], 0.0, atol=1e-5)
    assert np.allclose(alpha[interior], 0.0, atol=1e-5)


def test_body_accelerations_recovers_angular_velocity():
    """A constant spin about world-z is recovered as a constant omega_z."""
    omega_z = 1.25
    # A finer clock + small smoothing keeps the (Gaussian-prefiltered) quaternion
    # derivative from biasing the recovered rate; we still differentiate through the
    # real, noisy s.4 path, just with the smoothing kept local (s.6).
    pose = _spin_pose(omega_z, T=120, dt=0.005)
    a_com, alpha, omega = body_accelerations(pose, gravity=GRAVITY, accel_smooth_time=0.01)

    # Well inside the record, away from the smoother's one-sided boundary region.
    interior = slice(20, -20)
    assert np.allclose(omega[interior, 2], omega_z, atol=2e-3)
    assert np.allclose(omega[interior, 0:2], 0.0, atol=2e-3)
    # Constant spin => zero angular acceleration; zero translation => zero linear accel.
    assert np.allclose(alpha[interior], 0.0, atol=1e-2)
    assert np.allclose(a_com[interior], 0.0, atol=1e-4)


# --------------------------------------------------------------------------------------
# 3. solve_contact_implicit on a synthetic stationary box (s.8).
# --------------------------------------------------------------------------------------


def test_solve_contact_implicit_stationary_box():
    """4 corners at gap~0 carrying the weight: forces >= 0, sum ~m*g, residual ~0, all active."""
    mass = 3.0
    T = 30
    pose = _static_pose(T=T)
    inert = InertialParams(mass=mass, inertia=np.eye(3) * 0.01)
    G = _box_wrench_map(pose)
    w = required_wrench(pose, inert, gravity=GRAVITY)

    # All four corners closed (gap ~ 0, well inside the complementarity band).
    gaps = np.zeros((T, 4))
    params = InverseDynamicsParams()
    res = solve_contact_implicit(
        w, G, gaps, mu=0.6, params=params, t=pose.t, candidate_points=_BOX_CORNERS
    )

    interior = slice(3, -3)
    # Signorini: every recovered normal force is non-negative.
    assert np.all(res.contact_normal_force >= -1e-8)
    # The total normal force balances the weight m*g.
    assert np.allclose(res.total_normal_force[interior], mass * GRAVITY, atol=1e-3)
    # The forces explain the wrench (residual ~ 0).
    assert np.all(res.wrench_residual[interior] < 1e-3)
    # All four corners are active (each carries m*g/4 >> threshold).
    for ti in range(*interior.indices(T)):
        assert sorted(res.active_set[ti]) == [0, 1, 2, 3]


def test_solve_contact_implicit_flight_is_empty():
    """With all gaps large (flight), the complementarity mask zeros all forces."""
    mass = 1.0
    T = 25
    inert = InertialParams(mass=mass, inertia=np.eye(3) * 0.01)
    # Free-fall motion so the *required* wrench is ~0 too (belt and suspenders).
    pose_fall = _accel_pose(np.array([0.0, 0.0, -GRAVITY]), T=T)
    G = _box_wrench_map(pose_fall)
    w = required_wrench(pose_fall, inert, gravity=GRAVITY)

    params = InverseDynamicsParams()
    gaps = np.full((T, 4), 0.5)  # 0.5 m >> complementarity_gap (5 mm)
    res = solve_contact_implicit(
        w, G, gaps, mu=0.6, params=params, t=pose_fall.t, candidate_points=_BOX_CORNERS
    )

    assert np.allclose(res.contact_normal_force, 0.0, atol=1e-12)
    assert np.allclose(res.contact_tangent_force, 0.0, atol=1e-12)
    assert np.allclose(res.total_normal_force, 0.0, atol=1e-12)
    for ti in range(T):
        assert res.active_set[ti] == []


def test_solve_contact_implicit_respects_friction_cone():
    """A horizontal demand must be supplied by friction obeying ||f_t|| <= mu*f_n."""
    mass = 2.0
    T = 20
    mu = 0.4
    # A body accelerating horizontally while supported: required wrench has a lateral
    # force component that the contacts must supply via friction.
    a_lateral = np.array([1.5, 0.0, 0.0])
    pose = _accel_pose(a_lateral, T=T)
    inert = InertialParams(mass=mass, inertia=np.eye(3) * 0.01)
    G = _box_wrench_map(pose)
    w = required_wrench(pose, inert, gravity=GRAVITY)

    gaps = np.zeros((T, 4))
    params = InverseDynamicsParams()
    res = solve_contact_implicit(
        w, G, gaps, mu=mu, params=params, t=pose.t, candidate_points=_BOX_CORNERS
    )

    interior = slice(3, -3)
    fn = res.contact_normal_force
    ft = np.linalg.norm(res.contact_tangent_force, axis=-1)  # (T,K)
    # Coulomb cone, per candidate per frame (small slack for the SLSQP/projection tol).
    assert np.all(ft[interior] <= mu * fn[interior] + 1e-6)


def test_solve_contact_implicit_indeterminate_minimum_norm():
    """Symmetric 4-corner load is indeterminate (s.7); the regularizer picks min-norm.

    With four corners balancing a single 6-wrench the per-corner split has a null space
    (s.7). The Tikhonov term ``force_regularization*||f||^2`` selects the unique
    minimum-norm member, which for a symmetric load is the near-equal split. We check the
    split is finite, near-equal, and (because min-norm spreads load) lower-norm than an
    asymmetric split that produces the identical net wrench.
    """
    mass = 4.0
    T = 12
    pose = _static_pose(T=T)
    inert = InertialParams(mass=mass, inertia=np.eye(3) * 0.01)
    G = _box_wrench_map(pose)
    w = required_wrench(pose, inert, gravity=GRAVITY)

    gaps = np.zeros((T, 4))
    params = InverseDynamicsParams()
    res = solve_contact_implicit(
        w, G, gaps, mu=0.6, params=params, t=pose.t, candidate_points=_BOX_CORNERS
    )

    interior = slice(3, -3)
    fn = res.contact_normal_force[interior]  # (Ti, 4)
    expected_each = mass * GRAVITY / 4.0
    # Finite and near-equal across the four corners.
    assert np.all(np.isfinite(fn))
    assert np.allclose(fn, expected_each, atol=1e-2)
    # Min-norm property: the equal split has strictly smaller sum-of-squares than any
    # feasible asymmetric split that yields the same total m*g (e.g. all load on 2 corners).
    equal_sq = np.sum((np.full(4, expected_each)) ** 2)
    lopsided_sq = np.sum((np.array([0.5, 0.5, 0.0, 0.0]) * (mass * GRAVITY)) ** 2)
    assert equal_sq < lopsided_sq
    # The solver's recovered split realizes (essentially) that minimum.
    realized_sq = np.sum(fn[0] ** 2)
    assert realized_sq <= equal_sq + 1e-2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
