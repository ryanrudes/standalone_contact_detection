"""Tests for the impact (§6) and dynamics/material (§7) rungs of the ladder.

This suite exercises the two final rungs of THEORY.md §10 against both *synthetic*
signals (where the right answer is known in closed form) and the MuJoCo truth oracle
of THEORY.md §9 (where the simulator withholds the truth from the detector but not
from us). Two distinct claims are stressed:

* **§6 -- impacts are atoms in the force measure.** ``impacts.detect_impacts`` must
  localize the velocity-step arrest (``v+ = -e v-``) and read back its closing speed,
  restitution, and impulse, while *not* hallucinating an impact on a smooth signal.

* **§7 -- compliance restores observability.** ``dynamics.normal_force_from_penetration``
  turns penetration into a calibrated force gauge ``lambda = k*delta`` (and reports the
  force as unobservable -- NaN -- when stiffness is unknown); ``dynamics.friction_stick_slip``
  reads the stick->slip guard off the kinematics; and ``dynamics.observability_demo``
  exhibits the §7 theorem itself: the rigid-statics load split is unobservable (the
  equilibrium map has a nontrivial null space) yet compliance recovers each per-corner
  force from its own penetration.

MuJoCo is required for the scenario-backed tests; if it is absent those are skipped
(the synthetic tests of the same functions still run).
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

from contact import dynamics, impacts
from contact.config import ImpactParams, MaterialParams
from contact.types import ContactObservations

# A single fixed seed so every scenario-backed test is reproducible (the seed only
# drives the additive mocap noise in oracle.generate; the physics is deterministic).
SEED = 12345
HZ = 200.0


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

def _flat_obs(t: np.ndarray, v_normal: np.ndarray) -> ContactObservations:
    """A minimal ContactObservations carrying only ``t`` and ``v_normal``.

    ``detect_impacts`` consumes only those two channels (§6: the normal channel
    carries the arrest), so the tangential/angular channels can be zeros here.
    """
    T = t.shape[0]
    return ContactObservations(
        t=t,
        gap=np.zeros(T),
        v_normal=v_normal,
        v_tangent=np.zeros((T, 2)),
        omega_normal=np.zeros(T),
        omega_tangent=np.zeros((T, 2)),
    )


# --------------------------------------------------------------------------------------
# §6 -- impacts: synthetic signals with a known answer
# --------------------------------------------------------------------------------------

def test_detect_impacts_synthetic_bounce_recovers_one_impact():
    """A closing-then-rebound velocity step yields exactly one impact (THEORY.md §6).

    Construct the cleanest possible atom: the body closes at a constant 1.0 m/s
    (``v_normal = -1`` since +ve is separating, §1) and at one frame is reset to a
    rebound ``v+ = e * 1.0`` with ``e = 0.4``. This is the textbook reset map
    ``v+ = -e v-`` (here ``v- = -1`` so ``v+ = +0.4``). The detector must find a single
    arrest and read back the closing speed and the restitution it was built from.
    """
    dt = 1.0 / HZ
    T = 200
    t = np.arange(T) * dt
    e_true = 0.4
    v = np.full(T, -1.0)        # closing at 1 m/s
    v[100:] = e_true * 1.0      # reset to +0.4 (rebound) -- the velocity step / atom
    obs = _flat_obs(t, v)

    mass = 2.0
    found = impacts.detect_impacts(obs, ImpactParams(), mass=mass)

    assert len(found) == 1, "a single velocity step must produce a single atom (§6)"
    imp = found[0]
    # Closing speed is |v_before| = 1.0 m/s.
    assert imp.closing_speed == pytest.approx(1.0, abs=0.05)
    # Restitution e = -v_after/v_before = -(+0.4)/(-1.0) = 0.4.
    assert imp.restitution == pytest.approx(e_true, abs=0.05)
    # Impulse atom = m * (v_after - v_before) = 2.0 * (0.4 - (-1.0)) = 2.8 N*s (§6).
    assert imp.normal_impulse == pytest.approx(mass * (e_true + 1.0), abs=0.1)
    # The arrest is at frame 100; localized to within the template half-width.
    assert abs(imp.index - 100) <= 3


def test_detect_impacts_smooth_signal_returns_none():
    """A smooth, monotone velocity with no arrest yields no impacts (THEORY.md §6).

    An impact is an *arrest of approach* -- a genuine step in ``v_normal``. A smooth
    ramp (no step) and a steady closing velocity (no rise) are both non-events: the
    closing-speed / rise gates of the matched filter must reject them rather than
    inventing an atom from differentiation noise.
    """
    dt = 1.0 / HZ
    T = 200
    t = np.arange(T) * dt

    # (a) a smooth ramp through zero -- no arrest anywhere.
    v_ramp = np.linspace(0.5, -0.5, T)
    assert impacts.detect_impacts(_flat_obs(t, v_ramp), ImpactParams()) == []

    # (b) a steady closing velocity -- approaching but never arrested.
    v_const = np.full(T, -1.0)
    assert impacts.detect_impacts(_flat_obs(t, v_const), ImpactParams()) == []


# --------------------------------------------------------------------------------------
# Scenario-backed tests (MuJoCo required)
# --------------------------------------------------------------------------------------

mujoco = pytest.importorskip("mujoco")

from contact import geometry
import oracle  # noqa: E402  (after the skip guard)


def _observe(scenario_name: str):
    """generate -> observe one scenario (THEORY.md §9 observable channel)."""
    sc = oracle.generate(scenario_name, seed=SEED, hz=HZ)
    obs = geometry.observe(
        sc.moving, sc.support, sc.surface, sc.contact_point_local,
        geometry=getattr(sc, "geometry", None),
    )
    return sc, obs


def test_detect_impacts_bouncing_ball_several_impacts():
    """The bouncing ball produces several impacts with physical restitution (§6).

    A ball dropped onto a bouncy (low-damping) plane strikes it repeatedly, losing
    energy each bounce. ``detect_impacts`` should find several arrests, each with a
    positive closing speed, and -- where the rebound is resolvable -- a restitution in
    the physical open interval (0, 1) (a lossy bounce: leaves slower than it arrived).
    Late, weak bounces may arrest essentially to rest (a measured ``e ~ 0``, plastic),
    so we require ``e in [0, 1)`` for every resolved bounce and ``e in (0, 1)`` for at
    least one -- the signature that the restitution estimator is reading real physics,
    not a fabricated constant.
    """
    _sc, obs = _observe("bouncing_ball")
    found = impacts.detect_impacts(obs, ImpactParams())

    assert len(found) >= 2, "a bouncing ball strikes the plane several times (§6)"

    # Every detected impact is a genuine arrest of approach: positive closing speed.
    for imp in found:
        assert imp.closing_speed > 0.0

    finite_e = [imp.restitution for imp in found if np.isfinite(imp.restitution)]
    assert finite_e, "at least one bounce must yield a measurable restitution"
    # No measured restitution exceeds 1: a passive contact cannot add energy (§6/§8).
    for e in finite_e:
        assert 0.0 <= e < 1.0 + 1e-6
    # At least one bounce is a true (lossy, non-plastic) rebound with e strictly in (0,1).
    assert any(0.0 < e < 1.0 for e in finite_e)


# --------------------------------------------------------------------------------------
# §7 -- normal force from penetration (compliance as a calibrated force gauge)
# --------------------------------------------------------------------------------------

def test_normal_force_from_penetration_spring_law():
    """lambda = k*delta on contact, zero off-contact, NaN when stiffness unknown (§7).

    THEORY.md §7: under known compliance the penetration depth is a calibrated force
    gauge. We check the three regimes of the law directly:

      * a frame with a real *gap* (g > gap_bias) bears no force -- Signorini g*lambda=0;
      * a frame with penetration delta below the resting datum bears k*delta;
      * with no stiffness the magnitude is unobservable from kinematics -> all-NaN.
    """
    gap_bias = 0.0
    k = 1000.0
    # gap > bias (separation), gap < bias (penetration of 0.001 and 0.002 m).
    gap = np.array([0.005, -0.001, -0.002, 0.010])
    in_contact = np.array([False, True, True, False])

    mat = MaterialParams(stiffness=k)
    force = dynamics.normal_force_from_penetration(gap, gap_bias, in_contact, mat)

    # Off-contact frames (indices 0, 3): zero force regardless of gap sign.
    assert force[0] == pytest.approx(0.0)
    assert force[3] == pytest.approx(0.0)
    # On-contact penetration frames: lambda = k * delta = k * (-gap) here.
    assert force[1] == pytest.approx(k * 0.001)
    assert force[2] == pytest.approx(k * 0.002)

    # A contact frame with a genuine positive gap (above the datum) bears no force even
    # if flagged in_contact -- the delta clamp at 0 honours Signorini (§2).
    gap2 = np.array([0.003])
    f2 = dynamics.normal_force_from_penetration(
        gap2, gap_bias, np.array([True]), mat
    )
    assert f2[0] == pytest.approx(0.0)

    # No stiffness => force unobservable from kinematics alone (§7) => all-NaN.
    mat_none = MaterialParams(stiffness=None)
    f_none = dynamics.normal_force_from_penetration(
        gap, gap_bias, in_contact, mat_none
    )
    assert f_none.shape == gap.shape
    assert np.all(np.isnan(f_none))


# --------------------------------------------------------------------------------------
# §7 -- friction stick/slip on push_to_slide
# --------------------------------------------------------------------------------------

def test_friction_stick_slip_push_to_slide():
    """Early resting frames label "stick", later sliding frames label "slip" (§7).

    push_to_slide ramps a horizontal force on a seated box: it is STATIC (no tangential
    motion) until the demand reaches the friction-cone boundary, then SLIDES. The
    stick->slip guard of §5/§7 is the kinematic threshold on tangential speed, so the
    static-truth region must read all "stick" and the sliding-truth region all "slip".
    """
    sc, obs = _observe("push_to_slide")
    gt = sc.truth

    # Run purely kinematically (stiffness unknown): the always-observable channel is the
    # tangential speed, which is exactly what distinguishes static from sliding (§3/§7).
    mat = MaterialParams(stiffness=None)
    nf = dynamics.normal_force_from_penetration(
        obs.gap, 0.0, gt.in_contact, mat
    )
    labels = dynamics.friction_stick_slip(obs, nf, mat)

    assert len(labels) == len(gt.mode)

    # The truth modes give us the resting and sliding windows; the labels must agree.
    static_idx = [i for i, m in enumerate(gt.mode) if m == "static"]
    sliding_idx = [i for i, m in enumerate(gt.mode) if m == "sliding"]
    assert static_idx and sliding_idx, "scenario must contain both static and sliding"

    static_labels = [labels[i] for i in static_idx]
    sliding_labels = [labels[i] for i in sliding_idx]

    # The early static phase is overwhelmingly "stick" (the box does not move).
    n_stick = static_labels.count("stick")
    assert n_stick >= 0.9 * len(static_labels), (
        f"resting frames should be 'stick'; got {n_stick}/{len(static_labels)}"
    )
    # The later sliding phase is overwhelmingly "slip".
    n_slip = sliding_labels.count("slip")
    assert n_slip >= 0.9 * len(sliding_labels), (
        f"sliding frames should be 'slip'; got {n_slip}/{len(sliding_labels)}"
    )


# --------------------------------------------------------------------------------------
# §7 -- the observability theorem, exhibited on the indeterminate rig
# --------------------------------------------------------------------------------------

def test_observability_demo_indeterminate_rig():
    """The THEORY.md §7 observability theorem, demonstrated on a real rig.

    This is the THEORY.md §7 theorem demonstrated directly: a box on K = 4 corner
    contacts is statically indeterminate -- its 4 vertical corner-force unknowns are
    constrained by only 3 rigid-body static balance equations (sum F_z, sum M_x,
    sum M_y), so the rigid-statics equilibrium map ``A`` (3, K) has a nontrivial null
    space and the *load split* between corners is unobservable from kinematics alone.
    Granting each corner its own compliance ``f_i = k*delta_i`` collapses that
    indeterminacy: the per-corner force is pinned to its own measurable penetration and
    recovered to numerical precision.
    """
    sc = oracle.generate("indeterminate_rig", seed=SEED, hz=HZ)
    cp = sc.meta["contact_points"]
    k = sc.meta["stiffness"]

    pen = cp["penetration"]          # (K, T) per-corner penetration depth
    fn = cp["normal_force"]          # (K, T) per-corner measured normal force
    xy = cp["corners_local"][:, :2]  # (K, 2) planar corner positions for the eq map
    K = cp["n_corners"]
    assert K > 3, "the rig must over-determine the corner forces (K > 3) to be indeterminate"
    assert np.isfinite(k) and k > 0.0, "compliance stiffness must be identified"

    # Evaluate force recovery over the QUIET, SETTLED tail (last quarter), exactly the
    # window over which oracle.factory identified the stiffness slope: there the
    # velocity-dependent damper b*delta_dot has died out so the measured force is the
    # pure spring f = k*delta. The touchdown transient (force leads penetration) is
    # excluded -- evaluating compliance against the transient would unfairly inflate the
    # error and is a property of MuJoCo's damped solver, not of the §7 theorem.
    T = pen.shape[1]
    s0 = (3 * T) // 4

    result = dynamics.observability_demo(
        pen[:, s0:], fn[:, s0:], k, contact_xy=xy
    )

    # (a) Rigid statics are rank-deficient: load split UNOBSERVABLE.
    # rank(A) <= 3 with K unknowns => null space dimension >= K - 3 >= 1.
    assert result["equilibrium_rank"] <= 3
    assert result["null_space_dim"] >= 1, (
        "an indeterminate rig has a nontrivial load-split null space (§7)"
    )
    assert result["null_space_dim"] == K - result["equilibrium_rank"]
    # The reported basis really lies in the null space (A @ d ~ 0).
    assert result["null_space_residual"] < 1e-9
    assert result["observable_rigid"] is False

    # (b) Compliance RECOVERS the per-corner forces from their own penetrations.
    # Documented tolerance: over the settled tail (pure spring regime) the recovered
    # force matches the simulator's measured corner force to ~1e-6 relative; we allow a
    # generous 1e-3 to absorb residual damping and the least-squares-fit stiffness.
    REL_TOL = 1e-3
    assert result["max_rel_error"] < REL_TOL, (
        f"compliance must recover per-corner force (§7); "
        f"rel error {result['max_rel_error']:.2e} > {REL_TOL:.0e}"
    )
    assert result["observable_compliant"] is True
