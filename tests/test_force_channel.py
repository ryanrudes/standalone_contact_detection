"""Permanent validation of the force-emission channel (DESIGN.md PART II.A / Phase 4).

These promote the session's force-channel checks into self-contained tests against the math of
DESIGN.md PART II.A (the force-emission factor must be PROPER densities or it silently biases the
HMM):

* the per-state force densities (FREE half-normal, contact/impact Rayleigh) are proper -- they
  integrate to ~1 over their ``[0, inf)`` support;
* the whole term is GATED: force params with no ``normal_force`` observation contribute nothing
  (emissions are byte-identical to ``force=None``);
* the sustained-contact MIXTURE keeps a touching-but-unloaded contact NEAR-NEUTRAL versus FREE at
  ``f~0`` (the contact-vs-free log-ratio there is ~``log(w_unloaded)``, a small constant -- the
  fix for the "Rayleigh collapsed contact recall 452->16" bug) while decisively preferring contact
  at high force;
* the inferred-force VIRTUAL SENSOR tracks the simulator's true normal force on a supported body
  and degrades gracefully to ``None`` when the dynamics are unsupported.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("mujoco")  # the virtual-sensor test runs the MuJoCo harness; skip if absent

import oracle  # noqa: E402
from contact.config import DetectorConfig, EmissionParams, ForceEmissionParams  # noqa: E402
from contact.dynamics_id import infer_normal_force  # noqa: E402
from contact.emissions import (  # noqa: E402
    _force_log_density,
    _log_half_normal,
    _log_rayleigh,
    log_emissions,
)
from contact.types import ALL_STATES, FREE, STATIC, ContactObservations  # noqa: E402

# numpy 2.x renamed ``trapz`` -> ``trapezoid``; support either so the test is version-robust.
_TRAPZ = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _force_only_obs(normal_force) -> ContactObservations:
    """A ContactObservations carrying only the normal_force channel (other channels zeroed).

    The force densities read ``obs.normal_force`` alone, so the kinematic channels are irrelevant
    here. ``normal_force=None`` builds the gated-off case."""
    nf = None if normal_force is None else np.asarray(normal_force, dtype=float)
    n = 1 if normal_force is None else len(nf)
    z1 = np.zeros(n)
    z2 = np.zeros((n, 2))
    return ContactObservations(
        t=np.arange(float(n)), gap=z1.copy(), v_normal=z1.copy(), v_tangent=z2.copy(),
        omega_normal=z1.copy(), omega_tangent=z2.copy(), normal_force=nf,
    )


# ======================================================================================
# Proper densities (DESIGN.md PART II.A: every force term integrates to 1 over [0, inf))
# ======================================================================================


class TestForceDensitiesNormalized:
    def test_half_normal_integrates_to_one(self) -> None:
        """``exp(_log_half_normal)`` integrates to ~1 over [0, 30] (FREE's proper force density)."""
        f = np.linspace(0.0, 30.0, 300001)
        for sigma in (0.15, 1.0, 2.0):
            integral = float(_TRAPZ(np.exp(_log_half_normal(f, sigma)), f))
            assert abs(integral - 1.0) < 0.02, (
                f"the half-normal HN(sigma={sigma}) must integrate to 1 over its support; got {integral:.4f}"
            )

    def test_rayleigh_integrates_to_one(self) -> None:
        """``exp(_log_rayleigh)`` integrates to ~1 over [0, 30] (the contact/impact force density)."""
        f = np.linspace(0.0, 30.0, 300001)
        for scale in (1.0, 4.0):
            integral = float(_TRAPZ(np.exp(_log_rayleigh(f, scale)), f))
            assert abs(integral - 1.0) < 0.02, (
                f"the Rayleigh R(scale={scale}) must integrate to 1 over its support; got {integral:.4f}"
            )


# ======================================================================================
# Gated additivity (DESIGN.md §6.2 / §III.6 inv. 3: no channel => no factor)
# ======================================================================================


class TestForceGating:
    def test_force_term_is_gated_off_without_observation(self) -> None:
        """``log_emissions(force=ForceEmissionParams())`` == ``force=None`` when ``normal_force`` is None.

        The force term is gated on ``obs.normal_force is not None``, so supplying the params but no
        force observation must leave the emission matrix byte-identical to the kinematics-only one."""
        obs = _force_only_obs(None)  # no force channel present
        with_params = log_emissions(
            obs, EmissionParams(), 0.0, list(ALL_STATES), force=ForceEmissionParams()
        )
        without = log_emissions(obs, EmissionParams(), 0.0, list(ALL_STATES), force=None)
        assert np.array_equal(with_params, without), (
            "with no normal_force observation the force term must be gated off -> identical emissions"
        )


# ======================================================================================
# The contact-force MIXTURE allows an unloaded touch (DESIGN.md PART II.A, the corrected spec)
# ======================================================================================


class TestContactForceMixture:
    def test_mixture_allows_unloaded_contact_and_prefers_contact_under_load(self) -> None:
        """At ``f~0`` the contact-vs-FREE force log-ratio ~ ``log(w_unloaded)``; at high force contact > FREE.

        The sustained-contact density is a mixture ``w*HN + (1-w)*R``: at ``f~0`` its unloaded
        component makes the contact-vs-free log-ratio just ``log(w_unloaded)`` (a small constant, so
        the GAP decides an unloaded touch -- NOT a huge penalty that would collapse contact recall),
        while appreciable force pulls decisively to contact via the loaded Rayleigh."""
        fp = ForceEmissionParams()
        # median(positive force) == 1.0, so the internal robust normalization gives fn == these values.
        obs = _force_only_obs(np.array([0.0, 1.0, 1.0, 1.0, 10.0]))
        free = _force_log_density(obs, FREE, fp)
        contact = _force_log_density(obs, STATIC, fp)

        ratio_at_zero = float(contact[0] - free[0])
        assert abs(ratio_at_zero - np.log(fp.w_unloaded)) < 1e-3, (
            "an unloaded touch (f~0) must be near-neutral vs FREE: the contact-vs-free log-ratio "
            f"must be ~log(w_unloaded)={float(np.log(fp.w_unloaded)):.3f}; got {ratio_at_zero:.3f}"
        )
        assert ratio_at_zero > -2.0, (
            "f~0 must NOT be hugely below FREE (the mixture's unloaded component keeps it neutral)"
        )
        assert contact[-1] > free[-1], (
            "at high force the contact density must exceed FREE (the loaded Rayleigh component)"
        )


# ======================================================================================
# The inferred-force VIRTUAL SENSOR (DESIGN.md PART II.B / III.4 / Phase 4b)
# ======================================================================================


class TestVirtualForceSensor:
    def test_infer_normal_force_tracks_truth_on_drop_rest(self) -> None:
        """``infer_normal_force(drop_rest)`` correlates >0.8 with the true normal force in contact.

        A single rigid body with inertials: the contact-implicit inverse dynamics must reconstruct
        a force stream that tracks the simulator's true contact force on the in-contact frames,
        with no physical sensor."""
        cfg = DetectorConfig()
        raw = oracle.generate("drop_rest")
        inferred = infer_normal_force(raw, cfg)
        assert inferred is not None, (
            "drop_rest (single rigid body + inertials/candidates) must support inferred force"
        )
        truth = np.asarray(raw.truth.normal_force, dtype=float)[: len(inferred)]
        in_contact = truth > 0.5
        assert in_contact.any(), "drop_rest must have in-contact frames to correlate against"
        corr = float(np.corrcoef(inferred[in_contact], truth[in_contact])[0, 1])
        assert corr > 0.8, (
            f"the inferred-force virtual sensor must track the true force (corr>0.8); got {corr:.2f}"
        )

    def test_infer_normal_force_degrades_to_none_when_unsupported(self) -> None:
        """``infer_normal_force(rolling_ball)`` returns None (graceful no-op-when-absent).

        A scenario lacking the inertial/candidate metadata the solver needs must degrade to None so
        callers fall back to the kinematics-only estimate rather than trusting a bad inference."""
        cfg = DetectorConfig()
        assert infer_normal_force(oracle.generate("rolling_ball"), cfg) is None, (
            "a scenario without the required inertial/candidate metadata must return None gracefully"
        )
