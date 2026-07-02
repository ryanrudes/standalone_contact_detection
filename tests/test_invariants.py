"""Executable derivation of two structural invariants (THEORY.md §1 & §4).

(§1) observe() is SUPPORT-RELATIVE: a body rigidly co-moving with a translating support reads
      STATIC (relative twist ~ 0) despite large world motion -- the method's foundational claim
      (a foot on a moving skateboard is in solid contact though it screams across the world).
(§4) ROLLING is the one non-product mode; its coupled (v_t, omega_t) block is renormalized by
      Z_res so the column is a proper density. We recompute Z_res by an independent 2-D quadrature
      and assert the code's normalizer matches -- i.e. the coupled block integrates to 1.

These are pure-analytic checks (no MuJoCo); they cover in isolation what the moving_support and
rolling expectation scenarios exercise end-to-end.
"""

import numpy as np
import pytest

from contact import observe
from contact.emissions import _log_rolling_residual_normalizer
from contact.types import PoseTrajectory, SupportSurface


def test_observe_is_support_relative():
    hz = 100.0
    t = np.arange(0.0, 2.0, 1.0 / hz)
    T = t.shape[0]
    # A support that TRANSLATES ~2 m through the world (and bobs, so its speed is non-trivial).
    sup_pos = np.zeros((T, 3))
    sup_pos[:, 0] = 1.0 * t
    sup_pos[:, 1] = 0.3 * np.sin(2.0 * t)
    sup_pos[:, 2] = 0.5
    ident = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, 1))
    support = PoseTrajectory(t=t, position=sup_pos, quat=ident)
    # The body rides the support rigidly: a fixed world offset, same (identity) orientation.
    body = PoseTrajectory(t=t, position=sup_pos + np.array([0.1, 0.0, 0.05]), quat=ident.copy())
    surface = SupportSurface(point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]))
    obs = observe(body, support, surface, np.array([0.0, 0.0, -0.05]))

    world_speed = float(np.median(np.linalg.norm(np.diff(sup_pos, axis=0) * hz, axis=1)))
    rel = max(
        float(np.max(np.abs(obs.v_normal))),
        float(np.max(np.linalg.norm(obs.v_tangent, axis=1))),
        float(np.max(np.abs(obs.omega_normal))),
        float(np.max(np.linalg.norm(obs.omega_tangent, axis=1))),
    )
    assert world_speed > 0.9          # the support really is sweeping ~1 m/s through the world
    assert rel < 1e-9                 # yet the co-moving body reads static, in the support's frame


def test_rolling_zres_normalizer_is_proper():
    # EmissionParams defaults for the rolling block.
    sv, sw, rr, rs = 0.50, 3.00, 0.05, 0.03
    code_z = float(np.exp(_log_rolling_residual_normalizer(sv, sw, rr, rs)))
    # Independent 2-D Riemann quadrature over the Rayleigh magnitudes a = |v_t|, b = |omega_t|
    # (the code uses nested adaptive scipy.quad; a dense grid is a different method).
    n = 1200
    a = np.linspace(0.0, 8.0 * sv, n)
    b = np.linspace(0.0, 8.0 * sw, n)
    ray_a = a / (sv * sv) * np.exp(-a * a / (2.0 * sv * sv))
    ray_b = b / (sw * sw) * np.exp(-b * b / (2.0 * sw * sw))
    resid = 1.0 / (rs * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * ((a[:, None] - rr * b[None, :]) / rs) ** 2)
    grid_z = float(np.sum(ray_a[:, None] * ray_b[None, :] * resid) * (a[1] - a[0]) * (b[1] - b[0]))
    assert code_z == pytest.approx(grid_z, rel=5e-3)
