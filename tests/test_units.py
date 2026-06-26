"""Focused unit tests for the math / leaf modules.

These tests exercise the *leaf* building blocks of the detector in isolation, one
module per section, against hand-computed expectations:

* ``contact.hmm``       -- log-space HMM primitives (THEORY.md section 5).
* ``contact.emissions`` -- per-state emission log-likelihoods (THEORY.md sections 3 & 4).
* ``contact.geometry``  -- poses -> support-relative observations (THEORY.md sections 1 & 3).
* ``contact.signals``   -- time-aware differentiation (THEORY.md sections 4 & 6).

The emphasis is on *qualitative discrimination that must hold by construction* (e.g.
the right mode wins on a synthetic frame engineered to match that mode's twist
subspace), with tolerances chosen to be loose enough to survive numerical noise yet
tight enough to fail on a real regression. Higher-level integration (the assembled
``ContactDetector``, EM calibration, scenario scoring) is covered elsewhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from contact import emissions, geometry
from contact.config import EmissionParams
from contact.hmm import forward_backward, logsumexp, viterbi
from contact.signals import derivative
from contact.types import (
    ALL_STATES,
    FREE,
    PIVOTING,
    ROLLING,
    SLIDING,
    STATIC,
    ContactObservations,
    PoseTrajectory,
    SupportSurface,
)

# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def _quat_about_axis(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Scalar-first unit quaternion(s) rotating by ``angle`` about a fixed unit ``axis``.

    Vectorized over a (T,) angle array; returns (T, 4). Used to synthesize spinning /
    rolling pose streams for the geometry tests.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    angle = np.atleast_1d(np.asarray(angle, dtype=float))
    half = 0.5 * angle
    w = np.cos(half)
    xyz = np.sin(half)[:, None] * axis[None, :]
    return np.concatenate([w[:, None], xyz], axis=1)


def _const_quat_stream(T: int) -> np.ndarray:
    """(T, 4) stream of identity quaternions."""
    return np.tile(IDENTITY_QUAT, (T, 1))


def _zero_pose(t: np.ndarray) -> PoseTrajectory:
    """A PoseTrajectory pinned at the world origin with identity orientation."""
    T = t.shape[0]
    return PoseTrajectory(t=t, position=np.zeros((T, 3)), quat=_const_quat_stream(T))


def _single_frame_obs(
    gap: float,
    v_normal: float,
    v_tangent: tuple[float, float],
    omega_normal: float,
    omega_tangent: tuple[float, float],
) -> ContactObservations:
    """Build a length-1 ContactObservations from scalar channel values."""
    return ContactObservations(
        t=np.array([0.0]),
        gap=np.array([gap]),
        v_normal=np.array([v_normal]),
        v_tangent=np.array([v_tangent], dtype=float),
        omega_normal=np.array([omega_normal]),
        omega_tangent=np.array([omega_tangent], dtype=float),
    )


# ======================================================================================
# contact.hmm  (THEORY.md section 5)
# ======================================================================================


class TestLogsumexp:
    def test_matches_naive_on_safe_inputs(self) -> None:
        """logsumexp == log(sum(exp(.))) on small, overflow-safe values."""
        rng = np.random.default_rng(0)
        a = rng.uniform(-3.0, 3.0, size=(5, 4))

        # Whole-array reduction.
        assert np.isclose(
            float(logsumexp(a)), float(np.log(np.sum(np.exp(a)))), rtol=0, atol=1e-12
        ), "logsumexp over the whole array must match the naive formula on safe inputs"

        # Per-axis reductions agree with the naive per-axis computation.
        for axis in (0, 1):
            got = logsumexp(a, axis=axis)
            want = np.log(np.sum(np.exp(a), axis=axis))
            assert np.allclose(got, want, atol=1e-12), (
                f"logsumexp(axis={axis}) must match naive log-sum-exp"
            )

    def test_stable_under_large_shift(self) -> None:
        """The max-shift trick must keep logsumexp finite where the naive form overflows."""
        a = np.array([1000.0, 1000.0])
        # Naive exp(1000) overflows to inf; logsumexp stays finite and exact.
        got = float(logsumexp(a))
        want = 1000.0 + np.log(2.0)
        assert np.isclose(got, want, atol=1e-9), (
            "logsumexp([1000, 1000]) must equal 1000 + log(2) without overflow"
        )


class TestForwardBackward:
    def test_gamma_rows_sum_to_one(self) -> None:
        """The smoothed posterior gamma is a per-frame distribution: each row sums to 1."""
        rng = np.random.default_rng(1)
        T, S = 20, 3
        log_emission = rng.normal(size=(T, S))
        # A proper (row-normalized) random transition matrix, in log space.
        A = rng.uniform(size=(S, S))
        A /= A.sum(axis=1, keepdims=True)
        log_trans = np.log(A)
        log_init = np.log(np.full(S, 1.0 / S))

        gamma, total = forward_backward(log_emission, log_trans, log_init)
        assert gamma.shape == (T, S), "gamma must be (T, S)"
        assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-12), (
            "every gamma row must sum to exactly 1 (it is a posterior over states)"
        )
        assert np.all(gamma >= 0.0), "posterior probabilities must be non-negative"
        assert np.isfinite(total), "the total data log-likelihood must be finite"

    def test_recovers_obvious_segmentation(self) -> None:
        """On a 2-state chain with state-separating emissions, gamma & Viterbi recover the
        obvious half/half segmentation (THEORY.md section 5: persistence + evidence)."""
        T = 40
        S = 2
        # First half strongly favours state 0, second half strongly favours state 1.
        log_emission = np.full((T, S), -50.0)
        log_emission[: T // 2, 0] = 0.0
        log_emission[T // 2 :, 1] = 0.0

        # Sticky transition prior: strong tendency to persist (the section-5 prior).
        stay, switch = np.log(0.99), np.log(0.01)
        log_trans = np.array([[stay, switch], [switch, stay]])
        log_init = np.log(np.array([0.5, 0.5]))

        gamma, _ = forward_backward(log_emission, log_trans, log_init)
        map_path = viterbi(log_emission, log_trans, log_init)

        # gamma must be (near) certain of the correct state in each half.
        assert np.all(gamma[: T // 2, 0] > 0.99), "first half must be posterior state 0"
        assert np.all(gamma[T // 2 :, 1] > 0.99), "second half must be posterior state 1"

        expected = np.array([0] * (T // 2) + [1] * (T // 2))
        assert np.array_equal(map_path, expected), (
            "Viterbi must recover a single clean 0..0,1..1 segmentation"
        )

    def test_time_varying_transitions_accepted(self) -> None:
        """Time-varying (T, S, S) transitions are accepted and still yield valid posteriors
        (THEORY.md section 5: state-dependent guards => time-varying transition prior)."""
        T, S = 15, 2
        rng = np.random.default_rng(2)
        log_emission = rng.normal(size=(T, S))
        # A distinct proper transition matrix at every step.
        A = rng.uniform(size=(T, S, S))
        A /= A.sum(axis=2, keepdims=True)
        log_trans = np.log(A)
        log_init = np.log(np.full(S, 1.0 / S))

        gamma, total = forward_backward(log_emission, log_trans, log_init)
        path = viterbi(log_emission, log_trans, log_init)

        assert gamma.shape == (T, S)
        assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-12), (
            "time-varying transitions must still give row-normalized gamma"
        )
        assert path.shape == (T,) and path.dtype.kind == "i", (
            "Viterbi must return an int path of shape (T,) for time-varying transitions"
        )
        assert np.isfinite(total)


# ======================================================================================
# contact.emissions  (THEORY.md sections 3 & 4)
# ======================================================================================


class TestEmissions:
    def _params(self) -> EmissionParams:
        return EmissionParams()

    def test_shape(self) -> None:
        """log_emissions returns (T, len(states))."""
        p = self._params()
        T = 7
        obs = ContactObservations(
            t=np.linspace(0, 1, T),
            gap=np.zeros(T),
            v_normal=np.zeros(T),
            v_tangent=np.zeros((T, 2)),
            omega_normal=np.zeros(T),
            omega_tangent=np.zeros((T, 2)),
        )
        M = emissions.log_emissions(obs, p, gap_bias=0.0, states=list(ALL_STATES))
        assert M.shape == (T, len(ALL_STATES)), (
            f"emission matrix must be (T, len(states)); got {M.shape}"
        )
        assert np.all(np.isfinite(M)), "all emission log-densities must be finite"

    def test_static_wins_on_resting_frame(self) -> None:
        """A resting frame (gap ~ bias, ~0 velocity, ~0 angular rate) is best explained by
        STATIC (THEORY.md section 3: the whole twist is pinned to ~0)."""
        p = self._params()
        gap_bias = 0.001
        obs = _single_frame_obs(
            gap=gap_bias,           # sitting exactly at the resting contact
            v_normal=0.0,
            v_tangent=(0.0, 0.0),
            omega_normal=0.0,
            omega_tangent=(0.0, 0.0),
        )
        states = list(ALL_STATES)
        M = emissions.log_emissions(obs, p, gap_bias=gap_bias, states=states)
        winner = states[int(np.argmax(M[0]))]
        assert winner == STATIC, (
            f"a quiet resting frame must be best explained by STATIC; got {winner!r}"
        )

    def test_free_wins_on_high_clearance_fast_frame(self) -> None:
        """A frame high above the surface and moving fast must favour FREE (diffuse on
        every channel; THEORY.md section 4 -- nothing is pinned)."""
        p = self._params()
        gap_bias = 0.0
        obs = _single_frame_obs(
            gap=0.5,                # half a metre of clearance -- nowhere near contact
            v_normal=1.2,           # fast separating
            v_tangent=(0.8, -0.6),  # fast tangential
            omega_normal=2.0,
            omega_tangent=(1.5, 1.0),
        )
        states = list(ALL_STATES)
        M = emissions.log_emissions(obs, p, gap_bias=gap_bias, states=states)
        winner = states[int(np.argmax(M[0]))]
        assert winner == FREE, (
            f"a high-clearance fast-moving frame must be best explained by FREE; got {winner!r}"
        )

    def test_sliding_beats_static_with_large_tangential_speed(self) -> None:
        """When |v_tangent| is large (near slide_speed) but the gap is at the bias and there
        is no spin, SLIDING must out-score STATIC (THEORY.md section 3: sliding subspace)."""
        p = self._params()
        gap_bias = 0.0
        obs = _single_frame_obs(
            gap=gap_bias,
            v_normal=0.0,                       # not separating
            v_tangent=(p.slide_speed, 0.0),     # squarely on the sliding ring
            omega_normal=0.0,
            omega_tangent=(0.0, 0.0),
        )
        states = [STATIC, SLIDING]
        M = emissions.log_emissions(obs, p, gap_bias=gap_bias, states=states)
        ll = {s: M[0, j] for j, s in enumerate(states)}
        assert ll[SLIDING] > ll[STATIC], (
            "with a large tangential speed SLIDING must out-score STATIC "
            f"({ll[SLIDING]:.3f} !> {ll[STATIC]:.3f})"
        )

    def test_rolling_beats_sliding_on_coupled_frame(self) -> None:
        """When |v_tangent| ~ roll_radius * |omega_tangent| (the rolling constraint of
        THEORY.md section 3), ROLLING must beat SLIDING -- the defining cross-channel
        correlation that a per-channel model cannot represent."""
        p = self._params()
        gap_bias = 0.0
        omega_mag = 4.0                              # rad/s of tangential angular rate
        v_mag = p.roll_radius * omega_mag            # the locked tangential speed: v = r*omega
        obs = _single_frame_obs(
            gap=gap_bias,
            v_normal=0.0,
            v_tangent=(v_mag, 0.0),
            omega_normal=0.0,
            omega_tangent=(0.0, omega_mag),          # rolling axis in the tangent plane
        )
        states = [SLIDING, ROLLING]
        M = emissions.log_emissions(obs, p, gap_bias=gap_bias, states=states)
        ll = {s: M[0, j] for j, s in enumerate(states)}
        assert ll[ROLLING] > ll[SLIDING], (
            "on a frame satisfying |v_t| = r|omega_t| ROLLING must beat SLIDING "
            f"({ll[ROLLING]:.3f} !> {ll[SLIDING]:.3f})"
        )

    def test_pivoting_beats_static_on_spin(self) -> None:
        """A pure spin about the normal (large omega_normal, everything else ~0) favours
        PIVOTING over STATIC (THEORY.md section 3: normal-angular subspace)."""
        p = self._params()
        gap_bias = 0.0
        obs = _single_frame_obs(
            gap=gap_bias,
            v_normal=0.0,
            v_tangent=(0.0, 0.0),
            omega_normal=p.pivot_speed,   # spinning about the surface normal
            omega_tangent=(0.0, 0.0),
        )
        states = [STATIC, PIVOTING]
        M = emissions.log_emissions(obs, p, gap_bias=gap_bias, states=states)
        ll = {s: M[0, j] for j, s in enumerate(states)}
        assert ll[PIVOTING] > ll[STATIC], (
            "a pure normal spin must favour PIVOTING over STATIC "
            f"({ll[PIVOTING]:.3f} !> {ll[STATIC]:.3f})"
        )


# ======================================================================================
# contact.geometry  (THEORY.md sections 1 & 3)
# ======================================================================================


class TestGeometryGap:
    def test_gap_matches_hand_computed_plane_distance(self) -> None:
        """gap == signed distance to the plane, computed by hand for a static floor.

        Floor at z=0 with outward normal +z; the contact point rides at a constant
        height, so every frame's gap must equal that height (THEORY.md section 1)."""
        T = 10
        t = np.linspace(0.0, 1.0, T)
        height = 0.123
        moving = PoseTrajectory(
            t=t,
            position=np.column_stack([np.zeros(T), np.zeros(T), np.full(T, height)]),
            quat=_const_quat_stream(T),
        )
        support = _zero_pose(t)
        surface = SupportSurface(point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]))

        obs = geometry.observe(moving, support, surface, vel_smooth_time=0.0)
        assert np.allclose(obs.gap, height, atol=1e-9), (
            f"gap of a point at z={height} above the z=0 plane must equal {height}"
        )

    def test_gap_on_tilted_plane(self) -> None:
        """Hand-computed signed distance for a non-axis-aligned plane normal.

        Point at (1, 0, 0); plane through the origin with unit normal (1,1,0)/sqrt(2);
        signed distance = (point - p0) . n_hat = 1/sqrt(2)."""
        T = 4
        t = np.linspace(0.0, 1.0, T)
        moving = PoseTrajectory(
            t=t,
            position=np.tile(np.array([1.0, 0.0, 0.0]), (T, 1)),
            quat=_const_quat_stream(T),
        )
        support = _zero_pose(t)
        surface = SupportSurface(point=np.zeros(3), normal=np.array([1.0, 1.0, 0.0]))

        obs = geometry.observe(moving, support, surface, vel_smooth_time=0.0)
        expected = 1.0 / np.sqrt(2.0)
        assert np.allclose(obs.gap, expected, atol=1e-9), (
            f"signed distance to the tilted plane must be 1/sqrt(2)={expected:.6f}"
        )

    def test_rigid_ride_on_translating_support_has_zero_relative_motion(self) -> None:
        """A body rigidly riding a fast-translating support reads |v_tangent| ~ |v_normal| ~ 0.

        THEORY.md section 1: contact is support-RELATIVE. The foot-on-skateboard case --
        both bodies scream across the world, yet the relative twist is ~0."""
        T = 60
        t = np.linspace(0.0, 1.0, T)
        # Both bodies translate identically and fast in +x; the moving body sits a fixed
        # 1 m above the support origin (so it rides rigidly on the support's plane offset).
        x = 5.0 * t                                  # 5 m/s sweep across the world
        ride_height = 0.02
        sup_pos = np.column_stack([x, np.zeros(T), np.zeros(T)])
        mov_pos = np.column_stack([x, np.zeros(T), np.full(T, ride_height)])
        moving = PoseTrajectory(t=t, position=mov_pos, quat=_const_quat_stream(T))
        support = PoseTrajectory(t=t, position=sup_pos, quat=_const_quat_stream(T))
        surface = SupportSurface(point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]))

        obs = geometry.observe(moving, support, surface, vel_smooth_time=0.0)
        # Relative motion must be ~0 everywhere despite the huge world velocity.
        assert np.allclose(obs.v_normal, 0.0, atol=1e-6), (
            "a rigid ride must have ~0 relative NORMAL velocity"
        )
        assert np.allclose(obs.v_tangent, 0.0, atol=1e-6), (
            "a rigid ride must have ~0 relative TANGENTIAL velocity even at 5 m/s in world"
        )
        # And the gap is just the constant ride height.
        assert np.allclose(obs.gap, ride_height, atol=1e-9), (
            "gap must equal the constant ride height for a rigid ride"
        )

    def test_spin_about_normal_gives_omega_normal_dominant(self) -> None:
        """A body spinning about the surface normal yields |omega_normal| >> |omega_tangent|.

        THEORY.md section 3: spin about the normal is the pivoting channel; the tangential
        angular channels (the rolling axis) must stay ~0."""
        T = 80
        t = np.linspace(0.0, 1.0, T)
        spin_rate = 3.0                              # rad/s about +z (the surface normal)
        angle = spin_rate * t
        moving = PoseTrajectory(
            t=t,
            position=np.zeros((T, 3)),               # spinning in place at the origin
            quat=_quat_about_axis(np.array([0.0, 0.0, 1.0]), angle),
        )
        support = _zero_pose(t)
        surface = SupportSurface(point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]))

        obs = geometry.observe(moving, support, surface, vel_smooth_time=0.0)
        # Use interior frames to avoid finite-difference edge transients.
        sl = slice(5, T - 5)
        omega_normal = obs.omega_normal[sl]
        omega_tangent_mag = np.linalg.norm(obs.omega_tangent[sl], axis=1)

        assert np.allclose(np.abs(omega_normal), spin_rate, atol=1e-2), (
            f"omega_normal must recover the {spin_rate} rad/s spin about the normal"
        )
        assert np.all(omega_tangent_mag < 0.05 * spin_rate), (
            "spinning about the normal must leave |omega_tangent| ~ 0 "
            "(omega_normal >> omega_tangent)"
        )


# ======================================================================================
# contact.signals  (THEORY.md sections 4 & 6)
# ======================================================================================


class TestDerivative:
    @pytest.mark.parametrize("smooth_time", [0.0, 0.05])
    def test_linear_ramp_slope(self, smooth_time: float) -> None:
        """derivative of a linear ramp recovers the constant slope (both regimes)."""
        T = 50
        t = np.linspace(0.0, 2.0, T)
        slope = 3.7
        x = slope * t + 1.1
        d = derivative(x, t, smooth_time=smooth_time)
        assert d.shape == (T,), "derivative of a (T,) signal must be (T,)"
        assert np.allclose(d, slope, atol=1e-6), (
            f"derivative of a linear ramp must equal its slope {slope} "
            f"(smooth_time={smooth_time})"
        )

    def test_sinusoid_derivative(self) -> None:
        """derivative of sin(omega t) recovers omega*cos(omega t) within tolerance.

        Boundaries are excluded: one-sided fits / np.gradient end stencils are less
        accurate at the edges, which is expected, not a regression."""
        T = 400
        t = np.linspace(0.0, 2.0 * np.pi, T)
        omega = 2.0
        x = np.sin(omega * t)
        expected = omega * np.cos(omega * t)

        d = derivative(x, t, smooth_time=0.0)
        interior = slice(2, T - 2)
        # The signal is densely sampled, so a central difference is accurate interior.
        assert np.allclose(d[interior], expected[interior], atol=2e-3), (
            "finite-difference derivative of sin must approximate omega*cos interior"
        )

    def test_vector_signal_derivative_per_channel(self) -> None:
        """derivative differentiates each column of a (T, D) signal independently."""
        T = 30
        t = np.linspace(0.0, 1.0, T)
        slopes = np.array([2.0, -5.0, 0.5])
        x = t[:, None] * slopes[None, :] + np.array([0.0, 1.0, -2.0])
        d = derivative(x, t, smooth_time=0.0)
        assert d.shape == (T, 3), "derivative must preserve the (T, D) shape"
        assert np.allclose(d, slopes[None, :], atol=1e-6), (
            "each channel's derivative must equal that channel's slope"
        )
