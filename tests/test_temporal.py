"""Tests for the s.5 *temporal* upgrades: gap-gated guards and explicit-duration dwell.

THEORY.md section 5 says the HMM is the discrete shadow of a hybrid dynamical
system, and hands us two refinements "for free":

  * the free->contact transition prior should be *state-dependent* -- gated by the
    geometric guard ``g -> 0`` (the gap reaching zero), rather than a constant
    switch probability; and
  * dwell times are not memoryless, so the honest temporal prior is a
    *semi-Markov / explicit-duration* model with a (rising) hazard, which is the
    principled replacement for a hard "minimum contact duration" / blip drop.

This module exercises exactly those two objects, against properties that must hold
*by construction* rather than against tuned magic numbers:

  * ``contact.transitions.base_transition_matrix`` -- the time-homogeneous prior:
    proper row-stochasticity, strict positivity (finite log-space, s.4), and the
    s.6 timescale separation (IMPACT dwells shorter => higher self-exit than the
    sustained modes).
  * ``contact.transitions.gated_transition_tensor`` -- the gap-gated guard: the
    free->contact entry mass must be ~0 far above the surface, rise *monotonically*
    as the gap closes, and approach the ungated base level once inside ``gap_gate``;
    every row of every frame stays a proper distribution.
  * ``contact.hsmm.duration_logpmf`` -- a proper log-pmf over ``d >= 1`` with the
    requested mean, sharpening (more peaked) as the concentration grows.
  * ``contact.hsmm.hsmm_viterbi`` -- absorbs a 1-2 frame spurious blip that a plain
    per-frame argmax would keep, because the duration prior makes a length-1 segment
    intrinsically improbable.
"""

from __future__ import annotations

import numpy as np
import pytest

from contact.config import TransitionParams
from contact.hmm import viterbi
from contact.hsmm import duration_logpmf, hsmm_viterbi
from contact.transitions import base_transition_matrix, gated_transition_tensor
from contact.types import (
    ALL_STATES,
    CONTACT_MODES,
    FREE,
    IMPACT,
    ContactObservations,
)

# A frame period typical of optical mocap (~100 Hz). Small enough that one frame is a
# fraction of a dwell, so the dwell exponentials below are well away from 0 and 1.
DT = 0.01


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _gap_only_obs(gap: np.ndarray) -> ContactObservations:
    """A ContactObservations carrying only a gap ramp; other channels are zeroed.

    ``gated_transition_tensor`` consumes *only* ``obs.gap`` (it evaluates the gate on
    the signed distance, s.1), so the velocity/twist channels are irrelevant here and
    set to zero just to satisfy the dataclass contract.
    """
    gap = np.asarray(gap, dtype=float).ravel()
    T = gap.shape[0]
    t = np.arange(T, dtype=float) * DT
    return ContactObservations(
        t=t,
        gap=gap,
        v_normal=np.zeros(T),
        v_tangent=np.zeros((T, 2)),
        omega_normal=np.zeros(T),
        omega_tangent=np.zeros((T, 2)),
    )


# ======================================================================================
# base_transition_matrix  (the time-homogeneous temporal prior, s.5 / s.6)
# ======================================================================================

class TestBaseTransitionMatrix:
    def test_rows_sum_to_one(self) -> None:
        """Every row is a proper distribution over the next state (normalized, s.4)."""
        P = base_transition_matrix(ALL_STATES, DT, TransitionParams())
        assert P.shape == (len(ALL_STATES), len(ALL_STATES))
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12), (
            f"transition rows must sum to 1; got {P.sum(axis=1)}"
        )

    def test_strictly_positive(self) -> None:
        """Every entry is strictly > 0 so log-space has no -inf (s.4).

        The HMM must always be able to leave any state (however unlikely), so the
        smoother can recover from a surprising frame -- the ``_FLOOR`` in the module
        guarantees this.
        """
        P = base_transition_matrix(ALL_STATES, DT, TransitionParams())
        assert np.all(P > 0.0), f"all transition entries must be > 0; min={P.min()}"
        assert np.all(np.isfinite(np.log(P))), "log of every entry must be finite"

    def test_self_transition_is_dwell_survival(self) -> None:
        """The diagonal is the CT-Markov survival ``exp(-dt/tau)`` of the dwell (s.5).

        Each state's "stay" probability is the exponential survival of a jump process
        with that state's mean dwell; we check it matches for the baseline modes and
        the (shorter) impact dwell.
        """
        params = TransitionParams()
        P = base_transition_matrix(ALL_STATES, DT, params)
        idx = {s: i for i, s in enumerate(ALL_STATES)}
        expected_sustained = np.exp(-DT / params.mean_dwell_time)
        for s in ALL_STATES:
            if s == IMPACT:
                continue
            assert P[idx[s], idx[s]] == pytest.approx(expected_sustained, rel=1e-9), (
                f"{s} self-transition must equal exp(-dt/mean_dwell)"
            )
        expected_impact = np.exp(-DT / params.impact_dwell_time)
        assert P[idx[IMPACT], idx[IMPACT]] == pytest.approx(expected_impact, rel=1e-9)

    def test_impact_has_shorter_dwell_than_sustained(self) -> None:
        """IMPACT is a finer-timescale transient: shorter dwell => higher self-exit (s.6).

        s.6 places impact "at a finer timescale than the sustained modes," so it must
        persist *less*: its self-transition probability is strictly smaller (it exits
        sooner) than every sustained contact mode's.
        """
        P = base_transition_matrix(ALL_STATES, DT, TransitionParams())
        idx = {s: i for i, s in enumerate(ALL_STATES)}
        impact_stay = P[idx[IMPACT], idx[IMPACT]]
        impact_exit = 1.0 - impact_stay
        sustained = [m for m in CONTACT_MODES if m != IMPACT]
        for m in sustained:
            stay = P[idx[m], idx[m]]
            assert impact_stay < stay, (
                f"IMPACT must dwell shorter than sustained mode {m}: "
                f"impact_stay={impact_stay} vs {m}_stay={stay}"
            )
            assert impact_exit > (1.0 - stay), (
                "IMPACT must have a higher per-frame self-exit than sustained modes"
            )

    def test_impact_leads_the_free_entry(self) -> None:
        """FREE re-enters contact preferentially through IMPACT (the make guard, s.6).

        Not a tuned threshold: just the ordering the docstring promises -- FREE->IMPACT
        carries more mass than FREE->any single sustained mode.
        """
        P = base_transition_matrix(ALL_STATES, DT, TransitionParams())
        idx = {s: i for i, s in enumerate(ALL_STATES)}
        free_to_impact = P[idx[FREE], idx[IMPACT]]
        for m in [m for m in CONTACT_MODES if m != IMPACT]:
            assert free_to_impact > P[idx[FREE], idx[m]], (
                "FREE->IMPACT should lead FREE->sustained (make guard, s.6)"
            )


# ======================================================================================
# gated_transition_tensor  (the state-dependent gap guard, s.5)
# ======================================================================================

class TestGatedTransitionTensor:
    def test_every_row_every_frame_sums_to_one(self) -> None:
        """Each frame's matrix is row-stochastic (cross-state comparisons valid, s.4)."""
        gap = np.linspace(0.1, -0.01, 40)  # an arbitrary descent through the surface
        obs = _gap_only_obs(gap)
        params = TransitionParams()
        tensor = gated_transition_tensor(obs, ALL_STATES, DT, params)
        S = len(ALL_STATES)
        assert tensor.shape == (gap.shape[0], S, S)
        row_sums = tensor.sum(axis=2)  # (T, S)
        assert np.allclose(row_sums, 1.0, atol=1e-12), (
            f"every row of every frame must sum to 1; max dev "
            f"{np.abs(row_sums - 1.0).max()}"
        )

    def test_strictly_positive_everywhere(self) -> None:
        """Even with the gate shut, every entry stays > 0 (finite log, s.4)."""
        gap = np.linspace(0.5, -0.02, 30)
        obs = _gap_only_obs(gap)
        tensor = gated_transition_tensor(obs, ALL_STATES, DT, TransitionParams())
        assert np.all(tensor > 0.0), f"gated tensor must stay > 0; min={tensor.min()}"

    def test_entry_mass_near_zero_far_above_surface(self) -> None:
        """Far above the surface (gap >> gap_gate) the FREE->contact mass is ~0.

        The body is in flight; the make guard is *not* armed, so almost no probability
        leaks from FREE into a contact mode. Only the tiny ``_FLOOR`` residual survives.
        """
        params = TransitionParams()
        # A single frame sitting far above the gate (many softness-widths away).
        far = params.gap_gate + 50.0 * params.gap_gate_softness
        obs = _gap_only_obs(np.array([far]))
        tensor = gated_transition_tensor(obs, ALL_STATES, DT, params)
        free_i = ALL_STATES.index(FREE)
        contact_cols = [j for j, s in enumerate(ALL_STATES) if s != FREE]
        entry = float(tensor[0, free_i, contact_cols].sum())

        base = base_transition_matrix(ALL_STATES, DT, params)
        base_entry = float(base[free_i, contact_cols].sum())
        # The far-above entry mass should be a small fraction of the ungated base.
        assert entry < 0.1 * base_entry, (
            f"far-above entry mass {entry} should be << base entry {base_entry}"
        )

    def test_entry_mass_rises_toward_base_inside_gate(self) -> None:
        """Well inside the gate (gap << gap_gate) the entry mass approaches the base.

        Once the surface is within reach the gate is open (~1), so the gap-gated FREE
        row should recover essentially the full ungated free->contact entry mass.
        """
        params = TransitionParams()
        deep = params.gap_gate - 50.0 * params.gap_gate_softness  # well below the gate
        obs = _gap_only_obs(np.array([deep]))
        tensor = gated_transition_tensor(obs, ALL_STATES, DT, params)
        free_i = ALL_STATES.index(FREE)
        contact_cols = [j for j, s in enumerate(ALL_STATES) if s != FREE]
        entry = float(tensor[0, free_i, contact_cols].sum())

        base = base_transition_matrix(ALL_STATES, DT, params)
        base_entry = float(base[free_i, contact_cols].sum())
        assert entry == pytest.approx(base_entry, rel=0.05), (
            f"deep-inside entry mass {entry} should approach base entry {base_entry}"
        )

    def test_monotonic_gating_on_a_gap_ramp(self) -> None:
        """On a monotone gap descent, the FREE->contact entry mass rises monotonically.

        This is the core property of the s.5 guard: the chance of *entering* contact is
        a monotone increasing function of "how close are we" (decreasing gap). We feed a
        strictly-decreasing gap ramp from well above to well below the gate and assert
        the per-frame entry mass is non-decreasing throughout (and strictly grows across
        the transition region).
        """
        params = TransitionParams()
        # A strictly-decreasing ramp spanning many softness-widths on both sides of the
        # gate, so the logistic gate sweeps cleanly from ~0 to ~1.
        gap = np.linspace(
            params.gap_gate + 8.0 * params.gap_gate_softness,
            params.gap_gate - 8.0 * params.gap_gate_softness,
            60,
        )
        obs = _gap_only_obs(gap)
        tensor = gated_transition_tensor(obs, ALL_STATES, DT, params)
        free_i = ALL_STATES.index(FREE)
        contact_cols = [j for j, s in enumerate(ALL_STATES) if s != FREE]
        entry = tensor[:, free_i, contact_cols].sum(axis=1)  # (T,)

        diffs = np.diff(entry)
        # Monotone non-decreasing as the gap shrinks (allow a hair of float noise).
        assert np.all(diffs >= -1e-12), (
            f"entry mass must be non-decreasing as the gap closes; min diff {diffs.min()}"
        )
        # And it must actually move: low (start, far above) << high (end, deep inside).
        assert entry[-1] > 5.0 * entry[0], (
            f"entry mass must rise substantially across the gate: "
            f"start={entry[0]}, end={entry[-1]}"
        )

    def test_relative_mode_split_is_gate_independent(self) -> None:
        """The gate modulates the *total* entry, never *which* mode is entered.

        The relative split of the offered entry mass across contact modes is the base
        (IMPACT-led) shape at every frame, regardless of how far open the gate is.
        """
        params = TransitionParams()
        gap = np.array([0.5, params.gap_gate, params.gap_gate - 0.05])  # closed/edge/open
        obs = _gap_only_obs(gap)
        tensor = gated_transition_tensor(obs, ALL_STATES, DT, params)
        free_i = ALL_STATES.index(FREE)
        contact_cols = [j for j, s in enumerate(ALL_STATES) if s != FREE]

        shapes = []
        for t in range(gap.shape[0]):
            entry_vec = tensor[t, free_i, contact_cols]
            shapes.append(entry_vec / entry_vec.sum())  # normalized split
        for s in shapes[1:]:
            assert np.allclose(s, shapes[0], atol=1e-9), (
                "relative split across contact modes must be gate-independent"
            )

    def test_non_free_rows_are_frame_independent(self) -> None:
        """Non-FREE rows are copied straight from the base (governed by lambda->0, s.5).

        Only the FREE row is gap-gated; contact->contact / contact->free are the *force*
        break guard, not the gap guard, so they must equal the base matrix at every frame.
        """
        params = TransitionParams()
        gap = np.linspace(0.2, -0.02, 25)
        obs = _gap_only_obs(gap)
        tensor = gated_transition_tensor(obs, ALL_STATES, DT, params)
        base = base_transition_matrix(ALL_STATES, DT, params)
        free_i = ALL_STATES.index(FREE)
        for t in range(gap.shape[0]):
            for i in range(len(ALL_STATES)):
                if i == free_i:
                    continue
                assert np.allclose(tensor[t, i], base[i], atol=1e-12), (
                    f"non-FREE row {i} must equal the base row at frame {t}"
                )


# ======================================================================================
# duration_logpmf  (the explicit-duration dwell prior, s.5)
# ======================================================================================

class TestDurationLogpmf:
    def test_is_a_proper_log_pmf(self) -> None:
        """exp(log-pmf) over d=1..(large) sums to ~1 up to the truncation tail."""
        mean, conc = 12.0, 4.0
        d = np.arange(1, 20001)
        p = np.exp(duration_logpmf(d, mean, conc))
        assert np.all(p >= 0.0), "probabilities must be non-negative"
        assert abs(p.sum() - 1.0) < 1e-4, f"pmf must sum to ~1; got {p.sum()}"

    def test_has_the_requested_mean(self) -> None:
        """The distribution's mean matches the requested ``mean_dwell_frames``."""
        for mean in (5.0, 12.0, 40.0):
            d = np.arange(1, 40001)
            p = np.exp(duration_logpmf(d, mean, 4.0))
            got_mean = float((d * p).sum())
            assert got_mean == pytest.approx(mean, rel=1e-2), (
                f"duration mean must equal requested {mean}; got {got_mean}"
            )

    def test_supported_only_on_integers_ge_one(self) -> None:
        """A 0-frame (or non-integer) segment is meaningless => log-prob ~ -inf."""
        assert duration_logpmf(0, 12.0, 4.0) < -1e29, "d=0 must have ~zero probability"
        assert duration_logpmf(-3, 12.0, 4.0) < -1e29, "d<0 must have ~zero probability"
        assert duration_logpmf(2.5, 12.0, 4.0) < -1e29, "non-integer d must be ~zero"
        # A legitimate d=1 is finite (just possibly small).
        assert np.isfinite(duration_logpmf(1, 12.0, 4.0))

    def test_higher_concentration_is_more_peaked(self) -> None:
        """Larger concentration concentrates mass around the mean (smaller variance).

        s.5: ``concentration ~ 1`` is the memoryless geometric dwell; higher values are
        a sum of more geometric "stages" whose CLT effect tightens the distribution
        (coefficient of variation ~ 1/sqrt(concentration)). We assert the variance is
        strictly monotone-decreasing in concentration.
        """
        mean = 15.0
        d = np.arange(1, 60001)

        def variance(conc: float) -> float:
            p = np.exp(duration_logpmf(d, mean, conc))
            m = float((d * p).sum())
            return float(((d - m) ** 2 * p).sum())

        concs = [1.0, 4.0, 16.0, 64.0]
        variances = [variance(c) for c in concs]
        for lo, hi in zip(variances, variances[1:]):
            assert hi < lo, (
                f"variance must shrink as concentration grows; got {variances}"
            )

    def test_blip_is_improbable_relative_to_mean_at_high_concentration(self) -> None:
        """At high concentration a 1-frame dwell is far less probable than the mean.

        This is *why* the HSMM absorbs blips: P(d=1) is crushed relative to P(d~=mean)
        when the dwell is concentrated, so a length-1 segment carries a large prior cost.
        """
        mean, conc = 20.0, 8.0
        lp1 = duration_logpmf(1, mean, conc)
        lp_mean = duration_logpmf(int(round(mean)), mean, conc)
        assert lp1 < lp_mean, "a 1-frame dwell must be less probable than the mean dwell"
        # And much less so as concentration grows (the prior cost of a blip increases).
        gap_low = duration_logpmf(int(round(mean)), mean, 2.0) - duration_logpmf(1, mean, 2.0)
        gap_high = lp_mean - lp1
        assert gap_high > gap_low, (
            "higher concentration must widen the log-prob gap between d=1 and the mean"
        )


# ======================================================================================
# hsmm_viterbi  (explicit-duration MAP path absorbs spurious blips, s.5)
# ======================================================================================

class TestHsmmViterbiBlipAbsorption:
    @staticmethod
    def _two_state_blip(blip_len: int, T: int = 60, blip_at: int = 30) -> np.ndarray:
        """Emissions: a mild FREE preference everywhere, with a strong CONTACT blip.

        State 0 = FREE (long expected dwell), state 1 = CONTACT. Off the blip the FREE
        emission mildly dominates; across ``blip_len`` frames the CONTACT emission is so
        confident that a per-frame argmax (and a plain Markov Viterbi) would flip there.
        """
        S = 2
        log_emission = np.empty((T, S), dtype=float)
        log_emission[:, 0] = np.log(0.6)  # mild FREE preference
        log_emission[:, 1] = np.log(0.4)
        sl = slice(blip_at, blip_at + blip_len)
        log_emission[sl, 0] = np.log(0.02)  # strong, confident CONTACT for a few frames
        log_emission[sl, 1] = np.log(0.98)
        return log_emission

    def test_argmax_would_keep_the_blip(self) -> None:
        """Sanity: per-frame argmax *does* flip on the blip (the thing we must fix)."""
        for blip_len in (1, 2):
            log_emission = self._two_state_blip(blip_len)
            argmax_path = np.argmax(log_emission, axis=1)
            assert np.any(argmax_path == 1), (
                "argmax baseline must keep the blip (otherwise the test is vacuous)"
            )

    def test_hsmm_absorbs_one_and_two_frame_blips(self) -> None:
        """The duration prior makes a 1-2 frame contact segment too short to survive.

        With a concentrated dwell on the CONTACT state, ``hsmm_viterbi`` returns an
        all-FREE path: the short segment's duration-prior cost outweighs its confident
        emissions, so the blip is absorbed (the principled replacement for blip-dropping).
        """
        log_trans = np.log([[0.5, 0.5], [0.5, 0.5]])
        log_init = np.log([0.5, 0.5])
        mean_dwell = np.array([60.0, 12.0])  # FREE long-lived; CONTACT ~12-frame typical
        for blip_len in (1, 2):
            log_emission = self._two_state_blip(blip_len)
            path = hsmm_viterbi(
                log_emission, log_trans, log_init, mean_dwell, concentration=8.0
            )
            assert np.all(path == 0), (
                f"{blip_len}-frame blip must be absorbed by the HSMM; got {path}"
            )

    def test_plain_viterbi_keeps_what_hsmm_absorbs(self) -> None:
        """A plain (memoryless) Viterbi keeps the blip the HSMM removes.

        This pins the *difference* the explicit-duration prior makes: on the identical
        emissions and a mild persistence prior, plain Viterbi flips on the confident
        blip, while the HSMM (above) does not.
        """
        log_emission = self._two_state_blip(blip_len=1)
        # A mild self-persistence prior (still memoryless): not enough to beat a single
        # very-confident frame, so plain Viterbi keeps the blip.
        p_stay = 0.7
        log_trans = np.log([[p_stay, 1 - p_stay], [1 - p_stay, p_stay]])
        log_init = np.log([0.5, 0.5])
        plain_path = viterbi(log_emission, log_trans, log_init)
        assert np.any(plain_path == 1), (
            "plain Viterbi should keep the confident 1-frame blip (memoryless prior)"
        )

    def test_genuine_bout_is_preserved(self) -> None:
        """A real multi-frame contact bout is *not* absorbed (no over-smoothing).

        The duration prior must suppress only implausibly-short segments; a genuine
        20-frame contact (comfortably longer than the mean dwell) is recovered.
        """
        T, S = 60, 2
        log_emission = np.full((T, S), np.log(0.5))
        log_emission[20:40, 0] = np.log(0.2)
        log_emission[20:40, 1] = np.log(0.8)  # genuine 20-frame contact
        log_trans = np.log([[0.5, 0.5], [0.5, 0.5]])
        log_init = np.log([0.5, 0.5])
        mean_dwell = np.array([60.0, 12.0])
        path = hsmm_viterbi(
            log_emission, log_trans, log_init, mean_dwell, concentration=8.0
        )
        assert path[20:40].mean() > 0.8, f"genuine 20-frame bout must survive; got {path}"
        assert path[:15].mean() < 0.2 and path[45:].mean() < 0.2, (
            "frames outside the bout should remain FREE"
        )
