"""Tests for the s.8 RESEARCH-FRONTIER scaling + uncertainty layer.

THEORY.md section 8 makes "richer contact information" precise: the hidden object a
contact graph infers is a *structure* (which set of edges is simultaneously active),
as a calibrated Bayesian posterior, and section 10 spells out the tractability fork --
exact ``2**E`` enumeration is preferred for the small graphs the package ships with, but
large ``E`` needs a sampling method that never materializes the ``2**E`` alphabet.

This module exercises the two research-frontier pieces against properties that must hold
*by construction* rather than tuned magic numbers:

STRUCTURE (:mod:`contact.structure_inference`)
  * the particle SMOOTHER ``particle_filter_active_sets`` agrees with the exact
    reference ``exact_active_sets`` in the large-particle limit (documented tolerance:
    mean abs diff < 0.05 at ``n_particles >= 512`` on a random ``E=3`` problem);
  * both return valid probabilities in ``[0, 1]``;
  * the particle filter runs on ``E=8`` -- where exact enumeration would be ``2**8 =
    256`` states -- and returns a valid ``(T, 8)`` posterior, demonstrating it scales
    *without* enumerating the subset alphabet.

UNCERTAINTY (:mod:`contact.uncertainty`)
  * ``emission_tempering`` is all-ones when ``meas_cov is None`` (exact backward-compat,
    a no-op);
  * with an occluded window (inflated ``meas_cov``) the weight is ~1 outside and < 0.5
    inside;
  * ``apply_tempering`` shrinks the *magnitude* of the occluded rows;
  * feeding tempered emissions through :func:`contact.hmm.forward_backward` makes the
    occluded frames defer to the temporal prior -- their posterior moves toward their
    (clean) neighbors rather than following the corrupted local likelihood.
"""

from __future__ import annotations

import numpy as np

from contact.config import DetectorConfig, InferenceParams
from contact.hmm import forward_backward
from contact.model import ContactDetector
from contact.structure_inference import (
    exact_active_sets,
    particle_filter_active_sets,
)
from contact.types import ContactObservations
from contact.uncertainty import (
    apply_tempering,
    emission_tempering,
    gap_twist_variance,
    simulate_occlusion,
)

# Documented PF<->exact agreement tolerance on E=3 (see module + structure_inference
# docstrings: mean-abs-diff < 0.05 at n_particles >= 512).
_PF_TOL = 0.05
_N_PARTICLES = 512


# ======================================================================================
# Helpers
# ======================================================================================


def _random_log_evidence(
    T: int, E: int, seed: int, scale: float = 2.0
) -> np.ndarray:
    """A random ``(T, E, 2)`` log-evidence with structure the posterior can latch onto.

    Each edge has a true active/inactive trajectory; the active column is biased toward
    the truth by ``scale`` so the data is informative (not a flat tie), plus noise. Only
    the active-vs-inactive *difference* matters to the estimators, so the absolute level
    is irrelevant -- this is the abstract structure-posterior contract of s.8.
    """
    rng = np.random.default_rng(seed)
    # A persistent random active/inactive sequence per edge (block-structured so the
    # dwell prior is meaningful), as the latent truth.
    truth = np.zeros((T, E), dtype=bool)
    for e in range(E):
        state = rng.random() < 0.5
        for t in range(T):
            if rng.random() < 0.2:  # occasional switch -> a few dwell segments
                state = not state
            truth[t, e] = state
    log_ev = rng.normal(0.0, 0.5, size=(T, E, 2))
    # Push the column matching the truth up by `scale` so the evidence is informative.
    for e in range(E):
        log_ev[truth[:, e], e, 1] += scale
        log_ev[~truth[:, e], e, 0] += scale
    return log_ev


def _synthetic_obs(T: int, *, meas_cov: np.ndarray | None = None) -> ContactObservations:
    """A clean, length-T single-pair observation record (resting-contact-like).

    Channels are tiny/quiet (a body at rest in contact): small gap, near-zero
    velocities. Only the contact-relevant scale matters for the qualitative tests.
    """
    t = np.linspace(0.0, (T - 1) * 0.01, T)
    z = np.zeros(T)
    return ContactObservations(
        t=t,
        gap=z.copy(),
        v_normal=z.copy(),
        v_tangent=np.zeros((T, 2)),
        omega_normal=z.copy(),
        omega_tangent=np.zeros((T, 2)),
        meas_cov=meas_cov,
    )


# ======================================================================================
# STRUCTURE: scaling the active-set posterior beyond enumeration (THEORY.md s.8/s.10)
# ======================================================================================


class TestStructureScaling:
    def test_pf_matches_exact_on_E3(self) -> None:
        """PF smoother matches the exact 2**E reference within the documented tolerance.

        On a fixed-seed random E=3 problem, the Rao-Blackwellized particle smoother and
        the exact enumeration estimate the *same* smoothing marginal, so at
        n_particles >= 512 the mean absolute difference is < 0.05 (the documented bound;
        s.8 -- the PF is not approximating a different prior, only Monte-Carlo sampling
        the same one).
        """
        log_ev = _random_log_evidence(T=30, E=3, seed=7)
        log_dwell_stay = np.log(0.9)

        post_exact, _ = exact_active_sets(log_ev, log_dwell_stay, seed=0)
        post_pf, _ = particle_filter_active_sets(
            log_ev, log_dwell_stay, n_particles=_N_PARTICLES, seed=0
        )

        assert post_exact.shape == (30, 3)
        assert post_pf.shape == (30, 3)
        mean_abs_diff = float(np.mean(np.abs(post_exact - post_pf)))
        assert mean_abs_diff < _PF_TOL, (
            f"PF<->exact mean abs diff {mean_abs_diff:.4f} exceeds {_PF_TOL}"
        )

    def test_both_return_valid_probabilities(self) -> None:
        """Both estimators return per-edge marginals in [0, 1]."""
        log_ev = _random_log_evidence(T=25, E=3, seed=11)
        log_dwell_stay = np.log(0.85)

        post_exact, sets_exact = exact_active_sets(log_ev, log_dwell_stay)
        post_pf, sets_pf = particle_filter_active_sets(
            log_ev, log_dwell_stay, n_particles=_N_PARTICLES, seed=0
        )

        for post in (post_exact, post_pf):
            assert np.all(post >= 0.0) and np.all(post <= 1.0)
            assert np.all(np.isfinite(post))
        # MAP sets are valid edge subsets in [0, E).
        for sets in (sets_exact, sets_pf):
            assert len(sets) == 25
            for s in sets:
                assert all(0 <= e < 3 for e in s)

    def test_pf_scales_to_E8_without_enumeration(self) -> None:
        """PF runs on E=8 (where exact enumeration is 2**8 = 256 states) and is valid.

        The particle smoother never materializes the 2**E subset alphabet -- it only
        ever touches the handful of data-supported sets its cloud visits -- so it
        produces a valid (T, 8) posterior at E=8 where exact enumeration would have to
        carry 256 subset states (and grows exponentially past that). This is the s.8
        "particle filter over the discrete structure" / s.10 large-E rung: the PF
        *scales*.
        """
        E = 8
        # 2**8 = 256 subsets -- the exact reference would enumerate all of these; the PF
        # below does not, which is exactly the point of this test.
        assert (1 << E) == 256

        log_ev = _random_log_evidence(T=20, E=E, seed=3)
        log_dwell_stay = np.log(0.9)

        post_pf, sets_pf = particle_filter_active_sets(
            log_ev, log_dwell_stay, n_particles=_N_PARTICLES, seed=0
        )

        assert post_pf.shape == (20, E)
        assert np.all(post_pf >= 0.0) and np.all(post_pf <= 1.0)
        assert np.all(np.isfinite(post_pf))
        assert len(sets_pf) == 20
        for s in sets_pf:
            assert all(0 <= e < E for e in s)

    def test_pf_is_deterministic_given_seed(self) -> None:
        """The PF uses an isolated, seeded RNG: identical seed => identical output."""
        log_ev = _random_log_evidence(T=15, E=3, seed=5)
        log_dwell_stay = np.log(0.9)
        a, _ = particle_filter_active_sets(log_ev, log_dwell_stay, n_particles=128, seed=42)
        b, _ = particle_filter_active_sets(log_ev, log_dwell_stay, n_particles=128, seed=42)
        assert np.array_equal(a, b)


# ======================================================================================
# UNCERTAINTY: per-frame measurement-uncertainty propagation (THEORY.md s.8)
# ======================================================================================


class TestEmissionTempering:
    def test_all_ones_when_meas_cov_none(self) -> None:
        """meas_cov is None => tempering weight is all-ones (exact backward-compat)."""
        obs = _synthetic_obs(T=20, meas_cov=None)
        w = emission_tempering(obs)
        assert w.shape == (20,)
        assert np.allclose(w, 1.0)
        # And the variance proxy is all-zeros (no extra uncertainty).
        assert np.allclose(gap_twist_variance(obs), 0.0)

    def test_weight_high_outside_low_inside_occlusion(self) -> None:
        """An occluded window has weight ~1 outside and < 0.5 inside.

        Build clean data, occlude a middle window with a large inflated meas_cov, and
        confirm the shrinkage weight w = base/(base+meas) is ~1 on the clean frames and
        falls below 0.5 (measurement noise dominating the base channel noise) on the
        occluded ones.
        """
        T = 30
        window = (10, 18)
        obs = _synthetic_obs(T)
        # A base channel noise std; the inflated variance is chosen well above base^2 so
        # the occluded weight is forced below 0.5 (meas_var > base_var).
        base_noise = 0.05
        inflate = 10.0 * base_noise * base_noise  # meas_var >> base_var on the window
        occ = simulate_occlusion(obs, [window], inflate=inflate, seed=0)

        w = emission_tempering(occ, base_noise=base_noise)
        assert w.shape == (T,)
        assert np.all(w > 0.0) and np.all(w <= 1.0)

        inside = np.zeros(T, dtype=bool)
        inside[window[0] : window[1]] = True
        # Clean frames: barely tempered.
        assert np.all(w[~inside] > 0.99)
        # Occluded frames: strongly down-weighted.
        assert np.all(w[inside] < 0.5)

    def test_apply_tempering_shrinks_occluded_rows(self) -> None:
        """apply_tempering scales the occluded rows toward zero (magnitude shrinks).

        Tempering multiplies each frame row by w(t); since w<1 on the occluded frames,
        the magnitude of those log-emission rows shrinks (toward the flat constant 0)
        while the clean rows (w~1) are essentially unchanged.
        """
        T = 24
        window = (8, 16)
        rng = np.random.default_rng(0)
        log_em = rng.normal(0.0, 3.0, size=(T, 6))  # (T, S) arbitrary log-emissions

        obs = _synthetic_obs(T)
        base_noise = 0.05
        inflate = 20.0 * base_noise * base_noise
        occ = simulate_occlusion(obs, [window], inflate=inflate, seed=0)
        w = emission_tempering(occ, base_noise=base_noise)

        tempered = apply_tempering(log_em, w)
        assert tempered.shape == log_em.shape

        inside = np.zeros(T, dtype=bool)
        inside[window[0] : window[1]] = True

        row_mag = np.abs(log_em).sum(axis=1)
        tempered_mag = np.abs(tempered).sum(axis=1)
        # Occluded rows shrink strictly; clean rows are (numerically) unchanged.
        assert np.all(tempered_mag[inside] < row_mag[inside] - 1e-9)
        assert np.allclose(tempered_mag[~inside], row_mag[~inside])
        # The tempered occluded rows equal the original scaled by w (the definition).
        assert np.allclose(tempered[inside], log_em[inside] * w[inside, None])


class TestTemperingDefersToPrior:
    def test_occluded_frames_defer_to_temporal_prior(self) -> None:
        """Tempered occluded frames follow the temporal prior, not the corrupt evidence.

        Two-state HMM (0 vs 1). The (clean) evidence says state 1 everywhere; we then
        plant a corrupt block that *locally* shouts state 0. With a sticky temporal
        prior:

          * WITHOUT tempering the corrupt block drags the posterior toward state 0
            (the local likelihood wins);
          * WITH tempering the occluded rows are flattened (w -> 0), so those frames
            defer to the temporal prior and their state-1 posterior moves back toward
            their state-1 neighbors.

        We assert the qualitative behavior: tempering raises the occluded frames'
        state-1 posterior (toward the neighbors) relative to the untempered case.
        """
        T = 30
        window = (12, 20)
        S = 2

        # Sticky temporal prior: strong self-transition so neighbors carry the state.
        p_stay = 0.95
        log_trans = np.log(
            np.array([[p_stay, 1 - p_stay], [1 - p_stay, p_stay]])
        )
        log_init = np.log(np.array([0.5, 0.5]))

        # Clean evidence: state 1 favored everywhere (log-LR ~ +3 for state 1).
        log_em = np.zeros((T, S))
        log_em[:, 1] = 3.0
        log_em[:, 0] = 0.0
        # Corrupt block: a measurement glitch that *locally* shouts state 0 hard.
        log_em[window[0] : window[1], 0] = 8.0
        log_em[window[0] : window[1], 1] = 0.0

        # Occlude exactly that window so tempering knows those frames are untrustworthy.
        obs = _synthetic_obs(T)
        base_noise = 0.05
        inflate = 50.0 * base_noise * base_noise  # meas_var >> base_var: w -> ~0 inside
        occ = simulate_occlusion(obs, [window], inflate=inflate, seed=0)
        w = emission_tempering(occ, base_noise=base_noise)

        inside = np.zeros(T, dtype=bool)
        inside[window[0] : window[1]] = True
        # Sanity: outside w~1, inside w strongly < 1 (so the glitch is suppressed).
        assert np.all(w[~inside] > 0.99)
        assert np.all(w[inside] < 0.2)

        gamma_plain, _ = forward_backward(log_em, log_trans, log_init)
        gamma_temp, _ = forward_backward(apply_tempering(log_em, w), log_trans, log_init)

        p1_plain = gamma_plain[inside, 1]
        p1_temp = gamma_temp[inside, 1]

        # Tempering pulls the occluded frames back toward their state-1 neighbors:
        # their state-1 posterior strictly increases vs the untempered (glitch-following)
        # case, on every occluded frame.
        assert np.all(p1_temp > p1_plain)
        # And it actually defers to the prior: the occluded frames now mostly believe
        # state 1 (the neighbor-carried state), where untempered they did not.
        assert np.mean(p1_temp) > 0.5
        assert np.mean(p1_plain) < 0.5


class TestDetectorTemperingAtSystemScale:
    """The PRODUCTION path (ContactDetector.detect) tempers at the system's own scale.

    The two tempering tests above hand-pass ``base_noise=0.05``; this one drives the real
    detector wiring (``ContactDetector.detect`` -> ``uncertainty.emission_tempering``) so
    it guards that the model feeds the tempering the *representative channel scale*
    (``EmissionParams.vel_sigma``), not a unit base noise. With a unit base noise a
    realistic occlusion variance at the detector's scale (``vel_sigma=0.05``) is barely
    tempered (``w ~ 1``) and this test's behavioural assertion fails -- it is exactly the
    regression the wiring must not reintroduce.
    """

    @staticmethod
    def _contact_with_occluded_glitch(
        T: int, window: tuple[int, int], inflate: float
    ) -> ContactObservations:
        """Sustained-contact record (gap ~ 0) with a 'looks-free' glitch on a window.

        The window's gap is lifted to a clear separation (so its raw emission reads FREE)
        and its ``meas_cov`` is inflated to ``inflate`` (a per-frame scalar variance), so a
        tempering that trusts the measurement scale will discount the glitch. Outside the
        window the record is a clean, quiet contact; the channel *values* are set directly
        (not via :func:`simulate_occlusion`) so the test isolates the meas_cov-driven
        tempering from value corruption.
        """
        t = np.linspace(0.0, (T - 1) * 0.005, T)  # ~200 Hz
        gap = np.zeros(T)
        gap[window[0] : window[1]] = 0.01  # 1 cm separation -> reads FREE locally
        meas_cov = np.zeros(T)
        meas_cov[window[0] : window[1]] = float(inflate)
        return ContactObservations(
            t=t,
            gap=gap,
            v_normal=np.zeros(T),
            v_tangent=np.zeros((T, 2)),
            omega_normal=np.zeros(T),
            omega_tangent=np.zeros((T, 2)),
            meas_cov=meas_cov,
        )

    def test_enabling_uncertainty_flattens_occluded_frames_in_detect(self) -> None:
        """Enabling ``use_uncertainty`` makes the detector defer occluded frames to the prior.

        A sustained contact with a brief 'looks-free' glitch that is *also* flagged
        occluded (inflated ``meas_cov`` at the system scale, a multiple of
        ``vel_sigma**2``):

          * WITHOUT uncertainty the glitch wins locally -- the detector's contact
            posterior on the window collapses toward 0 (it believes the spurious free);
          * WITH uncertainty enabled the production path tempers those frames at the
            channel scale (``vel_sigma``), flattening them so the sticky temporal prior
            carries the (contact-believing) neighbours and the window's contact posterior
            rises toward 1.

        Clean frames outside the window are essentially unchanged either way.
        """
        T = 40
        window = (16, 24)
        cfg_scale = DetectorConfig()
        # Occlusion variance well above the base channel noise vel_sigma**2 so the
        # tempering weight is driven near 0 -- but ONLY if the detector uses vel_sigma as
        # the base scale (the fix). A unit base noise would leave w ~ 1 here.
        inflate = 20.0 * cfg_scale.emission.vel_sigma ** 2
        obs = self._contact_with_occluded_glitch(T, window, inflate)

        inside = np.zeros(T, dtype=bool)
        inside[window[0] : window[1]] = True

        # Uncertainty OFF (default): the flag is unset, so meas_cov is ignored entirely.
        res_off = ContactDetector(DetectorConfig()).detect(obs)
        # Uncertainty ON: identical config except the opt-in flag.
        cfg_on = DetectorConfig()
        cfg_on.inference = InferenceParams(use_uncertainty=True)
        res_on = ContactDetector(cfg_on).detect(obs)

        post_off = res_off.contact_posterior
        post_on = res_on.contact_posterior

        # WITHOUT tempering the glitch is believed: low contact posterior on the window.
        assert np.mean(post_off[inside]) < 0.2
        # WITH tempering the window defers to its contact neighbours: high posterior. This
        # only happens if the detector tempered at the channel scale (the fix); a unit
        # base noise would leave the window near post_off and fail here.
        assert np.mean(post_on[inside]) > 0.8
        # And the effect is concentrated on the occluded frames: clean frames barely move.
        assert np.allclose(post_off[~inside], post_on[~inside], atol=0.05)
