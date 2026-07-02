"""The assembled detector: the generative-HMM core wired end to end.

This module is the integration point of THEORY.md §4-§8 -- where the
leaf pieces (support-relative geometry, per-state emission likelihoods, the
reusable HMM, the make/break event refiner) compose into the single estimator
THEORY.md §8 names: a probabilistic hybrid dynamical system inferred as a
posterior over the active contact mode at every frame.

The pipeline ``detect(obs)`` runs is exactly the pragmatic rung 1-4 ladder of
THEORY.md §10:

  1. **States** = ``ALL_STATES`` (``free`` + the five contact modes of §3).
  2. **A temporal prior** -- a continuous-time Markov jump discretized per median
     ``dt`` (THEORY.md §5): ``P(stay over dt) = exp(-dt/dwell)``, the leftover mass
     split among the other states with a *structure* that mirrors the hybrid
     system's guards (free is the natural gateway; ``impact`` is a short-lived
     transient that bridges free <-> sustained contact).
  3. **EM self-calibration** of the resting-gap bias (THEORY.md §7 & §8): the
     contact-state mean gap, estimated by posterior responsibility, which is the
     principled replacement for the toy script's circular quiet-frame median.
  4. **Smoothed inference**: forward-backward gives the calibrated per-frame
     posterior P(contact) (§5), Viterbi gives the clean contiguous mode
     segmentation (§5), the make/break instants are refined by the event detector
     (§6), and -- if a material stiffness is known -- penetration becomes a
     calibrated force gauge ``lambda = k * delta`` (§7).

It is the only module permitted to import every other ``contact`` submodule.
"""

from __future__ import annotations

import copy

import numpy as np

from . import (
    dynamics,
    emissions,
    events,
    hmm,
    hsmm,
    impacts,
    mode_discovery,
    transitions,
    uncertainty,
)
from .config import DetectorConfig
from .types import (
    ALL_STATES,
    FREE,
    IMPACT,
    ContactEvent,
    ContactImpulse,
    ContactInterval,
    ContactObservations,
    DetectionResult,
    DiscoveredModeResult,
)

__all__ = ["ContactDetector"]


# --------------------------------------------------------------------------------------
# Transition prior (THEORY.md §5: the discrete shadow of the hybrid system's
# guards). The transition structure itself now lives in :mod:`contact.transitions`,
# which builds (a) the time-homogeneous base matrix and (b) the per-frame, gap-*gated*
# tensor in which free->contact entry rises as the gap nears zero (the §5 guard). This
# module only chooses *which* prior to feed the inference and converts it to log-space.
# --------------------------------------------------------------------------------------


def _median_dt(t: np.ndarray) -> float:
    """Median sampling interval of ``t`` (s), used to discretize the CT Markov jump.

    THEORY.md §5: the transition prior is a continuous-time Markov *jump process*
    discretized per frame, so its per-step stay probability depends on the frame
    spacing. We use the median ``dt`` (robust to a stray dropped frame) as the single
    representative step; for the near-uniform clocks here this equals the nominal
    period.
    """
    t = np.asarray(t, dtype=float).ravel()
    if t.shape[0] < 2:
        return 1.0
    dts = np.diff(t)
    dts = dts[dts > 0.0]
    if dts.size == 0:
        return 1.0
    return float(np.median(dts))


# --------------------------------------------------------------------------------------
# Interval extraction (THEORY.md §5: Viterbi gives the clean contiguous
# segmentation that replaces the toy script's morphological cleanup).
# --------------------------------------------------------------------------------------


def _intervals_from_map(
    t: np.ndarray, map_labels: list[str]
) -> list[ContactInterval]:
    """Contiguous non-FREE runs of the MAP path, each tagged with its dominant mode.

    THEORY.md §5: the Viterbi path is already temporally coherent, so a contact
    interval is simply a maximal run of frames whose MAP label is not FREE. The run's
    reported ``mode`` is the most frequent non-FREE label inside it (a run can contain
    e.g. a leading IMPACT then STATIC; the dominant label names the interval).
    """
    t = np.asarray(t, dtype=float).ravel()
    n = len(map_labels)
    intervals: list[ContactInterval] = []
    i = 0
    while i < n:
        if map_labels[i] == FREE:
            i += 1
            continue
        j = i
        while j < n and map_labels[j] != FREE:
            j += 1
        # Dominant (most frequent) non-FREE label across [i, j).
        run = map_labels[i:j]
        counts: dict[str, int] = {}
        for lbl in run:
            counts[lbl] = counts.get(lbl, 0) + 1
        dominant = max(counts, key=lambda k: counts[k])
        intervals.append(
            ContactInterval(t_start=float(t[i]), t_end=float(t[j - 1]), mode=dominant)
        )
        i = j
    return intervals


# --------------------------------------------------------------------------------------
# The detector
# --------------------------------------------------------------------------------------


def _emission_scaled_to_motion(cfg: DetectorConfig, obs) -> DetectorConfig:
    """A copy of ``cfg`` whose sliding/free velocity scales match THIS pair's own motion.

    The package defaults are foot-speed; a body moving well above ``slide_speed`` (a skidding
    box, a struck ball, a sliding deck) would otherwise fall outside the sliding ring and read
    FREE. We size the sliding scale to the 90th-percentile support-relative tangential speed --
    a physically interpretable, data-driven choice -- and never narrow below the defaults (the
    validated slow regime is untouched). Applied universally here so scenarios and scene edges
    are scaled identically.
    """
    vt = np.linalg.norm(np.asarray(obs.v_tangent, dtype=float), axis=1)
    if vt.size == 0:
        return cfg
    scale = float(np.percentile(vt, 90))
    if scale <= cfg.emission.slide_speed:
        return cfg
    c = copy.deepcopy(cfg)
    c.emission.slide_speed = scale
    c.emission.free_vel_sigma = max(c.emission.free_vel_sigma, 2.0 * scale)
    return c


class ContactDetector:
    """Infer the per-frame contact state from support-relative observations.

    Wraps the whole generative-HMM estimator of THEORY.md §4-§8 behind a single
    :meth:`detect` call. The pipeline now exercises the full §5-§7 upgrade stack:

      * **Gap-gated transitions (§5).** Instead of a fixed transition matrix the prior
        is the per-frame ``(T, S, S)`` tensor of :func:`contact.transitions.gated_transition_tensor`,
        whose free->contact entry mass *rises as the gap nears zero* -- the hybrid
        system's state-dependent make guard. forward-backward (and the EM bias loop)
        consume this gated tensor.
      * **Semi-Markov decoding (§5).** When ``config.transition.use_semi_markov`` is
        set, the MAP path is decoded with the explicit-duration
        :func:`contact.hsmm.hsmm_viterbi` (per-state mean dwell in *frames*), whose
        duration prior makes 1-frame blips intrinsically improbable -- the principled
        replacement for a hard minimum-contact-duration. Otherwise the plain
        :func:`contact.hmm.viterbi` runs on the gated tensor. The *per-frame posterior*
        always comes from forward-backward on the gated tensor.
      * **Impact atoms (§6).** :func:`contact.impacts.detect_impacts` characterizes the
        velocity-step atoms of the force measure (closing speed / restitution /
        impulse), reported in ``DetectionResult.impulses`` alongside the make/break
        events.
      * **Dynamics & material (§7).** Given a material stiffness, penetration becomes a
        calibrated force gauge via :func:`contact.dynamics.normal_force_from_penetration`,
        and :func:`contact.dynamics.friction_stick_slip` labels each contact frame
        stick/slip in ``DetectionResult.slip_state``.
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        """Store the configuration bundle (a fresh :class:`DetectorConfig` by default)."""
        self.config = config if config is not None else DetectorConfig()

    def detect(self, obs: ContactObservations) -> DetectionResult:
        """Run the full detection pipeline on one body-pair's observations.

        Parameters
        ----------
        obs:
            Per-frame support-relative observations (``contact.types.ContactObservations``),
            typically produced by :func:`contact.geometry.observe`.

        Returns
        -------
        DetectionResult
            The fully-populated result: calibrated contact posterior, full state
            posterior, MAP mode labels (plain-HMM or semi-Markov), the derived boolean
            contact mask, contact intervals with dominant modes, make/break events, the
            characterized impact atoms (§6), the EM-recovered resting bias, and -- if
            ``config.material.stiffness`` is set -- the estimated normal force per frame
            and the per-frame stick/slip labels (§7).
        """
        cfg = _emission_scaled_to_motion(self.config, obs)
        states = list(ALL_STATES)  # (a) FREE + the five contact modes
        contact_state_idx = [i for i, s in enumerate(states) if s != FREE]

        t = np.asarray(obs.t, dtype=float).ravel()
        gap = np.asarray(obs.gap, dtype=float).ravel()
        S = len(states)
        free_idx = states.index(FREE)

        # --- (b) Transition prior (THEORY.md §5). --------------------------------------
        # Two log-space objects are built once and reused throughout:
        #   * log_trans_gated : the per-frame (T, S, S) gap-GATED tensor -- free->contact
        #     entry rises as gap -> 0 (the state-dependent make guard, §5). It drives the
        #     EM bias loop and the final forward-backward smoothing, where the gate's
        #     frame-by-frame shaping of the entry probability is exactly what we want.
        #   * log_trans_base  : the time-homogeneous (S, S) base matrix -- the gate-free
        #     prior the semi-Markov decoder requires (the HSMM owns persistence in its
        #     duration model and only accepts a homogeneous (S, S); the gap gate, which
        #     mainly *shapes entry*, is instead carried by the gated tensor's posterior
        #     and by the emissions, which already pin touchdown to gap ~ 0).
        dt = _median_dt(t)
        gated = transitions.gated_transition_tensor(obs, states, dt, cfg.transition)
        log_trans_gated = np.log(gated)  # gated is strictly positive (§4).
        base = transitions.base_transition_matrix(states, dt, cfg.transition)
        log_trans_base = np.log(base)

        # Initial-state prior: most trajectories begin free (a body approaching, or
        # already resting). We use a gently FREE-favouring prior rather than a hard one
        # so a recording that *starts* mid-contact is not fought by the prior; the
        # emissions dominate after the first frame anyway. (THEORY.md §5.)
        init = np.full(S, (1.0 - 0.5) / (S - 1), dtype=float)
        init[free_idx] = 0.5
        log_init = np.log(init)

        # The per-frame smoother: an HMM over the modes whose prior is the gap-gated guard.
        # Built once and reused for the EM responsibilities and the final posterior (§5) --
        # the emissions are the only thing that changes between calls.
        smoother = hmm.HMM(log_trans_gated, log_init)

        # --- (b') Per-frame measurement-uncertainty tempering (THEORY.md §8). ----------
        # OFF by default and a strict no-op unless BOTH the flag is set AND the
        # observations carry a per-frame measurement covariance. When enabled we compute
        # one tempering weight per frame, w(t) in (0, 1], from obs.meas_cov; a clean frame
        # has w ~ 1 (untouched) and a badly occluded frame has w -> 0, which flattens that
        # frame's emission row so the temporal prior (§5) -- not the corrupted
        # measurement -- carries the state. The weight is folded into EVERY assembled
        # emission matrix below (the EM loop AND the final pass) via uncertainty.apply_tempering.
        # With the flag off (or meas_cov None) `temper_w` is None and the emission matrix is
        # used byte-for-byte as before -- the existing behaviour is exactly preserved.
        #
        # We pass cfg.EMISSION (not cfg.inference): emission_tempering needs the
        # representative channel NOISE SCALE that the measurement variance competes
        # against, which lives on EmissionParams (vel_sigma). Passing cfg.inference, which
        # carries no such scale, would silently fall back to a unit base noise (sigma=1)
        # and leave the feature inert at the system's scale (vel_sigma=0.05): a realistic
        # occlusion variance would barely flatten anything. (use_uncertainty -- the flag --
        # still lives on cfg.inference; only the scale comes from cfg.emission.)
        temper_w = None
        if getattr(cfg.inference, "use_uncertainty", False) and obs.meas_cov is not None:
            temper_w = uncertainty.emission_tempering(obs, cfg.emission)

        # --- (c) EM self-calibration of the resting-gap bias (THEORY.md §7 & §8). The
        # principled replacement for the toy script's circular quiet-frame median -- the
        # contact responsibilities that weight the bias come from the model itself.
        gap_bias = self._calibrate_gap_bias(
            obs, cfg, states, gap, smoother, temper_w, contact_state_idx
        )

        # --- (d) Final smoothed inference with the calibrated bias.
        log_em = emissions.log_emissions(
            obs, cfg.emission, gap_bias, states, cfg.material, force=cfg.force
        )
        if temper_w is not None:
            log_em = uncertainty.apply_tempering(log_em, temper_w)
        # Per-frame posterior ALWAYS from forward-backward on the gated tensor, whether
        # or not we decode the MAP path with the semi-Markov model (the gated guard is
        # exactly the per-frame prior we want for the calibrated P(contact)).
        gamma, _loglik = smoother.posterior(log_em)
        # Calibrated per-frame contact posterior = 1 - P(free) (THEORY.md §4/§5).
        contact_posterior = 1.0 - gamma[:, free_idx]

        # MAP segmentation (THEORY.md §5): the single most likely contiguous mode
        # sequence. Either the explicit-duration semi-Markov decoder (blips suppressed by
        # the duration prior, not by morphology) or the plain Viterbi on the gated tensor.
        if cfg.transition.use_semi_markov:
            # Per-state mean dwell in FRAMES: tau / dt for every state, with IMPACT's
            # shorter transient dwell (§6). dt is the median frame period; guard dt > 0.
            dt_safe = max(dt, 1e-9)
            mean_dwell_frames = np.full(
                S, float(cfg.transition.mean_dwell_time) / dt_safe, dtype=float
            )
            if IMPACT in states:
                mean_dwell_frames[states.index(IMPACT)] = (
                    float(cfg.transition.impact_dwell_time) / dt_safe
                )
            # The HSMM owns persistence in its duration model, so it takes the *base*
            # (homogeneous) matrix -- a time-varying guard would change meaning under the
            # segmental factorization (see contact.hsmm). The gap gate still shapes
            # touchdown via the emissions (gap ~ 0) and the reported posterior above.
            decoder = hsmm.SemiMarkovHMM(
                log_trans_base,
                log_init,
                mean_dwell_frames,
                concentration=float(cfg.transition.dwell_concentration),
            )
            path = decoder.map_path(log_em)
        else:
            path = smoother.map_path(log_em)
        map_state = [states[int(s)] for s in path]
        in_contact = np.array([s != FREE for s in map_state], dtype=bool)
        intervals = _intervals_from_map(t, map_state)

        # --- (e) Make/break event refinement (THEORY.md §6).
        # Touchdown/liftoff events come from the MAP boolean mask (the contact-state onset
        # of §5); the impact ATOMS are the finer-timescale characterization of the make
        # instants (closing speed, restitution, impulse). They are complementary: events
        # mark *that* a transition happened, impulses mark *how hard* it hit (§6). Mass
        # is unknown from the observable channel here, so impulse magnitudes are NaN (§7,
        # the force-as-measure atom is unobservable from kinematics alone without mass).
        ev: list[ContactEvent] = events.detect_events(obs, in_contact, t=t)
        impulses: list[ContactImpulse] = impacts.detect_impacts(obs, cfg.impact, mass=None)

        # --- (f) Dynamics & material (THEORY.md §7). -----------------------------------
        # Under known compliance the penetration depth is a calibrated force gauge,
        # lambda = k * delta, delta measured relative to the resting bias (so the resting
        # offset is not mistaken for a load) and zeroed off the Viterbi contact mask
        # (Signorini, §2). With the force in hand the friction cone labels each contact
        # frame stick/slip; off-contact frames get "". When stiffness is unknown the force
        # is unobservable, so normal_force stays None and the slip labels fall back to the
        # always-available kinematic (||v_t|| vs slip_speed_threshold) rule, masked to the
        # detector's own contact frames so we report friction state only where we believe
        # a contact exists (we do NOT let the cone fight the HMM/HSMM MAP -- the motion is
        # the hard evidence; the cone only refines a borderline stick, per §7).
        if cfg.material.stiffness is not None:
            nf = dynamics.normal_force_from_penetration(
                gap, gap_bias, in_contact, cfg.material
            )
            normal_force = np.asarray(nf, dtype=float)
            slip = dynamics.friction_stick_slip(obs, normal_force, cfg.material)
        else:
            normal_force = None
            # No force gauge: friction_stick_slip returns the kinematic stick/slip label
            # for every frame; we mask it to the contact frames the MAP path believes in
            # so off-contact frames read "" (no friction state without a contact, §2).
            kin = dynamics.friction_stick_slip(
                obs, np.full(t.shape[0], np.nan, dtype=float), cfg.material
            )
            slip = [
                kin[i] if in_contact[i] else "" for i in range(len(kin))
            ]

        return DetectionResult(
            t=t,
            contact_posterior=np.asarray(contact_posterior, dtype=float),
            state_posterior=np.asarray(gamma, dtype=float),
            map_state=map_state,
            in_contact=in_contact,
            intervals=intervals,
            events=ev,
            resting_bias=float(gap_bias),
            normal_force=normal_force,
            states=states,
            impulses=impulses,
            slip_state=slip,
        )

    @staticmethod
    def _calibrate_gap_bias(obs, cfg, states, gap, smoother, temper_w, contact_state_idx):
        """EM self-calibration of the resting-gap bias (THEORY.md §7/§8).

        Each EM step: (E) the smoothed posterior under the current bias (the gated guard in
        ``smoother`` informs the responsibilities); (M) re-estimate the bias as the
        posterior-contact-weighted mean of the observed gap, clipped to the plausible band.
        Returns the calibrated bias (0.0 if there is essentially no contact mass to weight).
        """
        max_bias = abs(float(cfg.calibration.max_resting_bias))
        gap_bias = 0.0
        for _ in range(max(0, int(cfg.calibration.em_iters))):
            log_em = emissions.log_emissions(
                obs, cfg.emission, gap_bias, states, cfg.material, force=cfg.force
            )
            if temper_w is not None:
                log_em = uncertainty.apply_tempering(log_em, temper_w)
            gamma, _ = smoother.posterior(log_em)
            w = gamma[:, contact_state_idx].sum(axis=1)  # contact responsibilities
            wsum = float(w.sum())
            if wsum > 1e-12:  # else: no contact mass -> leave the bias unchanged
                gap_bias = float(np.clip(np.dot(w, gap) / wsum, -max_bias, max_bias))
        return gap_bias

    def discover_modes(
        self, obs: ContactObservations, seed: int = 0
    ) -> DiscoveredModeResult:
        """Discover the contact-mode vocabulary from data, label-free (THEORY.md §8).

        A thin entrypoint onto :func:`contact.mode_discovery.discover_modes`: instead of
        presupposing the canonical five modes of §3, fit a truncated (weak-limit) sticky
        HDP-HMM to the per-frame twist feature and let the data say *how many* distinct
        contact regimes the clip contains and which frames belong to each (THEORY.md §8:
        "discover the mode vocabulary from data instead of presupposing it"; scaling
        beyond a fixed enumeration). The HDP truncation level / concentration / stickiness
        come from this detector's :class:`~contact.config.InferenceParams`
        (``self.config.inference``), so the same config bundle that drives the supervised
        :meth:`detect` also governs discovery.

        This is **opt-in research surface** -- it is never invoked by :meth:`detect` and
        does not touch the supervised pipeline. The returned ``alignment`` (a nearest
        canonical-signature heuristic) is for validation/reporting only and plays no part
        in the fit; the discovery itself is unsupervised.

        Parameters
        ----------
        obs:
            Per-frame support-relative observations (length ``T``), as
            :meth:`detect` consumes.
        seed:
            RNG seed for the Gibbs sampler (default ``0`` => deterministic output).

        Returns
        -------
        DiscoveredModeResult
            The discovered per-frame ``labels``, the count ``n_modes`` of populated modes,
            each mode's mean ``signatures`` (raw physical-unit twist feature), and the
            validation-only ``alignment`` to canonical mode names.
        """
        return mode_discovery.discover_modes(obs, self.config.inference, seed=seed)
