"""Unsupervised contact-mode discovery via a sticky HDP-HMM (THEORY.md section 8).

THEORY.md section 8 makes a research-frontier demand that the rest of the package
quietly sidesteps: the five canonical modes of section 3 (free / static / sliding /
pivoting / rolling, plus impact) are *presupposed* by ``contact.emissions`` and
``contact.types.ALL_STATES``. But section 8 wants the estimator to "*discover* the
mode vocabulary from data instead of presupposing it" -- both to scale beyond a fixed
enumeration and to be honest when a recording contains a regime the hand-built five do
not name. That is precisely what a *Bayesian nonparametric* hidden Markov model buys:
the number of states is itself inferred, growing only as the data demand.

The object section 8 names for this is the **sticky HDP-HMM** (Fox, Sudderth, Jordan &
Willsky, 2008):

  * a **Hierarchical Dirichlet Process** ties the per-state transition distributions to
    a shared, *infinite* menu of states, so the model can use as many or as few states
    as the data support -- the DP concentration ``hdp_concentration`` controls how
    readily a fresh state is spawned;
  * a **sticky** self-transition bias ``hdp_stickiness`` (the ``kappa`` of Fox et al.)
    adds prior mass to staying put, which is the nonparametric analogue of the dwell
    prior of THEORY.md section 5 -- without it an HDP-HMM notoriously fragments a single
    physical regime into many rapidly-alternating redundant states (it "flickers").

This module is honest about being a *tractable approximation* of that object; see the
``discover_modes`` docstring for the precise inference scheme and its documented limits.

Relationship to the rest of the package
---------------------------------------
This is a *label-free* alternative to the supervised emission bank of
``contact.emissions``. It never consumes ``ALL_STATES`` or the canonical emission
builders; it clusters the raw per-frame twist feature directly. The only place the
canonical names reappear is :func:`_align_signature` -- a **validation-only** heuristic
that names each discovered state by its nearest canonical signature, so a test can
check "did the unsupervised model rediscover sliding where we planted sliding?". The
discovery itself does not use, need, or trust those names.

Backward compatibility
-----------------------
Nothing here is invoked by the default detection path; it is opt-in research code. It
imports only :mod:`contact.types`, :mod:`contact.config`, :mod:`markovlib` (whose
:func:`~markovlib.sample_path` draws the label path), and numpy.

Public API
----------
* :func:`mode_feature_vector` -- ``obs -> (T, 5)`` raw per-frame twist feature.
* :func:`discover_modes`      -- ``(obs, params, seed) -> DiscoveredModeResult``.
"""

from __future__ import annotations

import markovlib as _markovlib
import numpy as np

from .config import InferenceParams
from .types import (
    FREE,
    IMPACT,
    PIVOTING,
    ROLLING,
    SLIDING,
    STATIC,
    ContactObservations,
    DiscoveredModeResult,
)

__all__ = ["mode_feature_vector", "discover_modes"]


# --------------------------------------------------------------------------------------
# The per-frame feature (THEORY.md section 3: a contact mode is a pattern in the twist).
# --------------------------------------------------------------------------------------
#
# The canonical emissions (contact.emissions) keep the *full* twist with its 2-D
# tangential vectors so that direction-aware modes (the ring density for sliding, the
# rolling coupling) are representable. For *clustering*, however, what distinguishes the
# physical regimes of section 3 is the set of channel MAGNITUDES that are excited:
#
#     gap            -- separation vs. resting (free vs. any contact)
#     |v_normal|     -- making/breaking contact (impact)
#     |v_tangent|    -- sliding
#     |omega_normal| -- pivoting / twisting
#     |omega_tangent|-- rolling axis
#
# so the feature is the 5-vector of those magnitudes (gap kept signed -- its sign is the
# Signorini branch and is informative). This is exactly the small signature stored in
# DiscoveredModeResult.signatures, which keeps discovery and reporting in one space.


def mode_feature_vector(obs: ContactObservations) -> np.ndarray:
    """Per-frame twist feature ``[gap, |v_n|, |v_t|, |omega_n|, |omega_t|]`` (``(T, 5)``).

    THEORY.md section 3: the contact mode is the pattern of *which twist channels are
    excited*, so the clustering feature is the vector of channel magnitudes (the gap is
    kept signed because its sign is the Signorini separation/penetration branch of
    section 2 and carries the free-vs-contact distinction). The 2-D tangential channels
    are reduced to their Euclidean norm: for discovering *which kind of motion* a regime
    is, the tangential *direction* is uninformative (it is exactly the nuisance the
    section-4 ring/mixture densities integrate over), while the *speed* is the signal.

    The raw feature is returned in physical units; per-channel standardization (so that
    a 1 cm gap and a 1 rad/s spin are commensurate before clustering) is applied inside
    :func:`discover_modes`, not here, so callers can inspect the feature in real units
    and so the stored signatures stay physically interpretable.

    Parameters
    ----------
    obs:
        Per-frame support-relative observations (length ``T``).

    Returns
    -------
    np.ndarray
        ``(T, 5)`` float array; columns ``[gap, |v_normal|, |v_tangent|, |omega_normal|,
        |omega_tangent|]``.
    """
    gap = np.asarray(obs.gap, dtype=float).ravel()
    v_n = np.abs(np.asarray(obs.v_normal, dtype=float).ravel())
    v_t = np.linalg.norm(np.atleast_2d(np.asarray(obs.v_tangent, dtype=float)), axis=-1)
    w_n = np.abs(np.asarray(obs.omega_normal, dtype=float).ravel())
    w_t = np.linalg.norm(np.atleast_2d(np.asarray(obs.omega_tangent, dtype=float)), axis=-1)
    return np.stack([gap, v_n, v_t, w_n, w_t], axis=-1)


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel standardize ``(x - mean) / scale``; return ``(z, mean, scale)``.

    Channels of the feature live in incommensurate units (metres, m/s, rad/s), so a
    raw Euclidean/Gaussian model would let whichever channel happens to have the largest
    numeric spread dominate the clustering. We z-score each channel to a common scale.
    A floor on the scale (a small fraction of the largest channel scale, with an
    absolute floor) keeps a channel that is essentially constant across the clip from
    blowing up tiny numerical jitter into apparent structure.
    """
    x = np.asarray(x, dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    # Floor near-constant channels: avoid dividing by ~0 and amplifying jitter.
    floor = max(1e-9, 1e-3 * float(np.max(scale)) if scale.size else 1e-9)
    scale = np.where(scale < floor, floor, scale)
    return (x - mean) / scale, mean, scale


# --------------------------------------------------------------------------------------
# Sticky HDP-HMM, weak-limit (truncated) approximation, blocked Gibbs sampling.
# --------------------------------------------------------------------------------------
#
# We use the *weak-limit* approximation of the HDP (Fox et al. 2008, sec. VI-B): the
# infinite menu of states is truncated to L = params.max_modes, with a finite Dirichlet
# whose total concentration equals the DP concentration approximating the top-level DP.
# This is the standard, tractable surrogate for the full HDP and converges to it as
# L -> infinity; it lets us run an exact-per-step blocked Gibbs sampler with the HMM
# forward-backward machinery already in the package.
#
# Generative model (truncated):
#     beta ~ Dir(gamma_conc / L, ..., gamma_conc / L)            top-level state menu
#     pi_k ~ Dir(alpha_conc * beta + kappa * e_k)  for k=1..L    sticky rows
#     theta_k = (mu_k, var_k)  Gaussian emission params per state (diagonal cov)
#     z_t | z_{t-1} ~ pi_{z_{t-1}};  x_t | z_t ~ N(mu_{z_t}, var_{z_t})
#
# kappa is the sticky bias added to the self-transition (THEORY.md s.5's dwell prior in
# nonparametric clothing). alpha_conc spreads the rows over the shared menu; gamma_conc
# governs how concentrated the menu itself is (how readily a *new* state earns mass).
#
# Inference: a partially-collapsed blocked Gibbs sweep. Each sweep:
#   (1) sample the whole label sequence z_{1:T} jointly given (pi, theta) by
#       forward-FILTER / backward-SAMPLE (the stochastic analogue of forward-backward,
#       reusing the package's log-space alpha recursion);
#   (2) set the transition rows pi_k to the sticky prior MEAN
#       (alpha*beta + kappa*e_k)/Z (see _sticky_rows) -- a deterministic-given-beta
#       choice rather than a fresh Dirichlet draw from transition counts. The data enter
#       the transitions through beta (step 3), not through a per-row count posterior;
#   (3) resample beta from its Dirichlet posterior (Antoniak/weak-limit: counts of
#       states used + gamma_conc/L); this is the standard weak-limit simplification and
#       is the sole stochastic, data-driven update of the transition structure;
#   (4) resample theta_k (conjugate Normal-Inverse-Gamma posterior, per channel).
# We keep the single MAP sample (highest joint log-prob over the sweeps) as the result.


def _gaussian_emission_loglik(
    z: np.ndarray, mu: np.ndarray, var: np.ndarray
) -> np.ndarray:
    """``(T, L)`` log N(z_t; mu_k, diag(var_k)) for standardized features ``z``.

    Diagonal-covariance Gaussian emissions per discovered state (THEORY.md s.4 keeps the
    twist's cross-channel *correlations* for the supervised modes; here, on the reduced
    *magnitude* feature, a diagonal Gaussian per state is the tractable, standard HDP-HMM
    emission and is sufficient to separate the section-3 regimes, which differ in *which*
    magnitude channels are large).
    """
    z = np.asarray(z, dtype=float)            # (T, D)
    mu = np.asarray(mu, dtype=float)          # (L, D)
    var = np.asarray(var, dtype=float)        # (L, D)
    # (T, L, D): squared standardized residual per channel.
    diff = z[:, None, :] - mu[None, :, :]
    quad = np.sum(diff * diff / var[None, :, :], axis=-1)          # (T, L)
    log_norm = np.sum(np.log(2.0 * np.pi * var), axis=-1)          # (L,)
    return -0.5 * (quad + log_norm[None, :])


def _dirichlet(alpha: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Dirichlet sample via independent Gammas (with a tiny floor on the shapes)."""
    a = np.maximum(np.asarray(alpha, dtype=float), 1e-6)
    g = rng.gamma(a)
    s = g.sum()
    return g / s if s > 0 else np.full_like(a, 1.0 / a.size)


def discover_modes(
    obs: ContactObservations,
    params: InferenceParams,
    seed: int = 0,
) -> DiscoveredModeResult:
    """Discover the contact-mode vocabulary from data with a sticky HDP-HMM (s.8).

    Fits a **truncated (weak-limit) sticky HDP-HMM** to the per-frame twist feature of
    :func:`mode_feature_vector`, learning *how many* modes the clip needs and which
    frames belong to each -- without ever being told the canonical mode list (THEORY.md
    section 8: discover the vocabulary instead of presupposing it).

    Model (truncation ``L = params.max_modes``)
        ``beta  ~ Dir(gamma/L , ... )`` -- a shared top-level menu over the ``L`` states,
        ``gamma = params.hdp_concentration`` (smaller => fewer states earn mass);
        ``pi_k  ~ Dir(alpha * beta + kappa * e_k)`` -- per-state transition row with the
        **sticky** self-bias ``kappa = params.hdp_stickiness`` (Fox et al. 2008), which
        is the nonparametric dwell prior of THEORY.md section 5 and is what keeps the
        labels piecewise-constant rather than flickering;
        emissions are per-state **diagonal Gaussians** on the *standardized* feature.

    Inference -- **partially-collapsed blocked Gibbs sampling** (this is the documented
    approximation):
        each sweep (a) jointly samples the label path by forward-filter/backward-sample
        (markovlib's :func:`~markovlib.sample_path`), (b) sets the sticky
        transition rows to their prior *mean* ``(alpha*beta + kappa*e_k)/Z`` --
        deterministic given ``beta`` (see :func:`_sticky_rows`), *not* a fresh Dirichlet
        draw from transition counts -- so the data reach the transitions only through
        ``beta``, (c) resamples the shared menu ``beta`` from the per-state usage counts
        (the sole stochastic, data-driven update of the transition structure), and (d)
        resamples the conjugate Normal/Inverse-Gamma emission parameters. We run a short
        burn-in then keep the **single highest-joint-log-probability sample** (a MAP-style
        point estimate) as the returned labelling. The sampler is seeded (``seed``), so
        the result is deterministic for a given input and seed.

    Honest limits of this approximation
        * It is the *weak-limit* truncation of the HDP, not the full infinite model: it
          can use at most ``L`` states and only *approaches* the true HDP as ``L`` grows.
          ``n_modes`` is the number of states the MAP sample actually populates, which is
          typically well below ``L``.
        * It returns a point estimate (best Gibbs sample), not the full posterior over
          segmentations; mixing of a Gibbs sampler over discrete states is not guaranteed,
          so on hard/ambiguous clips the answer can depend on ``seed`` and iteration count.
        * The emission is a diagonal Gaussian on channel *magnitudes*: it cannot, by
          construction, represent the rolling *coupling* (``|v_t| = r|omega_t|``) as a
          first-class constraint the way ``contact.emissions.rolling_logpdf`` does -- it
          can only notice that both tangential channels are jointly excited. Rolling
          alignment is therefore the weakest of the validation labels.
        * It is unsupervised and label-free; the canonical-name ``alignment`` is a
          post-hoc nearest-signature heuristic for *validation only* and plays no part in
          the fit.

    Parameters
    ----------
    obs:
        Per-frame support-relative observations (length ``T``).
    params:
        Research-frontier inference knobs (``contact.config.InferenceParams``): uses
        ``max_modes`` (truncation ``L``), ``hdp_concentration`` (``gamma``), and
        ``hdp_stickiness`` (``kappa``).
    seed:
        RNG seed for the Gibbs sampler. Default ``0`` => deterministic output.

    Returns
    -------
    DiscoveredModeResult
        ``labels`` ``(T,)`` int state id per frame (remapped to ``0..n_modes-1`` in order
        of first appearance), ``n_modes`` the count of populated states, ``signatures``
        ``{id: mean raw feature}`` (physical units), and ``alignment`` ``{id: canonical
        name}`` from :func:`_align_signature` (validation only).
    """
    rng = np.random.default_rng(int(seed))

    feat = mode_feature_vector(obs)                 # (T, 5) raw, physical units
    T, D = feat.shape
    z_std, _f_mean, _f_scale = _standardize(feat)   # standardized for clustering

    L = max(1, int(params.max_modes))
    gamma_conc = float(params.hdp_concentration)
    kappa = float(params.hdp_stickiness)
    # alpha: how strongly each row is tied to the shared menu. A modest value lets the
    # data shape transitions while keeping the HDP coupling; tie it to gamma so a single
    # knob (hdp_concentration) scales the whole prior sensibly.
    alpha = max(1.0, gamma_conc)

    if T == 1:
        # Degenerate clip: one frame is one mode. Skip the sampler.
        sig = {0: feat[0].copy()}
        return DiscoveredModeResult(
            labels=np.zeros(1, dtype=int),
            n_modes=1,
            signatures=sig,
            alignment={0: _align_signature(feat[0])},
        )

    # --- Emission hyperpriors (Normal-Inverse-Gamma, per channel, on standardized z). ---
    # Standardized features have ~unit global variance, so a unit-scale prior is natural.
    mu0 = np.zeros(D)            # prior mean (z-scored => 0)
    kappa0 = 0.1                 # prior strength on the mean (weak)
    a0 = 2.0                     # inverse-gamma shape (finite variance prior)
    b0 = 1.0                     # inverse-gamma scale (~unit variance)

    # --- Initialize via a simple k-means-ish seeding so Gibbs starts from a sane point.
    mu, var = _init_emissions(z_std, L, rng, b0 / (a0 - 1.0))
    beta = np.full(L, 1.0 / L)
    log_init = np.log(np.full(L, 1.0 / L))

    n_sweeps = 60
    n_burn = 30
    best_logjoint = -np.inf
    best_z = np.zeros(T, dtype=np.intp)

    for sweep in range(n_sweeps):
        # (1) Sample the label path given current emissions + sticky transitions.
        log_em = _gaussian_emission_loglik(z_std, mu, var)        # (T, L)
        pi = _sticky_rows(beta, alpha, kappa, L)                  # (L, L) rows sum to 1
        log_trans = np.log(pi)
        chain = _markovlib.DiscreteChain(log_init=log_init, log_trans=log_trans)
        z = _markovlib.sample_path(chain, log_em, rng=rng)

        # (2) Resample the shared menu beta from the per-state usage counts. The sticky
        # transition rows themselves are NOT resampled from a Dirichlet posterior: they
        # are set to the sticky prior MEAN (deterministic given beta) by _sticky_rows in
        # step (1) above, so beta -- driven here by the usage counts -- is what carries
        # the data into the transitions. (No transition-count statistic is needed for the
        # prior-mean row, so none is accumulated.)
        usage = np.bincount(z, minlength=L).astype(float)
        # (3) Weak-limit beta posterior: menu mass ~ Dir(gamma/L + usage). This is the
        # tractable surrogate for the Antoniak table-count update of the full HDP.
        beta = _dirichlet(gamma_conc / L + usage, rng)

        # (4) Resample emissions from the conjugate NIG posterior per populated state.
        mu, var = _resample_emissions(
            z_std, z, L, mu0, kappa0, a0, b0, rng
        )

        # Track the best (MAP-ish) sample by joint log-probability after burn-in.
        if sweep >= n_burn:
            lj = _joint_logprob(z_std, z, mu, var, beta, alpha, kappa, L, log_init)
            if lj > best_logjoint:
                best_logjoint = lj
                best_z = z.copy()

    # --- Compact the labels: keep only populated states, relabel 0..n-1 by first use.
    labels, signatures = _compact(best_z, feat)
    n_modes = len(signatures)
    alignment = {k: _align_signature(v) for k, v in signatures.items()}
    return DiscoveredModeResult(
        labels=labels,
        n_modes=n_modes,
        signatures=signatures,
        alignment=alignment,
    )


# --------------------------------------------------------------------------------------
# Gibbs-sweep helpers.
# --------------------------------------------------------------------------------------


def _sticky_rows(beta: np.ndarray, alpha: float, kappa: float, L: int) -> np.ndarray:
    """Sticky HDP transition rows ``pi_k = (alpha*beta + kappa*e_k)`` normalized.

    The prior mean of each transition row under the sticky HDP (Fox et al. 2008): the
    shared menu ``beta`` scaled by ``alpha`` plus a self-transition spike ``kappa`` on the
    diagonal. Using the prior *mean* (rather than a fresh Dirichlet draw) as the working
    transition matrix in the forward-filter is a light, deterministic-given-beta choice
    that keeps the sticky persistence while letting ``beta`` (resampled each sweep) and
    the data-driven labels carry the stochasticity. Each row is renormalized to sum to 1.
    """
    base = alpha * beta[None, :] + kappa * np.eye(L)
    return base / base.sum(axis=1, keepdims=True)


def _init_emissions(
    z: np.ndarray, L: int, rng: np.random.Generator, var0: float
) -> tuple[np.ndarray, np.ndarray]:
    """Seed emission means by farthest-point sampling; unit-ish variances.

    A k-means++-style spread of initial means over the observed standardized features so
    distinct regimes are separated from the first sweep (Gibbs over discrete states mixes
    poorly from a degenerate start). Pure-numpy, seeded by ``rng``.
    """
    T, D = z.shape
    idx = [int(rng.integers(T))]
    for _ in range(1, L):
        d2 = np.min(
            np.stack([np.sum((z - z[j]) ** 2, axis=1) for j in idx], axis=0), axis=0
        )
        total = float(d2.sum())
        if total <= 0:
            idx.append(int(rng.integers(T)))
            continue
        probs = d2 / total
        idx.append(int(rng.choice(T, p=probs)))
    mu = z[np.array(idx)].copy()
    var = np.full((L, D), max(var0, 1e-3), dtype=float)
    return mu, var


def _resample_emissions(
    z_std: np.ndarray,
    z: np.ndarray,
    L: int,
    mu0: np.ndarray,
    kappa0: float,
    a0: float,
    b0: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample per-state Gaussian (mu, var) from the conjugate Normal-Inverse-Gamma.

    Standard NIG update per channel (diagonal covariance): for the frames assigned to a
    state, the posterior mean shrinks the sample mean toward ``mu0`` by ``kappa0``, and
    the variance is an inverse-gamma whose scale accumulates the within-state spread plus
    the prior. Empty states fall back to a draw from the prior so they remain available
    for the sampler to repopulate (this is what lets the truncated model *grow* usage).
    """
    D = z_std.shape[1]
    mu = np.empty((L, D), dtype=float)
    var = np.empty((L, D), dtype=float)
    for k in range(L):
        mask = z == k
        n = int(mask.sum())
        if n == 0:
            # Prior draw: keeps the state alive but data-agnostic.
            var[k] = 1.0 / rng.gamma(a0, 1.0 / b0, size=D)
            mu[k] = mu0 + np.sqrt(var[k] / kappa0) * rng.standard_normal(D)
            continue
        xk = z_std[mask]
        xbar = xk.mean(axis=0)
        kappa_n = kappa0 + n
        mu_n = (kappa0 * mu0 + n * xbar) / kappa_n
        a_n = a0 + 0.5 * n
        ss = np.sum((xk - xbar) ** 2, axis=0)
        b_n = b0 + 0.5 * ss + 0.5 * (kappa0 * n / kappa_n) * (xbar - mu0) ** 2
        var[k] = 1.0 / rng.gamma(a_n, 1.0 / np.maximum(b_n, 1e-12), size=D)
        var[k] = np.maximum(var[k], 1e-4)  # floor: no zero-width clusters
        mu[k] = mu_n + np.sqrt(var[k] / kappa_n) * rng.standard_normal(D)
    return mu, var


def _joint_logprob(
    z_std: np.ndarray,
    z: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
    beta: np.ndarray,
    alpha: float,
    kappa: float,
    L: int,
    log_init: np.ndarray,
) -> float:
    """Joint log p(x, z | theta, pi) for the current sample (for MAP selection).

    Emission term (Gaussian on the standardized feature) plus the transition term under
    the sticky-row matrix; used only to pick the best Gibbs sample, so an unnormalized
    additive constant common to all samples is harmless and omitted where convenient.
    """
    log_em = _gaussian_emission_loglik(z_std, mu, var)            # (T, L)
    pi = _sticky_rows(beta, alpha, kappa, L)
    log_trans = np.log(pi)
    lj = float(log_init[z[0]] + log_em[0, z[0]])
    for t in range(1, len(z)):
        lj += float(log_trans[z[t - 1], z[t]] + log_em[t, z[t]])
    return lj


def _compact(
    z: np.ndarray, feat: np.ndarray
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Relabel a sampled path to ``0..n-1`` by first appearance; build raw signatures.

    Only the states actually used in the MAP path survive (this is the model's discovered
    ``n_modes``). Each surviving state's *signature* is the mean of the **raw** (physical-
    unit) feature over its frames -- so signatures are directly interpretable and feed
    both ``DiscoveredModeResult.signatures`` and the canonical-name alignment.
    """
    z = np.asarray(z, dtype=int)
    order: list[int] = []
    remap: dict[int, int] = {}
    for s in z:
        if s not in remap:
            remap[s] = len(order)
            order.append(int(s))
    labels = np.array([remap[s] for s in z], dtype=int)
    signatures: dict[int, np.ndarray] = {}
    for old, new in remap.items():
        signatures[new] = feat[z == old].mean(axis=0)
    return labels, signatures


# --------------------------------------------------------------------------------------
# Validation-only alignment (THEORY.md section 3 canonical signatures).
# --------------------------------------------------------------------------------------
#
# IMPORTANT: this is *not* part of discovery. Discovery is label-free. This routine
# exists so a validation harness can ask "which canonical regime does this discovered
# state most resemble?" by matching its mean signature against the section-3 archetypes.
# It is a deterministic rule-based classifier on the 5-vector signature, not a learned
# component, and it never influences the fitted labels.


def _align_signature(
    sig: np.ndarray,
    *,
    gap_quiet: float = 0.01,
    vel_quiet: float = 0.05,
    omega_quiet: float = 0.30,
    roll_radius: float = 0.05,
    impact_speed: float = 0.30,
) -> str:
    """Name a discovered state's mean signature by its nearest canonical mode (s.3).

    A rule-based reading of the signature ``[gap, |v_n|, |v_t|, |omega_n|, |omega_t|]``
    against the THEORY.md section-3 archetypes, applied **for validation only**:

      * **FREE**     -- the gap is clearly open (separation) and the motion is quiet, or
                        equivalently the frame is well above the surface;
      * **IMPACT**   -- a large normal closing/rebound speed (the section-6 transient);
      * **ROLLING**  -- both tangential channels excited *and* consistent with the
                        coupling ``|v_t| ~= roll_radius * |omega_t|`` (the only coupled
                        archetype -- checked before sliding/pivoting, which are its
                        single-channel degenerations);
      * **SLIDING**  -- tangential linear speed dominates;
      * **PIVOTING** -- spin about the normal dominates;
      * **STATIC**   -- gap ~ 0 and everything quiet.

    Thresholds default to the package's physical scales (``EmissionParams`` defaults) and
    are kept loose because this is a coarse archetype match, not a calibrated detector.
    """
    gap, v_n, v_t, w_n, w_t = (float(sig[0]), float(sig[1]), float(sig[2]),
                               float(sig[3]), float(sig[4]))

    quiet = (v_n < vel_quiet and v_t < vel_quiet
             and w_n < omega_quiet and w_t < omega_quiet)

    # FREE: clearly separated and not driven into the surface.
    if gap > gap_quiet and quiet:
        return FREE

    # IMPACT: a strong normal velocity transient dominates everything else.
    if v_n > impact_speed and v_n > v_t and v_n > roll_radius * w_t:
        return IMPACT

    # ROLLING: tangential linear AND angular both active and coupled (v ~= r*omega).
    rolling_pred = roll_radius * w_t
    if (v_t > vel_quiet and w_t > omega_quiet
            and abs(v_t - rolling_pred) < 0.5 * max(v_t, rolling_pred, 1e-9)):
        return ROLLING

    # SLIDING vs PIVOTING: whichever single tangential/spin channel dominates.
    sliding_score = v_t / max(vel_quiet, 1e-9)
    pivot_score = w_n / max(omega_quiet, 1e-9)
    if sliding_score > 1.0 and sliding_score >= pivot_score:
        return SLIDING
    if pivot_score > 1.0 and pivot_score > sliding_score:
        return PIVOTING

    # Default: a quiet contact at/near the surface.
    return STATIC
