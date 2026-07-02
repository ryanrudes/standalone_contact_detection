"""Per-frame measurement-uncertainty propagation into the detector (THEORY.md §8).

THEORY.md §8 (and the workflow of §9) insists that the detector be
honest about *what is observable given the sensor*: real captures have noisy and
**occluded** frames (a marker briefly leaves the cameras, or a fit degrades), and
those frames carry less information about the contact state than clean ones. A
detector that weighs every frame equally lets a single garbage frame shout as
loudly as a confident one. The principled fix is to fold the *per-frame
measurement covariance* into the generative model so a noisy frame's emission is
correspondingly less certain and the **temporal prior** (THEORY.md §5) carries
the state across it.

What this module does -- and what it approximates
-------------------------------------------------
The fully principled object would *inflate the emission noise per frame*: a frame
with measurement covariance ``R(t)`` on the contact point should have every
emission channel's variance grow by the part of ``R(t)`` that projects onto that
channel (``sigma_eff^2 = sigma_base^2 + sigma_meas^2(t)``), re-evaluating each
state's density with its own widened scales. That is exact but couples this module
into every channel's normalizer in :mod:`contact.emissions`, and the widened
normalizers change the cross-state likelihood *ratio* in a channel-dependent way.

We instead implement a tractable, well-documented **likelihood-tempering**
approximation. We compute a single per-frame scalar measurement-variance proxy
``meas_var(t)`` (from ``obs.meas_cov``), turn it into a tempering weight

    w(t) = base_noise_var / (base_noise_var + meas_var(t))   in (0, 1],

and *temper* the whole emission row by raising the per-frame likelihood to the
power ``w(t)`` -- i.e. multiply the log-emission row by ``w(t)``. A clean frame
(``meas_var -> 0``) has ``w -> 1`` and is untouched; a badly occluded frame
(``meas_var`` dominating the channel noise) has ``w -> 0``, which *flattens* that
frame's log-emission toward a constant so every state becomes (nearly) equally
likely there and the transition prior of THEORY.md §5 dictates the state.

HONEST limitations (no overclaiming):
  * Tempering scales *all* states' log-densities by the same ``w(t)``. It does not
    selectively widen only the channels that the measurement actually corrupts, nor
    does it re-derive each state's normalization constant. It is a *uniform*
    down-weighting of a frame's evidence, not a per-channel noise inflation. The two
    agree in the two limits (``w=1`` exact; ``w=0`` fully flattened) and the
    monotone interpolation between them is reasonable, but the interior is an
    approximation.
  * Because the row is scaled uniformly, tempering preserves the *shape* of the
    cross-state likelihood ratio at a frame and only attenuates its *magnitude*
    (``w * (logp_a - logp_b)``). It cannot, for instance, decide that an occlusion
    corrupts the gap channel but not the twist. Full per-channel inflation can; this
    cannot. We accept that for tractability and state it plainly.
  * ``meas_var`` is reduced to a single scalar per frame (the mean variance of the
    contact-point position, i.e. ``trace(R)/3`` for a 3x3 ``R``). This discards the
    anisotropy of ``R``; it is the right *summary* for a scalar tempering weight but
    is not the full covariance.

Backward compatibility
-----------------------
Everything here is inert unless ``obs.meas_cov`` is provided *and* the caller opts
in via ``config.inference.use_uncertainty`` (the model wires that). With
``meas_cov is None`` the variance proxy is all-zeros, the tempering weight is
all-ones, and :func:`apply_tempering` returns the log-emission matrix unchanged --
the exact pre-existing behaviour.

Public API
----------
``gap_twist_variance(obs) -> (T,)``
    Per-frame scalar measurement-variance proxy from ``obs.meas_cov`` (zeros if None).
``emission_tempering(obs, params, base_noise=None) -> (T,)``
    Per-frame tempering weight ``w(t) in (0, 1]`` (all-ones if ``meas_cov`` is None).
``apply_tempering(log_emission, w) -> (T, S)``
    The log-emission matrix with each frame row scaled by ``w(t)``.
``simulate_occlusion(obs, windows, inflate) -> ContactObservations``
    A copy of ``obs`` with ``meas_cov`` inflated (and noise added) over given windows;
    a helper to MAKE occluded test data.

This module imports only :mod:`contact.types`, :mod:`contact.config`, and numpy.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .config import EmissionParams, InferenceParams
from .types import ContactObservations

__all__ = [
    "gap_twist_variance",
    "emission_tempering",
    "apply_tempering",
    "simulate_occlusion",
]


def _n_frames(obs: ContactObservations) -> int:
    """Number of frames T, read off the always-present ``gap`` channel."""
    return int(np.asarray(obs.gap, dtype=float).shape[0])


def gap_twist_variance(obs: ContactObservations) -> np.ndarray:
    """Per-frame scalar measurement-variance proxy for the observation channels.

    THEORY.md §8: ``obs.meas_cov`` is the per-frame measurement variance of the
    tracked contact *point* (the quantity a mocap rig would report a covariance on).
    The observation channels the detector consumes (gap, the linear velocities,
    derived from differentiating that point) all inherit their uncertainty from it,
    so a single scalar variance per frame is the natural proxy that drives the
    scalar tempering weight of :func:`emission_tempering`.

    We accept ``obs.meas_cov`` in either form documented on
    :class:`~contact.types.ContactObservations`:

    * ``(T,)``    -- already a per-frame scalar position variance; returned as-is
      (coerced to float, non-negative).
    * ``(T, 3, 3)`` -- a per-frame position covariance matrix; reduced to a scalar
      by the **mean eigen-variance** ``trace(R) / 3`` (the average per-axis
      variance). This is the isotropic summary that matches a scalar weight; it
      deliberately discards the anisotropy of ``R`` (documented limitation in the
      module docstring).

    Parameters
    ----------
    obs:
        Per-frame observations. Only ``obs.meas_cov`` and the frame count are used.

    Returns
    -------
    np.ndarray
        Shape ``(T,)`` non-negative measurement-variance proxy. **All zeros** when
        ``obs.meas_cov is None`` (no extra uncertainty -- exact backward-compat).

    Raises
    ------
    ValueError
        If ``obs.meas_cov`` is present but not shaped ``(T,)`` or ``(T, 3, 3)``.
    """
    T = _n_frames(obs)
    if obs.meas_cov is None:
        return np.zeros(T, dtype=float)

    cov = np.asarray(obs.meas_cov, dtype=float)
    if cov.shape == (T,):
        var = cov
    elif cov.shape == (T, 3, 3):
        # Mean per-axis variance = trace(R)/3 (the isotropic summary of R).
        var = np.trace(cov, axis1=1, axis2=2) / 3.0
    else:
        raise ValueError(
            f"obs.meas_cov must have shape (T,)=({T},) or (T, 3, 3)=({T}, 3, 3); "
            f"got {cov.shape}"
        )
    # A variance is non-negative; clip away any tiny negative round-off so the
    # downstream weight stays in (0, 1].
    return np.maximum(np.asarray(var, dtype=float).ravel(), 0.0)


def emission_tempering(
    obs: ContactObservations,
    params: EmissionParams | InferenceParams | None = None,
    base_noise: float | None = None,
) -> np.ndarray:
    """Per-frame likelihood-tempering weight ``w(t) in (0, 1]`` (THEORY.md §8).

    The weight interpolates between trusting a frame fully and ignoring it, as a
    function of how the measurement variance compares to the detector's *intrinsic*
    channel noise. We use the standard **signal-vs-noise shrinkage** form

        w(t) = base_noise_var / (base_noise_var + meas_var(t)),

    a proper number in ``(0, 1]`` that is exactly the fraction of the total variance
    attributable to the model's own noise (the rest being measurement noise). It is

    * ``w -> 1`` when ``meas_var -> 0`` (a clean frame: trust the likelihood fully),
    * ``= 0.5`` when measurement noise equals the base channel noise,
    * ``-> 0`` when ``meas_var`` dominates (an occluded frame: flatten its
      likelihood so the temporal PRIOR of THEORY.md §5 carries the state).

    This is a likelihood-TEMPERING approximation to full per-frame emission-noise
    inflation -- see the module docstring for exactly how it differs and why we use
    it. It does NOT re-derive any per-state normalizer; it uniformly attenuates a
    frame's evidence.

    Parameters
    ----------
    obs:
        Per-frame observations (its ``meas_cov`` supplies ``meas_var(t)`` via
        :func:`gap_twist_variance`).
    params:
        Optional source of the default ``base_noise`` *standard deviation* when
        ``base_noise`` is not given explicitly. An :class:`EmissionParams` supplies
        its contact velocity noise ``vel_sigma`` (the representative channel scale
        the measurement variance competes against). Anything without a usable scale
        (e.g. ``InferenceParams`` or ``None``) falls back to a unit base noise.
    base_noise:
        Explicit base-noise **standard deviation** (same units as the contact-point
        position / velocity, so its square is comparable to ``meas_var``). Overrides
        ``params``. Must be positive when given.

    Returns
    -------
    np.ndarray
        Shape ``(T,)`` tempering weights in ``(0, 1]``. **All ones** when
        ``obs.meas_cov is None`` (exact backward-compat: tempering is then a no-op).

    Raises
    ------
    ValueError
        If an explicit ``base_noise`` is non-positive.
    """
    T = _n_frames(obs)
    if obs.meas_cov is None:
        # No measurement covariance => no extra uncertainty => tempering is a no-op.
        return np.ones(T, dtype=float)

    # Resolve the base-noise standard deviation.
    if base_noise is not None:
        sigma = float(base_noise)
        if not sigma > 0.0:
            raise ValueError(f"base_noise must be positive; got {base_noise!r}")
    elif isinstance(params, EmissionParams):
        # The contact velocity noise is the representative scale the measurement
        # variance competes against (gap/velocity channels are differentiated from
        # the same noisy contact point). Guard against a non-positive config value.
        sigma = float(params.vel_sigma)
        if not sigma > 0.0:
            sigma = 1.0
    else:
        # No usable scale (InferenceParams / None): unit base noise. meas_var is then
        # measured in those same (squared) units, which is the caller's contract.
        sigma = 1.0

    base_var = sigma * sigma
    meas_var = gap_twist_variance(obs)
    # w = base_var / (base_var + meas_var) in (0, 1]; base_var > 0 keeps it finite and
    # strictly positive, so a fully-flattened frame is approached but never NaN.
    w = base_var / (base_var + meas_var)
    return np.asarray(w, dtype=float)


def apply_tempering(log_emission: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Temper a log-emission matrix: scale each frame row by ``w(t)`` (THEORY.md §8).

    Tempering raises a frame's likelihood to the power ``w(t)``, which in log-space
    is a multiplication of that frame's whole row:

        log_em_tempered[t, s] = w(t) * log_em[t, s].

    Because the same ``w(t)`` multiplies every state ``s`` at frame ``t``, the
    cross-state log-likelihood *ratio* at that frame is scaled by ``w(t)`` too:
    ``w * (logp_a - logp_b)``. With ``w = 1`` the row is unchanged; with ``w -> 0``
    the row collapses toward the constant ``0`` so every state is (nearly) equally
    likely at that frame and the HMM's temporal prior (THEORY.md §5) -- not the
    corrupted measurement -- decides the state. That is exactly "an occluded frame
    contributes less" (THEORY.md §8).

    This operates purely on the assembled matrix and is the *only* place the
    approximation touches inference, so the per-state emission builders in
    :mod:`contact.emissions` stay untouched and exact.

    Parameters
    ----------
    log_emission:
        ``(T, S)`` log-emission likelihoods (as from
        :func:`contact.emissions.log_emissions`).
    w:
        ``(T,)`` per-frame tempering weights (typically from
        :func:`emission_tempering`).

    Returns
    -------
    np.ndarray
        A new ``(T, S)`` matrix with row ``t`` scaled by ``w[t]``. The input is not
        modified.

    Raises
    ------
    ValueError
        If shapes are inconsistent.
    """
    log_em = np.asarray(log_emission, dtype=float)
    if log_em.ndim != 2:
        raise ValueError(
            f"log_emission must be 2-D (T, S); got shape {log_em.shape}"
        )
    T, _S = log_em.shape
    w = np.asarray(w, dtype=float).ravel()
    if w.shape != (T,):
        raise ValueError(
            f"w must have shape (T,)=({T},) matching log_emission rows; got {w.shape}"
        )
    # Broadcast the per-frame weight across the state columns.
    return log_em * w[:, None]


def simulate_occlusion(
    obs: ContactObservations,
    windows: list[tuple[int, int]],
    inflate: float,
    *,
    seed: int | None = 0,
) -> ContactObservations:
    """Return a copy of ``obs`` with measurement uncertainty injected on windows.

    A test/data helper to MAKE occluded data (THEORY.md §9 domain-randomization:
    "marker noise and dropout"). Over each ``(start, end)`` half-open frame window it

    1. **inflates** ``meas_cov`` by ``inflate`` (a variance multiplier / additive
       floor) so those frames declare themselves uncertain, and
    2. **adds Gaussian noise** to the observation channels there, with a standard
       deviation set by the injected measurement variance, so the data is actually
       corrupted (not merely flagged). This makes a realistic occluded frame: both
       garbage *values* and an honest *covariance* saying so.

    The returned ``meas_cov`` is always a ``(T,)`` per-frame scalar variance: it
    starts from the existing ``meas_cov`` (reduced to a scalar via
    :func:`gap_twist_variance`, or zeros if none) and, on the windowed frames, is
    raised to ``max(existing, inflate)`` -- so ``inflate`` acts as a variance floor
    for the occluded frames while clean frames keep their original (possibly zero)
    variance. The added channel noise on those frames has standard deviation
    ``sqrt(inflate)`` (velocity channels) and a matched gap perturbation.

    Determinism: the RNG is seeded (``seed=0`` by default) so repeated calls produce
    identical corrupted data; pass ``seed=None`` for fresh randomness.

    Parameters
    ----------
    obs:
        The clean (or partially noisy) observations to corrupt. Not modified.
    windows:
        List of ``(start, end)`` half-open frame index ranges to occlude. Indices are
        clipped to ``[0, T]``; an empty list is a no-op (returns a copy with a
        scalar ``meas_cov``).
    inflate:
        Non-negative measurement-variance floor applied on the windows (m^2, matching
        the contact-point position units). Also sets the corrupting noise scale
        (std ``sqrt(inflate)``). ``inflate=0`` flags nothing and adds no noise.
    seed:
        RNG seed for the additive noise (default ``0`` -> deterministic).

    Returns
    -------
    ContactObservations
        A new observations object (a shallow ``dataclasses.replace`` copy) with fresh
        arrays for the corrupted channels and a ``(T,)`` ``meas_cov``.

    Raises
    ------
    ValueError
        If ``inflate`` is negative.
    """
    if inflate < 0.0:
        raise ValueError(f"inflate must be non-negative; got {inflate!r}")

    T = _n_frames(obs)
    rng = np.random.default_rng(seed)

    # Base per-frame scalar variance (existing meas_cov reduced to (T,), or zeros).
    meas_var = gap_twist_variance(obs).copy()

    # Mask of frames to occlude.
    mask = np.zeros(T, dtype=bool)
    for start, end in windows:
        s = int(np.clip(start, 0, T))
        e = int(np.clip(end, 0, T))
        if e > s:
            mask[s:e] = True

    # Inflate the declared variance on the windows (inflate is a floor, not a reset:
    # an already-noisier frame keeps its larger variance).
    meas_var = np.where(mask, np.maximum(meas_var, float(inflate)), meas_var)

    # Copy the channels we will corrupt so the input is untouched.
    gap = np.asarray(obs.gap, dtype=float).copy()
    v_normal = np.asarray(obs.v_normal, dtype=float).copy()
    v_tangent = np.asarray(obs.v_tangent, dtype=float).copy()
    omega_normal = np.asarray(obs.omega_normal, dtype=float).copy()
    omega_tangent = np.asarray(obs.omega_tangent, dtype=float).copy()

    if inflate > 0.0 and mask.any():
        sigma = float(np.sqrt(inflate))  # position-noise std (m); velocity inherits it.
        idx = np.where(mask)[0]
        n = idx.size
        gap[idx] += rng.normal(0.0, sigma, size=n)
        v_normal[idx] += rng.normal(0.0, sigma, size=n)
        v_tangent[idx] += rng.normal(0.0, sigma, size=(n, 2))
        omega_normal[idx] += rng.normal(0.0, sigma, size=n)
        omega_tangent[idx] += rng.normal(0.0, sigma, size=(n, 2))

    return replace(
        obs,
        gap=gap,
        v_normal=v_normal,
        v_tangent=v_tangent,
        omega_normal=omega_normal,
        omega_tangent=omega_tangent,
        meas_cov=meas_var,
    )
