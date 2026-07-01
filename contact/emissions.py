"""Per-state emission log-likelihoods (THEORY.md sections 3 & 4).

For every candidate contact state we write down a *generative* model: a proper
probability density over the observed (gap, twist) tuple. THEORY.md section 4
makes the central modeling commitments that govern this file:

  * The decision between states is a **likelihood ratio** -- "how much better does
    state A explain this frame than state B?". For that ratio to be meaningful the
    densities must be compared over the *same* observation space and each must be a
    *proper, normalized* density. Concretely this means we keep every normalization
    constant, including the ``-log(sigma * sqrt(2*pi))`` of each Gaussian factor: a
    state with a tight tolerance pays for that tightness through a *larger* constant,
    and dropping the constants would silently bias the ratio toward the sharp states.

  * **FREE is diffuse** -- the body could be at any height moving any which way, so
    every channel gets a broad prior (a uniform clearance and wide Gaussians).

  * **Contact states are sharp peaks on a twist subspace** (THEORY.md section 3).
    Each mode pins the gap near the resting bias and concentrates the relative twist
    on the subspace that mode allows, leaving the off-subspace channels broad. The
    modes are distinguished by *which* channels are pinned and -- for rolling -- by a
    *cross-channel correlation* (``|v_tangent| ~= r*|omega_tangent|``), which no
    per-channel-independent model could represent.

  * The **gap channel is asymmetric and bounded** (THEORY.md sections 2 & 4): tight
    tolerance on the ``g > 0`` (separation) side, looser tolerance on the ``g < 0``
    (penetration) side because sensor / plane-fit error squishes the point below the
    plane, and -- because section 2 forbids true rigid penetration -- the likelihood
    must *decay* for large negative gap rather than reward it. A two-piece (split)
    Gaussian about the resting bias delivers exactly this: gross penetration sits far
    out in a Gaussian tail and earns essentially zero contact likelihood.

Everything is done in **log space**; we never multiply raw likelihoods.

Public API
----------
``log_emissions(obs, params, gap_bias, states, material=None, force=None) -> (T, S)``
    The assembled emission-log-likelihood matrix, one column per requested state.

``MODES``
    A ``{mode_name: ContactMode}`` registry; each mode is a generative model whose
    ``log_density(obs, params, gap_bias, material, force) -> (T,)`` is independently testable.

The optional ``force`` (a :class:`~contact.config.ForceEmissionParams`) enables the
MEASURED-force channel (DESIGN.md PART II.A; PHASE 4a): when both ``force`` is given AND
``obs.normal_force`` is present, each builder adds one more proper density on ``[0, inf)`` over
the (robustly normalized) normal force -- a half-normal at 0 for FREE, a Rayleigh for the
sustained contact modes, a larger-scale Rayleigh spike for IMPACT. The term is purely additive
and gated: with no force channel it is never evaluated and the kinematic emissions are unchanged.

This module imports only :mod:`contact.types` and :mod:`contact.config`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np

from .config import EmissionParams, ForceEmissionParams, MaterialParams
from .types import (
    FREE,
    IMPACT,
    PIVOTING,
    ROLLING,
    SLIDING,
    STATIC,
    ContactObservations,
)

# --------------------------------------------------------------------------------------
# Low-level log-density primitives (each integrates to 1 over its stated domain)
# --------------------------------------------------------------------------------------

_LOG_2PI = float(np.log(2.0 * np.pi))
_LOG_SQRT_2PI = 0.5 * _LOG_2PI
#: log of sqrt(pi/2); the split-normal constant is -log(sqrt(pi/2)*(s_hi + s_lo)).
_LOG_SQRT_PI_OVER_2 = 0.5 * float(np.log(np.pi / 2.0))
#: log(2/pi); the half-normal constant is 0.5*log(2/pi) - log(sigma) (DESIGN.md PART II.A).
_LOG_2_OVER_PI = float(np.log(2.0 / np.pi))
#: floor inside the Rayleigh log so log(f) stays finite at f = 0 (where the density is exactly 0).
_FORCE_EPS = 1e-12


def _log_normal_1d(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    """log N(x; mean, sigma^2) for a scalar channel, with the full constant.

    A proper 1-D Gaussian log-density:
        -log(sigma) - 0.5*log(2*pi) - 0.5*((x-mean)/sigma)^2
    The ``-log(sigma) - 0.5*log(2*pi)`` term is the normalization we must keep so
    likelihood ratios across states stay calibrated (THEORY.md section 4).
    """
    z = (np.asarray(x, dtype=float) - mean) / sigma
    return -np.log(sigma) - _LOG_SQRT_2PI - 0.5 * z * z


def _log_normal_2d_iso(x: np.ndarray, sigma: float) -> np.ndarray:
    """log of an isotropic zero-mean 2-D Gaussian N(x; 0, sigma^2 I) on R^2.

    For a 2-vector channel (tangential velocity / tangential angular velocity) the
    emission is a distribution *on the vector as a whole* (THEORY.md section 4), here
    the isotropic case. Constant is two copies of the 1-D constant:
        -2*log(sigma) - log(2*pi) - 0.5*|x|^2/sigma^2
    ``x`` has shape (T, 2); returns (T,).
    """
    x = np.asarray(x, dtype=float)
    sq = np.sum(x * x, axis=-1)
    return -2.0 * np.log(sigma) - _LOG_2PI - 0.5 * sq / (sigma * sigma)


def _log_uniform(width: float) -> float:
    """log density of a uniform distribution of total width ``width`` (a constant).

    The diffuse FREE clearance prior (THEORY.md section 4): log density = -log(width)
    everywhere on the support. We treat the support as wide enough that observed gaps
    fall inside it, so every frame gets the same constant; this is intentional -- a
    flat prior contributes a constant offset that the likelihood *ratio* against the
    sharp contact gap density still resolves correctly.
    """
    return -float(np.log(width))


def _log_split_normal_gap(
    gap: np.ndarray, mean: float, sigma_hi: float, sigma_lo: float
) -> np.ndarray:
    """log of the two-piece (split) Gaussian gap density about ``mean``.

    THEORY.md sections 2 & 4: the contact gap density is asymmetric -- standard
    deviation ``sigma_hi`` for ``gap >= mean`` (a real *separation* above the resting
    contact quickly means "free", so this side is *tight*) and ``sigma_lo`` for
    ``gap < mean`` (penetration is tolerated more: sensor / plane-fit squish). Both
    sides decay as Gaussians, so gross penetration lands deep in the lower tail and
    earns essentially zero contact likelihood -- the bounded behaviour section 2
    demands, achieved without an ad-hoc clip.

    A split normal that is continuous at the mean has a single normalization constant
        Z = sqrt(pi/2) * (sigma_hi + sigma_lo)
    (each half contributes ``sqrt(pi/2)*sigma`` of mass), so
        log p(gap) = -log(Z) - 0.5*((gap-mean)/sigma_side)^2
    with ``sigma_side`` selected per sample. The two halves meet at the same peak
    height ``1/Z`` at ``gap == mean``, hence continuity.
    """
    gap = np.asarray(gap, dtype=float)
    sigma = np.where(gap >= mean, sigma_hi, sigma_lo)
    log_z = _LOG_SQRT_PI_OVER_2 + np.log(sigma_hi + sigma_lo)
    z = (gap - mean) / sigma
    return -log_z - 0.5 * z * z


def _log_mix_zero_1d(x: np.ndarray, sigma_tight: float, sigma_broad: float, w_broad: float) -> np.ndarray:
    """log of a zero-mean 1-D Gaussian MIXTURE: ``(1-w)*N(0,s_t^2) + w*N(0,s_b^2)``.

    A proper density that is sharply peaked at 0 (the tight component rewards "this channel
    rests") yet HEAVY-TAILED (the broad component bounds the penalty when the channel is in
    fact active). Used for a mode's OFF-subspace channel that should neither require activity
    nor catastrophically reject it -- e.g. sliding's rotation: a sliding body usually isn't
    spinning, but a struck ball slides WHILE spinning up, and a single tight Gaussian would
    drive that to ~-2000 and flip the frame to FREE.
    """
    lp_t = np.log1p(-w_broad) + _log_normal_1d(x, 0.0, sigma_tight)
    lp_b = np.log(w_broad) + _log_normal_1d(x, 0.0, sigma_broad)
    return np.logaddexp(lp_t, lp_b)


def _log_mix_zero_2d(x: np.ndarray, sigma_tight: float, sigma_broad: float, w_broad: float) -> np.ndarray:
    """log of a zero-mean isotropic 2-D Gaussian mixture (the R^2 analogue of the above)."""
    lp_t = np.log1p(-w_broad) + _log_normal_2d_iso(x, sigma_tight)
    lp_b = np.log(w_broad) + _log_normal_2d_iso(x, sigma_broad)
    return np.logaddexp(lp_t, lp_b)


def _log_offset_magnitude_1d(x: np.ndarray, speed: float, sigma: float) -> np.ndarray:
    """log density for a scalar whose *magnitude* should favour a nonzero ``speed``.

    Used by the modes whose signature is "a particular channel is *moving*, not
    resting" -- sliding's tangential speed, pivoting's spin rate, impact's normal
    closing speed (THEORY.md section 3). We need a proper density on the signed
    observation ``x in R`` that is peaked away from 0, symmetric (sign of the channel
    is not informative), and integrates to 1. We use a **symmetric two-component
    Gaussian mixture** with equal weight on +/-``speed``:
        p(x) = 0.5*N(x; +speed, sigma^2) + 0.5*N(x; -speed, sigma^2).
    This is a proper density on R (normalizes to 1), its modes sit near +/-``speed``
    (a trough at 0 once ``speed > sigma``), and it reduces gracefully to a single
    zero-mean Gaussian as ``speed -> 0``. We evaluate it stably with logaddexp.
    """
    x = np.asarray(x, dtype=float)
    log_half = -float(np.log(2.0))
    lp_plus = log_half + _log_normal_1d(x, +speed, sigma)
    lp_minus = log_half + _log_normal_1d(x, -speed, sigma)
    return np.logaddexp(lp_plus, lp_minus)


def _log_offset_magnitude_2d(x: np.ndarray, speed: float, sigma: float) -> np.ndarray:
    """log density for a 2-vector whose *magnitude* should favour nonzero ``speed``.

    The 2-D analogue used for sliding's tangential velocity vector (THEORY.md section
    3: the tangential SPEED is nonzero but its *direction* in the tangent plane is
    uninformative). We place the probability mass on a ring of radius ``speed``: take
    an isotropic Gaussian of width ``sigma`` and shift its *radial* coordinate to
    ``speed``. As a proper density on R^2 this is

        p(x) = C * exp( -0.5 * (|x| - speed)^2 / sigma^2 ),

    whose normalizer (integrating in polar coordinates, with the Jacobian ``r dr``)
    is
        Z = 2*pi * sigma * [ sigma*exp(-speed^2/(2*sigma^2))
                             + speed*sqrt(pi/2)*(1 + erf(speed/(sqrt(2)*sigma))) ].
    For ``speed=0`` this collapses to the isotropic Gaussian ``Z = 2*pi*sigma^2``.
    The density is peaked on the circle ``|x| = speed`` and is rotationally symmetric,
    exactly encoding "moving tangentially at ~``speed`` in some direction".

    ``x`` has shape (T, 2); returns (T,). The normalizer is computed once.
    """
    from scipy.special import erf  # local import: scipy only needed here

    x = np.asarray(x, dtype=float)
    r = np.sqrt(np.sum(x * x, axis=-1))
    s2 = sigma * sigma
    # Closed-form polar normalizer Z (see docstring).
    term_gauss = sigma * np.exp(-(speed * speed) / (2.0 * s2))
    term_ring = speed * np.sqrt(np.pi / 2.0) * (1.0 + erf(speed / (np.sqrt(2.0) * sigma)))
    z = 2.0 * np.pi * sigma * (term_gauss + term_ring)
    log_z = float(np.log(z))
    return -log_z - 0.5 * (r - speed) ** 2 / s2


@lru_cache(maxsize=None)
def _log_rolling_residual_normalizer(
    free_vel_sigma: float,
    free_omega_sigma: float,
    roll_radius: float,
    roll_sigma: float,
) -> float:
    """log of the normalizer the ROLLING tangential block needs to be a proper density.

    The ROLLING column (``rolling_logpdf``) multiplies two broad isotropic priors on the
    tangential vectors (``v_tangent ~ N2(0, free_vel_sigma^2 I)``,
    ``omega_tangent ~ N2(0, free_omega_sigma^2 I)``) by the coupling-residual factor
    ``N(|v_t| - roll_radius*|omega_t|; 0, roll_sigma^2)``. Because that residual is a
    function of *both* tangential vectors, multiplying it onto the priors removes mass
    from the joint, so the product does NOT integrate to 1 (it integrates to ``Z_res``,
    ~0.66 at the defaults). Every other state's column integrates to 1, so this missing
    normalizer would NOT cancel in the cross-state likelihood ratio of THEORY.md s.4 and
    would hand ROLLING a parameter-dependent, unearned offset on every frame. We restore
    properness by subtracting ``log(Z_res)`` (this helper) from the column.

    ``Z_res`` is the expectation of the residual factor under the two broad priors:

        Z_res = E_{v_t~N2(0,sv^2 I), w_t~N2(0,sw^2 I)}[ N(|v_t| - r*|w_t|; 0, roll_sigma^2) ]

    Since the priors are isotropic and the residual depends only on the magnitudes
    ``a = |v_t|`` and ``b = |w_t|``, the magnitudes are Rayleigh-distributed and the
    integral collapses to a 2-D quadrature over ``(a, b)`` of
        f_a(a) * f_b(b) * N(a - r*b; 0, roll_sigma^2),
    with ``f_a, f_b`` the Rayleigh densities. It depends only on the four scalar params,
    so we compute it once and memoize -- exactly as ``_log_offset_magnitude_2d`` computes
    its polar normalizer once.
    """
    from scipy import integrate  # local import: scipy only needed here

    sv = float(free_vel_sigma)
    sw = float(free_omega_sigma)
    rr = float(roll_radius)
    rs = float(roll_sigma)

    def f_rayleigh(x: float, scale: float) -> float:
        return x / (scale * scale) * np.exp(-x * x / (2.0 * scale * scale))

    def n1(x: float) -> float:
        return 1.0 / (rs * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (x / rs) ** 2)

    # 8-sigma bounds capture essentially all Rayleigh mass; nested fixed quadrature is
    # deterministic and avoids the adaptive-subdivision warnings dblquad emits here.
    def inner(b: float) -> float:
        val, _ = integrate.quad(
            lambda a: f_rayleigh(a, sv) * n1(a - rr * b), 0.0, 8.0 * sv, limit=200
        )
        return val

    z_res, _ = integrate.quad(
        lambda b: f_rayleigh(b, sw) * inner(b), 0.0, 8.0 * sw, limit=200
    )
    return float(np.log(z_res))


# --------------------------------------------------------------------------------------
# Force-channel log-densities on [0, inf) (DESIGN.md PART II.A; PHASE 4a). Both keep their
# normalization constants so they are PROPER densities (each integrates to ~1 over [0, inf)) and
# the cross-state force log-RATIO stays calibrated -- the same module invariant the kinematic
# Gaussians obey. The normal force is physically f >= 0, so both live on the shared support
# [0, inf): FREE peaks at 0 (a separated body carries no load), the contact modes are zero at 0
# with a mode away from it (contact carries load), and IMPACT is a large, brief spike. This is
# exactly the free-vs-contact-vs-impact discriminator kinematics lacks for the cradle.
# --------------------------------------------------------------------------------------


def _log_half_normal(f: np.ndarray, sigma: float) -> np.ndarray:
    """log of a half-normal density ``HN(sigma)`` on ``[0, inf)`` (FREE's force term).

    The half-normal is the folded zero-mean Gaussian restricted to ``f >= 0``:
        p(f) = sqrt(2/pi) / sigma * exp(-0.5 (f/sigma)^2),    f >= 0,
    so, evaluated at ``max(f, 0)`` (its support),
        log p(f) = 0.5*log(2/pi) - log(sigma) - 0.5*(f/sigma)^2.
    It keeps the full constant (``0.5*log(2/pi) - log(sigma)``) and integrates to 1 over
    ``[0, inf)``. Its mode is at ``f = 0`` -- a separated (FREE) body carries no load.
    """
    f = np.maximum(np.asarray(f, dtype=float), 0.0)
    z = f / sigma
    return 0.5 * _LOG_2_OVER_PI - np.log(sigma) - 0.5 * z * z


def _log_rayleigh(f: np.ndarray, scale: float) -> np.ndarray:
    """log of a Rayleigh density ``R(scale)`` on ``[0, inf)`` (the contact / impact force term).

    The Rayleigh density is proper on ``[0, inf)``, is exactly **zero at ``f = 0``**, and has its
    mode at ``scale``:
        p(f) = f / scale^2 * exp(-0.5 (f/scale)^2),    f >= 0,
        log p(f) = log(f) - 2*log(scale) - f^2 / (2 scale^2).
    We floor the log argument at ``max(f, eps)`` so the log stays finite at ``f = 0`` (where the
    true density vanishes, i.e. log -> -inf); the quadratic term uses the raw ``f``. It keeps the
    full constant (``-2*log(scale)``) and integrates to 1 over ``[0, inf)``. "Contact carries
    load" (mode at ``s_load``); an IMPACT uses a larger ``scale`` for a brief, large spike.
    """
    f = np.asarray(f, dtype=float)
    return np.log(np.maximum(f, _FORCE_EPS)) - 2.0 * np.log(scale) - (f * f) / (2.0 * scale * scale)


def _force_log_density(
    obs: ContactObservations, state: str, force: ForceEmissionParams
) -> np.ndarray:
    """Per-frame force log-density for ``state`` over the normalized normal force (DESIGN.md II.A).

    The observed ``obs.normal_force`` is normalized ONCE by a robust positive scale
    ``s = median(force[force > 0])`` (fallback ``1.0`` when there is no positive force or the
    median is non-finite/non-positive), so ``fn = force / s`` is dimensionless. Then the per-state
    density is selected:

      * ``FREE``                                   -> ``HN(sigma_free)``  (peaks at 0),
      * ``STATIC`` / ``SLIDING`` / ``PIVOTING`` / ``ROLLING`` -> a MIXTURE
            ``w*HN(sigma_free) + (1-w)*R(s_load)`` (``w = force.w_unloaded``): a contact may be
            UNLOADED (a resting touch, ``f ~ 0``) OR loaded, so the density must allow both. A pure
            Rayleigh is *zero* at ``f = 0`` and would pull a touching-but-unloaded body to FREE
            (it sank the cradle's resting contact 452 -> 16); the unloaded ``HN(sigma_free)``
            component makes ``f ~ 0`` ~neutral (the contact-vs-free log-ratio there is just
            ``log(w)``, a small constant), so the GAP decides an unloaded touch while appreciable
            force still pulls to contact via the loaded Rayleigh,
      * ``IMPACT``                                 -> ``R(s_impact)`` (a force spike, ``s_impact >> s_load``;
            an impact is never unloaded, so zero-at-0 is correct here).

    Because every sustained contact mode shares the SAME normal-force density, the factor moves
    probability free<->contact and <->impact, never spuriously between the sustained modes (force
    magnitude alone does not distinguish static from sliding -- DESIGN.md PART II.A). Returns ``(T,)``.
    """
    f = np.asarray(obs.normal_force, dtype=float).ravel()
    pos = f[f > 0.0]
    s = float(np.median(pos)) if pos.size > 0 else 1.0
    if not np.isfinite(s) or s <= 0.0:
        s = 1.0
    fn = f / s
    if state == FREE:
        return _log_half_normal(fn, force.sigma_free)
    if state == IMPACT:
        return _log_rayleigh(fn, force.s_impact)
    # Sustained contact: a proper mixture of an UNLOADED (free-like) and a LOADED (Rayleigh)
    # component, so a resting touch (f ~ 0) is not penalized against FREE -- the gap decides it.
    w = float(force.w_unloaded)
    return np.logaddexp(
        np.log(w) + _log_half_normal(fn, force.sigma_free),
        np.log1p(-w) + _log_rayleigh(fn, force.s_load),
    )


# --------------------------------------------------------------------------------------
# The channel densities as first-class objects (THEORY.md sections 3 & 4).
#
# Each contact mode is a PRODUCT of independent per-channel densities (a SUM of these
# log-densities); the modes differ only in WHICH density sits on each channel. Naming each
# density as a small immutable object lets a mode read as its generative signature (s.3), and
# keeps every normalization constant a property that can be tested in isolation (each .logpdf is
# a proper density -- see tests/test_density.py). Every .logpdf is a thin wrapper over the
# primitives above, so a composed mode is byte-identical to the inline accumulation it replaces.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Normal1D:
    """log N(x; mean, sigma^2) on R -- a proper density (normalizer included)."""

    mean: float
    sigma: float

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return _log_normal_1d(x, self.mean, self.sigma)


@dataclass(frozen=True)
class IsoNormal2D:
    """Isotropic zero-mean 2-D Gaussian N(0, sigma^2 I) on R^2."""

    sigma: float

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return _log_normal_2d_iso(x, self.sigma)


@dataclass(frozen=True)
class SplitNormalGap:
    """Two-piece Gaussian gap density: sigma_hi above the mean, sigma_lo below (s.2). Equal sigmas => N(mean, sigma^2)."""

    mean: float
    sigma_hi: float
    sigma_lo: float

    def logpdf(self, gap: np.ndarray) -> np.ndarray:
        return _log_split_normal_gap(gap, self.mean, self.sigma_hi, self.sigma_lo)


@dataclass(frozen=True)
class OffsetMagnitude1D:
    """Proper density on R peaked at +/- speed (sign uninformative). speed -> 0 => N(0, sigma^2)."""

    speed: float
    sigma: float

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return _log_offset_magnitude_1d(x, self.speed, self.sigma)


@dataclass(frozen=True)
class OffsetMagnitude2D:
    """Proper R^2 density on the ring |x| = speed. speed -> 0 => the isotropic Gaussian IsoNormal2D(sigma)."""

    speed: float
    sigma: float

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return _log_offset_magnitude_2d(x, self.speed, self.sigma)


@dataclass(frozen=True)
class MixZero1D:
    """Zero-mean 1-D Gaussian mixture (1-w)*N(0,s_t^2) + w*N(0,s_b^2). w -> 0 => N(0, s_t^2)."""

    sigma_tight: float
    sigma_broad: float
    w_broad: float

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return _log_mix_zero_1d(x, self.sigma_tight, self.sigma_broad, self.w_broad)


@dataclass(frozen=True)
class MixZero2D:
    """Zero-mean isotropic 2-D Gaussian mixture (the R^2 analogue of MixZero1D)."""

    sigma_tight: float
    sigma_broad: float
    w_broad: float

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return _log_mix_zero_2d(x, self.sigma_tight, self.sigma_broad, self.w_broad)


@dataclass(frozen=True)
class UniformClearance:
    """Diffuse uniform gap prior of total width ``width`` -- the FREE clearance (no surface pins the gap)."""

    width: float

    def logpdf(self, gap: np.ndarray) -> np.ndarray:
        return _log_uniform(self.width) * np.ones_like(np.asarray(gap, dtype=float))


def _compose(terms: tuple) -> np.ndarray:
    """Sum a sequence of ``(channel_value, Density)`` into one per-frame log-density.

    Reduced strictly left-to-right so the result is byte-identical to the equivalent
    ``lp = d0.logpdf(x0); lp = lp + d1.logpdf(x1); ...`` accumulation (float addition is
    not associative, so the order is load-bearing for the standalone-equivalence gate).
    """
    (x0, d0), rest = terms[0], terms[1:]
    lp = d0.logpdf(x0)
    for x, d in rest:
        lp = lp + d.logpdf(x)
    return lp


# --------------------------------------------------------------------------------------
# The contact modes (THEORY.md section 3) as generative models.
# --------------------------------------------------------------------------------------
#
# Each mode is a proper probability density over the WHOLE observation (gap, v_normal in R,
# v_tangent in R^2, omega_normal in R, omega_tangent in R^2) so the columns are comparable
# and their differences are the calibrated likelihood ratios of THEORY.md s.4. A mode's
# *signature* is the set of twist channels it PINS to ~0 vs. EXCITES (s.3); off-subspace
# channels get a broad FREE-scale density so the mode neither rewards nor penalizes motion
# it does not constrain. Each subclass writes that signature as ``kinematic_log_density``;
# the base adds the shared, optional, gated measured-force channel (DESIGN.md II.A) once.


class ContactMode:
    """A single latent contact mode, written as a generative model (THEORY.md s.3/s.4).

    A subclass IS its KINEMATIC density over the (gap, twist) observation -- which channels the
    mode pins and which it excites, i.e. its physical signature (s.3). The cross-cutting, optional
    MEASURED-FORCE channel (DESIGN.md II.A) is no longer folded in here; it is a separate
    ``ForceFactor`` in the emission sum (see ``log_emissions``), so a mode stays purely kinematic.
    ``name`` is the ``contact.types`` mode id -- both the registry key and the force-density selector.
    """

    name: str = ""

    def kinematic_log_density(
        self, obs: ContactObservations, params: EmissionParams, gap_bias: float
    ) -> np.ndarray:
        """``(T,)`` log p(gap, twist | mode) -- the mode's kinematic signature (s.3)."""
        raise NotImplementedError


class Free(ContactMode):
    """FREE: nothing is pinned -- a diffuse prior on every channel (THEORY.md s.4).

    The body could be at any clearance moving any way, so every channel is broad:
      gap           ~ Uniform over ``gap_free_range``
      v_normal      ~ N(0, free_vel_sigma^2)
      v_tangent     ~ N(0, free_vel_sigma^2 I)  on R^2
      omega_normal  ~ N(0, free_omega_sigma^2)
      omega_tangent ~ N(0, free_omega_sigma^2 I) on R^2
    ``gap_bias`` is unused (free has no resting contact).
    """

    name = FREE

    def kinematic_log_density(self, obs, params, gap_bias):
        return _compose((
            (obs.gap,           UniformClearance(params.gap_free_range)),
            (obs.v_normal,      Normal1D(0.0, params.free_vel_sigma)),
            (obs.v_tangent,     IsoNormal2D(params.free_vel_sigma)),
            (obs.omega_normal,  Normal1D(0.0, params.free_omega_sigma)),
            (obs.omega_tangent, IsoNormal2D(params.free_omega_sigma)),
        ))


class Static(ContactMode):
    """STATIC / sticking: the whole twist is pinned to ~0 -- a contact at rest (s.3).

    Channels:
      gap           ~ split-normal about gap_bias (sigma_gap above / sigma_pen below)
      v_normal      ~ N(0, vel_sigma^2)
      v_tangent     ~ N(0, vel_sigma^2 I)
      omega_normal  ~ N(0, omega_sigma^2)
      omega_tangent ~ N(0, omega_sigma^2 I)
    """

    name = STATIC

    def kinematic_log_density(self, obs, params, gap_bias):
        return _compose((
            (obs.gap,           SplitNormalGap(gap_bias, params.gap_sigma_gap, params.gap_sigma_pen)),
            (obs.v_normal,      Normal1D(0.0, params.vel_sigma)),
            (obs.v_tangent,     IsoNormal2D(params.vel_sigma)),
            (obs.omega_normal,  Normal1D(0.0, params.omega_sigma)),
            (obs.omega_tangent, IsoNormal2D(params.omega_sigma)),
        ))


class Sliding(ContactMode):
    """SLIDING: tangential-linear motion only (THEORY.md section 3).

    Channels:
      gap           ~ split-normal about gap_bias
      v_normal      ~ N(0, vel_sigma^2)              (still pinned: not separating)
      v_tangent     ~ ring density peaked at |v_tangent| = slide_speed (a proper R^2 density
                      favouring nonzero tangential SPEED, direction uninformative)
      omega_normal  ~ tight+broad mixture (small: not pivoting -- but a struck slider can spin)
      omega_tangent ~ tight+broad mixture (small: not rolling)
    """

    name = SLIDING

    def kinematic_log_density(self, obs, params, gap_bias):
        # v_tangent rides a BROAD ring (a slider sweeps a range of speeds; width floored at
        # vel_sigma). omega is OFF-subspace: usually resting but a struck ball slides while
        # spinning up, so a heavy-tailed tight+broad mixture, not a single tight Gaussian.
        slide_width = max(params.vel_sigma, params.slide_width_frac * params.slide_speed)
        wb = params.slide_omega_broad_weight
        return _compose((
            (obs.gap,           SplitNormalGap(gap_bias, params.gap_sigma_gap, params.gap_sigma_pen)),
            (obs.v_normal,      Normal1D(0.0, params.vel_sigma)),
            (obs.v_tangent,     OffsetMagnitude2D(params.slide_speed, slide_width)),
            (obs.omega_normal,  MixZero1D(params.omega_sigma, params.free_omega_sigma, wb)),
            (obs.omega_tangent, MixZero2D(params.omega_sigma, params.free_omega_sigma, wb)),
        ))


class Pivoting(ContactMode):
    """PIVOTING / twisting: normal-angular motion only (THEORY.md section 3).

    Channels:
      gap           ~ split-normal about gap_bias
      v_normal      ~ N(0, vel_sigma^2)
      v_tangent     ~ N(0, vel_sigma^2 I)            (small: not sliding)
      omega_normal  ~ two-component mixture peaked at +/- pivot_speed (spin about the
                      normal; sign/handedness uninformative)
      omega_tangent ~ N(0, omega_sigma^2 I)          (small: not rolling)
    """

    name = PIVOTING

    def kinematic_log_density(self, obs, params, gap_bias):
        return _compose((
            (obs.gap,           SplitNormalGap(gap_bias, params.gap_sigma_gap, params.gap_sigma_pen)),
            (obs.v_normal,      Normal1D(0.0, params.vel_sigma)),
            (obs.v_tangent,     IsoNormal2D(params.vel_sigma)),
            (obs.omega_normal,  OffsetMagnitude1D(params.pivot_speed, params.omega_sigma)),
            (obs.omega_tangent, IsoNormal2D(params.omega_sigma)),
        ))


class Rolling(ContactMode):
    """ROLLING: tangential-linear COUPLED to tangential-angular (THEORY.md section 3).

    Rolling is *defined* by the cross-channel constraint ``v = omega x r`` -- the tangential
    speed and the tangential spin are locked together, so a per-channel-independent model
    literally cannot represent it (s.3/s.4). We encode the coupling as a Gaussian on the
    rolling-constraint RESIDUAL

        r_res = |v_tangent| - roll_radius * |omega_tangent|   ~  N(0, roll_sigma^2)

    while leaving each tangential magnitude broad, so the mode rewards the *correlation*,
    not any particular speed.

    Channels:
      gap           ~ split-normal about gap_bias
      v_normal      ~ N(0, vel_sigma^2)              (pinned: not separating)
      v_tangent     ~ broad N(0, free_vel_sigma^2 I) magnitude prior + the coupling residual
      omega_tangent ~ broad N(0, free_omega_sigma^2 I) magnitude prior
      omega_normal  ~ N(0, omega_sigma^2)            (small: a pure roll has no spin)

    Properness: the residual is a 1-D Gaussian in a scalar that depends on BOTH tangential
    vectors, so multiplying it onto the broad 2-D priors removes mass -- the joint over
    ``(v_t, omega_t)`` integrates to ``Z_res`` (~0.66 at defaults), not 1. Since every other
    column integrates to 1, that offset would not cancel in the s.4 likelihood ratio, so we
    subtract ``log(Z_res)`` (computed once and cached by ``_log_rolling_residual_normalizer``)
    -- making this a proper coupling-aware joint density, not a product of independent
    marginals, and so comparable to the other modes' columns.
    """

    name = ROLLING

    def kinematic_log_density(self, obs, params, gap_bias):
        v_t = np.asarray(obs.v_tangent, dtype=float)
        w_t = np.asarray(obs.omega_tangent, dtype=float)
        speed_t = np.sqrt(np.sum(v_t * v_t, axis=-1))
        omega_t = np.sqrt(np.sum(w_t * w_t, axis=-1))
        residual = speed_t - params.roll_radius * omega_t  # the rolling-constraint residual
        log_z_res = _log_rolling_residual_normalizer(
            params.free_vel_sigma, params.free_omega_sigma, params.roll_radius, params.roll_sigma
        )
        # The ONE non-product mode: v_tangent and omega_tangent are coupled through the residual,
        # so the tangential block is renormalized by Z_res. In the composition the coupling is just
        # a derived channel (residual) and Z_res a trailing block normalizer -- no special case.
        return _compose((
            (obs.gap,           SplitNormalGap(gap_bias, params.gap_sigma_gap, params.gap_sigma_pen)),
            (obs.v_normal,      Normal1D(0.0, params.vel_sigma)),
            (v_t,               IsoNormal2D(params.free_vel_sigma)),
            (w_t,               IsoNormal2D(params.free_omega_sigma)),
            (residual,          Normal1D(0.0, params.roll_sigma)),
            (obs.omega_normal,  Normal1D(0.0, params.omega_sigma)),
        )) - log_z_res


class Impact(ContactMode):
    """IMPACT: a short-lived transient at touchdown/break (THEORY.md section 6).

    The signature is a large (closing) normal velocity at a gap near the bias but WIDER than
    sustained contact -- the body crosses g~0 with momentum, so the instantaneous gap is less
    tightly pinned. Other channels are broad.

    Channels:
      gap           ~ split-normal about gap_bias, but with WIDER scales (2x): impacts
                      straddle the surface rather than rest on it.
      v_normal      ~ two-component mixture peaked at +/- impact_speed (large closing/rebound)
      v_tangent     ~ broad N(0, free_vel_sigma^2 I)
      omega_normal  ~ broad N(0, free_omega_sigma^2)
      omega_tangent ~ broad N(0, free_omega_sigma^2 I)
    """

    name = IMPACT

    def kinematic_log_density(self, obs, params, gap_bias):
        return _compose((
            (obs.gap,           SplitNormalGap(gap_bias, 2.0 * params.gap_sigma_gap, 2.0 * params.gap_sigma_pen)),
            (obs.v_normal,      OffsetMagnitude1D(params.impact_speed, params.vel_sigma)),
            (obs.v_tangent,     IsoNormal2D(params.free_vel_sigma)),
            (obs.omega_normal,  Normal1D(0.0, params.free_omega_sigma)),
            (obs.omega_tangent, IsoNormal2D(params.free_omega_sigma)),
        ))


# --------------------------------------------------------------------------------------
# Registry + assembly
# --------------------------------------------------------------------------------------

#: mode-name -> the ContactMode generative model. ``log_emissions`` stacks the requested
#: columns in order; each mode is independently constructable and testable. ``force`` (a
#: ForceEmissionParams or None) is the optional gated measured-force term (DESIGN.md II.A).
MODES: dict[str, ContactMode] = {
    m.name: m for m in (Free(), Static(), Sliding(), Pivoting(), Rolling(), Impact())
}


# --------------------------------------------------------------------------------------
# Emission factors: the (T, S) grid as a SUM of independent log-contributions (THEORY.md s.4)
# --------------------------------------------------------------------------------------
#
# The emission log-likelihood is a sum of factors on the grid: the always-present kinematic mode
# bank, plus optional gated channels (the measured-force term here). An absent capability
# contributes the additive identity ZERO, so "no capability declared => the pure kinematic
# detector, byte-for-byte" (the DESIGN.md invariant) holds BY CONSTRUCTION rather than via a
# scattered ``if``; a new evidence channel is one more entry in the factor list. NOTE: measurement
# tempering (model.detect) is a *multiplicative* per-frame reweighting, not a summand, so it is
# deliberately NOT an EmissionFactor -- the additive monoid is the emission side only.

_EMISSION_ZERO = 0.0  # additive identity of the (T, S) emission grid


class EmissionFactor(Protocol):
    def contribute(
        self, obs: ContactObservations, params: EmissionParams, gap_bias: float, states: list[str]
    ) -> np.ndarray | float:
        """A (T, len(states)) log-contribution, or ``_EMISSION_ZERO`` when the factor is inactive."""
        ...


class KinematicFactor:
    """The mode bank (s.3): log p(gap, twist | state) for each requested state. Always present."""

    def contribute(self, obs, params, gap_bias, states):
        T = int(np.asarray(obs.gap, dtype=float).shape[0])
        out = np.empty((T, len(states)), dtype=float)
        for j, name in enumerate(states):
            try:
                mode = MODES[name]
            except KeyError as exc:  # pragma: no cover - defensive
                raise KeyError(
                    f"no contact mode registered for state {name!r}; known modes: {sorted(MODES)}"
                ) from exc
            out[:, j] = mode.kinematic_log_density(obs, params, gap_bias)
        return out


class ForceFactor:
    """The optional MEASURED-force channel (DESIGN.md II.A). ZERO when ``obs.normal_force is None``."""

    def __init__(self, force: ForceEmissionParams) -> None:
        self.force = force

    def contribute(self, obs, params, gap_bias, states):
        if obs.normal_force is None:
            return _EMISSION_ZERO
        T = int(np.asarray(obs.gap, dtype=float).shape[0])
        out = np.empty((T, len(states)), dtype=float)
        for j, name in enumerate(states):
            out[:, j] = _force_log_density(obs, name, self.force)
        return out


def _sum_emissions(factors, obs, params, gap_bias, states) -> np.ndarray:
    """Left-to-right sum of the factor contributions -- byte-identical to inline accumulation."""
    total = factors[0].contribute(obs, params, gap_bias, states)
    for f in factors[1:]:
        total = total + f.contribute(obs, params, gap_bias, states)
    return total


def log_emissions(
    obs: ContactObservations,
    params: EmissionParams,
    gap_bias: float,
    states: list[str],
    material: MaterialParams | None = None,
    force: ForceEmissionParams | None = None,
) -> np.ndarray:
    """Assemble the per-state emission log-likelihood matrix (THEORY.md sections 3 & 4).

    Evaluates, for each frame ``t`` and each requested state ``s``, the proper
    log-density ``log p(obs_t | state = s)``. Because every column is a normalized
    density over the same observation space, differences between columns are exactly
    the log-likelihood ratios the HMM / detector consumes (THEORY.md section 4).

    Parameters
    ----------
    obs :
        Per-frame support-relative observations (length T).
    params :
        Emission noise/speed scales.
    gap_bias :
        The resting-contact mean gap (m); the EM-calibrated offset that all contact
        modes' gap densities are centered on (THEORY.md sections 7 & 8). FREE ignores it.
    states :
        State names (subset / reordering of ``contact.types.ALL_STATES``). The output
        columns follow this exact order.
    material :
        Optional material parameters; accepted for interface symmetry (a future,
        compliance-aware gap/force term, THEORY.md section 7). The kinematic emissions
        here do not yet use it.
    force :
        Optional :class:`~contact.config.ForceEmissionParams` enabling the MEASURED-force
        channel (DESIGN.md PART II.A; PHASE 4a). When given AND ``obs.normal_force`` is present,
        every builder adds one proper force log-density (FREE half-normal / contact Rayleigh /
        IMPACT spike). ``None`` (or no ``obs.normal_force``) => no force factor; the kinematic
        emissions are byte-identical.

    Returns
    -------
    np.ndarray
        Shape ``(T, len(states))``; column ``j`` is ``log p(obs | states[j])``.

    Raises
    ------
    KeyError
        If a requested state has no registered builder.
    """
    factors: list[EmissionFactor] = [KinematicFactor()]
    if force is not None:
        factors.append(ForceFactor(force))
    return _sum_emissions(factors, obs, params, gap_bias, states)
