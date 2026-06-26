"""State-dependent (gap-gated) HMM transition tensors -- the hybrid guards of s.5.

THEORY.md section 5 says the HMM is the *discrete shadow* of a hybrid dynamical
system: continuous flows inside a mode, punctuated by discrete jumps at *guards*.
A plain time-homogeneous transition matrix captures only "the tendency to
persist." But s.5 hands us a sharper, free, refinement:

  > Because the guards are *state-dependent*, the transition prior should be too:
  > the probability of free->contact should rise as the gap approaches zero, which
  > is strictly more informative than a constant switch probability.

The free->contact guard is precisely the geometric zero-crossing ``g -> 0`` of
s.2/s.5. So this module builds two objects:

* :func:`base_transition_matrix` -- the time-homogeneous ``(S, S)`` continuous-time
  Markov jump discretized per frame (``P(stay) = exp(-dt / dwell)``), with the
  off-diagonal mass split along the hybrid-system's guard structure (FREE is the
  gateway, IMPACT bridges free<->sustained, sustained modes mostly break to FREE).
  This is the baseline temporal prior of s.5.

* :func:`gated_transition_tensor` -- the per-frame ``(T, S, S)`` upgrade in which
  the FREE->contact mass is *gated* by gap proximity: a logistic gate ``g(t)`` that
  is ~0 when the body is far above the surface and ~1 once the gap falls within
  ``gap_gate``. This makes touchdown physically gated by the gap reaching ~0 -- the
  hybrid-system guard -- instead of a body teleporting into contact from altitude.

Both are built in probability space (small, directly-interpretable stochastic
matrices) and returned in probability space. The HMM (``contact.hmm``) takes their
*log*; every entry is kept strictly positive by a small floor so the log is finite
and the smoother can always recover from a surprising frame (s.4 -- comparisons must
stay in valid log-space, never -inf). Every row sums to 1, so likelihoods across
states remain properly normalized and comparable (s.4).

The ``(T, S, S)`` tensor is consumed directly by ``contact.hmm.forward_backward``
and ``contact.hmm.viterbi``, which already accept time-varying transitions where
``log_trans[t]`` is the transition from step ``t`` to step ``t+1`` (see the
``contact.hmm`` module docstring for that step convention).

Public API
----------
* ``base_transition_matrix(states, dt, params) -> (S, S)``      -- time-homogeneous prior.
* ``gated_transition_tensor(obs, states, dt, params) -> (T, S, S)`` -- gap-gated prior.
"""

from __future__ import annotations

import numpy as np

from .config import TransitionParams
from .types import CONTACT_MODES, FREE, IMPACT, ContactObservations

__all__ = ["base_transition_matrix", "gated_transition_tensor"]

# A small floor on every transition propensity. It keeps the matrix strictly
# positive so log-space has no -inf (the HMM must always be able to leave any state,
# however unlikely; THEORY.md s.4 keeps everything in valid, finite log-space).
_FLOOR = 0.02


def base_transition_matrix(
    states: list[str], dt: float, params: TransitionParams
) -> np.ndarray:
    """Time-homogeneous transition matrix ``P[i, j] = P(state_{t+1}=j | state_t=i)``.

    THEORY.md s.5: a contact does not flicker on and off, so the temporal prior is
    "the tendency to persist." We model each state as a continuous-time Markov *jump
    process* with mean dwell ``tau``; over a step of length ``dt`` the survival
    probability of staying put is the exponential survival of that jump,

        P(stay) = exp(-dt / tau),

    and the leftover ``1 - P(stay)`` "I jumped" mass is split among the *other*
    states with a hand-designed (but clearly-commented) structure that mirrors the
    hybrid system's guards rather than spreading uniformly:

      * **FREE is the gateway.** A sustained contact mode (static/sliding/pivoting/
        rolling) leaves almost entirely back to FREE (the break guard ``lambda -> 0``
        of s.5), and FREE re-enters contact preferentially through the short-lived
        IMPACT transient (free -> impact -> established, the make guard of s.6) rather
        than landing directly in a sustained mode.
      * **IMPACT is a fast transient.** s.6 places impact "at a finer timescale than
        the sustained modes," so it dwells *shorter* -- ``params.impact_dwell_time``
        instead of ``params.mean_dwell_time`` -- and when it jumps it overwhelmingly
        *establishes* a sustained contact mode (the touch "takes") or falls back to
        FREE (it did not). It never hops to another transient.
      * **Sustained <-> sustained switches** (e.g. static <-> sliding at the
        friction-cone stick->slip guard, s.7) are allowed but kept modest -- such
        mid-contact mode changes happen, but persistence and release are more likely.

    The weights below are deliberately round numbers, not tuned to any scenario;
    they set only the *relative* propensity of each jump, and each row is renormalized
    so the off-diagonal mass sums to exactly ``1 - P(stay)``. Every entry is at least
    ``_FLOOR`` of the jump mass so ``log(P)`` is finite everywhere (s.4).

    Parameters
    ----------
    states:
        State ordering; rows and columns follow this order.
    dt:
        Discretization step (s) -- typically the median frame interval.
    params:
        Temporal-prior parameters (``config.TransitionParams``). Uses
        ``mean_dwell_time`` for every state and ``impact_dwell_time`` for IMPACT.

    Returns
    -------
    np.ndarray
        ``(S, S)`` row-stochastic matrix, strictly positive.
    """
    states = list(states)
    S = len(states)
    idx = {name: i for i, name in enumerate(states)}
    dt = max(float(dt), 0.0)

    tau = max(float(params.mean_dwell_time), 1e-6)
    tau_impact = max(float(params.impact_dwell_time), 1e-6)

    # Per-state dwell: IMPACT is a brief transient (finer timescale, s.6), so it uses
    # the short ``impact_dwell_time``; every other state uses the baseline ``tau``.
    dwell = {name: tau for name in states}
    if IMPACT in idx:
        dwell[IMPACT] = tau_impact

    # Sustained contact modes = all contact modes except the transient IMPACT.
    sustained = [m for m in CONTACT_MODES if m != IMPACT]

    def jump_weights(src: str) -> np.ndarray:
        """Unnormalized "where do I go *if* I jump?" propensities for row ``src``.

        These are relative weights only; the diagonal (stay) is set separately from
        ``exp(-dt / dwell)``. The ``_FLOOR`` baseline guarantees every off-diagonal
        entry is strictly positive (no -inf in log-space, s.4).
        """
        w = np.full(S, _FLOOR, dtype=float)
        if src == FREE:
            # FREE re-enters contact mainly through the IMPACT transient (the make
            # guard of s.6); a direct jump straight to a sustained mode is possible
            # but minor (a body can already be resting when the record starts).
            if IMPACT in idx:
                w[idx[IMPACT]] = 1.0
            for m in sustained:
                if m in idx:
                    w[idx[m]] = 0.15
        elif src == IMPACT:
            # IMPACT establishes a sustained contact (the touch "takes"), or falls
            # back to FREE if it did not. It does not jump to another transient.
            for m in sustained:
                if m in idx:
                    w[idx[m]] = 1.0
            if FREE in idx:
                w[idx[FREE]] = 0.6
        else:
            # A sustained mode mostly breaks back to FREE (the lambda -> 0 break
            # guard, s.5); mode<->mode switches (static<->sliding at the friction-cone
            # guard, s.7) are allowed but rarer than persistence or release.
            if FREE in idx:
                w[idx[FREE]] = 1.0
            for m in sustained:
                if m in idx and m != src:
                    w[idx[m]] = 0.25
            if IMPACT in idx:
                # A break can momentarily look like an impact (the reset map, s.6).
                w[idx[IMPACT]] = 0.20
        w[idx[src]] = 0.0  # stay mass is the diagonal, handled below -- not a "jump"
        return w

    P = np.zeros((S, S), dtype=float)
    for src in states:
        i = idx[src]
        stay = float(np.exp(-dt / dwell[src]))  # CT-Markov survival over dt (s.5)
        P[i, i] = stay
        w = jump_weights(src)
        total = float(w.sum())
        if total <= 0.0:
            # No off-diagonal mass to distribute (the only way ``total`` can be 0 is a
            # single-state model: the lone weight is the diagonal, forced to 0 above).
            if S > 1:  # pragma: no cover - with S>1 the _FLOOR keeps total > 0
                # Defensive: spread the jump mass uniformly over the other states.
                P[i] = (1.0 - stay) / (S - 1)
                P[i, i] = stay
            else:
                # A single state is *absorbing*: there is nowhere else to go, so the
                # whole row is the self-loop. P[i, i] = 1 keeps the row summing to 1
                # (the row-stochasticity contract), not ``stay`` < 1 which would leak.
                P[i, i] = 1.0
        else:
            P[i] += (1.0 - stay) * w / total  # split the leftover jump mass
    return P


def gated_transition_tensor(
    obs: ContactObservations,
    states: list[str],
    dt: float,
    params: TransitionParams,
) -> np.ndarray:
    """Per-frame ``(T, S, S)`` transitions with the FREE->contact entry *gap-gated*.

    THEORY.md s.5: the free->contact guard is the geometric zero-crossing ``g -> 0``,
    so the transition prior should be state-dependent -- the chance of *entering*
    contact must rise as the gap closes and vanish when the body is far above the
    surface. We start from the time-homogeneous :func:`base_transition_matrix` and, at
    each frame, scale only the FREE row's contact-entry mass by a logistic **gate**

        g(t) = sigmoid( (gap_gate - gap(t)) / gap_gate_softness )  in [0, 1],

    which is ~0 when ``gap(t)`` is well *above* ``gap_gate`` (still in flight: do not
    let FREE jump into contact) and ~1 once ``gap(t)`` falls within ``gap_gate`` (the
    surface is within reach: the make guard is armed). ``gap_gate_softness`` sets how
    abruptly the gate opens. This is exactly the hybrid-system guard: touchdown becomes
    physically gated by the gap reaching ~0 rather than a body teleporting into contact
    from altitude.

    Construction, row by row, per frame ``t``:

      * **FREE row.** Take the base FREE row. The "I stayed FREE" diagonal and the
        FREE->contact entries are *re-apportioned* by the gate: of the base
        off-diagonal (contact-entry) mass ``m0`` = ``1 - P_base(free|free)``, only the
        fraction ``g(t)`` is offered to the contact states; the remaining
        ``(1 - g(t)) * m0`` is returned to the FREE diagonal (you keep waiting in
        flight). Within the offered mass the *relative* split across contact modes is
        the base one (IMPACT-led, per the make guard), so we only modulate the total
        entry probability, never which mode is entered. A small ``_FLOOR``-scaled
        residual is retained on each contact entry even when the gate is shut, so the
        FREE row stays strictly positive (finite log, s.4) and a genuinely surprising
        touchdown is never made literally impossible.
      * **All other rows are frame-independent** -- copied straight from the base
        matrix. Contact->contact and contact->free are governed by the *force* break
        guard (``lambda -> 0``), not by the gap proximity gate, so they are left exactly
        as :func:`base_transition_matrix` set them (s.5).

    Each row is renormalized to sum to 1 every frame, keeping cross-state likelihood
    comparisons valid in log-space (s.4). The result is a ``(T, S, S)`` stack where
    ``[t]`` is the transition from step ``t`` to ``t+1`` -- exactly the layout
    ``contact.hmm.forward_backward`` / ``contact.hmm.viterbi`` consume.

    Parameters
    ----------
    obs:
        Support-relative observations; only ``obs.gap`` (the signed distance, s.1) is
        used here to evaluate the gate.
    states:
        State ordering; rows and columns follow this order.
    dt:
        Discretization step (s) for the base matrix -- typically the median frame
        interval.
    params:
        Temporal-prior parameters (``config.TransitionParams``); ``gap_gate`` and
        ``gap_gate_softness`` define the logistic gate.

    Returns
    -------
    np.ndarray
        ``(T, S, S)`` row-stochastic tensor, strictly positive.
    """
    states = list(states)
    S = len(states)
    idx = {name: i for i, name in enumerate(states)}
    gap = np.asarray(obs.gap, dtype=float).ravel()
    T = gap.shape[0]

    base = base_transition_matrix(states, dt, params)  # (S, S)

    # No FREE state (e.g. a contact-only sub-model) -> no free->contact guard to gate;
    # the prior is just the homogeneous base tiled across all frames.
    if FREE not in idx:
        return np.broadcast_to(base, (T, S, S)).copy()

    free_i = idx[FREE]
    contact_cols = np.array(
        [j for j, s in enumerate(states) if s != FREE], dtype=np.intp
    )

    # FREE is the *only* state: there is no contact column to gate (and the base FREE
    # row is the single absorbing self-loop). Nothing to modulate, so just tile the
    # homogeneous base across all frames -- mirroring the no-FREE guard above and
    # avoiding the empty-array divide that the per-frame loop would otherwise hit.
    if contact_cols.size == 0:
        return np.broadcast_to(base, (T, S, S)).copy()

    # --- The logistic gap gate g(t) in [0, 1] (s.5 hybrid guard) --------------------
    # g(t) = sigmoid((gap_gate - gap) / softness): ~0 far above the surface
    # (gap >> gap_gate), ~1 once gap has fallen within gap_gate. Softness sets the
    # sharpness. Computed with a numerically-stable sigmoid (no overflow for large |z|).
    softness = max(float(params.gap_gate_softness), 1e-9)
    z = (float(params.gap_gate) - gap) / softness  # (T,)
    gate = _sigmoid(z)  # (T,) in [0, 1]

    # --- Decompose the base FREE row -----------------------------------------------
    # base_free_diag : "I stayed FREE" mass = P_base(free | free).
    # base_contact   : base FREE->contact entries (the contact-entry propensities).
    # m0             : total base contact-entry mass = sum(base_contact).
    base_free_row = base[free_i].copy()  # (S,)
    base_free_diag = float(base_free_row[free_i])
    base_contact = base_free_row[contact_cols].copy()  # (n_contact,)
    m0 = float(base_contact.sum())
    # Relative split of the entry mass across contact modes (IMPACT-led make guard).
    # Falls back to uniform if the base somehow offered zero entry mass.
    if m0 > 0.0:
        contact_shape = base_contact / m0
    else:  # pragma: no cover - base always offers positive contact-entry mass
        contact_shape = np.full(contact_cols.shape[0], 1.0 / contact_cols.shape[0])

    # A floor so the gated-shut FREE row keeps a sliver of contact-entry mass: even
    # with the gate fully closed, a surprising touchdown stays *possible* (finite log,
    # s.4). This is a small fraction of the base entry mass, not of the whole row.
    floor_frac = _FLOOR

    # --- Build the (T, S, S) stack --------------------------------------------------
    tensor = np.broadcast_to(base, (T, S, S)).copy()  # all rows = base by default
    for t in range(T):
        g = float(gate[t])
        # Effective gate keeps a small residual open even when shut (floor), and is
        # capped at 1 -- so the offered entry fraction lives in [floor_frac, 1].
        g_eff = floor_frac + (1.0 - floor_frac) * g
        offered = m0 * g_eff  # contact-entry mass actually offered this frame
        returned = m0 - offered  # the rest goes back to "keep waiting in FREE"
        row = np.empty(S, dtype=float)
        row[free_i] = base_free_diag + returned
        row[contact_cols] = offered * contact_shape
        # Renormalize defensively so the FREE row sums to exactly 1 (s.4).
        row /= row.sum()
        tensor[t, free_i] = row
    return tensor


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically-stable logistic ``1 / (1 + exp(-z))``, elementwise.

    Splits on the sign of ``z`` so neither ``exp(z)`` nor ``exp(-z)`` overflows for
    large ``|z|`` (the gate saturates cleanly at 0 and 1 far from the surface).
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0.0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out
