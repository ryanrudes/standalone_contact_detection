"""Generic explicit-duration (semi-Markov) decoding in log-space (numpy only).

This module is the principled upgrade of the plain HMM in ``contact.hmm`` that
THEORY.md section 5 demands:

    "A plain Markov prior says dwell times are memoryless, which is wrong -- the
     chance a contact ends depends on how long it has lasted and how loaded it
     is.  The honest version is a semi-Markov / explicit-duration model with a
     hazard rate, which is also the principled replacement for the toy script's
     hard 'minimum contact duration'."

Why a plain HMM is not enough (s.5).  In an HMM the only knob on persistence is
the per-frame self-transition probability ``p``.  That makes the dwell time in a
state *geometrically* distributed: ``P(stay exactly d frames) = (1-p) p^(d-1)``.
The geometric distribution is *memoryless* -- its hazard (the instantaneous
probability of leaving given you have lasted this long) is the constant ``1-p``,
independent of how long you have already been in the state.  Physically that is
wrong: a contact that has lasted 200 ms is *not* equally likely to end in the
next frame as one that began 1 ms ago.  And operationally it is exactly why a
plain HMM still admits 1-frame blips -- nothing about the model says a contact
of length 1 is intrinsically improbable; only the (constant) transition cost
discourages it, and a single sufficiently confident emission frame can overpower
that cost.

The fix is an **explicit-duration HMM (EDHMM / HSMM)**: a *segment* of state ``s``
that spans ``d`` consecutive frames carries an explicit duration prior
``duration_logpmf(d | s)`` on top of the emissions and the inter-segment
transition.  By choosing a duration distribution whose mass is concentrated away
from ``d = 1`` we make short spurious segments intrinsically expensive -- this is
the principled, model-based replacement for the morphological ``drop_short_runs``
post-process.

Deliberately **contact-free**, exactly like ``contact.hmm``: states are abstract
integers ``0..S-1``; there is no notion of gap, twist, or mode here.  The contact
layer supplies per-state ``mean_dwell_frames`` (e.g. from
``TransitionParams.mean_dwell_time`` divided by the frame period) and the global
``concentration`` (e.g. ``TransitionParams.dwell_concentration``) and sits on top.

Everything is in log-space (``logsumexp`` reused from ``contact.hmm``) because
likelihoods over a whole trajectory underflow catastrophically if multiplied raw
(THEORY.md s.4 & s.5).

The duration model (negative binomial)
--------------------------------------
We use a **shifted negative-binomial** duration distribution over ``d >= 1``.
A negative binomial ``NB(r, p)`` is the number of successes before the ``r``-th
failure; equivalently it is the *sum of ``r`` i.i.d. geometric variables*.  That
"sum of ``r`` geometrics" reading is exactly what we want for a dwell time:

* ``r = 1``  -> a single geometric -> the **memoryless** geometric dwell of a
  plain HMM.  So ``concentration = 1`` recovers the HMM as a special case, and
  smaller ``r`` keeps the hazard flat / near-memoryless.
* larger ``r`` -> a sum of many small geometric "stages" -> by the central-limit
  effect the distribution **concentrates around its mean**, the coefficient of
  variation shrinks like ``1/sqrt(r)``, and the hazard becomes *rising* (an
  Erlang/gamma-like, non-memoryless dwell).  This is precisely the "higher =>
  tighter around the mean, more deterministic dwell" behaviour the spec asks for.

It is the discrete analogue of a Gamma (Erlang) holding time, so it is the
natural discrete-time shadow of a continuous-time hazard model (s.5).

Parameterisation.  We work with ``k = d - 1 >= 0`` (extra frames beyond the
minimum dwell of one frame) and put ``k ~ NB(r, p)`` with ``r = concentration``.
``NB(r, p)`` has mean ``r (1 - p) / p``; we want ``E[d] = 1 + E[k] =
mean_dwell_frames``, i.e. ``E[k] = mean_dwell_frames - 1``, which fixes

    p = r / (r + (mean_dwell_frames - 1)).

The log-pmf is, with ``C(k + r - 1, k)`` the binomial coefficient,

    log P(k) = lgamma(k + r) - lgamma(r) - lgamma(k + 1)
               + r log p + k log(1 - p),     k = d - 1.

Public API
----------
* ``duration_logpmf(d, mean_dwell_frames, concentration)`` -> float / array.
* ``hsmm_viterbi(log_emission, log_trans, log_init, mean_dwell_frames,
                 concentration, max_dur=None)`` -> int path ``(T,)``.
* ``hsmm_posteriors(...)`` -> ``(gamma (T, S), loglik)``  -- segmental
  forward-backward giving per-frame state posteriors.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from .hmm import logsumexp

__all__ = ["duration_logpmf", "hsmm_viterbi", "hsmm_posteriors"]

# Finite stand-in for log(0); same sentinel convention as contact.hmm so the two
# modules compose without -inf/nan surprises.
_LOG_ZERO = -1e30


# ======================================================================================
# Duration distribution
# ======================================================================================

def duration_logpmf(
    d: int | np.ndarray,
    mean_dwell_frames: float,
    concentration: float,
) -> float | np.ndarray:
    """Log-pmf of the dwell duration ``d >= 1`` (frames) for one state (THEORY.md s.5).

    A **shifted negative-binomial** distribution: with ``k = d - 1`` we take
    ``k ~ NB(r, p)`` where ``r = concentration`` and ``p`` is chosen so the mean
    dwell equals ``mean_dwell_frames`` (see module docstring for the derivation).

    Behaviour vs ``concentration``:

    * ``concentration ~ 1`` -> geometric / **memoryless** dwell (recovers the
      plain-HMM self-transition prior; flat hazard).
    * larger ``concentration`` -> the mass tightens around ``mean_dwell_frames``
      (coefficient of variation ``~ 1/sqrt(concentration)``; rising hazard), so
      short dwells -- and ``d = 1`` blips in particular -- become improbable.

    Parameters
    ----------
    d:
        Duration(s) in frames. Scalar or integer array. Values ``< 1`` get
        log-prob ``_LOG_ZERO`` (a 0-frame segment is meaningless).
    mean_dwell_frames:
        Target mean dwell ``E[d]`` in frames (must be ``>= 1``). Values below 1
        are clamped to 1 (a degenerate spike at ``d = 1``).
    concentration:
        Negative-binomial shape ``r > 0``; higher is sharper. Clamped to a small
        positive floor for safety.

    Returns
    -------
    float | np.ndarray
        ``log P(d)``, matching the shape of ``d`` (scalar in -> float out).
    """
    d_arr = np.asarray(d, dtype=float)
    scalar_in = d_arr.ndim == 0

    r = float(max(concentration, 1e-6))
    mean = float(max(mean_dwell_frames, 1.0))

    # k = d - 1 is the number of extra frames beyond the mandatory single frame.
    # E[k] = mean - 1; NB mean is r(1-p)/p, so p = r / (r + E[k]).
    mean_k = mean - 1.0
    # When mean_k -> 0 the dwell collapses to a deterministic single frame; guard p->1.
    p = r / (r + mean_k) if mean_k > 0.0 else 1.0
    p = min(max(p, 1e-12), 1.0 - 1e-15)  # keep both logs finite
    log_p = np.log(p)
    log_1mp = np.log1p(-p)

    k = d_arr - 1.0
    valid = (d_arr >= 1.0) & (np.abs(d_arr - np.round(d_arr)) < 1e-9)

    # NB log-pmf: lgamma(k+r) - lgamma(r) - lgamma(k+1) + r*log p + k*log(1-p).
    with np.errstate(invalid="ignore"):
        k_safe = np.where(valid, k, 0.0)  # avoid lgamma of negatives for invalid entries
        logpmf = (
            gammaln(k_safe + r)
            - gammaln(r)
            - gammaln(k_safe + 1.0)
            + r * log_p
            + k_safe * log_1mp
        )
    logpmf = np.where(valid, logpmf, _LOG_ZERO)

    if scalar_in:
        return float(logpmf)
    return logpmf


def _duration_logsf(
    d: int,
    mean_dwell_frames: float,
    concentration: float,
) -> float:
    """Log survival function ``log P(duration >= d)`` for one state.

    This is the right-censored ("the bout has lasted at least ``d`` frames and has
    not yet ended") duration mass, the complement of the CDF:
    ``log(1 - CDF(d - 1)) = log sum_{j >= d} pmf(j)``. It is what a segment that
    hits the ``max_dur`` cap must score instead of the point pmf, so a genuine bout
    longer than the cap is represented as a *censored* segment (which may continue)
    rather than being forced to end exactly at the cap (see :func:`_duration_table`
    and the censoring discussion in :func:`hsmm_viterbi`).

    Computed as ``log P(>= d) = log(pmf(d)) + log(1 + sum_{j>d} pmf(j)/pmf(d))`` via
    a stable ``logsumexp`` over a tail long enough to capture essentially all the
    remaining mass (the negative-binomial tail decays geometrically).
    """
    mean = float(max(mean_dwell_frames, 1.0))
    # Tail length: a few multiples of the mean beyond d comfortably covers the NB
    # mass (the survival is dominated by the first ~ a few means of the tail).
    tail = int(max(1, np.ceil(40.0 * mean)))
    js = np.arange(d, d + tail, dtype=float)
    logpmf = np.asarray(duration_logpmf(js, mean, concentration))  # array in -> array out
    return float(logsumexp(logpmf))


def _duration_table(
    mean_dwell_frames: np.ndarray,
    concentration: float,
    max_dur: int,
) -> np.ndarray:
    """Precompute the censored duration table ``log_dur[s, d-1]`` for ``d = 1..max_dur``.

    Returns a ``(S, max_dur)`` table so the segmental DP can index durations in
    O(1). Entries for ``d = 1 .. max_dur - 1`` are the ordinary duration log-pmf
    ``duration_logpmf(d | s)`` (the segment *ended* at exactly ``d`` frames). The
    final entry ``d = max_dur`` is the **survival** ``log P(duration >= max_dur)``
    (the log survival function), i.e. the segment is *right-censored* at the cap and
    has not necessarily ended.

    This is the standard EDHMM duration-censoring treatment and it is what lets a
    state persist *beyond* ``max_dur``: a censored (length-``max_dur``) segment may
    chain into a same-state continuation, so a genuine bout longer than the cap is
    represented exactly instead of being corrupted by a spurious state flip at the
    boundary (see :func:`hsmm_viterbi`). The segmental DP pairs the ``d = max_dur``
    column with same-state continuation; the ``d < max_dur`` columns force a switch.
    """
    S = mean_dwell_frames.shape[0]
    table = np.empty((S, max_dur), dtype=float)
    if max_dur >= 2:
        durations = np.arange(1, max_dur, dtype=float)  # d = 1 .. max_dur-1
        for s in range(S):
            table[s, : max_dur - 1] = duration_logpmf(
                durations, float(mean_dwell_frames[s]), concentration
            )
    for s in range(S):
        # The boundary (cap) segment is right-censored: score its survival mass.
        table[s, max_dur - 1] = _duration_logsf(
            max_dur, float(mean_dwell_frames[s]), concentration
        )
    return table


def _default_max_dur(mean_dwell_frames: np.ndarray, max_dur: int | None) -> int:
    """Choose the duration cap ``D`` (a generous multiple of the largest mean dwell).

    ``max_dur`` is a *tractability* cap on the per-segment duration loop, NOT a hard
    maximum bout length: thanks to duration censoring (a segment that hits the cap is
    scored by its survival mass and may continue as the same state -- see
    :func:`_duration_table` and :func:`hsmm_viterbi`), a single state can persist for
    arbitrarily many frames, so a genuine contact longer than ``D`` is represented
    exactly rather than truncated. The default is scaled off the *largest* per-state
    mean (so no state's typical dwell is unduly censored) and kept generous; an
    explicit ``max_dur`` is honoured but never silently corrupts long bouts.
    """
    if max_dur is not None:
        return int(max(1, max_dur))
    # Several times the largest mean comfortably covers the bulk of the NB mass, so
    # only genuinely long (and then censored, not truncated) bouts ever hit the cap.
    return int(max(1, np.ceil(5.0 * float(np.max(mean_dwell_frames)))))


def _prep(
    log_emission: np.ndarray,
    log_trans: np.ndarray,
    log_init: np.ndarray,
    mean_dwell_frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Coerce/validate the common HSMM inputs; return ``(emit, A, init, mean, T, S)``.

    Only the *time-homogeneous* ``(S, S)`` transition layout is supported here.
    The semi-Markov factorisation pulls the persistence (self-transition) out of
    the matrix and into the explicit duration model, so the transition matrix is
    used purely for *inter-segment* (state-to-state, ``s != s'``) jumps; a
    time-varying guard would change meaning under that factorisation and is left
    to the plain HMM in ``contact.hmm``.
    """
    log_emission = np.asarray(log_emission, dtype=float)
    if log_emission.ndim != 2:
        raise ValueError(f"log_emission must be 2-D (T, S); got {log_emission.shape}")
    T, S = log_emission.shape
    if T == 0 or S == 0:
        raise ValueError(f"log_emission must be non-empty; got {log_emission.shape}")

    log_trans = np.asarray(log_trans, dtype=float)
    if log_trans.shape != (S, S):
        raise ValueError(
            f"hsmm requires a time-homogeneous transition matrix of shape "
            f"(S, S)=({S}, {S}); got {log_trans.shape}"
        )

    log_init = np.asarray(log_init, dtype=float)
    if log_init.shape != (S,):
        raise ValueError(f"log_init must have shape (S,)=({S},); got {log_init.shape}")

    mean = np.asarray(mean_dwell_frames, dtype=float)
    if mean.shape != (S,):
        raise ValueError(
            f"mean_dwell_frames must have shape (S,)=({S},); got {mean.shape}"
        )

    return log_emission, log_trans, log_init, mean, T, S


def _interseg_logtrans(log_trans: np.ndarray) -> np.ndarray:
    """Transition matrix restricted to *between-segment* (state-changing) jumps.

    In an HSMM the dwell is owned by the duration model, so a "transition" between
    two *naturally ended* segments is by definition a move to a *different* state.
    We therefore zero out the diagonal (set ``log P(s -> s) = -inf``) and renormalise
    each row over the off-diagonal successors, so the inter-segment transition is a
    proper distribution over the ``s' != s`` it can jump to.

    Same-state continuation is NOT carried here: it only ever happens across a
    *duration-censored* boundary (a segment that hit the ``max_dur`` cap and so has
    not necessarily ended), where the continuation cost is 0 (no jump occurred). The
    segmental DPs handle that case explicitly, which is what lets a single state --
    including the degenerate single-state (``S == 1``) model, whose only row here is
    all ``-inf`` -- persist past the cap (see :func:`hsmm_viterbi`).
    """
    A = np.array(log_trans, dtype=float, copy=True)
    np.fill_diagonal(A, _LOG_ZERO)
    # Renormalise each row over its (off-diagonal) successors.
    row_norm = logsumexp(A, axis=1)  # (S,)
    # Rows that are entirely -inf (a single-state model) stay -inf; guard the divide.
    safe = np.where(np.isfinite(row_norm) & (row_norm > _LOG_ZERO / 2), row_norm, 0.0)
    A = A - safe[:, None]
    return A


# ======================================================================================
# Segmental Viterbi (MAP path)
# ======================================================================================

def hsmm_viterbi(
    log_emission: np.ndarray,
    log_trans: np.ndarray,
    log_init: np.ndarray,
    mean_dwell_frames: np.ndarray,
    concentration: float,
    max_dur: int | None = None,
) -> np.ndarray:
    """MAP state path under an explicit-duration semi-Markov model (THEORY.md s.5).

    The trajectory is modelled as a sequence of *segments*, each a maximal run of
    one state. A segment of state ``s`` covering frames ``[a, b]`` (length
    ``d = b - a + 1``) contributes three terms to the log-score:

        sum_{t=a..b} log_emission[t, s]          (the segment's emissions)
        + duration_logpmf(d | s)                 (the explicit dwell prior, s.5)
        + log P(prev_state -> s)                 (the inter-segment transition)

    and the very first segment uses ``log_init[s]`` in place of the transition.
    This is the standard segmental ("generalized Viterbi") dynamic program, with one
    essential refinement: **duration censoring**. The ``max_dur`` cap is a tractability
    bound on the inner duration loop, *not* a hard ceiling on bout length. A segment
    that runs the full ``D = max_dur`` frames is treated as *right-censored* (it has
    lasted at least ``D`` frames and has not necessarily ended): it scores the survival
    mass ``log P(d >= D)`` instead of the point pmf, and -- crucially -- it may be
    *continued by a same-state segment at zero transition cost*. A segment of length
    ``d < D`` ended naturally and must hand off to a *different* state. This censoring
    is what lets a single state span a run longer than ``D`` without the spurious
    state flip that a hard cap would otherwise force into the middle of a real bout.

    We track, per ``(t, s)``, two "flavours" of segment ending:

        V_end[t, s]  = best score of a path covering frames 0..t-1 whose last segment
                       is state s and *ended naturally* (its length d was < D).
        V_cens[t, s] = ... whose last segment is state s and *hit the cap* (d == D,
                       right-censored, so it may continue as the same state).

    With prefix sums ``E[t, s] = sum_{u<t} emit[u, s]`` and the off-diagonal
    (state-changing) transition ``A[s', s]``, a new segment of state s starting at
    ``t - d`` may be entered from:

        * the start of the record (only if ``t - d == 0``): cost ``log_init[s]``;
        * a *switch* from any other state's best path ending at ``t - d``:
          ``max_{s' != s} ( max(V_end, V_cens)[t-d, s'] + A[s', s] )``; or
        * a same-state *continuation* across a censored boundary:
          ``V_cens[t-d, s]`` at zero transition cost.

    and the segment then adds ``(E[t, s] - E[t-d, s]) + log_dur[s, d-1]`` (its
    emissions and the duration term, which is the pmf for ``d < D`` and the survival
    for ``d == D``). The result lands in ``V_cens`` if ``d == D`` else ``V_end``.

    We store, for each ``(t, s, flavour)``, the chosen duration, predecessor state and
    predecessor flavour to backtrace whole segments. The result is the single most
    likely *contiguous* mode sequence -- the clean boolean segmentation of s.5 -- with
    short spurious segments suppressed by the duration prior (not a morphological
    clean-up) and long genuine bouts represented exactly (not truncated at the cap).
    Single-state (``S == 1``) models work too: every entry into a state is then a
    censored continuation, so the state simply persists for the whole record.

    Complexity is ``O(T * S * max_dur * S)`` in the worst case (the inner argmax
    over predecessors is vectorised), which the ``max_dur`` cap keeps tractable.

    Parameters
    ----------
    log_emission:
        ``(T, S)`` log-emission likelihoods.
    log_trans:
        ``(S, S)`` time-homogeneous log-transitions. The diagonal (self-loop) is
        ignored -- persistence is the duration model's job -- and each row is
        renormalised over its off-diagonal successors.
    log_init:
        ``(S,)`` log initial-state prior (the first segment's entry cost).
    mean_dwell_frames:
        ``(S,)`` target mean dwell per state, in frames.
    concentration:
        Duration-distribution sharpness (see ``duration_logpmf``).
    max_dur:
        Duration-loop cap in frames (a tractability bound, NOT a maximum bout
        length: censored segments persist past it). Defaults to ``ceil(5 * max mean
        dwell)``.

    Returns
    -------
    np.ndarray
        MAP state path, ``int`` array of shape ``(T,)``.
    """
    emit, log_trans, log_init, mean, T, S = _prep(
        log_emission, log_trans, log_init, mean_dwell_frames
    )
    D = _default_max_dur(mean, max_dur)
    log_dur = _duration_table(mean, concentration, D)            # (S, D); last col = survival
    A = _interseg_logtrans(log_trans)                            # (S', S), off-diagonal switch

    # Prefix sums of emissions: E[t, s] = sum_{u=0..t-1} emit[u, s], so the
    # emission mass of frames [t-d .. t-1] is E[t] - E[t-d]. Shape (T+1, S).
    E = np.zeros((T + 1, S), dtype=float)
    np.cumsum(emit, axis=0, out=E[1:])

    NEG = _LOG_ZERO
    # Two flavours of "a segment of state s ends at frame t-1": natural (d<D) and
    # censored (d==D). Defined for t = 1..T; t=0 is the empty prefix.
    V_end = np.full((T + 1, S), NEG, dtype=float)
    V_cens = np.full((T + 1, S), NEG, dtype=float)
    # Backtrace per flavour: chosen duration d, predecessor state (-1 = "start"),
    # and predecessor flavour (0 = natural-end, 1 = censored).
    bd_end = np.zeros((T + 1, S), dtype=np.intp)
    bp_end = np.full((T + 1, S), -1, dtype=np.intp)
    bf_end = np.zeros((T + 1, S), dtype=np.intp)
    bd_cens = np.zeros((T + 1, S), dtype=np.intp)
    bp_cens = np.full((T + 1, S), -1, dtype=np.intp)
    bf_cens = np.zeros((T + 1, S), dtype=np.intp)

    s_range = np.arange(S)
    for t in range(1, T + 1):
        d_max = min(t, D)
        # seg_emit[d-1, s] = E[t, s] - E[t-d, s]  for d = 1..d_max  -> (d_max, S).
        seg_emit = E[t][None, :] - E[t - d_max : t][::-1, :]
        dur_term = log_dur[:, :d_max].T                          # (d_max, S)

        # best path ending at frame (t-d)-1 in any flavour:
        # tot_prev[d-1, s'] = max(V_end, V_cens)[t-d, s'].
        tot = np.maximum(V_end, V_cens)                          # (T+1, S)
        tot_prev = tot[t - d_max : t][::-1, :]                   # (d_max, S')
        cens_prev = V_cens[t - d_max : t][::-1, :]               # (d_max, S) same-state cont.

        # --- entry value into state s for each duration d ------------------------
        # switch_in[d-1, s] = max_{s'} ( tot_prev[d-1, s'] + A[s', s] ). A has -inf on
        # its diagonal, so s' == s is automatically excluded (a switch must change state).
        trans_score = tot_prev[:, :, None] + A[None, :, :]       # (d_max, S', S)
        switch_in = np.max(trans_score, axis=1)                  # (d_max, S)
        switch_idx = np.argmax(trans_score, axis=1)              # (d_max, S) -> s'

        # cont_in[d-1, s] = V_cens[t-d, s]  (same-state continuation, zero cost).
        cont_in = cens_prev                                      # (d_max, S)

        # The best predecessor for entering (d, s): a switch (flavour irrelevant for
        # the *successor*; we record the source state) or a censored continuation.
        entry = np.maximum(switch_in, cont_in)                   # (d_max, S)

        # --- first-segment (start) candidate: only when t - d == 0, i.e. d == t ---
        if d_max == t:
            entry = entry.copy()
            start_vals = np.maximum(log_init, entry[t - 1])
            entry[t - 1] = start_vals

        cand = entry + seg_emit + dur_term                       # (d_max, S)

        # Resolve, per duration and state, which predecessor won (for the backtrace):
        # prev_state, prev_flavour. Default: a switch -> source state, prev flavour is
        # whichever flavour was larger at that source; same-state continuation -> s, cens.
        use_cont = cont_in >= switch_in                          # (d_max, S)
        # for the switch winner, prev flavour = argmax of (V_end, V_cens) at source.
        src = switch_idx                                         # (d_max, S)
        # gather V_end/V_cens at (t-d, src) to decide the source flavour
        idx_d = np.arange(d_max)[:, None]
        # absolute time index of predecessor end = t - (d) = t - (idx_d+1)
        t_pred = (t - (idx_d + 1))                               # (d_max, 1)
        ve_src = V_end[t_pred, src]                              # (d_max, S)
        vc_src = V_cens[t_pred, src]                             # (d_max, S)
        switch_pflav = (vc_src > ve_src).astype(np.intp)         # (d_max, S)

        pstate = np.where(use_cont, s_range[None, :], src)       # (d_max, S)
        pflav = np.where(use_cont, 1, switch_pflav)              # (d_max, S)

        if d_max == t:
            # at d == t the start candidate may beat both; mark predecessor = -1.
            is_start = start_vals >= np.maximum(switch_in[t - 1], cont_in[t - 1])
            pstate[t - 1] = np.where(is_start, -1, pstate[t - 1])
            pflav[t - 1] = np.where(is_start, 0, pflav[t - 1])

        # --- split candidate scores into the two successor flavours --------------
        # d == D (index D-1, present only when d_max == D) is the censored flavour;
        # all shorter d are natural-end. Maximise within each flavour over duration.
        is_cap = d_max == D
        if is_cap:
            # natural-end candidates: durations 1..D-1 (indices 0..D-2).
            nat = cand[: D - 1] if D >= 2 else np.full((0, S), NEG)
            cens = cand[D - 1]                                   # (S,) the single cap duration
        else:
            nat = cand                                           # all durations are natural
            cens = None

        # natural-end winner
        if nat.shape[0] > 0:
            best_nat = np.argmax(nat, axis=0)                    # (S,)
            V_end[t] = nat[best_nat, s_range]
            bd_end[t] = best_nat + 1
            bp_end[t] = pstate[best_nat, s_range]
            bf_end[t] = pflav[best_nat, s_range]
        # censored winner (single duration d == D)
        if is_cap and cens is not None:
            V_cens[t] = cens
            bd_cens[t] = D
            bp_cens[t] = pstate[D - 1]
            bf_cens[t] = pflav[D - 1]

    # --- Backtrace whole segments from the best terminal (state, flavour) at T -----
    term_end = V_end[T]
    term_cens = V_cens[T]
    s = int(np.argmax(np.maximum(term_end, term_cens)))
    flav = 1 if term_cens[s] > term_end[s] else 0

    path = np.empty(T, dtype=np.intp)
    t = T
    while t > 0:
        # Pick the backtrace tables for the flavour of the segment ending at frame t-1.
        bd, bp, bf = (bd_cens, bp_cens, bf_cens) if flav == 1 else (bd_end, bp_end, bf_end)
        d = int(bd[t, s])
        prev_s = int(bp[t, s])
        prev_f = int(bf[t, s])
        path[t - d : t] = s
        t -= d
        if prev_s < 0:
            break  # reached the start segment
        s = prev_s
        flav = prev_f
    return path.astype(int)


# ======================================================================================
# Segmental forward-backward (per-frame posteriors)
# ======================================================================================

def hsmm_posteriors(
    log_emission: np.ndarray,
    log_trans: np.ndarray,
    log_init: np.ndarray,
    mean_dwell_frames: np.ndarray,
    concentration: float,
    max_dur: int | None = None,
) -> tuple[np.ndarray, float]:
    """Per-frame state posteriors under the explicit-duration model (THEORY.md s.5).

    This is the *exact* segmental (explicit-duration) forward-backward with the same
    **duration censoring** as :func:`hsmm_viterbi` (a segment that runs the full
    ``D = max_dur`` frames is right-censored -- scored by the survival ``log P(d>=D)``
    and free to continue as the *same* state -- so a bout longer than the cap is
    represented exactly, not corrupted by a phantom flip). We therefore split the
    standard EDHMM end-variable into a *naturally-ended* and a *censored* flavour, all
    in log-space, with ``seg(a, b, s) = sum_{u=a..b-1} emit[u, s]`` and the off-diagonal
    (state-changing) transition ``A[s', s]``:

        alpha_star[t, s] = log p(o_0..o_{t-1}, a segment of state s *begins* at t)
        alpha_end[t, s]  = log p(o_0..o_{t-1}, a *natural* (d<D) segment of s ends at t-1)
        alpha_cens[t, s] = log p(o_0..o_{t-1}, a *censored* (d==D) segment of s ends at t-1)

        alpha_star[0, s] = log_init[s]
        alpha_end[t, s]  = logsumexp_{d=1..min(t,D-1)} ( alpha_star[t-d, s]
                              + duration_logpmf(d|s) + seg(t-d, t, s) )
        alpha_cens[t, s] = alpha_star[t-D, s] + logsf(D|s) + seg(t-D, t, s)   (t >= D)
        alpha_star[t, s] = logsumexp_{s'!=s} ( logaddexp(alpha_end, alpha_cens)[t, s']
                              + A[s', s] )                       # a switch into s
                           (+) alpha_cens[t, s]                  # same-state continuation

    The total likelihood is ``logsumexp_s logaddexp(alpha_end, alpha_cens)[T, s]`` (a
    segment ends exactly at the last frame, in either flavour). The backward pass uses
    the post-end continuation values

        cont_nat[u, s]  = value after a *natural* end of s at u: terminate (0) if u==T,
                          else logsumexp_{s'!=s} ( A[s, s'] + beta[u, s'] )    (must switch)
        cont_cens[u, s] = value after a *censored* end of s at u: terminate (0) if u==T,
                          else logaddexp( the switch term above , beta[u, s] ) (switch or stay)
        beta[t, s]      = log p(o_t..o_{T-1} | a segment of state s begins at t)
                        = logsumexp_{d=1..min(T-t,D-1)} ( duration_logpmf(d|s)
                              + seg(t, t+d, s) + cont_nat[t+d, s] )
                          (+) [ if T-t >= D: logsf(D|s) + seg(t, t+D, s) + cont_cens[t+D, s] ]

    The per-frame posterior accumulates, in log-space, the occupancy of every
    (start t', duration d) segment of state s -- weight
    ``alpha_star[t', s] + dur(d|s) + seg(t', t'+d, s) + cont_*[t'+d, s] - loglik`` with
    ``cont_cens`` for the censored ``d == D`` and ``cont_nat`` otherwise -- and scatters
    it uniformly across frames ``t' .. t'+d-1``; normalising per frame gives
    ``gamma[t, s] = P(state_t = s | all observations)``.

    Complexity ``O(T * S * D)`` for the recursions (transitions add ``* S``),
    kept tractable by the ``max_dur`` cap.

    Parameters
    ----------
    log_emission, log_trans, log_init, mean_dwell_frames, concentration, max_dur:
        As in :func:`hsmm_viterbi`.

    Returns
    -------
    gamma:
        ``(T, S)`` per-frame posterior, row-normalised to sum to 1.
    loglik:
        Scalar ``log p(observations)``.
    """
    emit, log_trans, log_init, mean, T, S = _prep(
        log_emission, log_trans, log_init, mean_dwell_frames
    )
    D = _default_max_dur(mean, max_dur)
    log_dur = _duration_table(mean, concentration, D)            # (S, D); col D-1 = survival
    A = _interseg_logtrans(log_trans)                            # (S', S), off-diagonal switch

    # Prefix sums: E[t, s] = sum_{u<t} emit[u, s]; seg(a,b,s) = E[b,s]-E[a,s].
    E = np.zeros((T + 1, S), dtype=float)
    np.cumsum(emit, axis=0, out=E[1:])

    NEG = _LOG_ZERO

    # --- Forward pass -----------------------------------------------------------
    alpha_star = np.full((T + 1, S), NEG, dtype=float)  # segment of s begins at t
    alpha_end = np.full((T + 1, S), NEG, dtype=float)   # natural (d<D) end at t-1
    alpha_cens = np.full((T + 1, S), NEG, dtype=float)  # censored (d==D) end at t-1
    alpha_star[0] = log_init

    for t in range(1, T + 1):
        # Natural-end durations d = 1 .. min(t, D-1) use the duration *pmf*.
        dn = min(t, D - 1)
        if dn >= 1:
            seg_emit = E[t][None, :] - E[t - dn : t][::-1, :]    # (dn, S)
            as_prev = alpha_star[t - dn : t][::-1, :]            # (dn, S)
            dur_term = log_dur[:, :dn].T                         # (dn, S) pmf for d=1..dn
            alpha_end[t] = logsumexp(as_prev + dur_term + seg_emit, axis=0)  # (S,)
        # Censored duration d == D uses the survival mass (log_dur column D-1).
        if t >= D:
            seg_emit = E[t] - E[t - D]                           # (S,)
            alpha_cens[t] = alpha_star[t - D] + log_dur[:, D - 1] + seg_emit

        # A new segment can begin at t only if frames remain (t < T). A begin is either
        # a switch from any other state's end (either flavour) or a same-state
        # continuation across a censored boundary (zero transition cost).
        if t < T:
            tot_end = np.logaddexp(alpha_end[t], alpha_cens[t])  # (S,)
            switch = logsumexp(tot_end[:, None] + A, axis=0)     # (S,) over s' (A excludes s'=s)
            alpha_star[t] = np.logaddexp(switch, alpha_cens[t])  # switch (+) continuation

    loglik = float(logsumexp(np.logaddexp(alpha_end[T], alpha_cens[T])))

    # --- Backward pass ----------------------------------------------------------
    # beta[t, s] = log p(o_t..o_{T-1} | a segment of state s begins at t). The post-end
    # continuation values cont_nat / cont_cens are computed inline from beta[u].
    beta = np.full((T + 1, S), NEG, dtype=float)

    def _cont_nat(u: int) -> np.ndarray:
        """Value following a *natural* end of each state at frame u (must switch)."""
        if u == T:
            return np.zeros(S, dtype=float)  # the record terminates here
        return logsumexp(A + beta[u][None, :], axis=1)           # (S,) over s' (A excludes s'=s)

    def _cont_cens(u: int) -> np.ndarray:
        """Value following a *censored* end of each state at u (switch OR continue)."""
        if u == T:
            return np.zeros(S, dtype=float)
        switch = logsumexp(A + beta[u][None, :], axis=1)         # (S,)
        return np.logaddexp(switch, beta[u])                     # switch (+) same-state stay

    for t in range(T - 1, -1, -1):
        terms = []
        dn = min(T - t, D - 1)
        if dn >= 1:
            seg_emit = E[t + 1 : t + dn + 1, :] - E[t][None, :]  # (dn, S)
            dur_term = log_dur[:, :dn].T                         # (dn, S) pmf
            # cont_nat at each natural-end frame t+1 .. t+dn.
            cont = np.stack([_cont_nat(t + d) for d in range(1, dn + 1)], axis=0)  # (dn, S)
            terms.append(logsumexp(dur_term + seg_emit + cont, axis=0))            # (S,)
        if T - t >= D:
            seg_emit = E[t + D, :] - E[t, :]                     # (S,)
            terms.append(log_dur[:, D - 1] + seg_emit + _cont_cens(t + D))         # (S,)
        if terms:
            beta[t] = logsumexp(np.stack(terms, axis=0), axis=0)

    # --- Per-frame occupancy posterior -----------------------------------------
    # A segment of state s with start t' and duration d has normalised log-probability
    #   alpha_star[t', s] + log_dur[s, d-1] + seg(t', t'+d, s) + cont[t'+d, s] - loglik,
    # with cont = cont_cens for the censored d == D and cont_nat otherwise. It covers
    # frames t' .. t'+d-1; we scatter its (<= 1) probability uniformly across them.
    # Subtracting loglik makes the scores <= 0, so exponentiating is safe.
    cont_nat_tab = np.stack([_cont_nat(u) for u in range(T + 1)], axis=0)   # (T+1, S)
    cont_cens_tab = np.stack([_cont_cens(u) for u in range(T + 1)], axis=0)  # (T+1, S)

    gamma = np.zeros((T, S), dtype=float)
    for s in range(S):
        for tp in range(T):
            d_max = min(T - tp, D)
            if d_max < 1:
                continue
            seg_emit = E[tp + 1 : tp + d_max + 1, s] - E[tp, s]  # (d_max,)
            dur = log_dur[s, :d_max].copy()                      # (d_max,)
            # cont per duration: cont_cens for d == D (index D-1), cont_nat otherwise.
            cont = cont_nat_tab[tp + 1 : tp + d_max + 1, s].copy()  # (d_max,)
            if d_max == D:
                cont[D - 1] = cont_cens_tab[tp + D, s]
            seg_logp = alpha_star[tp, s] + dur + seg_emit + cont - loglik  # (d_max,)
            seg_p = np.exp(np.minimum(seg_logp, 0.0))            # (d_max,) probabilities
            # Segment of duration d (index d-1) covers frames tp .. tp+d-1. Adding
            # seg_p[d-1] to each of those frames is a suffix-cumulative scatter:
            # frame tp+j receives the sum of seg_p over d-1 >= j, i.e. reverse-cumsum.
            contrib = np.cumsum(seg_p[::-1])[::-1]               # (d_max,)
            gamma[tp : tp + d_max, s] += contrib

    # Normalise per frame (guard against tiny numerical drift / all-zero rows).
    row = gamma.sum(axis=1, keepdims=True)
    row = np.where(row > 0.0, row, 1.0)
    gamma /= row
    return gamma, loglik


# ======================================================================================
# Self-check (run: `python -m contact.hsmm`)
# ======================================================================================

def _selfcheck() -> None:
    """Sanity-check the duration model and the suppression of short spurious segments.

    Verifies (a) ``duration_logpmf`` is a normalised pmf with the requested mean,
    reduces to the *memoryless geometric* at ``concentration = 1`` (flat hazard) and
    sharpens as concentration grows; and (b) a single-frame blip in an otherwise-free
    record is absorbed by ``hsmm_viterbi`` / ``hsmm_posteriors`` while a genuine
    multi-frame bout is preserved -- the principled replacement for ``drop_short_runs``.
    """
    # -- duration distribution -------------------------------------------------
    d = np.arange(1, 30001)
    p = np.exp(duration_logpmf(d, 10.0, 4.0))
    assert abs(p.sum() - 1.0) < 1e-4, p.sum()
    assert abs((d * p).sum() - 10.0) < 1e-2, (d * p).sum()
    p1 = np.exp(duration_logpmf(d, 10.0, 1.0))           # concentration=1 => geometric
    surv = 1.0 - np.cumsum(p1) + p1
    hazard = p1[:60] / surv[:60]
    assert hazard.std() < 1e-6, hazard.std()             # memoryless: constant hazard
    var = lambda c: (lambda q, m: ((d - m) ** 2 * q).sum())(
        np.exp(duration_logpmf(d, 10.0, c)),
        (d * np.exp(duration_logpmf(d, 10.0, c))).sum(),
    )
    assert var(16.0) < var(1.0)                          # higher conc => tighter
    assert duration_logpmf(0, 10.0, 4.0) < -1e29 and duration_logpmf(2.5, 10.0, 4.0) < -1e29

    # -- blip absorption vs genuine bout --------------------------------------
    T, S = 60, 2
    log_trans = np.log([[0.5, 0.5], [0.5, 0.5]])
    log_init = np.log([0.5, 0.5])
    mean_dwell = np.array([60.0, 12.0])  # free likes long dwells; contact ~12-frame typical

    blip = np.full((T, S), 0.0)
    blip[:, 0], blip[:, 1] = np.log(0.6), np.log(0.4)    # mild free preference everywhere
    blip[30, 0], blip[30, 1] = np.log(0.02), np.log(0.98)  # strong 1-frame contact blip
    path = hsmm_viterbi(blip, log_trans, log_init, mean_dwell, concentration=8.0)
    assert np.all(path == 0), f"1-frame blip not absorbed: {path}"
    gamma, ll = hsmm_posteriors(blip, log_trans, log_init, mean_dwell, concentration=8.0)
    assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-6) and np.isfinite(ll)
    assert gamma[30, 1] < 0.5, gamma[30]

    bout = np.full((T, S), np.log(0.5))
    bout[20:40, 0], bout[20:40, 1] = np.log(0.2), np.log(0.8)  # genuine 20-frame contact
    path2 = hsmm_viterbi(bout, log_trans, log_init, mean_dwell, concentration=8.0)
    assert path2[20:40].mean() > 0.8, f"genuine bout lost: {path2}"
    gamma2, _ = hsmm_posteriors(bout, log_trans, log_init, mean_dwell, concentration=8.0)
    assert gamma2[20:40, 1].mean() > gamma2[:10, 1].mean()
    assert (np.argmax(gamma2, axis=1) == path2).mean() > 0.9  # viterbi/posterior agree

    print("contact.hsmm self-check passed: duration pmf normalized & geometric@conc=1; "
          "1-frame blip absorbed; 20-frame bout recovered; posteriors normalized.")


if __name__ == "__main__":
    _selfcheck()
