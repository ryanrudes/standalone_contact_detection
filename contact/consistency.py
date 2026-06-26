"""Soft global physics priors for the multi-body contact graph (THEORY.md section 8).

THEORY.md s.8 names two *global* consistency checks that link all the edges of the
contact graph together — quantities no single per-edge detector can see, because they
are properties of the whole scene at once:

  * an **energy / dissipation budget** — "a static contact dissipates nothing, sliding
    dissipates ``mu*lambda_n*||v_slip||``, an impact dissipates ``1/2 m v_n^2 (1-e^2)``"
    (s.8). Read as an inference factor: the scene's total mechanical energy
    ``E_mech(t) = sum_b (m_b g h_b + 1/2 m_b ||v_b||^2)`` may only *decrease* through a
    genuinely dissipative active contact (sliding / impact). An energy drop with *no*
    active contact to absorb it, or energy *spontaneously increasing*, is physically
    incoherent and should be (gently) penalized.
  * a **CoM-over-support-polygon balance** check — during a quasi-static stance the
    centre of mass should project inside the polygon spanned by the *active* contact
    points; an active set whose support polygon contains the CoM projection is rewarded.

Both are **best-effort soft factors** (a nudge, never a veto): they are log-additive
terms over the candidate active-set states that :mod:`contact.graph` enumerates, and
they return **all-zeros** (a true no-op) whenever the inputs are too poor to evaluate
them honestly. This is deliberate — s.8 lists them as *consistency checks* layered on
top of the per-edge emission/temporal evidence, not as a source of evidence on their
own. The per-edge :class:`contact.model.ContactDetector` posteriors remain the primary
signal; these factors only break ties the kinematics leave open.

Honest approximations (documented in full at each function):

  * **Masses.** If ``masses`` is ``None`` we use *unit* masses and label the energies
    ``"relative"``: the kinetic/potential *shape* over time is still meaningful (and the
    sign of dE/dt is unaffected by a global mass scale when all masses are equal), but
    the absolute Joules are not. The dissipation factor only ever uses the *sign and
    relative magnitude* of dE/dt, so it degrades gracefully.
  * **Body velocity.** We finite-difference the (already smoothed) body-origin world
    positions to get a translational kinetic energy ``1/2 m ||v_com||^2``. We *omit*
    rotational kinetic energy (we have no inertia tensors here) — a documented
    under-estimate of KE for spinning bodies. Potential energy is ``m g h`` with ``h``
    the world z of the body origin (gravity assumed ``-z``).
  * **Attribution.** We do not try to compute the *exact* dissipation rate of each
    contact (that needs the unobservable force magnitude, s.7). We only ask the much
    weaker, observable question: *is there an active dissipative edge available to
    explain the observed energy loss?* That keeps the factor honest about s.7's
    observability limits.
  * **Support polygon / CoM.** Without per-body masses the CoM is the unweighted mean of
    the body origins (documented). The support polygon is the convex hull (in the
    world horizontal plane) of the *active* edges' contact points; "inside" is tested
    with a standard point-in-convex-polygon test, softened by a margin so the reward is
    graded, not a step.

Public API (consumed by :mod:`contact.graph`):

    energy_budget(scene, masses=None) -> dict
    energy_log_factor(scene, edges, subset_index_per_state, masses=None) -> np.ndarray
    balance_log_factor(scene, edges, subset_index_per_state, support_polygon=None) -> np.ndarray

Only :mod:`contact.types`, :mod:`contact.config`, and numpy are imported (plus the leaf
:mod:`contact.signals` for time-aware smoothing/differentiation, with a pure-numpy
fallback so this module imports stand-alone).
"""

from __future__ import annotations

import numpy as np

from .types import ContactEdge, MultiBodyScene

__all__ = ["energy_budget", "energy_log_factor", "balance_log_factor"]

# Standard gravity (m/s^2); potential energy is m * G * h with h the world-z height.
_G = 9.81


# --------------------------------------------------------------------------------------
# Leaf numeric helpers — delegated to contact.signals where present (THEORY.md s.4: we
# smooth before differentiating because raw differentiation amplifies mocap noise). A
# pure-numpy fallback keeps this module importable on its own.
# --------------------------------------------------------------------------------------


def _resolve_signals():
    """Return ``(smooth_fn, derivative_fn)`` from :mod:`contact.signals`, else fallbacks.

    Mirrors :func:`contact.geometry._resolve_signals` so this module differentiates body
    trajectories through the same time-aware leaf the rest of the package uses.
    """

    def _local_smooth(x: np.ndarray, t: np.ndarray, sigma_time: float) -> np.ndarray:
        if sigma_time <= 0.0:
            return np.asarray(x, dtype=float)
        x = np.asarray(x, dtype=float)
        t = np.asarray(t, dtype=float)
        dt = t[:, None] - t[None, :]
        w = np.exp(-0.5 * (dt / sigma_time) ** 2)
        w /= w.sum(axis=1, keepdims=True)
        return w @ x

    def _local_deriv(x: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.gradient(np.asarray(x, dtype=float), np.asarray(t, dtype=float), axis=0)

    smooth_fn = _local_smooth
    deriv_fn = _local_deriv
    try:
        from . import signals as _signals
    except Exception:
        return smooth_fn, deriv_fn
    fn = getattr(_signals, "gaussian_smooth", None)
    if callable(fn):
        smooth_fn = fn
    fn = getattr(_signals, "derivative", None)
    if callable(fn):
        # contact.signals.derivative takes an optional smooth_time kwarg; the bare
        # (x, t) call uses np.gradient on the non-uniform clock, which is what we want
        # after we have already Gaussian-smoothed the positions.
        deriv_fn = fn
    return smooth_fn, deriv_fn


def _edge_ids(edges) -> list[str]:
    """Normalize the ``edges`` argument to an ordered list of edge-id strings.

    :mod:`contact.graph` may pass either a list of :class:`contact.types.ContactEdge`
    (the scene's ordered edges) or a list of edge-id strings. We accept both so the
    contract is forgiving; the *order* defines the column/edge ordering everything else
    aligns to.
    """
    ids: list[str] = []
    for e in edges:
        if isinstance(e, ContactEdge):
            ids.append(e.edge_id)
        else:
            ids.append(str(e))
    return ids


def _body_mass(masses, name: str) -> float:
    """Mass of body ``name`` from a ``masses`` mapping, defaulting to unit mass.

    ``masses`` may be a dict body-name -> mass (kg). A missing body or ``None`` map
    yields unit mass (the "relative energy" regime — see module docstring).
    """
    if masses is None:
        return 1.0
    try:
        m = float(masses[name])
    except (KeyError, TypeError, ValueError):
        return 1.0
    return m if np.isfinite(m) and m > 0.0 else 1.0


# --------------------------------------------------------------------------------------
# Energy budget (THEORY.md s.8: the raw material for the dissipation check).
# --------------------------------------------------------------------------------------


def energy_budget(scene: MultiBodyScene, masses=None) -> dict:
    """Per-body kinetic + potential energy over time, plus the scene total (THEORY.md s.8).

    For each body ``b`` with world-origin trajectory ``x_b(t)`` we compute

        PE_b(t) = m_b * G * h_b(t),         h_b = world-z of the body origin
        KE_b(t) = 1/2 * m_b * ||v_b(t)||^2, v_b = d/dt (smoothed) x_b
        E_b(t)  = PE_b(t) + KE_b(t)

    and the scene total mechanical energy ``E_mech(t) = sum_b E_b(t)``. Velocities come
    from Gaussian-smoothing the body positions (THEORY.md s.4) then differentiating on
    the body's own (possibly non-uniform) clock.

    Approximations (honest, per the module docstring):

      * **Translational KE only.** We have no inertia tensors, so rotational KE
        ``1/2 omega^T I omega`` is omitted. This *under-estimates* the KE of spinning
        bodies; the dissipation factor only uses the sign/shape of dE/dt, which is robust
        to a missing additive rotational term that is itself slowly varying for the quiet
        stances and steady slides s.8 cares about.
      * **Unit masses => "relative".** With ``masses=None`` every ``m_b = 1`` and the
        returned ``units`` field is ``"relative"`` (Joules only up to the unknown mass
        scale). With a ``masses`` map the energies are in Joules (``units="joule"``).
      * **Gravity is -z.** ``h`` is the raw world z of the body origin; an arbitrary
        datum only shifts PE by a constant and never affects dE/dt.

    Parameters
    ----------
    scene : MultiBodyScene
        Bodies (each a ``PoseTrajectory`` sharing a common time base) and candidate edges.
    masses : dict[str, float] | None
        Optional body-name -> mass (kg). ``None`` => unit masses (relative energy).

    Returns
    -------
    dict
        ``{"t": (T,), "bodies": {name: {"pe","ke","total","speed"}}, ``
        ``"E_mech": (T,), "dE": (T,), "units": "joule"|"relative", "masses": {name: m}}``.
        ``dE[k] = E_mech[k] - E_mech[k-1]`` (with ``dE[0] = 0``) is the per-frame change
        used by :func:`energy_log_factor`. Returns ``{}``-ish degenerate arrays (length-0
        or all-zero) gracefully when the scene has no usable bodies.
    """
    smooth_fn, deriv_fn = _resolve_signals()

    bodies = getattr(scene, "bodies", {}) or {}
    # Common time base: take it from any body (all share it per the MultiBodyScene
    # contract). If there are none, return an empty, harmless budget.
    t_ref: np.ndarray | None = None
    for traj in bodies.values():
        t_ref = np.asarray(traj.t, dtype=float).ravel()
        break
    if t_ref is None or t_ref.size == 0:
        return {
            "t": np.zeros(0),
            "bodies": {},
            "E_mech": np.zeros(0),
            "dE": np.zeros(0),
            "units": "relative" if masses is None else "joule",
            "masses": {},
        }

    T = t_ref.shape[0]
    # A representative smoothing time: short, like the per-edge geometry default (0.05 s),
    # but converted to be safe on the body's own clock. We smooth in real time (s.4).
    if T >= 2:
        dts = np.diff(t_ref)
        dts = dts[dts > 0.0]
        med_dt = float(np.median(dts)) if dts.size else 0.0
    else:
        med_dt = 0.0
    sigma_time = max(0.05, 3.0 * med_dt) if med_dt > 0.0 else 0.05

    per_body: dict[str, dict] = {}
    used_masses: dict[str, float] = {}
    E_mech = np.zeros(T, dtype=float)

    for name, traj in bodies.items():
        pos = np.asarray(traj.position, dtype=float)
        if pos.ndim != 2 or pos.shape[0] != T or pos.shape[1] < 3:
            # Body trajectory does not match the common length / is malformed: skip it
            # (its energy contribution is simply absent — a documented graceful no-op).
            continue
        m = _body_mass(masses, name)
        used_masses[name] = m

        # Smooth then differentiate the world position to get the COM velocity (s.4).
        pos_s = np.asarray(smooth_fn(pos, t_ref, sigma_time), dtype=float)
        vel = np.asarray(deriv_fn(pos_s, t_ref), dtype=float)        # (T, 3) world
        speed2 = np.sum(vel * vel, axis=1)                            # (T,) ||v||^2
        ke = 0.5 * m * speed2                                         # (T,)
        h = pos[:, 2]                                                 # world z (gravity -z)
        pe = m * _G * h                                               # (T,)
        total = pe + ke

        per_body[name] = {
            "pe": pe,
            "ke": ke,
            "total": total,
            "speed": np.sqrt(speed2),
        }
        E_mech += total

    # Per-frame change in total mechanical energy (dE[0] := 0). This is the quantity the
    # dissipation factor reads: dE < 0 is an energy LOSS (needs a dissipative contact),
    # dE > 0 is a spontaneous GAIN (physically incoherent without an external drive).
    dE = np.zeros(T, dtype=float)
    if T >= 2:
        dE[1:] = np.diff(E_mech)

    return {
        "t": t_ref,
        "bodies": per_body,
        "E_mech": E_mech,
        "dE": dE,
        "units": "relative" if masses is None else "joule",
        "masses": used_masses,
        "sigma_time": sigma_time,
    }


# --------------------------------------------------------------------------------------
# Internal: which states carry a *dissipative* edge?
# --------------------------------------------------------------------------------------


def _normalize_subset_index(
    subset_index_per_state,
    valid_ids: set[str],
    ids_order: list[str] | None = None,
) -> list[set[str]]:
    """Coerce ``subset_index_per_state`` into a list of *sets of active edge ids*.

    :mod:`contact.graph` documents ``subset_index_per_state`` as "a list aligned with the
    state alphabet" mapping each joint-inference state index to the set of active edges.
    We accept the natural encodings — a list/tuple of iterables, or of single entries, or
    even a ``dict`` keyed by state index — where each *entry element* is either an edge-id
    string OR an **integer edge index** into ``ids_order`` (the encoding that
    :func:`contact.graph._enumerate_subsets` actually emits: tuples of integer indices
    like ``(), (0,), (1,), (0, 1)``). Integer indices are resolved to their edge-id string
    via ``ids_order`` (the ordered edge-id list). Everything is intersected against
    ``valid_ids`` so a stray/unknown id never crashes the factor (it is simply ignored).
    """
    if subset_index_per_state is None:
        return []
    ids_order = list(ids_order) if ids_order is not None else None

    def _resolve(x):
        """Map one entry element to an edge-id string (index -> ids_order[x], else str)."""
        # bool is an int subclass; exclude it so a stray True/False is not an index.
        if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
            i = int(x)
            if ids_order is not None and 0 <= i < len(ids_order):
                return ids_order[i]
            return str(i)
        return str(x)

    # dict keyed by integer state index -> iterable of edge entries.
    if isinstance(subset_index_per_state, dict):
        n = (max(subset_index_per_state.keys()) + 1) if subset_index_per_state else 0
        items = [subset_index_per_state.get(i, ()) for i in range(n)]
    else:
        items = list(subset_index_per_state)

    out: list[set[str]] = []
    for entry in items:
        if entry is None:
            out.append(set())
            continue
        if isinstance(entry, str):
            ids = {entry}
        elif isinstance(entry, (int, np.integer)) and not isinstance(entry, bool):
            ids = {_resolve(entry)}
        else:
            try:
                ids = {_resolve(x) for x in entry}
            except TypeError:
                ids = {_resolve(entry)}
        out.append({i for i in ids if i in valid_ids})
    return out


# --------------------------------------------------------------------------------------
# Energy / dissipation log-factor (THEORY.md s.8).
# --------------------------------------------------------------------------------------


def energy_log_factor(
    scene: MultiBodyScene,
    edges,
    subset_index_per_state,
    masses=None,
) -> np.ndarray:
    """Soft per-state log-factor enforcing the s.8 energy/dissipation budget.

    The physics (THEORY.md s.8): mechanical energy is conserved within a frictionless
    free flight, *dissipated* by a sliding or impacting contact, and *cannot
    spontaneously increase* in a passive scene. Turned into a soft factor over the
    candidate active-set states:

      * **Energy dropping (``dE < 0``).** A state whose active set contains at least one
        *potentially dissipative* edge is rewarded (it can absorb the loss); a state with
        an empty active set is penalized (nothing to dissipate into). The reward scales
        with the size of the drop, saturating so it stays gentle.
      * **Energy roughly constant (``dE ~ 0``).** Nearly no preference — a quiet static
        stance neither needs nor forbids an active contact on energy grounds. (Existence
        evidence for the stance comes from the per-edge detectors, not here.)
      * **Energy increasing (``dE > 0``).** Mild *uniform* penalty on the magnitude of the
        gain regardless of the active set, because a passive scene should not gain energy;
        we do not prefer any particular set here, we only down-weight the frame's overall
        plausibility (a soft "this frame is surprising" term that cancels in the per-frame
        normalization but keeps the factor honest about sign).

    "Potentially dissipative" is judged *kinematically* (the only observable channel,
    s.7): an active edge can dissipate if its moving body is actually moving relative to
    the scene — we proxy this by the moving body's COM speed exceeding a small threshold.
    We deliberately do **not** try to compute ``mu*lambda_n*||v_slip||`` exactly, because
    ``lambda_n`` (the force magnitude) is unobservable from kinematics alone (s.7); the
    factor only asks the weaker, recoverable question *is a dissipative channel available?*

    Gentleness. The reward/penalty is a bounded logistic-style nudge scaled by
    ``_ENERGY_GAIN`` (a fraction of a nat), so it can tip a genuinely ambiguous tie toward
    the physically coherent active set but can never override confident per-edge evidence.
    Per-frame the returned column is **mean-centred** across states, so it is a pure
    *relative* preference (it adds no constant offset to the joint log-posterior).

    Parameters
    ----------
    scene : MultiBodyScene
    edges :
        Ordered candidate edges — a list of :class:`contact.types.ContactEdge` or of
        edge-id strings. Defines which moving body each edge id refers to.
    subset_index_per_state :
        Maps each joint-inference state index -> the set of active edge ids (see
        :func:`_normalize_subset_index` for accepted encodings).
    masses : dict[str, float] | None
        Optional body masses; ``None`` => unit masses (relative energy, see
        :func:`energy_budget`).

    Returns
    -------
    np.ndarray
        Shape ``(T, n_states)`` log-factor (per frame, per candidate active set). All
        zeros (a no-op) when it cannot be computed — no usable bodies/edges/states, or a
        degenerate single-state alphabet. Never a hard veto.
    """
    ids = _edge_ids(edges)
    states = _normalize_subset_index(subset_index_per_state, set(ids), ids_order=ids)
    n_states = len(states)

    budget = energy_budget(scene, masses=masses)
    t = np.asarray(budget.get("t", np.zeros(0)), dtype=float)
    T = t.shape[0]
    dE = np.asarray(budget.get("dE", np.zeros(0)), dtype=float)

    # No-op guards: nothing to score over.
    if T == 0 or n_states == 0:
        return np.zeros((max(T, 0), max(n_states, 0)), dtype=float)

    # --- per-edge "is this edge currently a dissipative channel?" (kinematic proxy) ---
    # An edge can dissipate only if its moving body is actually moving relative to the
    # scene. We use the moving body's COM speed from the energy budget. The threshold is
    # set from the energy budget's own velocity scale so it is unit-agnostic.
    per_body = budget.get("bodies", {})
    # Robust speed scale: the median nonzero body speed across the scene (fallback 0.05).
    all_speed = []
    for b in per_body.values():
        all_speed.append(np.asarray(b.get("speed", np.zeros(T)), dtype=float))
    if all_speed:
        sp = np.concatenate(all_speed)
        sp = sp[sp > 0.0]
        speed_scale = float(np.median(sp)) if sp.size else 0.05
    else:
        speed_scale = 0.05
    move_thresh = max(0.02, 0.25 * speed_scale)

    edge_to_moving: dict[str, str] = {}
    for e in edges:
        if isinstance(e, ContactEdge):
            edge_to_moving[e.edge_id] = e.moving_body

    # (T, n_edges-ish) -> but we only need, per state, whether ANY of its active edges is
    # moving above threshold at frame k. Precompute per-edge moving-mask once.
    edge_moving: dict[str, np.ndarray] = {}
    for eid in ids:
        mb = edge_to_moving.get(eid)
        spd = None
        if mb is not None and mb in per_body:
            spd = np.asarray(per_body[mb].get("speed", np.zeros(T)), dtype=float)
        if spd is None or spd.shape[0] != T:
            # Unknown moving body / speed: assume it *could* dissipate (be permissive so
            # we never wrongly penalize an edge whose body we simply cannot resolve).
            edge_moving[eid] = np.ones(T, dtype=bool)
        else:
            edge_moving[eid] = spd > move_thresh

    # state -> (T,) bool: does this state have a dissipative channel available each frame?
    state_dissipative = np.zeros((n_states, T), dtype=bool)
    for s_idx, active in enumerate(states):
        if not active:
            continue  # empty set: no dissipative channel, leave all-False
        avail = np.zeros(T, dtype=bool)
        for eid in active:
            avail |= edge_moving.get(eid, np.ones(T, dtype=bool))
        state_dissipative[s_idx] = avail

    # --- shape the energy change into a bounded "loss magnitude" signal --------------
    # Scale dE by a robust scale of |dE| so the factor is unit-agnostic (works for both
    # "relative" unit-mass energies and real Joules). drops below this scale are gentle;
    # bigger drops saturate via tanh so the factor never explodes.
    abs_dE = np.abs(dE)
    e_scale = float(np.median(abs_dE[abs_dE > 0.0])) if np.any(abs_dE > 0.0) else 0.0
    out = np.zeros((T, n_states), dtype=float)
    if e_scale <= 0.0:
        # Energy is flat across the whole record: nothing to arbitrate -> pure no-op.
        return out

    # Per-frame loss/gain in saturated, dimensionless units (in [-1, 1]).
    drop = np.tanh(np.maximum(0.0, -dE) / e_scale)   # >0 when energy is LOST
    gain = np.tanh(np.maximum(0.0, dE) / e_scale)    # >0 when energy is GAINED

    # Reward a dissipative-capable state on loss frames; penalize an incapable one.
    # +g*drop if dissipative else -g*drop ; the uniform gain penalty is added to all
    # states (it cancels under per-frame mean-centring but documents the sign of physics).
    for s_idx in range(n_states):
        diss = state_dissipative[s_idx].astype(float)         # (T,) 1 if can dissipate
        reward = _ENERGY_GAIN * drop * (2.0 * diss - 1.0)     # +/- on loss frames
        penalty = -_ENERGY_GAIN * gain                         # uniform on gain frames
        out[:, s_idx] = reward + penalty

    # Per-frame mean-centre across states: the factor is a *relative* preference only, it
    # must not bias the overall per-frame evidence mass (that belongs to the emissions).
    out -= out.mean(axis=1, keepdims=True)
    return out


#: Maximum per-frame nudge (nats) of the energy factor. A fraction of a nat: enough to
#: tip a genuine tie, far too small to override confident per-edge likelihood ratios.
_ENERGY_GAIN = 0.5


# --------------------------------------------------------------------------------------
# Geometry helpers for the balance factor.
# --------------------------------------------------------------------------------------


def _point_in_convex_polygon_margin(p_xy: np.ndarray, poly_xy: np.ndarray) -> float:
    """Signed inside-margin of point ``p_xy`` w.r.t. convex polygon ``poly_xy`` (m).

    Returns the signed distance from the point to the nearest polygon edge: ``> 0`` when
    the point is strictly inside (the value is how deep), ``<= 0`` when on/outside. Works
    for the degenerate low-vertex cases too:

      * 0 vertices -> ``-inf`` (no support at all),
      * 1 vertex   -> ``-distance`` to that point (a point support never contains an
        off-point CoM; only a CoM exactly over it has margin ~0),
      * 2 vertices -> ``-distance`` to the segment (a line support, never strictly
        contains a point off the line).

    For 3+ vertices we build the convex hull, orient it CCW, and take the minimum signed
    distance to its edges (positive interior). This is a standard, dependency-free
    point-in-convex-polygon margin.
    """
    poly = np.asarray(poly_xy, dtype=float).reshape(-1, 2)
    p = np.asarray(p_xy, dtype=float).reshape(2)
    n = poly.shape[0]
    if n == 0:
        return float("-inf")
    if n == 1:
        return -float(np.hypot(*(p - poly[0])))
    if n == 2:
        a, b = poly[0], poly[1]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 <= 1e-18:
            return -float(np.hypot(*(p - a)))
        s = float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
        proj = a + s * ab
        return -float(np.hypot(*(p - proj)))

    # 3+ points: convex hull (Andrew's monotone chain), then min signed edge distance.
    hull = _convex_hull(poly)
    if hull.shape[0] < 3:
        # Collinear points collapsed to a segment.
        return _point_in_convex_polygon_margin(p, hull)

    # Orient CCW so "inside" is consistently the +left side of every directed edge.
    if _signed_area(hull) < 0.0:
        hull = hull[::-1]

    margin = float("inf")
    m = hull.shape[0]
    for i in range(m):
        a = hull[i]
        b = hull[(i + 1) % m]
        e = b - a
        elen = float(np.hypot(*e))
        if elen <= 1e-18:
            continue
        # Signed distance to the line through edge (a->b), +ve on the interior (left) side.
        nrm = np.array([-e[1], e[0]]) / elen        # left normal of the directed edge
        d = float((p - a) @ nrm)
        margin = min(margin, d)
    return margin


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """2D convex hull (Andrew's monotone chain). Returns CCW-ish hull vertices ``(H, 2)``.

    Pure-numpy, no scipy dependency required for the small point sets here.
    """
    pts = np.unique(np.asarray(points, dtype=float).reshape(-1, 2), axis=0)
    if pts.shape[0] <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    return hull


def _signed_area(poly: np.ndarray) -> float:
    """Shoelace signed area of polygon ``poly`` (``(N, 2)``); ``> 0`` for CCW order."""
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _edge_contact_point_world(scene: MultiBodyScene, edge: ContactEdge, k: int) -> np.ndarray | None:
    """World position of an edge's tracked contact point at frame ``k`` (or ``None``).

    The contact point is the moving body's material point ``contact_point_local`` carried
    into the world by that body's pose at frame ``k`` (THEORY.md s.1, identical to
    :func:`contact.geometry.observe` step 1). Returns ``None`` if the body or frame is
    unavailable.
    """
    bodies = getattr(scene, "bodies", {}) or {}
    traj = bodies.get(edge.moving_body)
    if traj is None:
        return None
    pos = np.asarray(traj.position, dtype=float)
    quat = np.asarray(traj.quat, dtype=float)
    if pos.ndim != 2 or k >= pos.shape[0] or quat.ndim != 2 or k >= quat.shape[0]:
        return None
    cpl = np.asarray(edge.contact_point_local, dtype=float).reshape(3)
    q = quat[k]
    q = q / (np.linalg.norm(q) + 1e-300)
    w, x, y, z = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return pos[k] + R @ cpl


def _scene_com_world(scene: MultiBodyScene, masses, k: int) -> np.ndarray | None:
    """World centre of mass of the scene at frame ``k`` (or ``None`` if unavailable).

    With a ``masses`` map this is the mass-weighted mean of the body origins; with
    ``masses=None`` it is the *unweighted* mean (documented approximation — the
    geometric centroid of the body origins). Returns ``None`` when no body has a valid
    frame ``k``.
    """
    bodies = getattr(scene, "bodies", {}) or {}
    num = np.zeros(3, dtype=float)
    den = 0.0
    for name, traj in bodies.items():
        pos = np.asarray(traj.position, dtype=float)
        if pos.ndim != 2 or k >= pos.shape[0] or pos.shape[1] < 3:
            continue
        m = _body_mass(masses, name)
        num += m * pos[k]
        den += m
    if den <= 0.0:
        return None
    return num / den


# --------------------------------------------------------------------------------------
# Balance log-factor (THEORY.md s.8: CoM over the support polygon).
# --------------------------------------------------------------------------------------


def balance_log_factor(
    scene: MultiBodyScene,
    edges,
    subset_index_per_state,
    support_polygon=None,
) -> np.ndarray:
    """Soft per-state log-factor: CoM should project inside the active support polygon (s.8).

    The physics (THEORY.md s.8): during a *quasi-static* stance the scene's centre of mass
    must project (along gravity, i.e. into the world horizontal plane) inside the support
    polygon — the convex hull of the *active* contact points. An active set whose polygon
    contains the CoM projection is statically balanced and is rewarded; one that leaves the
    CoM projection outside its polygon is unbalanced and is penalized. This is a structural
    discriminator: it can favour "both feet active" over "one foot active" when only the
    two-foot polygon brackets the CoM.

    We build the support polygon per state per frame from that state's *active* edges'
    world contact points (their horizontal x/y projection, :func:`_edge_contact_point_world`),
    unless a fixed ``support_polygon`` is supplied (then that polygon is used for *every*
    active state and only the empty set is treated as "no support"). The CoM is
    :func:`_scene_com_world` (mass-weighted if ``masses`` are in ``scene.meta``, else the
    unweighted body-origin centroid — documented). "Inside" uses the graded signed margin
    of :func:`_point_in_convex_polygon_margin`, squashed by ``tanh(margin / scale)`` so the
    reward is smooth, not a step at the polygon boundary.

    Honest limits / no-op conditions (returns all-zeros, never a veto):

      * needs >= 1 active edge with a resolvable contact point to form any support;
      * a 1- or 2-point support can never strictly contain an off-axis CoM, so it gets a
        graded *negative* margin (correct: a single contact point is not a stable stance);
      * **quasi-static only.** The CoM-over-polygon law is a *static* balance statement; a
        dynamically accelerating scene legitimately violates it (a running CoM leaves the
        stance polygon). We therefore *down-weight* the whole factor on frames where the
        scene is clearly non-quasi-static, by scaling each frame's factor by a quiescence
        weight derived from CoM speed (fast CoM => factor -> 0). This keeps the prior from
        fighting honest dynamics, exactly the "best-effort soft factor" the spec asks for;
      * if masses/geometry are insufficient (no bodies, no CoM, no edges) the whole factor
        is zero.

    Parameters
    ----------
    scene : MultiBodyScene
    edges :
        Ordered candidate edges (``ContactEdge`` list or edge-id strings).
    subset_index_per_state :
        State index -> set of active edge ids.
    support_polygon : np.ndarray | None
        Optional fixed ``(P, 2)`` (or ``(P, 3)``, z ignored) support-polygon corners in
        world horizontal coords, overriding the per-state hull of active contact points.

    Returns
    -------
    np.ndarray
        Shape ``(T, n_states)`` log-factor, per-frame mean-centred across states (a pure
        relative preference). All zeros when it cannot be computed.
    """
    edge_list = [e for e in edges if isinstance(e, ContactEdge)]
    ids = _edge_ids(edges)
    id_to_edge = {e.edge_id: e for e in edge_list}
    states = _normalize_subset_index(subset_index_per_state, set(ids), ids_order=ids)
    n_states = len(states)

    bodies = getattr(scene, "bodies", {}) or {}
    # Common time base.
    t_ref: np.ndarray | None = None
    for traj in bodies.values():
        t_ref = np.asarray(traj.t, dtype=float).ravel()
        break
    if t_ref is None or t_ref.size == 0 or n_states == 0 or not edge_list:
        T0 = 0 if t_ref is None else int(t_ref.shape[0])
        return np.zeros((T0, n_states), dtype=float)
    T = t_ref.shape[0]

    # Masses (for the CoM) may live in scene.meta. None => unweighted centroid (documented).
    meta = getattr(scene, "meta", {}) or {}
    masses = meta.get("masses")

    # Optional fixed support polygon (world horizontal coords).
    fixed_poly = None
    if support_polygon is not None:
        fp = np.asarray(support_polygon, dtype=float)
        if fp.ndim == 2 and fp.shape[0] >= 1 and fp.shape[1] >= 2:
            fixed_poly = fp[:, :2]

    # --- quiescence weight (quasi-static gate, s.8): fast CoM => down-weight the factor.
    com_xy = np.full((T, 2), np.nan, dtype=float)
    for k in range(T):
        c = _scene_com_world(scene, masses, k)
        if c is not None:
            com_xy[k] = c[:2]
    # CoM horizontal speed from finite differences (no extra smoothing needed; this is a
    # coarse quasi-static gate, not a precise velocity).
    com_speed = np.zeros(T, dtype=float)
    if T >= 2:
        valid = np.all(np.isfinite(com_xy), axis=1)
        dt = np.diff(t_ref)
        dxy = np.diff(com_xy, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            spd = np.hypot(dxy[:, 0], dxy[:, 1]) / np.where(dt > 0, dt, np.nan)
        com_speed[1:] = np.where(np.isfinite(spd), spd, 0.0)
        com_speed[~valid] = 0.0
    # quiescence in [0, 1]: ~1 when the CoM is nearly still (quasi-static), -> 0 when it
    # moves fast. Scale relative to a slow walk (~0.3 m/s) so a planted stance counts.
    quies = np.exp(-(com_speed / 0.3) ** 2)

    # A length scale for the margin squash: a fraction of the support-polygon extent so
    # "well inside" saturates. Derived per frame from the active points; fall back to 0.1 m.
    out = np.zeros((T, n_states), dtype=float)

    for k in range(T):
        if not np.all(np.isfinite(com_xy[k])):
            continue
        p = com_xy[k]
        any_signal = False
        col = np.zeros(n_states, dtype=float)
        for s_idx, active in enumerate(states):
            if not active:
                # Empty set: no support at all -> strongly-but-softly "unbalanced".
                col[s_idx] = -_BALANCE_GAIN
                any_signal = True
                continue
            if fixed_poly is not None:
                poly = fixed_poly
            else:
                pts = []
                for eid in active:
                    e = id_to_edge.get(eid)
                    if e is None:
                        continue
                    cp = _edge_contact_point_world(scene, e, k)
                    if cp is not None:
                        pts.append(cp[:2])
                if not pts:
                    # Active set but no resolvable contact points: no opinion this state.
                    continue
                poly = np.asarray(pts, dtype=float)
            # Margin scale: half the polygon's bounding extent (>= 5 cm) so the squash is
            # sized to the stance, not to absolute metres.
            ext = float(np.max(poly.max(axis=0) - poly.min(axis=0))) if poly.shape[0] >= 2 else 0.0
            scale = max(0.05, 0.5 * ext)
            margin = _point_in_convex_polygon_margin(p, poly)
            if not np.isfinite(margin):
                col[s_idx] = -_BALANCE_GAIN
            else:
                col[s_idx] = _BALANCE_GAIN * float(np.tanh(margin / scale))
            any_signal = True
        if any_signal:
            col -= col.mean()                     # relative preference only
            out[k] = quies[k] * col               # quasi-static gate (s.8)

    return out


#: Maximum per-frame nudge (nats) of the balance factor — like the energy factor, a
#: fraction of a nat so it nudges ties without overriding per-edge evidence.
_BALANCE_GAIN = 0.5
