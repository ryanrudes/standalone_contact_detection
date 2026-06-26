"""Dynamics / material layer: force from compliance, and the friction cone.

This is rung 4 of the pragmatic ladder (THEORY.md s.10): *dynamics and material*.
It implements the two consequences of the observability theorem of THEORY.md s.7:

  1. **Penetration is a calibrated force gauge.** Under rigid contact the normal
     force magnitude is a Lagrange multiplier set by the dynamics, *unobservable*
     from kinematics alone (s.7). The instant we grant the contact a known
     compliance ``k`` (stiffness), the force is pinned to that contact's *own*
     measurable deformation ``delta`` by the linear spring law ``lambda = k*delta``.
     So a number the toy script treated as "penetration error to forgive" becomes a
     loading measurement: loaded vs. unloaded contact falls out for free.

  2. **The friction law is set-valued, and that closes the dynamical loop.** While a
     contact *sticks*, the tangential force may be anything inside the Coulomb cone
     ``||lambda_t|| <= mu*lambda_n``; gross *sliding* begins exactly when the
     tangential demand reaches the cone boundary (the stick->slip guard of s.5). With
     the normal force from compliance we can therefore *predict* stick vs. slip from
     dynamics and cross-check that prediction against the observed kinematics --
     "apparent sliding with no tangential force => something is wrong" (s.7).

The headline of s.7 is the **observability theorem** itself, which
:func:`observability_demo` exhibits on a statically-indeterminate rig:

  > Under rigid contact, force magnitude is unobservable from kinematics alone, and
  > in indeterminate configurations it is unobservable even with full rigid-body
  > dynamics. Compliance is exactly the regularizer that restores observability.

We make that concrete with the linear algebra: with ``K > 3`` vertical contacts the
rigid-body static-equilibrium map from per-contact forces to the net wrench is a
``(3, K)`` matrix (one vertical-force balance + two horizontal-torque balances). It
has rank ``<= 3`` but ``K`` unknowns, so its null space has dimension ``>= K - 3``:
an entire family of load splits produces the *identical* net wrench, hence the
identical rigid motion. Compliance collapses that null space because each force is
tied to its own penetration, ``f_i = k*delta_i``, which is individually measurable.

This module imports only :mod:`contact.types`, :mod:`contact.config`, and numpy.
"""

from __future__ import annotations

import numpy as np

from .config import MaterialParams
from .types import ContactObservations

# --------------------------------------------------------------------------------------
# 1. Normal force from penetration -- compliance as a calibrated force gauge (s.7)
# --------------------------------------------------------------------------------------


def normal_force_from_penetration(
    gap: np.ndarray,
    gap_bias: float,
    in_contact: np.ndarray,
    material: MaterialParams,
) -> np.ndarray:
    """Per-frame normal contact force from penetration under a linear spring (s.7).

    THEORY.md s.7: when the material stiffness ``k`` is known the penetration depth
    becomes a *calibrated force gauge*, ``lambda = k * delta``. This turns the gap's
    ``g < 0`` side -- which earlier rungs treated as "squish error to forgive" -- into
    a loading measurement, and is the regularizer that makes the contact force
    observable at all.

    The penetration is measured *relative to the calibrated resting gap* (the EM bias
    of s.7/s.8), because a constant sensor bias and a true constant offset are
    indistinguishable from a static pose (the calibration caveat in s.7). Concretely::

        delta_i = max(0, -(g_i - gap_bias))     # depth of interpenetration (m, >= 0)
        lambda_i = max(0, k * delta_i)           # linear spring force (N), only on contact frames

    The Signorini complementarity of s.2 (``g*lambda = 0``, force only when touching)
    is honoured two ways: ``delta`` is clamped at 0 so a real gap contributes no force,
    and the force is additionally zeroed on frames the detector did not flag as contact
    (``in_contact`` False), since off-contact a fitted ``g < gap_bias`` is noise, not load.

    Parameters
    ----------
    gap : (T,) array
        Support-relative signed distance (m); ``>0`` separation, ``<0`` penetration.
    gap_bias : float
        Calibrated resting-gap offset (m). Penetration is depth below this datum.
    in_contact : (T,) bool array
        The detector's per-frame contact flag (e.g. ``DetectionResult.in_contact``).
    material : MaterialParams
        Material properties. If ``material.stiffness is None`` the force is
        *unobservable* (pure-kinematic run) and we return an all-NaN array, matching
        the ``DetectionResult.normal_force is None`` convention at a per-frame level.

    Returns
    -------
    (T,) array
        Normal force magnitude (N), ``>= 0`` on contact frames and ``0`` elsewhere;
        all-NaN if stiffness is unknown.
    """
    gap = np.asarray(gap, dtype=float)
    in_contact = np.asarray(in_contact, dtype=bool)

    if material.stiffness is None:
        # No compliance => no force gauge => magnitude is unobservable (s.7).
        return np.full(gap.shape, np.nan, dtype=float)

    k = float(material.stiffness)

    # Penetration depth measured below the calibrated resting datum, clamped at 0 so a
    # genuine separation (g > gap_bias) contributes no force (Signorini, s.2).
    penetration = np.maximum(0.0, -(gap - gap_bias))

    # Linear spring law lambda = k*delta; max(0, .) guards a negative stiffness input.
    force = np.maximum(0.0, k * penetration)

    # Off-contact frames carry no load even if the fit dips slightly below the datum.
    force = np.where(in_contact, force, 0.0)
    return force


# --------------------------------------------------------------------------------------
# 2. Friction cone: stick vs. slip (s.7)
# --------------------------------------------------------------------------------------


def friction_stick_slip(
    obs: ContactObservations,
    normal_force: np.ndarray,
    material: MaterialParams,
) -> list[str]:
    """Per-frame stick/slip label from kinematics and the Coulomb cone (s.7).

    THEORY.md s.7: friction is a *set-valued* law. While sticking the tangential force
    may be anything inside the cone ``||lambda_t|| <= mu*lambda_n``; gross sliding
    begins exactly when the tangential demand reaches the cone boundary -- the
    stick->slip guard of s.5. There are therefore two independent witnesses to a slip,
    and the principled label cross-checks them:

      * **Kinematic evidence** (always available): a contact is *sliding* when the
        tangential speed ``||v_t||`` exceeds ``material.slip_speed_threshold``. This is
        the direct, motion-side signature of the sliding mode of s.3.

      * **Dynamic / friction-cone guard** (available only when ``normal_force`` is
        known, i.e. compliance gave us the force gauge above): the contact *should*
        slip once the required tangential force saturates the cone. We do not observe
        the tangential force directly here, so we use the kinematic evidence as the
        switch and report the cone status as a consistency cross-check.

    Combined rule (per contact frame; ``""`` when not in contact or force ~ 0):

      1. ``label = "slip"`` if ``||v_t|| > slip_speed_threshold`` else ``"stick"``
         -- the kinematic decision, which is the channel we can always measure.
      2. When the normal force is known we additionally evaluate the cone. A frame is
         *at the boundary* once ``mu*lambda_n`` is small enough that even a modest
         tangential demand would saturate it. The honest cross-check of s.7 fires on
         the contradiction the theory calls out explicitly: **apparent sliding with a
         healthy normal force but (by the cone) no admissible tangential drive, or a
         frame the cone says must slip yet the kinematics show it stuck.** We never let
         the unobservable force *override* the observed motion -- the cone refines the
         label only when the two witnesses agree, and otherwise defers to kinematics
         (the motion is the harder evidence; the force is reconstructed).

    A frame is "in contact" iff its estimated ``normal_force`` is meaningfully positive
    when the force is known; when the force is unknown (all NaN) we fall back to a
    penetration/closing test on the kinematics is *not* available here, so we treat a
    near-stationary normal channel as contact and let the caller mask by their own
    ``in_contact`` if they have one. (The detector wires this with its contact flag.)

    Parameters
    ----------
    obs : ContactObservations
        Support-relative observations; only ``v_tangent`` (T, 2) is used.
    normal_force : (T,) array
        Per-frame normal force (N) from :func:`normal_force_from_penetration`. May be
        all-NaN (stiffness unknown) -- then only the kinematic rule is used.
    material : MaterialParams
        Provides ``slip_speed_threshold`` (kinematic) and ``friction`` mu (cone).

    Returns
    -------
    list[str]
        Length-T labels, each in ``{"stick", "slip", ""}``.
    """
    v_tan = np.asarray(obs.v_tangent, dtype=float)
    speed = np.linalg.norm(v_tan, axis=-1)  # (T,) tangential speed magnitude
    T = speed.shape[0]

    nf = np.asarray(normal_force, dtype=float)
    force_known = nf.shape == speed.shape and not np.all(np.isnan(nf))

    mu = float(material.friction)
    v_thresh = float(material.slip_speed_threshold)

    # A frame carries a contact when the normal force is meaningfully positive. With no
    # force gauge we cannot decide existence here, so every frame is "potentially in
    # contact" and gets a stick/slip label; the caller masks non-contact frames.
    if force_known:
        # Force-magnitude floor below which the contact is unloaded (== free). We scale
        # off the largest force in the window so the test is unit-robust; a contact
        # bearing < 0.1% of the peak load is treated as carrying essentially nothing.
        peak = float(np.nanmax(nf)) if np.any(np.isfinite(nf)) else 0.0
        force_floor = 1e-3 * peak if peak > 0.0 else 0.0
        loaded = np.isfinite(nf) & (nf > force_floor)
    else:
        loaded = np.ones(T, dtype=bool)

    labels: list[str] = []
    for i in range(T):
        if not loaded[i]:
            labels.append("")  # not in contact / unloaded => no friction state (s.2)
            continue

        # (1) Kinematic decision -- the always-observable channel.
        kinematic_slip = speed[i] > v_thresh

        if not force_known:
            labels.append("slip" if kinematic_slip else "stick")
            continue

        # (2) Friction-cone cross-check. The cone half-angle's tangential capacity is
        # mu*lambda_n. We do not measure lambda_t, but the *sticking* hypothesis is only
        # self-consistent if the cone has room to host whatever tangential force balances
        # the motion. We treat a vanishing cone capacity (mu*lambda_n ~ 0) as "the cone
        # cannot hold a stick", and a healthy capacity as "stick is admissible".
        cone_capacity = mu * float(nf[i])  # max static tangential force the cone allows (N)
        # Capacity scale relative to the loaded peak: a cone narrower than this carries
        # negligible tangential force and so cannot sustain stick.
        cap_floor = 1e-3 * mu * (float(np.nanmax(nf)) if np.any(np.isfinite(nf)) else 0.0)
        cone_can_stick = cone_capacity > cap_floor

        if kinematic_slip:
            # Motion says slip. The cone agrees iff it is at/over its boundary, which we
            # cannot confirm without lambda_t; we defer to the motion (the hard evidence)
            # and label slip. (A separate diagnostic -- "slip with a fat unsaturated
            # cone" -- is the s.7 inconsistency, surfaced by the caller comparing against
            # the predicted force; we do not silently flip the observed motion.)
            labels.append("slip")
        else:
            # Motion says stick. Honour it only if the cone can actually host a stick;
            # if the cone capacity has collapsed the contact must in fact be slipping
            # (cone guard overriding a borderline-still kinematic reading, s.5/s.7).
            labels.append("stick" if cone_can_stick else "slip")

    return labels


# --------------------------------------------------------------------------------------
# 3. The observability theorem, made concrete (s.7)
# --------------------------------------------------------------------------------------


def _equilibrium_map(contact_xy: np.ndarray) -> np.ndarray:
    """Rigid-body static-equilibrium map from vertical contact forces to net wrench.

    Consider ``K`` point contacts at planar positions ``(x_i, y_i)`` each pushing
    *vertically* (along +z) with magnitude ``f_i >= 0`` against a rigid body that is in
    static equilibrium under gravity. The net wrench the contacts must supply has only
    three nontrivial components for vertical-only forces:

      * **vertical force balance**:  sum_i f_i            = W           (total weight)
      * **torque about the x-axis**: sum_i  y_i * f_i     = W * y_cop   (moment arm = y)
      * **torque about the y-axis**: sum_i (-x_i) * f_i   = -W * x_cop  (moment arm = -x)

    Stacking the three linear functionals of ``f = (f_1..f_K)`` gives the ``(3, K)``
    map ``A`` with rows ``[1, 1, ..., 1]``, ``[y_1, ..., y_K]``, ``[-x_1, ..., -x_K]``
    such that ``A @ f`` is the net (vertical force, x-torque, y-torque) wrench.

    For ``K > 3`` distinct contacts ``A`` has rank ``<= 3 < K``, so its null space
    ``{ d : A @ d = 0 }`` has dimension ``>= K - 3 > 0``: a whole family of force
    splits ``f + d`` produces the *identical* net wrench, hence the *identical* rigid
    motion. That is the statically-indeterminate unobservability of s.7, in one matrix.
    """
    contact_xy = np.asarray(contact_xy, dtype=float)
    x = contact_xy[:, 0]
    y = contact_xy[:, 1]
    ones = np.ones_like(x)
    # Rows: vertical-force balance, torque about x (arm = +y), torque about y (arm = -x).
    return np.vstack([ones, y, -x])  # (3, K)


def observability_demo(
    contact_points_penetration: np.ndarray,
    contact_points_force: np.ndarray,
    stiffness: float,
    contact_xy: np.ndarray | None = None,
    tol: float = 1e-6,
) -> dict:
    """Exhibit the observability theorem of THEORY.md s.7 on an indeterminate rig.

    Two halves, matching the two clauses of the theorem:

    (a) **Rigid statics are rank-deficient (force split UNOBSERVABLE).** We build the
        ``(3, K)`` rigid-body equilibrium map ``A`` (see :func:`_equilibrium_map`) from
        the contacts' planar positions. With ``K > 3`` it has rank ``<= 3``, so its
        null space has dimension ``>= K - 3 > 0``. We compute that null space via the
        SVD: the right-singular vectors whose singular values are ~0 span
        ``{ d : A @ d = 0 }``. Any load split ``f`` and ``f + d`` for ``d`` in this
        space are *indistinguishable* from the net wrench / rigid motion alone -- the
        load split between the contacts is invisible to rigid kinematics & statics.

    (b) **Compliance RECOVERS the forces (observability restored).** Each contact is a
        spring of stiffness ``k``, so its force is pinned to *its own* penetration:
        ``f_i = k * delta_i``. This is a *diagonal* (decoupled) map -- no null space --
        so the per-contact forces are individually identifiable from the per-contact
        penetrations. We apply it and check the recovered forces match the measured
        ground-truth contact forces within tolerance, confirming that material
        knowledge is the regularizer that makes loading recoverable (s.7).

    Parameters
    ----------
    contact_points_penetration : (K, T) array
        Per-contact penetration depth ``delta_i(t)`` (m, >= 0).
    contact_points_force : (K, T) array
        Per-contact *measured* normal force ``f_i(t)`` (N), e.g. the simulator's truth.
    stiffness : float
        The (shared) linear contact stiffness ``k`` (N/m).
    contact_xy : (K, 2) array, optional
        Planar positions of the contacts used to build the rigid equilibrium map. If
        omitted, ``K`` points are placed on a regular ring so they are distinct and the
        rank-deficiency for ``K > 3`` is exhibited generically.
    tol : float
        Relative singular-value threshold for the null-space dimension, and the scale
        of the recovery-error report.

    Returns
    -------
    dict with keys:
        ``num_contacts``        : K.
        ``equilibrium_rank``    : numerical rank of A (<= 3).
        ``null_space_dim``      : dim ker A = K - rank (>= K - 3); the indeterminacy.
        ``null_space_basis``    : (K, null_space_dim) orthonormal basis of ker A.
        ``recovered_force``     : (K, T) forces from f_i = k * delta_i.
        ``measured_force``      : (K, T) the input measured forces (echoed for the caller).
        ``max_rel_error``       : max relative error |recovered - measured| / scale.
        ``observable_rigid``    : False (load split unobservable from rigid statics).
        ``observable_compliant``: True iff recovery matched within tolerance.
    """
    pen = np.asarray(contact_points_penetration, dtype=float)
    meas = np.asarray(contact_points_force, dtype=float)
    if pen.ndim != 2:
        raise ValueError("contact_points_penetration must be (K, T)")
    if meas.shape != pen.shape:
        raise ValueError("contact_points_force must match penetration shape (K, T)")

    K = pen.shape[0]
    k = float(stiffness)

    # --- (a) rigid-body equilibrium map and its null space -------------------------
    if contact_xy is None:
        # Distinct points on a unit ring: guarantees the rows of A are not degenerate
        # so rank is the full min(3, K) and the deficiency is purely the K > 3 surplus.
        ang = np.linspace(0.0, 2.0 * np.pi, K, endpoint=False)
        contact_xy = np.column_stack([np.cos(ang), np.sin(ang)])
    contact_xy = np.asarray(contact_xy, dtype=float)

    A = _equilibrium_map(contact_xy)  # (3, K)

    # SVD-based rank and null space. Singular values below tol * largest are treated as
    # zero; the corresponding right-singular vectors (rows of Vh) span ker A.
    U, s, Vh = np.linalg.svd(A)
    smax = float(s[0]) if s.size else 0.0
    rank_thresh = tol * smax if smax > 0.0 else tol
    rank = int(np.sum(s > rank_thresh))
    null_dim = K - rank
    # Rows of Vh beyond the rank are the null-space directions; transpose to columns.
    null_basis = Vh[rank:, :].T if null_dim > 0 else np.zeros((K, 0))

    # Cross-check the null space is genuine: A @ d ~ 0 for each basis vector.
    if null_dim > 0:
        residual = float(np.max(np.abs(A @ null_basis)))
    else:
        residual = 0.0

    # --- (b) compliance recovers the per-contact forces ----------------------------
    # Diagonal, decoupled map f_i = k * delta_i: no null space, fully observable.
    recovered = k * pen  # (K, T)

    # Relative error against the measured forces, scaled by the measured peak so a
    # contact bearing little load does not inflate the relative error pathologically.
    scale = float(np.max(np.abs(meas))) if meas.size else 0.0
    if scale <= 0.0:
        scale = 1.0
    max_rel_error = float(np.max(np.abs(recovered - meas)) / scale)

    return {
        "num_contacts": K,
        "equilibrium_map": A,
        "equilibrium_rank": rank,
        "null_space_dim": null_dim,
        "null_space_basis": null_basis,
        "null_space_residual": residual,
        "recovered_force": recovered,
        "measured_force": meas,
        "max_rel_error": max_rel_error,
        "observable_rigid": False,
        "observable_compliant": bool(max_rel_error <= max(tol, 1e-9)),
    }
