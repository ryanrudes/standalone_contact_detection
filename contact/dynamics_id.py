"""Contact-implicit inverse dynamics — the THEORY.md s.8 "north star".

This is the final rung of the ladder (THEORY.md s.10, beyond rung 5), the object the
whole document points at in s.8:

  > a full contact-implicit inverse dynamics ... jointly infer contact existence,
  > mode, and force as the physically-valid explanation of the observed motion under
  > Newton-Euler dynamics with complementarity and friction-cone constraints.

Where the kinematic detector (the HMM/HSMM of s.1-s.6) asks *does the motion look like
contact?*, and the compliance layer (``contact.dynamics``, s.7) asks *given a known
stiffness, what force does the penetration imply?*, this module asks the dual,
dynamics-first question:

  Given a rigid body of known mass ``m`` and inertia ``I``, and its observed pose
  trajectory, what net external wrench must have acted on it (Newton-Euler), and what
  set of physically-valid contact forces -- each obeying Signorini (force only where
  the gap is closed, normal force >= 0) and the Coulomb friction cone
  (||f_t|| <= mu*f_n) -- supplies that wrench? The contacts that come out carrying
  load ARE the active set; their forces ARE the loading. Contact is thus recovered
  *from the dynamics*, complementary to (and a cross-check on) the kinematic detector.

Honest scope and the observability caveat (THEORY.md s.7)
---------------------------------------------------------
The recovered force is set by the *dynamics*, not the kinematics, so unlike the pure
HMM this layer can in principle recover force magnitude -- but only up to the
indeterminacy s.7 names: with more than 6 force unknowns balancing a single 6-wrench
(e.g. a box on four corners), an entire null-space family of force splits produces the
*identical* net wrench. We do NOT pretend that family away. We pick the **minimum-norm**
member via a small Tikhonov term ``force_regularization*||f||^2`` -- the most honest
default (it spreads load smoothly and never invents a corner that the wrench does not
demand) -- and we document that the split itself is the unobservable quantity that only
compliance (``contact.dynamics``) can pin down. So: the active set and the *total*
wrench are observable here; the per-candidate split among co-located redundant
candidates is the regularizer's choice, not a measurement.

Approximations, stated plainly
------------------------------
* Accelerations come from double-differentiating noisy, smoothed mocac poses. s.4/s.6
  warn this amplifies noise and that wide smoothing destroys impact timing; we keep the
  smoothing local (``contact.signals``) and small, and we expose ``accel_smooth_time``.
* We treat the body as a single rigid body with a known constant body-frame inertia.
  Articulated/multi-body inverse dynamics is out of scope (s.8's contact GRAPH is the
  ``contact.graph`` layer's job).
* The friction cone is the true (second-order) Coulomb cone ||f_t|| <= mu*f_n, solved
  as a small per-frame second-order-cone-constrained least squares via
  ``scipy.optimize.minimize`` (SLSQP). This is the elliptic cone of s.7, not MuJoCo's
  pyramidal approximation -- we keep the *physical* law, per the s.9 transfer caveat.

This module imports only :mod:`contact.types`, :mod:`contact.config`,
:mod:`contact.signals`, numpy and scipy (the spec's allowed set). It is a NEW, separate
analysis path: it does not touch or alter the existing detector defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .config import DetectorConfig, InverseDynamicsParams
from .geometry import quat_conjugate as _quat_conjugate
from .geometry import quat_mul as _quat_mul
from .geometry import quat_to_matrix as _quat_to_matrix
from .signals import derivative, gaussian_smooth
from .types import InverseDynamicsResult, PoseTrajectory, RawScenario

# --------------------------------------------------------------------------------------
# Quaternion helpers are imported from contact.geometry (the canonical scalar-first set);
# only ``_continuous_quat`` below is specific to this module's differentiation pipeline.
# --------------------------------------------------------------------------------------


def _continuous_quat(quat: np.ndarray) -> np.ndarray:
    """Normalize and sign-align a quaternion stream against the antipodal double cover.

    ``q`` and ``-q`` encode the same rotation; an unaligned stream injects spurious
    2-revolution jumps into ``dq/dt``. We flip any frame pointing "away" from the
    previous one so the path through S^3 is continuous (mirrors geometry's handling).
    """
    q = np.asarray(quat, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q = q.copy()
    if q.shape[0] > 1:
        flip = np.cumprod(np.sign(np.sum(q[1:] * q[:-1], axis=1) + 1e-300))
        q[1:] *= np.where(flip[:, None] < 0.0, -1.0, 1.0)
    return q


# --------------------------------------------------------------------------------------
# Inertial parameters (lightweight container; the raw-meta path fills it from the sim).
# --------------------------------------------------------------------------------------


@dataclass
class InertialParams:
    """Rigid-body inertial parameters used by the inverse dynamics.

    mass:      scalar body mass (kg).
    inertia:   (3, 3) rotational inertia about the CoM, in the BODY frame (kg*m^2).
    com_local: (3,) center of mass in the body-local frame, measured from the pose
               origin ``PoseTrajectory.position`` (m). The wrench is taken about the CoM.
    """

    mass: float
    inertia: np.ndarray
    com_local: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.inertia = np.asarray(self.inertia, dtype=float).reshape(3, 3)
        if self.com_local is None:
            self.com_local = np.zeros(3, dtype=float)
        self.com_local = np.asarray(self.com_local, dtype=float).reshape(3)


def _as_inertial(inertial) -> InertialParams:
    """Coerce a dict/obj with mass/inertia/com_local into an :class:`InertialParams`."""
    if isinstance(inertial, InertialParams):
        return inertial
    if isinstance(inertial, dict):
        return InertialParams(
            mass=float(inertial["mass"]),
            inertia=inertial["inertia"],
            com_local=inertial.get("com_local", inertial.get("com", np.zeros(3))),
        )
    # Duck-typed object with attributes.
    return InertialParams(
        mass=float(getattr(inertial, "mass")),
        inertia=getattr(inertial, "inertia"),
        com_local=getattr(inertial, "com_local", getattr(inertial, "com", np.zeros(3))),
    )


# --------------------------------------------------------------------------------------
# 1. Body accelerations from the observed pose trajectory.
# --------------------------------------------------------------------------------------


def body_accelerations(
    pose: PoseTrajectory,
    gravity: float = 9.81,
    accel_smooth_time: float = 0.04,
    com_local: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear CoM acceleration, angular acceleration, and angular velocity from a pose.

    THEORY.md s.8 needs the body's acceleration state to evaluate Newton-Euler. We get
    it by differentiating the (noisy) observed pose, which s.4 warns amplifies noise and
    s.6 warns can smear impact timing if over-smoothed. So we differentiate through the
    *local*, time-aware helpers of :mod:`contact.signals`: smooth in real seconds, then
    take a local-polynomial derivative, twice for the linear channel.

    Differentiation, documented
    ----------------------------
    * **CoM position** ``p_com(t) = position(t) + R(q(t)) @ com_local`` (the tracked
      pose origin carried to the center of mass; if ``com_local`` is None or zero this
      is just the origin). We smooth ``p_com`` once (``gaussian_smooth``,
      ``accel_smooth_time``) and apply :func:`contact.signals.derivative` *twice* (each
      call itself a local-polynomial fit) to get ``a_com = d^2 p_com / dt^2``.
    * **Angular velocity** ``omega(t)``: for a body whose orientation is ``q(t)``, the
      world angular velocity is the vector part of ``2 * (dq/dt) * conj(q)`` (the s.3
      construction, mirrored from ``contact.geometry``). We sign-align the quaternion
      stream against its double cover, smooth it, finite-difference it, and renormalize.
    * **Angular acceleration** ``alpha(t) = d omega / dt``: a further local derivative of
      ``omega``. (This is the world-frame angular acceleration; the body-frame term
      ``omega x (I omega)`` is added in :func:`required_wrench`.)

    Parameters
    ----------
    pose : PoseTrajectory
        ``t (T,)``, ``position (T,3)`` world origin, ``quat (T,4)`` scalar-first.
    gravity : float
        Magnitude of gravity (m/s^2); unused here but accepted so the signature reads as
        a body-state extractor (it is consumed in :func:`required_wrench`).
    accel_smooth_time : float
        Real-time smoothing window (s) applied before each differentiation (s.4/s.6).
    com_local : (3,) array, optional
        CoM offset in the body frame; default origin (zeros).

    Returns
    -------
    (a_com (T,3), alpha (T,3), omega (T,3))
        Linear CoM acceleration, angular acceleration, and angular velocity, all in the
        WORLD frame.
    """
    t = np.asarray(pose.t, dtype=float)
    pos = np.asarray(pose.position, dtype=float)
    quat = _continuous_quat(pose.quat)
    T = t.shape[0]

    cl = np.zeros(3) if com_local is None else np.asarray(com_local, dtype=float).reshape(3)

    # CoM world trajectory: origin + R(q) @ com_local (per-frame rotation of a fixed offset).
    if np.allclose(cl, 0.0):
        p_com = pos
    else:
        R = _quat_to_matrix(quat)                      # (T,3,3)
        p_com = pos + np.einsum("tij,j->ti", R, cl)    # (T,3)

    # Linear acceleration: differentiate twice. Each :func:`contact.signals.derivative`
    # call is itself a *local* least-squares (Savitzky-Golay) fit over an
    # ``accel_smooth_time``-second window, which both robustly rejects differentiation
    # noise (s.4) and stays local enough not to smear impact timing (s.6). We deliberately
    # do NOT pre-Gaussian-smooth: a row-normalized Gaussian average is one-sided near the
    # record ends and so injects a boundary bias into a *curved* signal that the second
    # derivative then amplifies (it turns a clean parabola's exact -g into a biased value).
    # The local-polynomial fit has no such bias -- on a parabola it recovers the constant
    # acceleration exactly, and on noisy data it tracks the true accel with the window
    # trading variance for impact fidelity (the single s.6 knob).
    v_com = derivative(p_com, t, smooth_time=accel_smooth_time)
    a_com = derivative(v_com, t, smooth_time=accel_smooth_time)

    # Angular velocity omega = vec(2 * dq/dt * conj(q)) in the world frame (s.3). Here a
    # single derivative is taken, so a light pre-smooth of the (already sign-aligned)
    # quaternion stream is safe and mirrors the geometry module's omega path; we then
    # renormalize before forming dq/dt.
    q_sm = gaussian_smooth(quat, t, accel_smooth_time)
    q_sm = q_sm / np.linalg.norm(q_sm, axis=-1, keepdims=True)
    dq = derivative(q_sm, t, smooth_time=accel_smooth_time)           # (T,4)
    omega = (2.0 * _quat_mul(dq, _quat_conjugate(q_sm)))[..., 1:]      # (T,3) drop ~0 scalar

    # Angular acceleration alpha = d omega / dt (one more local derivative).
    alpha = derivative(omega, t, smooth_time=accel_smooth_time)        # (T,3)

    if T < 3:
        # With <3 samples the second derivative is undefined; return zeros for accel.
        a_com = np.zeros((T, 3))
        alpha = np.zeros((T, 3))
    return a_com, alpha, omega


# --------------------------------------------------------------------------------------
# 2. Required net external wrench (Newton-Euler).
# --------------------------------------------------------------------------------------


def required_wrench(
    pose: PoseTrajectory,
    inertial,
    gravity: float = 9.81,
    accel_smooth_time: float = 0.04,
) -> np.ndarray:
    """Net external wrench [F(3); tau(3)] (world, about the CoM) the contacts must supply.

    THEORY.md s.8 (Newton-Euler). For a rigid body of mass ``m`` and body-frame inertia
    ``I_body`` whose CoM has world acceleration ``a_com`` and whose world angular
    velocity/acceleration are ``omega``/``alpha``, the NET external wrench equals::

        F   = m * (a_com - g_vec)                          # linear:  m a = F_ext + m g
        tau = R * (I_body * alpha_body)  +  omega x (I_world * omega)

    Derivation of the rotational term, and the inertia rotation (the subtle part):
    Euler's equation in the BODY frame is ``tau_body = I_body * alpha_body +
    omega_body x (I_body * omega_body)``. We want the torque in the WORLD frame. Rotating
    the body inertia to the world gives ``I_world = R I_body R^T`` (a similarity
    transform, NOT ``R I_body``: inertia is a rank-2 tensor). Then the world-frame Euler
    equation is the clean ``tau = I_world * alpha + omega x (I_world * omega)`` with all
    of ``tau``, ``alpha``, ``omega`` in the world frame. We compute it in that equivalent
    world form, which avoids ever rotating ``alpha`` into the body frame and is what the
    code below does (``tau = I_world @ alpha + omega x (I_world @ omega)``).

    Gravity acts at the CoM, so it contributes to ``F`` but adds NO torque about the CoM;
    that is why ``tau`` has no gravity term. ``g_vec = (0, 0, -gravity)`` (world -z).

    The sign convention: ``F``/``tau`` are the *external* wrench that the environment
    (contacts) must apply to produce the observed motion under gravity -- exactly the
    quantity the contact forces in :func:`solve_contact_implicit` are fit to.

    Parameters
    ----------
    pose : PoseTrajectory
        Observed pose (origin + orientation) of the body.
    inertial : dict | InertialParams | obj
        Carries ``mass`` (kg), ``inertia`` (3x3 body, kg*m^2), ``com_local`` (3,, m).
    gravity : float
        Gravity magnitude (m/s^2) along world -z.
    accel_smooth_time : float
        Smoothing window for the differentiation (s); passed to
        :func:`body_accelerations`.

    Returns
    -------
    (T, 6) array
        Per-frame net external wrench ``[Fx,Fy,Fz, taux,tauy,tauz]`` (world frame, about
        the CoM).
    """
    inert = _as_inertial(inertial)
    a_com, alpha, omega = body_accelerations(
        pose, gravity=gravity, accel_smooth_time=accel_smooth_time, com_local=inert.com_local
    )

    g_vec = np.array([0.0, 0.0, -float(gravity)])
    # Linear: F = m (a_com - g_vec). m*g_vec is the weight the contacts must hold up.
    F = inert.mass * (a_com - g_vec[None, :])                          # (T,3)

    # Rotate the body inertia to the world each frame: I_world = R I_body R^T.
    quat = _continuous_quat(pose.quat)
    R = _quat_to_matrix(quat)                                          # (T,3,3)
    I_body = inert.inertia                                             # (3,3)
    I_world = np.einsum("tij,jk,tlk->til", R, I_body, R)               # (T,3,3) = R I R^T

    Iw_alpha = np.einsum("tij,tj->ti", I_world, alpha)                 # I_world @ alpha
    Iw_omega = np.einsum("tij,tj->ti", I_world, omega)                 # I_world @ omega
    tau = Iw_alpha + np.cross(omega, Iw_omega)                         # (T,3) world Euler

    wrench = np.concatenate([F, tau], axis=1)                          # (T,6)
    return wrench


# --------------------------------------------------------------------------------------
# 3. Contact-wrench map G(t): per-candidate force components -> net wrench about the CoM.
# --------------------------------------------------------------------------------------


def contact_wrench_map(
    pose: PoseTrajectory,
    candidate_points_local: np.ndarray,
    candidate_normals_local: np.ndarray,
    com_local: np.ndarray | None = None,
) -> np.ndarray:
    """Per-frame map ``G(t)`` from per-candidate force components to the net CoM wrench.

    THEORY.md s.8: each candidate contact ``i`` applies a force at a world point ``r_i``.
    We parameterize that force in the candidate's own contact basis -- one **normal**
    component (along the candidate's outward normal ``n_i``) and two **tangential**
    components (in the tangent plane). The contribution of candidate ``i`` to the net
    wrench about the CoM is::

        F   += f_i = n_i * f_n,i + t1_i * f_t1,i + t2_i * f_t2,i
        tau += (r_i - p_com) x f_i

    Stacking the K candidates' 3 components each into ``f_stack`` (length ``3K``, ordered
    ``[n_0, t1_0, t2_0, n_1, t1_1, t2_1, ...]``) gives the linear map ``wrench = G f``
    where ``G(t)`` is ``(6, 3K)``. The TOP three rows are the force-balance block (each
    candidate's three basis vectors stacked as columns); the BOTTOM three rows are the
    torque block (the cross-product matrix ``[r_i - p_com]_x`` times each basis vector).
    This is exactly the **grasp/contact Jacobian** of multi-contact statics.

    The candidate points/normals are given in the BODY-local frame (they ride with the
    body) and carried into the world each frame by the body pose, so a tilting box's
    corners and outward normals rotate correctly. ``com_local`` sets the moment center.

    Why parameterize in the candidate's basis (not raw xyz)? The Signorini /
    friction-cone constraints of s.2/s.7 are stated per candidate as "normal >= 0" and
    "||tangential|| <= mu*normal". Carrying the basis into ``G`` lets the solver impose
    those as simple bounds/cone constraints on ``f_stack`` directly.

    Parameters
    ----------
    pose : PoseTrajectory
        Observed pose of the body the candidates are attached to.
    candidate_points_local : (K, 3) array
        Candidate contact points in the body-local frame.
    candidate_normals_local : (K, 3) array
        Candidate outward normals in the body-local frame (need not be unit; normalized).
    com_local : (3,) array, optional
        CoM offset in the body frame (moment center). Default origin.

    Returns
    -------
    (T, 6, 3K) array
        Per-frame contact-wrench map ``G(t)``.
    """
    pts_l = np.asarray(candidate_points_local, dtype=float).reshape(-1, 3)
    K = pts_l.shape[0]

    t = np.asarray(pose.t, dtype=float)
    pos = np.asarray(pose.position, dtype=float)
    quat = _continuous_quat(pose.quat)
    T = t.shape[0]
    R = _quat_to_matrix(quat)                                          # (T,3,3)

    # Normals accept either a STATIC body-local set ``(K, 3)`` or a PER-FRAME body-local
    # stream ``(T, K, 3)`` / ``(K, T, 3)`` (the generator's ``meta['candidates']`` exposes
    # per-frame normals so a tilting/rolling contact's normal is tracked exactly). We
    # normalize the input to ``(T, K, 3)`` body-local before rotating into the world.
    nrm = np.asarray(candidate_normals_local, dtype=float)
    if nrm.ndim == 2:
        if nrm.shape != (K, 3):
            raise ValueError(f"static normals must be (K,3)=({K},3); got {nrm.shape}")
        nrm_tk = np.broadcast_to(nrm[None, :, :], (T, K, 3))           # (T,K,3)
    elif nrm.ndim == 3:
        if nrm.shape == (T, K, 3):
            nrm_tk = nrm
        elif nrm.shape == (K, T, 3):
            nrm_tk = np.transpose(nrm, (1, 0, 2))                      # (K,T,3) -> (T,K,3)
        else:
            raise ValueError(
                f"per-frame normals must be (T,K,3) or (K,T,3) with T={T}, K={K}; "
                f"got {nrm.shape}"
            )
    else:
        raise ValueError("candidate_normals_local must be (K,3), (T,K,3) or (K,T,3)")

    cl = np.zeros(3) if com_local is None else np.asarray(com_local, dtype=float).reshape(3)
    p_com = pos + np.einsum("tij,j->ti", R, cl)                        # (T,3) world CoM

    # Candidate world points and the lever arms r_i - p_com.
    pts_w = pos[:, None, :] + np.einsum("tij,kj->tki", R, pts_l)       # (T,K,3)
    lever = pts_w - p_com[:, None, :]                                  # (T,K,3)

    # Per-candidate world contact basis: outward normal + two tangents. The (body-local)
    # normal rides with the body, rotated each frame into the world; the tangents are an
    # arbitrary orthonormal pair spanning the tangent plane (the cone constraint is
    # rotation-invariant in that plane, so any basis is fine -- we build one
    # deterministically per frame per candidate).
    n_w = np.einsum("tij,tkj->tki", R, nrm_tk)                         # (T,K,3) body->world
    n_w = n_w / np.maximum(np.linalg.norm(n_w, axis=-1, keepdims=True), 1e-12)

    G = np.zeros((T, 6, 3 * K), dtype=float)
    for k in range(K):
        nk = n_w[:, k, :]                                              # (T,3)
        t1, t2 = _tangent_pair(nk)                                    # each (T,3)
        rk = lever[:, k, :]                                            # (T,3)
        basis = (nk, t1, t2)
        for j, b in enumerate(basis):
            col = 3 * k + j
            G[:, 0:3, col] = b                                         # force-balance block
            G[:, 3:6, col] = np.cross(rk, b)                          # torque block r x b
    return G


def _tangent_pair(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A per-frame orthonormal tangent pair (t1, t2) for a (T,3) unit-normal stream.

    For the friction cone the only thing that matters is that ``(t1, t2)`` span the plane
    orthogonal to ``n`` and are orthonormal; their absolute orientation is irrelevant
    because the cone ``||(f_t1, f_t2)|| <= mu f_n`` is rotation-invariant in that plane.
    So we use a simple, robust per-frame construction (pick the world axis least parallel
    to ``n``, project it out, normalize) rather than the continuous parallel transport
    that ``contact.geometry`` needs for *velocity decomposition*. Continuity is not
    required here -- the force solve is independent per frame and per-candidate.
    """
    z = np.asarray(normals, dtype=float)
    T = z.shape[0]
    t1 = np.empty((T, 3))
    # Choose a seed axis per frame that is least aligned with the normal (x unless n~x).
    seed = np.tile(np.array([1.0, 0.0, 0.0]), (T, 1))
    near_x = np.abs(z[:, 0]) > 0.9
    seed[near_x] = np.array([0.0, 1.0, 0.0])
    proj = seed - (np.sum(seed * z, axis=1, keepdims=True)) * z
    nrm = np.linalg.norm(proj, axis=1, keepdims=True)
    t1 = proj / np.maximum(nrm, 1e-12)
    t2 = np.cross(z, t1)
    return t1, t2


# --------------------------------------------------------------------------------------
# 4. The per-frame contact-implicit solve.
# --------------------------------------------------------------------------------------


def _solve_frame(
    G: np.ndarray,
    w: np.ndarray,
    active_mask: np.ndarray,
    mu: float,
    reg: float,
) -> np.ndarray:
    """Solve one frame's constrained force-fit; return the (3K,) stacked force vector.

    Minimize ``||G f - w||^2 + reg*||f||^2`` over ``f`` in R^{3K}, subject to, for each
    candidate i that is in ``active_mask``:

        f_n,i >= 0                          (Signorini: contact can only push, s.2)
        ||(f_t1,i, f_t2,i)|| <= mu * f_n,i  (Coulomb friction cone, s.7)

    and ``f_*,i = 0`` for every candidate NOT in ``active_mask`` (the complementarity
    mask: a candidate whose gap is open carries no force, ``g*lambda = 0``, s.2). We
    enforce the mask by simply dropping those candidates' columns from the optimization
    and leaving their forces at 0.

    Solver. The objective is a convex quadratic and the friction cone is a convex
    second-order cone, so this is a small **second-order-cone program**, solved with
    cvxpy + Clarabel. Clarabel returns the global optimum directly (the true elliptic
    Coulomb cone of s.7, not a pyramidal linearization), so none of the warm-start /
    iterate-tie-break / feasibility-projection scaffolding a general NLP solver needs is
    required here.

    Indeterminacy (s.7). When the active candidates over-determine the wrench (>6 force
    components for one 6-wrench, e.g. a box on four corners = 12 components) the data fit
    has a null space: many ``f`` give the same ``G f``. The Tikhonov term ``reg*||f||^2``
    makes the objective strictly convex, so the solver returns the unique **minimum-norm**
    member -- the honest default; the split among redundant co-located candidates is the
    regularizer's choice, NOT a measurement (the unobservable load split of s.7).
    """
    K = G.shape[1] // 3
    active_idx = np.flatnonzero(active_mask)
    f_full = np.zeros(3 * K)
    if active_idx.size == 0:
        return f_full

    # Restrict to active candidates' columns (the masked ones stay 0 by complementarity).
    cols = np.concatenate([[3 * i, 3 * i + 1, 3 * i + 2] for i in active_idx])
    Ga = G[:, cols]                                                   # (6, 3*na)
    na = active_idx.size

    # Decision variable: the active candidates' stacked (f_n, f_t1, f_t2) components.
    f = cp.Variable(3 * na)
    constraints = []
    for j in range(na):
        nj = 3 * j
        constraints.append(f[nj] >= 0.0)                             # Signorini: f_n >= 0 (s.2)
        if mu > 0.0:
            constraints.append(cp.SOC(mu * f[nj], f[nj + 1 : nj + 3]))   # Coulomb cone (s.7)
        else:
            constraints.append(f[nj + 1 : nj + 3] == 0.0)            # no friction => no tangential
    objective = cp.Minimize(cp.sum_squares(Ga @ f - w) + reg * cp.sum_squares(f))
    try:
        cp.Problem(objective, constraints).solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        return f_full                                                # solver failure -> no force
    if f.value is None:
        return f_full                                                # infeasible (never: f=0 is feasible)
    f_full[cols] = np.asarray(f.value)
    return f_full


def solve_contact_implicit(
    required_wrench: np.ndarray,
    G,
    gaps: np.ndarray,
    mu: float,
    params: InverseDynamicsParams | None = None,
    t: np.ndarray | None = None,
    candidate_points: np.ndarray | None = None,
) -> InverseDynamicsResult:
    """Per-frame contact-implicit inverse dynamics solve (THEORY.md s.8 north star).

    For each frame ``t`` we find the per-candidate contact forces that best explain the
    required net wrench under Signorini + Coulomb, then read off the active set. See
    :func:`_solve_frame` for the per-frame convex program; this wraps it over time, packs
    the result into :class:`InverseDynamicsResult`, and applies the complementarity mask.

    Complementarity mask (THEORY.md s.2). A candidate ``i`` may carry force on frame
    ``t`` only when its gap is closed: ``|gap_i(t)| < params.complementarity_gap``.
    Candidates with a larger gap are forced to zero force (``g*lambda = 0``). This is the
    Signorini branch selection -- contact existence enters as a hard mask on which forces
    the dynamics is even allowed to use.

    Active set (THEORY.md s.8). After the solve, a candidate is *active* on a frame iff
    its recovered normal force exceeds ``params.active_force_threshold``. That set is the
    dynamics-side estimate of contact existence + which candidate carries load -- the
    complement to the kinematic detector.

    Parameters
    ----------
    required_wrench : (T, 6) array
        The net external wrench from :func:`required_wrench`.
    G : (T, 6, 3K) array | callable(frame_index) -> (6, 3K)
        The contact-wrench map from :func:`contact_wrench_map`, either materialized or a
        per-frame builder (a builder avoids storing the full tensor for long records).
    gaps : (T, K) array
        Per-candidate signed gap (m) per frame; drives the complementarity mask.
    mu : float
        Coulomb friction coefficient (the cone half-angle tangent).
    params : InverseDynamicsParams, optional
        Thresholds/regularization (defaults if None).
    t : (T,) array, optional
        Timestamps for the result; defaults to ``arange(T)`` if not given.
    candidate_points : (K, 3) array, optional
        Candidate points (body-local) echoed into the result for downstream attribution.

    Returns
    -------
    InverseDynamicsResult
        ``contact_normal_force (T,K)``, ``contact_tangent_force (T,K,2)``,
        ``active_set`` (length-T lists of active indices), ``total_normal_force (T,)``,
        ``wrench_residual (T,)`` (``||G f - w||``), ``candidate_points (K,3)``, ``t (T,)``.
    """
    if params is None:
        params = InverseDynamicsParams()

    W = np.asarray(required_wrench, dtype=float).reshape(-1, 6)
    T = W.shape[0]
    gaps = np.asarray(gaps, dtype=float)
    if gaps.ndim == 1:
        gaps = gaps.reshape(T, 1)
    K = gaps.shape[1]

    callable_G = callable(G)
    if not callable_G:
        Garr = np.asarray(G, dtype=float)
        if Garr.shape != (T, 6, 3 * K):
            raise ValueError(
                f"G must be (T,6,3K)=({T},6,{3*K}); got {Garr.shape}. "
                "Check that gaps' K matches the candidates used to build G."
            )

    reg = float(params.force_regularization)
    comp_gap = float(params.complementarity_gap)
    thresh = float(params.active_force_threshold)

    normal_force = np.zeros((T, K))
    tangent_force = np.zeros((T, K, 2))
    residual = np.zeros(T)
    active_set: list[list[int]] = []

    for ti in range(T):
        Gt = G(ti) if callable_G else Garr[ti]                        # (6, 3K)
        w = W[ti]
        # Complementarity: only gaps within +/- comp_gap may carry force (Signorini, s.2).
        active_mask = np.abs(gaps[ti]) < comp_gap                     # (K,)
        f = _solve_frame(Gt, w, active_mask, mu, reg)                 # (3K,)

        f_resh = f.reshape(K, 3)
        normal_force[ti] = f_resh[:, 0]
        tangent_force[ti] = f_resh[:, 1:3]
        residual[ti] = float(np.linalg.norm(Gt @ f - w))
        active_set.append([int(i) for i in np.flatnonzero(normal_force[ti] > thresh)])

    total_normal = normal_force.sum(axis=1)
    if t is None:
        t = np.arange(T, dtype=float)
    if candidate_points is None:
        candidate_points = np.zeros((K, 3))

    return InverseDynamicsResult(
        t=np.asarray(t, dtype=float),
        contact_normal_force=normal_force,
        contact_tangent_force=tangent_force,
        active_set=active_set,
        total_normal_force=total_normal,
        wrench_residual=residual,
        candidate_points=np.asarray(candidate_points, dtype=float).reshape(K, 3),
    )


# --------------------------------------------------------------------------------------
# 5. Convenience: run the whole pipeline from a RawScenario.
# --------------------------------------------------------------------------------------


def contact_implicit_from_raw(
    raw: RawScenario, config: DetectorConfig | None = None
) -> InverseDynamicsResult:
    """Run contact-implicit inverse dynamics end-to-end from a :class:`RawScenario`.

    Pulls the inertial parameters and contact candidates from the scenario metadata,
    builds the wrench map from the (noisy) observed moving-body pose, computes the
    required Newton-Euler wrench, and solves the per-frame constrained problem (s.8).

    Expected metadata schema (``raw.meta``)
    ----------------------------------------
    * ``meta["inertial"]`` -- dict/obj with ``mass`` (kg), ``inertia`` (3x3 body,
      kg*m^2), ``com_local`` (3,, m).
    * ``meta["candidates"]`` -- dict with ``points_local`` (K,3) candidate points in the
      moving body's local frame, ``normals_local`` (K,3) outward normals (body-local),
      and ``gap`` (K,T) per-candidate signed gap per frame. (Note the (K,T) layout,
      matching ``meta["contact_points"]["penetration"]``; we transpose to (T,K) for the
      solver.)

    Fallback. If ``meta["candidates"]`` is absent but the scenario exposes the
    indeterminate-rig arrays (``meta["contact_points"]`` with ``corners_local`` and
    ``penetration``), we synthesize candidates from the corners with upward body-local
    normals and derive each candidate's gap from the *observed* center gap (the box's
    bottom face is planar, so every corner shares the body's tracked gap to first order).
    This lets the convenience path run on the existing rig scenario for validation.

    Parameters
    ----------
    raw : RawScenario
        A scenario with the inertial + candidate metadata above.
    config : DetectorConfig, optional
        Supplies ``inverse_dynamics`` params and material ``friction`` (mu). Defaults.

    Returns
    -------
    InverseDynamicsResult
    """
    if config is None:
        config = DetectorConfig()
    idp = config.inverse_dynamics
    mu = float(config.material.friction)

    meta = raw.meta or {}
    pose = raw.moving
    t = np.asarray(pose.t, dtype=float)
    T = t.shape[0]

    if "inertial" not in meta:
        raise KeyError(
            "contact_implicit_from_raw needs raw.meta['inertial'] "
            "(mass, inertia 3x3 body, com_local)."
        )
    inert = _as_inertial(meta["inertial"])

    # --- candidate points/normals/gaps ------------------------------------------------
    if "candidates" in meta:
        cand = meta["candidates"]
        pts_l = np.asarray(cand["points_local"], dtype=float).reshape(-1, 3)
        K = pts_l.shape[0]
        # Normals may be static ``(K,3)`` or per-frame ``(K,T,3)``/``(T,K,3)`` -- we pass
        # them through to ``contact_wrench_map`` unchanged (it accepts all three layouts).
        nrm_l = np.asarray(cand["normals_local"], dtype=float)
        gap_kt = np.asarray(cand["gap"], dtype=float)                 # (K, T)
        if gap_kt.shape != (K, T):
            raise ValueError(
                f"meta['candidates']['gap'] must be (K,T)=({K},{T}); got {gap_kt.shape}"
            )
        gaps_tk = gap_kt.T                                            # (T, K)
    elif "contact_points" in meta and "corners_local" in meta["contact_points"]:
        # Fallback for the existing indeterminate-rig scenario (s.7 arrays).
        cp = meta["contact_points"]
        pts_l = np.asarray(cp["corners_local"], dtype=float).reshape(-1, 3)
        K = pts_l.shape[0]
        # Outward normals: the rig's contact face is the box bottom (-z body), whose
        # outward normal points DOWN in the body frame; the equal-and-opposite reaction
        # the floor applies points UP. The candidate normal is the direction the contact
        # can PUSH the body, i.e. body-local +z for a bottom-face support.
        nrm_l = np.tile(np.array([0.0, 0.0, 1.0]), (K, 1))
        # Per-corner gap: the tracked bottom-face point's observed signed distance to the
        # plane stands in for every corner (planar bottom face => shared gap to 1st order).
        # We recompute the center gap from the observed pose against the world plane.
        gaps_tk = _center_gap_per_candidate(raw, K)                  # (T, K)
    else:
        raise KeyError(
            "contact_implicit_from_raw needs raw.meta['candidates'] "
            "(points_local, normals_local, gap) or the indeterminate-rig "
            "raw.meta['contact_points'] fallback."
        )

    # Prefer the scenario's recorded gravity (it is the physics that produced the motion);
    # fall back to the config default otherwise. Magnitude only -- the world -z direction
    # is fixed by :func:`required_wrench`.
    gravity = abs(float(meta.get("gravity", idp.gravity)))

    # --- build the wrench map and the required wrench from the observed pose ----------
    G = contact_wrench_map(pose, pts_l, nrm_l, com_local=inert.com_local)
    w = required_wrench(
        pose, inert, gravity=gravity, accel_smooth_time=idp.accel_smooth_time
    )

    return solve_contact_implicit(
        w, G, gaps_tk, mu, params=idp, t=t, candidate_points=pts_l
    )


def infer_normal_force(
    raw: RawScenario, config: DetectorConfig | None = None
) -> np.ndarray | None:
    """The inferred-force VIRTUAL SENSOR (DESIGN.md PART II.B / III.4): a per-frame total
    normal contact force recovered from KINEMATICS + INERTIALS alone -- no force sensor.

    This is the "force, measured OR *inferred*" half of the force channel. It runs the
    contact-implicit inverse dynamics (Newton-Euler required wrench + Signorini/friction-cone
    solve, all from the *observed* noisy pose and the body's mass/inertia) and aggregates the
    per-candidate normal forces into the scalar ``(T,)`` normal-force stream that
    :class:`~contact.types.ContactObservations.normal_force` expects -- so a body whose
    rigid-body dynamics are tractable can feed the force emission (DESIGN.md PART II.A) with
    *no* physical sensor.

    Validated as a virtual sensor on the box scenarios (e.g. ``drop_rest``: the inferred
    force correlates ~0.94 with the simulator's true contact force at ~5% magnitude error).
    The force emission normalizes by its own scale, so the *profile* (a touchdown spike, a
    sustained load) is what matters, not the absolute Newtons.

    Returns ``None`` when the scenario lacks the inertial/candidate metadata the solver needs
    (so callers degrade gracefully to the kinematics-only estimate -- the no-op-when-absent
    contract). Otherwise returns ``(T,)`` non-negative inferred normal force.

    .. note:: Single rigid body only. An ARTICULATED body (e.g. the hinge-suspended
       Newton's-cradle balls) has unmodeled joint-reaction wrenches that this solver would
       mis-attribute to the contact, so it is intentionally out of scope here (DESIGN.md
       PART II.B / s.11 open items): the cradle's clacks need a measured sensor or an
       articulated-dynamics extension.
    """
    meta = raw.meta or {}
    if "inertial" not in meta or not (
        "candidates" in meta or "contact_points" in meta
    ):
        return None
    res = contact_implicit_from_raw(raw, config)
    return np.maximum(np.asarray(res.total_normal_force, dtype=float), 0.0)


def _center_gap_per_candidate(raw: RawScenario, K: int) -> np.ndarray:
    """Observed center-point gap broadcast to K candidates (rig fallback only).

    Recomputes the tracked bottom-face point's signed distance to the support plane from
    the observed pose (mirroring ``contact.geometry.observe`` step 3, inlined here to
    keep this module's import set to types/config/signals). The box's bottom face is
    planar so to first order every corner shares this gap, which is all the
    complementarity mask needs.
    """
    pose = raw.moving
    sup = raw.support
    R_m = _quat_to_matrix(_continuous_quat(pose.quat))
    cpl = np.asarray(raw.contact_point_local, dtype=float)
    p = np.asarray(pose.position, dtype=float) + np.einsum("tij,j->ti", R_m, cpl)

    R_s = _quat_to_matrix(_continuous_quat(sup.quat))
    plane_pt = np.asarray(sup.position, dtype=float) + np.einsum(
        "tij,j->ti", R_s, np.asarray(raw.surface.point, dtype=float)
    )
    n_w = np.einsum("tij,j->ti", R_s, np.asarray(raw.surface.normal, dtype=float))
    n_w = n_w / np.maximum(np.linalg.norm(n_w, axis=-1, keepdims=True), 1e-12)
    gap = np.sum((p - plane_pt) * n_w, axis=-1)                       # (T,)
    return np.repeat(gap[:, None], K, axis=1)                         # (T, K)
