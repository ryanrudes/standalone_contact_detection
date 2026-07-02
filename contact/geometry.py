"""The relative-frame core: poses -> support-relative ContactObservations.

This module is the concrete realization of THEORY.md §1 & §3. §1
("contact is *relative* and *geometric*") forces every quantity to be measured in
the frame of one body of the pair — the *support* — never in the world: a foot on a
fast-moving skateboard is in solid contact even though it screams across the world.
§3 ("a contact constrains *motion*") forces us to keep the relative motion as
a 6-component **twist** (3 linear + 3 angular) decomposed against the contact frame,
because the *correlations* between its components are exactly what name the mode.

So `observe(...)` does, per frame:

    1. place the tracked material point in the world,                       (§1)
    2. carry the support's plane into the world (it may translate/rotate),  (§1)
    3. measure the support-relative *gap* (signed plane distance),          (§1 / §2)
    4. build a contact frame (z = surface normal, x/y a continuous tangent),(§3)
    5. compute the RELATIVE linear velocity of the coincident material
       points and split it into normal / tangent in that frame,            (§3)
    6. compute the RELATIVE angular velocity and split it likewise.         (§3)

The key subtlety, and the whole reason this file exists, is step 5: the velocity that
distinguishes sliding from rolling is the velocity of the *material point currently at
the contact* — `v = v_origin + omega x r` (THEORY.md §3, final paragraph) — and it
must be taken *relative to the coincident point on the support*. Differencing world
velocities is wrong the instant the support moves.

Frame conventions (mirroring contact/types.py):
* World is a fixed inertial frame.
* Quaternions are scalar-first unit quaternions ``q = (w, x, y, z)`` and rotate a
  body-local vector into the world: ``v_world = R(q) @ v_local``.
* The contact frame's columns ``[x_hat, y_hat, z_hat]`` are the world-frame axes of
  the support's instantaneous tangent/normal basis; a world vector ``u`` expressed in
  the contact frame is ``[x_hat . u, y_hat . u, z_hat . u]``.

Differentiation of the (noisy) pose streams is delegated to ``contact.signals`` (the
only cross-module import this file's spec permits): positions and quaternions are
Gaussian-smoothed *first*, then finite-differenced, because differentiation amplifies
sensor noise (THEORY.md §4). We fall back to local equivalents only if that module
is unavailable, so this leaf stays importable on its own.
"""

from __future__ import annotations

import numpy as np

from .signals import derivative, gaussian_smooth
from .types import ContactGeometry, ContactObservations, PoseTrajectory, SupportSurface

# --------------------------------------------------------------------------------------
# Differentiation/smoothing is delegated to the time-aware leaf helpers in contact.signals
# (THEORY.md §4: smooth in real time before differentiating, never raw finite differences).
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# Quaternion helpers — scalar-first (w, x, y, z), unit norm. Vectorized over the leading
# time axis. A quaternion rotates a *body-local* vector into the *world*:
# v_world = R(q) @ v_local. These are the elementary group operations on SO(3) used to
# carry the support's plane into the world (§1) and to read body angular rates (§3).
# --------------------------------------------------------------------------------------


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate (= inverse for unit quaternions) of scalar-first ``q``.

    THEORY.md §3: the inverse rotation; used to map world vectors back into a body
    frame and to form ``dq/dt * conj(q)`` for angular velocity.

    Parameters
    ----------
    q : np.ndarray
        ``(4,)`` or ``(..., 4)`` scalar-first quaternion(s) ``(w, x, y, z)``.

    Returns
    -------
    np.ndarray
        Same shape as ``q`` with the vector part negated.
    """
    q = np.asarray(q, dtype=float)
    out = q.copy()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` of scalar-first quaternions (composition of rotations).

    THEORY.md §3: rotation composition; ``R(a*b) = R(a) R(b)``. Broadcasts over any
    leading time axis.

    Parameters
    ----------
    a, b : np.ndarray
        ``(4,)`` or ``(..., 4)`` scalar-first quaternions; broadcast against each other.

    Returns
    -------
    np.ndarray
        ``(..., 4)`` product quaternion(s).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return np.stack([w, x, y, z], axis=-1)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix/matrices ``R(q)`` with ``v_world = R(q) @ v_local``.

    THEORY.md §1: the body->world orientation map used to place local geometry
    (contact point, plane point, plane normal) into the world frame. Input is
    normalized defensively so non-unit inputs still give a proper rotation.

    Parameters
    ----------
    q : np.ndarray
        ``(4,)`` or ``(T, 4)`` scalar-first quaternion(s).

    Returns
    -------
    np.ndarray
        ``(3, 3)`` or ``(T, 3, 3)`` rotation matrix/matrices.
    """
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # Standard quaternion -> rotation matrix (right-handed, body-local to world).
    R = np.empty(q.shape[:-1] + (3, 3), dtype=float)
    R[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[..., 0, 1] = 2.0 * (x * y - z * w)
    R[..., 0, 2] = 2.0 * (x * z + y * w)
    R[..., 1, 0] = 2.0 * (x * y + z * w)
    R[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[..., 1, 2] = 2.0 * (y * z - x * w)
    R[..., 2, 0] = 2.0 * (x * z - y * w)
    R[..., 2, 1] = 2.0 * (y * z + x * w)
    R[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return R


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate body-local vector(s) ``v`` into the world by ``q``: ``R(q) @ v``.

    THEORY.md §1: applies the body->world rotation. Broadcasts over a leading time
    axis so a ``(T, 4)`` pose stream rotates either one fixed ``(3,)`` vector or a
    ``(T, 3)`` stream of vectors.

    Parameters
    ----------
    q : np.ndarray
        ``(4,)`` or ``(T, 4)`` scalar-first quaternion(s).
    v : np.ndarray
        ``(3,)`` or ``(T, 3)`` vector(s) in the body-local frame.

    Returns
    -------
    np.ndarray
        Rotated vector(s) in the world frame, shape ``(3,)`` or ``(T, 3)``.
    """
    R = quat_to_matrix(q)                          # (3,3) or (T,3,3)
    v = np.asarray(v, dtype=float)
    # einsum handles every (scalar/stream) x (scalar/stream) combination uniformly.
    if R.ndim == 2 and v.ndim == 1:
        return R @ v
    if R.ndim == 2 and v.ndim == 2:
        return v @ R.T                             # (T,3) each rotated by the same R
    if R.ndim == 3 and v.ndim == 1:
        return np.einsum("tij,j->ti", R, v)        # one fixed vector per-frame rotation
    return np.einsum("tij,tj->ti", R, v)           # per-frame vector, per-frame rotation


def _angular_velocity_world(
    quat: np.ndarray, t: np.ndarray, sigma_time: float
) -> np.ndarray:
    """World-frame angular velocity ``omega(t)`` from a quaternion sequence.

    THEORY.md §3 (and the spec): for a body whose orientation evolves as ``q(t)``, the
    body's world angular velocity is the vector part of ``2 * (dq/dt) * conj(q)``. We
    smooth the quaternion stream first (in real time) and finite-difference it, exactly
    as we do for positions, because differentiating raw noisy orientation is hopeless
    (§4). The quaternion is renormalized and sign-aligned across frames so the double
    cover (q and -q encode the same rotation) does not inject a spurious 2-revolution
    jump into dq/dt.

    Returns
    -------
    np.ndarray
        ``(T, 3)`` angular velocity in the WORLD frame (rad/s).
    """
    q = np.asarray(quat, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    # Resolve the antipodal double cover: flip any frame that points "away" from the
    # previous one so the path through S^3 is continuous and dq/dt is meaningful.
    q = q.copy()
    flip = np.cumprod(np.sign(np.sum(q[1:] * q[:-1], axis=1) + 1e-300))
    q[1:] *= np.where(flip[:, None] < 0.0, -1.0, 1.0)

    q_smooth = gaussian_smooth(q, t, sigma_time)
    q_smooth = q_smooth / np.linalg.norm(q_smooth, axis=-1, keepdims=True)
    dq = derivative(q_smooth, t)                                 # (T,4)
    # omega_quat = 2 * dq * conj(q); its scalar part is ~0, vector part is omega_world.
    omega_quat = 2.0 * quat_mul(dq, quat_conjugate(q_smooth))    # (T,4)
    return omega_quat[..., 1:]                                   # drop the (~0) scalar


# --------------------------------------------------------------------------------------
# Plane gap (THEORY.md §1 / §2): the gap function is the support-relative signed
# distance along the contact normal. For a planar support it is the dot of (point -
# plane_point) with the (unit) outward normal: > 0 separation, < 0 penetration.
# --------------------------------------------------------------------------------------


def plane_gap(
    points_world: np.ndarray,
    plane_point_world: np.ndarray,
    plane_normal_world: np.ndarray,
) -> np.ndarray:
    """Signed distance of points to a (possibly moving) plane; ``+`` on the normal side.

    THEORY.md §1: this is the gap function ``g`` specialized to a plane — the value of
    the support's signed-distance field at the contact point. ``g > 0`` means the point
    is on the outward-normal side (separation), ``g < 0`` means penetration (§2).

    Parameters
    ----------
    points_world : np.ndarray
        ``(T, 3)`` (or ``(3,)``) contact point(s) in the world frame.
    plane_point_world : np.ndarray
        ``(T, 3)`` (or ``(3,)``) a point on the plane, in the world frame (per frame
        if the support moves).
    plane_normal_world : np.ndarray
        ``(T, 3)`` (or ``(3,)``) outward plane normal in the world frame; need not be
        pre-normalized (it is normalized here so the result is a true metric distance).

    Returns
    -------
    np.ndarray
        ``(T,)`` signed distances (m).
    """
    p = np.atleast_2d(np.asarray(points_world, dtype=float))
    p0 = np.atleast_2d(np.asarray(plane_point_world, dtype=float))
    n = np.atleast_2d(np.asarray(plane_normal_world, dtype=float))
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    # Signed distance = projection of (point - plane_point) onto the unit normal.
    return np.sum((p - p0) * n, axis=-1)


# --------------------------------------------------------------------------------------
# Contact frame (THEORY.md §3): z = world surface normal; x, y span the tangent plane.
# We need the tangent basis to be *continuous across frames* (no sign flips), otherwise
# v_tangent / omega_tangent would acquire spurious jumps that look like motion. We get
# continuity by *transporting* a single tangent reference along the normal's path rather
# than recomputing an arbitrary basis independently each frame.
# --------------------------------------------------------------------------------------


def _tangent_basis(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Continuous orthonormal tangent basis ``(x_hat, y_hat)`` for a normal stream.

    THEORY.md §3: the tangent plane carries the sliding/rolling components, so its axes
    must not flip frame-to-frame. We pick one tangent at the first frame, then for every
    later frame project the previous ``x_hat`` onto the new tangent plane and re-normalize
    (a discrete parallel transport). ``y_hat = z_hat x x_hat`` completes a right-handed
    frame. The result is smooth as long as the normal is.

    Parameters
    ----------
    normals : np.ndarray
        ``(T, 3)`` unit outward normals in the world frame.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``x_hat`` ``(T, 3)`` and ``y_hat`` ``(T, 3)``, each orthonormal to the normal.
    """
    z = np.asarray(normals, dtype=float)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    T = z.shape[0]
    x_hat = np.empty((T, 3), dtype=float)
    y_hat = np.empty((T, 3), dtype=float)

    # Seed: pick the world axis least parallel to the first normal, then orthogonalize.
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, z[0])) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    x0 = seed - np.dot(seed, z[0]) * z[0]
    x0 /= np.linalg.norm(x0)
    x_hat[0] = x0
    y_hat[0] = np.cross(z[0], x_hat[0])

    # Transport: project the previous x_hat into each new tangent plane (Gram-Schmidt).
    for i in range(1, T):
        x = x_hat[i - 1] - np.dot(x_hat[i - 1], z[i]) * z[i]
        nrm = np.linalg.norm(x)
        if nrm < 1e-12:
            # Degenerate (normal flipped ~180 deg between frames); reseed from y_hat.
            x = y_hat[i - 1] - np.dot(y_hat[i - 1], z[i]) * z[i]
            nrm = np.linalg.norm(x)
            if nrm < 1e-12:
                s = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(s, z[i])) > 0.9:
                    s = np.array([0.0, 1.0, 0.0])
                x = s - np.dot(s, z[i]) * z[i]
                nrm = np.linalg.norm(x)
        x_hat[i] = x / nrm
        y_hat[i] = np.cross(z[i], x_hat[i])
    return x_hat, y_hat


# --------------------------------------------------------------------------------------
# The relative-frame core (THEORY.md §1 & §3): poses -> ContactObservations.
# --------------------------------------------------------------------------------------


def observe(
    moving: PoseTrajectory,
    support: PoseTrajectory,
    surface: SupportSurface,
    contact_point_local: np.ndarray = np.zeros(3),
    vel_smooth_time: float = 0.05,
    geometry: ContactGeometry | None = None,
) -> ContactObservations:
    """Turn a moving body + (possibly moving) support into support-relative observations.

    This is the relative-frame core of THEORY.md §1 (contact is relative and
    geometric) and 3 (a contact constrains motion; keep the relative twist with its
    correlations). Every output channel lives in the support's instantaneous contact
    frame, so a body riding a fast-moving support reads ~0 relative motion — the whole
    point of measuring support-relative rather than in the world.

    Steps (matching the module docstring):

    1. World contact point ``p(t) = moving.position + R(moving.quat) @ contact_point_local``.
    2. The SUPPORT plane carried into the world each frame: a point on the plane becomes
       ``support.position + R(support.quat) @ surface.point`` and the normal becomes
       ``R(support.quat) @ surface.normal`` (a free vector, so no translation).
    3. ``gap(t)`` = signed distance of ``p(t)`` to that moving plane (``plane_gap``).
    4. Contact frame: ``z_hat`` = world surface normal; ``x_hat, y_hat`` a continuous
       tangent basis (``_tangent_basis``, no sign flips across frames).
    5. RELATIVE linear velocity of the coincident material points: the velocity of the
       moving material point minus the velocity of the support point momentarily at the
       same location, ``v_support_point = v_support_origin + omega_support x r`` with
       ``r = p - support.position``. Both velocities come from smoothing then
       differentiating the pose streams via ``contact.signals``. The relative velocity is
       split into ``v_normal`` (scalar, ``+`` = separating) and ``v_tangent`` (2-vector).
    6. RELATIVE angular velocity ``omega_moving - omega_support`` (from the quaternion
       streams) split into ``omega_normal`` (scalar, spin about z) and ``omega_tangent``
       (2-vector, the rolling axis in the tangent plane).

    Parameters
    ----------
    moving : PoseTrajectory
        Pose of the body whose contact we test. ``t``/``position``/``quat`` are ``(T,)``/
        ``(T, 3)``/``(T, 4)``.
    support : PoseTrajectory
        Pose of the support body (identity-ish poses for a static floor). Must share the
        same length / timebase as ``moving``.
    surface : SupportSurface
        The plane (point + outward normal) in the support body's *local* frame.
    contact_point_local : np.ndarray, optional
        ``(3,)`` tracked material point on the moving body, in its local frame. Default
        is the body origin.
    vel_smooth_time : float, optional
        Gaussian smoothing time (s) applied to positions and quaternions before
        differentiation (THEORY.md §4). Default 0.05 s.
    geometry : ContactGeometry | None, optional
        The per-frame, world-frame contact-geometry resolver (DESIGN.md III.1/III.2). When
        ``None`` (the default) steps 1-3 below are produced by a
        :class:`contact.geometry_resolvers.FlatRegion` wrapping ``surface`` +
        ``contact_point_local`` — bit-identical to the pre-refactor pipeline. A non-``None``
        resolver supplies the per-frame ``(point, normal, gap)`` instead (``surface`` /
        ``contact_point_local`` are then unused); the twist decomposition is unchanged.

    Returns
    -------
    ContactObservations
        Per-frame, support-relative ``(t, gap, v_normal, v_tangent, omega_normal,
        omega_tangent)`` in the contact frame.
    """
    # The narrow waist (DESIGN.md III.1/III.2): the per-frame world contact point, outward
    # normal and signed gap come from a ContactGeometry resolver. The legacy
    # (surface, contact_point_local) spec is just the configuration of the default
    # FlatRegion resolver, whose arithmetic reproduces steps 1-3 below bit-for-bit, so the
    # geometry=None path is byte-identical to the pre-refactor pipeline.
    if geometry is None:
        from .geometry_resolvers import FlatRegion

        geometry = FlatRegion(surface, contact_point_local)

    t = np.asarray(moving.t, dtype=float)
    mov_pos = np.asarray(moving.position, dtype=float)            # (T,3) world (moving body origin)
    sup_pos = np.asarray(support.position, dtype=float)            # (T,3) world
    sup_quat = np.asarray(support.quat, dtype=float)               # (T,4)
    mov_quat = np.asarray(moving.quat, dtype=float)                # (T,4)

    # --- Steps 1-3 (delegated to the resolver): the world contact point p(t), the support
    # plane's world outward normal, and the support-relative signed gap, one ContactFrame
    # per recorded frame. Stack the (single-point) frames back into the (T,3)/(T,3)/(T,)
    # arrays the twist decomposition consumes. With the default FlatRegion these equal:
    #   p          = moving.pos  + R(moving.quat)  @ contact_point_local   (step 1)
    #   plane_pt_w = support.pos + R(support.quat) @ surface.point         (step 2)
    #   normal_w   = normalize(R(support.quat) @ surface.normal)           (step 2)
    #   gap        = plane_gap(p, plane_pt_w, normal_w)                    (step 3)
    frames = geometry.resolve(moving, support)                     # list[ContactFrame], length T

    # --- Multi-point aggregation (DESIGN.md PART II.C): a face/edge contact is several points
    # but ONE kinematic mode, so reduce each frame to a single representative for the twist
    # decomposition. The representative is the MIN-gap (lowest/closest) contact point: its
    # point and (shared) plane normal drive the frame, and its gap is the frame's gap. A
    # single-point frame (FlatRegion / Sphere*) has exactly one point, for which `min` returns
    # `frame[0]` -- so `p`, `normal_w`, `gap` are byte-identical to the pre-Phase-2 pipeline.
    reps = [min(frame, key=lambda cp: cp.gap) for frame in frames]   # one representative / frame
    p = np.stack([rep.point for rep in reps])                      # (T,3) world contact point
    normal_w = np.stack([rep.normal for rep in reps])              # (T,3) world unit normal
    gap = np.array([rep.gap for rep in reps], dtype=float)         # (T,) signed plane distance

    # --- Step 4: contact frame. z = world normal; x/y a continuous tangent basis.
    z_hat = normal_w                                               # (T,3)
    x_hat, y_hat = _tangent_basis(z_hat)                           # each (T,3)

    # The moving body's WORLD angular velocity. Computed here (ahead of step 5) because a
    # MIGRATING resolver needs it for the analytic moving-point velocity below; step 6 reuses
    # this exact same array. Moving the call up does not change its value (it is a pure function
    # of the quaternion stream), so the non-migrating path stays byte-identical.
    omega_moving = _angular_velocity_world(mov_quat, t, vel_smooth_time)  # (T,3) world

    # --- Step 5: RELATIVE linear velocity of the coincident material points.
    # (a) Velocity of the moving material point.
    if getattr(geometry, "migrating", False):
        # MIGRATING resolver (e.g. BoxPlane): the contact point JUMPS between corners, so its
        # world trajectory teleports and DIFFERENTIATING it would manufacture a spike at every
        # switch. Instead take the velocity of the body's material point CURRENTLY at the
        # contact analytically from the rigid-body twist: v = v_com + omega x (p - com), with
        # com = moving body origin and v_com its (smoothed/differentiated) linear velocity.
        # (For a FIXED point this equals d/dt(p) since d/dt(R @ cpl) = omega x (R @ cpl); the
        # formulas diverge only where the point genuinely migrates -- DESIGN.md PART II.D.)
        v_com = derivative(gaussian_smooth(mov_pos, t, vel_smooth_time), t)  # (T,3) world COM velocity
        v_moving_point = v_com + np.cross(omega_moving, p - mov_pos)  # (T,3) world
    else:
        # FIXED material point (FlatRegion / Sphere*): differentiate its smooth world
        # trajectory (THEORY.md §4). This is the verbatim pre-Phase-2 path -> bit-identical.
        v_moving_point = derivative(gaussian_smooth(p, t, vel_smooth_time), t)  # (T,3) world

    # (b) Velocity of the support point momentarily coincident with p. A rigid body's
    #     material-point velocity is v_origin + omega x r, where r is the lever arm from
    #     the support origin to the contact point (THEORY.md §3). v_origin and
    #     omega_support both come from the smoothed/differentiated support pose.
    v_sup_origin = derivative(gaussian_smooth(sup_pos, t, vel_smooth_time), t)  # (T,3) world
    omega_support = _angular_velocity_world(sup_quat, t, vel_smooth_time)  # (T,3) world
    r = p - sup_pos                                                # (T,3) lever arm (world)
    v_support_point = v_sup_origin + np.cross(omega_support, r)    # (T,3) world

    # (c) Relative velocity = moving material point minus coincident support point.
    #     If the moving body is rigidly fixed to the support, these are identical and
    #     v_rel == 0 even when both scream across the world — the relative-frame check.
    v_rel = v_moving_point - v_support_point                       # (T,3) world

    # Decompose v_rel into the contact frame. The normal component is +ve when the
    # bodies are separating (moving along the outward normal); the two tangent
    # components are the sliding velocity in the surface plane.
    v_normal = np.sum(v_rel * z_hat, axis=-1)                      # (T,) + = separating
    v_tangent = np.stack(
        [np.sum(v_rel * x_hat, axis=-1), np.sum(v_rel * y_hat, axis=-1)], axis=-1
    )                                                              # (T,2)

    # --- Step 6: RELATIVE angular velocity = omega_moving - omega_support.
    # omega from a quaternion stream is the vector part of 2 * dq/dt * conj(q) (world);
    # omega_moving was computed once above (and, for a migrating resolver, also fed the
    # analytic moving-point velocity) -- reuse it verbatim here.
    omega_rel = omega_moving - omega_support                       # (T,3) world

    # Decompose into the contact frame: the normal component is spin/pivot about the
    # surface normal; the two tangent components are the rolling axis in the surface.
    omega_normal = np.sum(omega_rel * z_hat, axis=-1)              # (T,)
    omega_tangent = np.stack(
        [np.sum(omega_rel * x_hat, axis=-1), np.sum(omega_rel * y_hat, axis=-1)], axis=-1
    )                                                              # (T,2)

    return ContactObservations(
        t=t,
        gap=gap,
        v_normal=v_normal,
        v_tangent=v_tangent,
        omega_normal=omega_normal,
        omega_tangent=omega_tangent,
    )
