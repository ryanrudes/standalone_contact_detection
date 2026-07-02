"""Convex collision queries for the mesh resolvers (DESIGN.md PART II.D / III.5).

The highest-fidelity rung of the resolver ladder. Two queries back the mesh resolvers in
:mod:`contact.geometry_resolvers` (``MeshPlane`` / ``MeshConvex``):

* :func:`convex_plane` -- a convex vertex cloud against a single plane, the exact
  generalization of ``BoxPlane``'s per-corner signed distance: ``d_i = (vertᵢ − plane_pt)·n``,
  ``gap = min_i d_i``, and the contact set is every vertex within ``eps`` of that minimum (so a
  box's 8 corners reproduce ``BoxPlane`` bit-for-bit). Kept as a tiny closed form because it
  returns the *full* multi-vertex contact set, which a general collision library does not.

* :func:`gjk_distance` -- the signed distance (with witness/contact point and normal) between
  two convex vertex clouds, delegated to the **coal** library (the maintained GJK + EPA
  collision kernel, formerly hpp-fcl). coal builds a convex hull per cloud and returns the
  signed minimum distance (``> 0`` separated, ``< 0`` penetration), the two nearest/witness
  points, and the contact normal -- exactly the watertight query the hand-rolled GJK + EPA +
  SAT kernel used to compute, and validated *identical* to it on the resolver fixtures and a
  randomized separated/penetrating battery (max gap diff ~1e-14).

The contact normal is geometry-derived (it carries no body-fixed spin artifact) and oriented
``support -> moving`` (from hull ``B`` to hull ``A``), the package's outward-normal convention.
Clouds arrive already world-placed; this module only ever sees vertex clouds and planes.
"""

from __future__ import annotations

import numpy as np

__all__ = ["convex_plane", "gjk_distance"]


def convex_plane(
    verts_world: np.ndarray,
    plane_pt: np.ndarray,
    plane_normal: np.ndarray,
    eps: float = 1e-9,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Signed gap of a convex vertex cloud to a plane, plus the contacting vertices.

    A *single-frame* helper (the per-frame loop body of :class:`MeshPlane`). Each vertex's
    signed plane distance is ``d_i = (vertᵢ − plane_pt) · normal`` (``> 0`` separation, ``< 0``
    penetration); the gap is the closest vertex ``gap = min_i d_i`` and the contact set is every
    vertex within ``eps`` of that minimum (1 = a single touching vertex, 2 = an edge resting on
    the plane, 3+ = a face). This is exactly ``BoxPlane``'s per-corner arithmetic
    (``geometry_resolvers.BoxPlane.resolve``) lifted to an arbitrary cloud, so the 8 box
    corners reproduce ``BoxPlane`` bit-for-bit.

    The ``normal`` is taken AS GIVEN (assumed already unit) and returned unchanged -- the caller
    (the resolver) carries the support plane's world normal in, so re-normalizing here would
    perturb the bit-identical match with ``BoxPlane``.

    Parameters
    ----------
    verts_world : np.ndarray
        ``(V, 3)`` cloud vertices in the world frame.
    plane_pt : np.ndarray
        ``(3,)`` a point on the plane, world frame.
    plane_normal : np.ndarray
        ``(3,)`` the (unit) outward plane normal, world frame.
    eps : float, optional
        Grouping tolerance (m): vertices with ``d_i ≤ gap + eps`` are reported as simultaneous
        contacts. Default ``1e-9``.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        ``(gap, indices, normal)`` -- the minimum signed distance, the indices (into
        ``verts_world``, ascending) of the contacting vertices, and the plane normal as given.
    """
    verts = np.asarray(verts_world, dtype=float)
    p0 = np.asarray(plane_pt, dtype=float)
    n = np.asarray(plane_normal, dtype=float)
    d = (verts - p0) @ n                               # (V,) signed plane distance per vertex
    gap = float(d.min())
    idxs = np.nonzero(d <= gap + eps)[0]               # the contacting vertices (>=1)
    return gap, idxs, n


def _hull(verts: np.ndarray):
    """A coal ``Convex`` hull from an ``(V, 3)`` world-frame vertex cloud."""
    import coal  # local import: the collision backend loads only when a mesh query runs

    pts = coal.StdVec_Vec3s()
    for v in np.asarray(verts, dtype=float):
        pts.append(v)
    return coal.Convex.convexHull(pts, False, "")


def _vertex_nearest(
    vA: np.ndarray, vB: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Closest-vertex-pair fallback for a degenerate (non-3-D) cloud coal cannot hull.

    Correct for the separated regime the estimator gates on; approximate under penetration.
    Not reached by the box/sphere clouds the resolvers actually pass.
    """
    d2 = np.sum((vA[:, None, :] - vB[None, :, :]) ** 2, axis=-1)
    i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
    diff = vA[i] - vB[j]
    dist = float(np.linalg.norm(diff))
    normal = diff / dist if dist > 1e-12 else np.array([0.0, 0.0, 1.0])
    return dist, 0.5 * (vA[i] + vB[j]), normal


def gjk_distance(
    vA_world: np.ndarray, vB_world: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Signed distance (+ contact point and normal) between two convex vertex clouds.

    Delegates to coal's GJK/EPA kernel on the convex hulls of the two clouds (the Mesh/SDF rung
    of DESIGN.md II.D / III.5).

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        ``(gap, point, normal)``:

        * ``gap``    -- signed separation; ``> 0`` distance between hulls, ``< 0`` penetration.
        * ``point``  -- world contact point: the midpoint of the two nearest/witness points.
        * ``normal`` -- world unit ``support -> moving`` normal (from hull ``B`` to hull ``A``),
          oriented by the inter-centroid direction. Position-derived -- no body-fixed spin.
    """
    import coal  # local import: the collision backend loads only when a mesh query runs

    vA = np.asarray(vA_world, dtype=float)
    vB = np.asarray(vB_world, dtype=float)
    req = coal.DistanceRequest()
    req.enable_signed_distance = True          # signed: > 0 separated, < 0 penetrating
    res = coal.DistanceResult()
    identity = coal.Transform3s()              # clouds are already world-placed
    try:
        coal.distance(_hull(vA), identity, _hull(vB), identity, req, res)
    except Exception:
        return _vertex_nearest(vA, vB)         # degenerate (non-3-D) cloud

    gap = float(res.min_distance)
    point = 0.5 * (np.asarray(res.getNearestPoint1()) + np.asarray(res.getNearestPoint2()))
    normal = np.asarray(res.normal, dtype=float)
    # Orient support(B) -> moving(A): align with the inter-centroid direction (B -> A).
    if float(normal @ (vA.mean(axis=0) - vB.mean(axis=0))) < 0.0:
        normal = -normal
    return gap, point, normal
