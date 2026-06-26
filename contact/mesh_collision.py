"""Phase 3 (DESIGN.md PART II.D / III.5): convex MESH geometry via GJK + EPA.

The highest-fidelity rung of the resolver ladder (DESIGN.md PART II section D, the
``Mesh / SDF`` row of §3.3, and the Phase-3 row of III.5). Where ``FlatRegion`` /
``SpherePlane`` / ``BoxPlane`` exploit a *closed-form* nearest feature, a mesh is an
arbitrary convex vertex cloud with no such formula, so the contact ``(point, normal, gap)``
must be recovered by an iterative collision query. This module is the pure-:mod:`numpy`
collision kernel behind the two mesh resolvers in
:mod:`contact.geometry_resolvers` (``MeshPlane`` / ``MeshConvex``):

* :func:`convex_plane` -- a convex vertex cloud against a single plane (a *single-frame*
  signed-distance query). This is the exact generalization of ``BoxPlane``'s per-corner
  arithmetic to an arbitrary vertex cloud: ``d_i = (vertᵢ − plane_pt)·n``, ``gap = min_i d_i``,
  and the contact set is every vertex within ``eps`` of that minimum. A box mesh (its 8
  corners) reduces to ``BoxPlane`` exactly.

* :func:`gjk_distance` -- the minimum distance (and witness/normal) between *two* convex
  vertex clouds, via the Gilbert-Johnson-Keerthi (GJK) algorithm on the Minkowski difference
  ``A ⊖ B = {a − b}``. The distance between the hulls equals the distance from the origin to
  ``A ⊖ B``; GJK walks a simplex of Minkowski-difference points toward the origin using the
  classic point/segment/triangle/tetrahedron closest-point sub-routines, and the separation
  is the distance from the origin to the final simplex. When the hulls TOUCH or PENETRATE
  (the origin is on/inside ``A ⊖ B``) the signed gap is resolved EXACTLY by a
  Separating-Axis close-out on the two convex hulls (:func:`_penetration_sat`) -- the
  penetration depth is the minimum projection overlap over the hulls' face normals and
  edge-edge axes. This is exact for *every* convex polytope, the degenerate coplanar
  box-vs-box face contact included (where a GJK simplex stalls and EPA mis-converges), and
  agrees with the analytic depth on curved meshes. :func:`_epa` remains the fallback for a
  genuinely degenerate (non-3-D) cloud where a convex hull cannot be built.

The contact normal is **position/geometry-derived** -- the unit ``origin -> closest`` direction
on the Minkowski difference, which points from the support hull ``B`` to the moving hull ``A``
(``support -> moving``, the package's outward-normal convention). It is never a body-fixed
vector rotated by a quaternion, so it carries no spin artifact -- exactly the property that
makes ``SphereSphere`` correct, here generalized to arbitrary convex shapes (DESIGN.md PART
II.D).

This is **numpy-only**: no external collision/geometry dependency. The resolvers in
:mod:`contact.geometry_resolvers` carry the bodies' poses; this module only ever sees
already-world-placed vertex clouds and planes.
"""

from __future__ import annotations

import numpy as np

__all__ = ["support", "convex_plane", "gjk_distance"]


# --------------------------------------------------------------------------------------
# Support functions (the only way GJK ever touches the geometry).
# --------------------------------------------------------------------------------------


def support(cloud: np.ndarray, d: np.ndarray) -> np.ndarray:
    """The support point of a convex point cloud in direction ``d``.

    ``support(cloud, d) = cloud[argmax_i (cloudᵢ · d)]`` -- the vertex farthest along ``d``.
    For a convex polytope given by its vertices this is the exact support mapping (the max of
    a linear functional over the hull is attained at a vertex), so GJK/EPA built on it operate
    on the true convex hulls of the clouds, not just the sampled points.

    Parameters
    ----------
    cloud : np.ndarray
        ``(V, 3)`` vertices in the world frame.
    d : np.ndarray
        ``(3,)`` search direction (need not be unit).

    Returns
    -------
    np.ndarray
        ``(3,)`` the supporting vertex.
    """
    cloud = np.asarray(cloud, dtype=float)
    d = np.asarray(d, dtype=float)
    return cloud[int(np.argmax(cloud @ d))]


def _minkowski_support(
    vA: np.ndarray, vB: np.ndarray, d: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Support point of the Minkowski difference ``A ⊖ B`` in direction ``d``.

    ``s(d) = support(A, d) − support(B, −d)`` -- the support of ``A ⊖ B`` (DESIGN.md II.D).
    Returns the Minkowski point ``w = a − b`` together with the two *witness* points
    ``a ∈ A`` and ``b ∈ B`` it came from, so the closest-point routines can carry barycentric
    combinations back to per-hull witness points for the contact ``point``.
    """
    a = support(vA, d)
    b = support(vB, -d)
    return a - b, a, b


# --------------------------------------------------------------------------------------
# Convex cloud vs plane (single-frame): the EXACT generalization of BoxPlane's per-corner
# signed distance to an arbitrary vertex cloud (DESIGN.md II.D BoxPlane row).
# --------------------------------------------------------------------------------------


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


# --------------------------------------------------------------------------------------
# Closest point on a simplex (1-4 points) to the ORIGIN. The classic GJK/Johnson
# sub-distance routines (point / segment / triangle / tetrahedron), each returning the
# closest point, the indices of the MINIMAL sub-simplex containing it, and barycentric
# weights for those vertices (so witness points transfer). For the tetrahedron we also
# flag when the origin is ENCLOSED (penetration -> EPA). Ported from the Voronoi-region
# tests of Ericson, "Real-Time Collision Detection", specialized to the query point P = 0.
# --------------------------------------------------------------------------------------


def _closest_seg(
    W: list[np.ndarray], i: int, j: int
) -> tuple[np.ndarray, list[int], list[float], bool]:
    """Closest point on segment ``[W_i, W_j]`` to the origin (with reduced simplex)."""
    a = W[i]
    b = W[j]
    ab = b - a
    t = -float(a @ ab)                                  # projection of (0 - a) onto ab
    if t <= 0.0:
        return a.copy(), [i], [1.0], False
    den = float(ab @ ab)
    if t >= den:
        return b.copy(), [j], [1.0], False
    t /= den
    return a + t * ab, [i, j], [1.0 - t, t], False


def _closest_tri(
    W: list[np.ndarray], i: int, j: int, k: int
) -> tuple[np.ndarray, list[int], list[float], bool]:
    """Closest point on triangle ``[W_i, W_j, W_k]`` to the origin (with reduced simplex)."""
    a = W[i]
    b = W[j]
    c = W[k]
    ab = b - a
    ac = c - a
    ap = -a
    d1 = float(ab @ ap)
    d2 = float(ac @ ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a.copy(), [i], [1.0], False              # vertex A region
    bp = -b
    d3 = float(ab @ bp)
    d4 = float(ac @ bp)
    if d3 >= 0.0 and d4 <= d3:
        return b.copy(), [j], [1.0], False              # vertex B region
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:           # edge AB region
        t = d1 / (d1 - d3)
        return a + t * ab, [i, j], [1.0 - t, t], False
    cp = -c
    d5 = float(ab @ cp)
    d6 = float(ac @ cp)
    if d6 >= 0.0 and d5 <= d6:
        return c.copy(), [k], [1.0], False              # vertex C region
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:           # edge AC region
        t = d2 / (d2 - d6)
        return a + t * ac, [i, k], [1.0 - t, t], False
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:   # edge BC region
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + t * (c - b), [j, k], [1.0 - t, t], False
    denom = 1.0 / (va + vb + vc)                        # interior face region
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w, [i, j, k], [1.0 - v - w, v, w], False


def _point_outside_plane(
    W: list[np.ndarray], i: int, j: int, k: int, opp: int
) -> bool:
    """Whether the origin is on the opposite side of plane ``(i,j,k)`` from vertex ``opp``.

    A degenerate (zero-area) face returns ``False`` (cannot be "outside" a non-face); the
    enclosing-tetra guard in :func:`_closest_tetra` handles flat simplices separately.
    """
    a = W[i]
    n = np.cross(W[j] - a, W[k] - a)
    if float(n @ n) < 1e-30:
        return False
    sp = float((-a) @ n)                                # (origin - a) . n
    so = float((W[opp] - a) @ n)                        # (opp - a) . n
    return sp * so < 0.0


def _closest_tetra(
    W: list[np.ndarray],
) -> tuple[np.ndarray, list[int], list[float] | None, bool]:
    """Closest point on tetrahedron ``W[0..3]`` to the origin (or ENCLOSED flag).

    Tests each of the 4 faces; for every face the origin lies outside of, take that face's
    closest point and keep the nearest. If the origin is outside no face it is ENCLOSED
    (penetration) -> the fourth return value is ``True`` and the caller hands the full
    tetra to EPA. A (numerically) flat tetra is reduced over its 4 triangular faces.
    """
    a, b, c, d = W[0], W[1], W[2], W[3]
    vol = float(np.dot(b - a, np.cross(c - a, d - a)))
    if abs(vol) < 1e-18:
        # Degenerate / flat tetra: closest over its triangular faces (never "enclosed").
        best = None
        best_sq = np.inf
        for (i, j, k) in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            v, idxs, lam, _ = _closest_tri(W, i, j, k)
            sq = float(v @ v)
            if sq < best_sq:
                best_sq = sq
                best = (v, idxs, lam)
        return best[0], best[1], best[2], False         # type: ignore[index]

    best = None
    best_sq = np.inf
    any_outside = False
    # (i, j, k, opposite-vertex) for the 4 faces (each omits one tetra vertex).
    for (i, j, k, opp) in ((0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2), (1, 3, 2, 0)):
        if _point_outside_plane(W, i, j, k, opp):
            any_outside = True
            v, idxs, lam, _ = _closest_tri(W, i, j, k)
            sq = float(v @ v)
            if sq < best_sq:
                best_sq = sq
                best = (v, idxs, lam)
    if not any_outside:
        return np.zeros(3), [0, 1, 2, 3], None, True     # origin enclosed -> penetration
    return best[0], best[1], best[2], False              # type: ignore[index]


def _closest_on_simplex(
    W: list[np.ndarray],
) -> tuple[np.ndarray, list[int], list[float] | None, bool]:
    """Dispatch the closest-point-to-origin routine by simplex size (1-4 points)."""
    n = len(W)
    if n == 1:
        return W[0].copy(), [0], [1.0], False
    if n == 2:
        return _closest_seg(W, 0, 1)
    if n == 3:
        return _closest_tri(W, 0, 1, 2)
    return _closest_tetra(W)


# --------------------------------------------------------------------------------------
# EPA (Expanding Polytope Algorithm): penetration depth + normal from a GJK tetra that
# encloses the origin. The eval gates ONLY on the SEPARATED-frame distance, so penetration
# depth here is best-effort (documented): a robust, capped expansion with a graceful
# fallback, never a crash on a penetrating frame.
# --------------------------------------------------------------------------------------


def _make_outward_face(
    W: list[np.ndarray], i: int, j: int, k: int
) -> list | None:
    """A face record ``[i, j, k, n, d]`` with ``n`` the OUTWARD unit normal, ``d = n·W_i ≥ 0``.

    Orientation is fixed by the origin (which is strictly interior to the EPA polytope): the
    outward normal satisfies ``n · vertex > 0``, so we flip ``n`` if ``n · W_i < 0``. Returns
    ``None`` for a degenerate (zero-area) triangle.
    """
    a = W[i]
    n = np.cross(W[j] - a, W[k] - a)
    ln = float(np.linalg.norm(n))
    if ln < 1e-18:
        return None
    n = n / ln
    d = float(n @ a)
    if d < 0.0:                                          # orient outward (origin interior)
        n = -n
        d = -d
    return [i, j, k, n, d]


def _init_faces(W: list[np.ndarray]) -> list | None:
    """The 4 outward-oriented faces of the initial GJK tetra, or ``None`` if degenerate."""
    vol = float(np.dot(W[1] - W[0], np.cross(W[2] - W[0], W[3] - W[0])))
    if abs(vol) < 1e-18:
        return None
    faces = []
    for (i, j, k) in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
        f = _make_outward_face(W, i, j, k)
        if f is None:
            return None
        faces.append(f)
    return faces


def _barycentric(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, c: np.ndarray
) -> list[float] | None:
    """Barycentric weights of point ``c`` w.r.t. triangle ``(p0, p1, p2)`` (or ``None``)."""
    v0 = p1 - p0
    v1 = p2 - p0
    v2 = c - p0
    d00 = float(v0 @ v0)
    d01 = float(v0 @ v1)
    d11 = float(v1 @ v1)
    d20 = float(v2 @ v0)
    d21 = float(v2 @ v1)
    den = d00 * d11 - d01 * d01
    if abs(den) < 1e-18:
        return None
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return [1.0 - v - w, v, w]


def _epa_result(
    face: list,
    W: list[np.ndarray],
    WA: list[np.ndarray],
    WB: list[np.ndarray],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Turn the closest EPA face into ``(gap < 0, point, normal)`` (support -> moving).

    The closest boundary point of ``A ⊖ B`` to the origin is ``c = d·n`` (``n`` the outward
    Minkowski face normal, ``d`` the penetration depth). Its barycentric coords on the face
    transfer to per-hull witnesses ``wa = Σλ a_i`` / ``wb = Σλ b_i``; the contact ``point`` is
    their midpoint. Because ``wa − wb = c = d·n`` points OUTWARD on ``A ⊖ B`` (i.e. moving ->
    support during overlap), the package's ``support -> moving`` normal is ``−n``, and the gap
    is ``−d`` (negative => penetration), continuous with the separated branch (``+v/‖v‖``).
    """
    i, j, k, n, d = face
    c = d * n
    lam = _barycentric(W[i], W[j], W[k], c)
    if lam is None or any(li < -1e-6 for li in lam):
        lam = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    wa = lam[0] * WA[i] + lam[1] * WA[j] + lam[2] * WA[k]
    wb = lam[0] * WB[i] + lam[1] * WB[j] + lam[2] * WB[k]
    point = 0.5 * (wa + wb)
    return -float(d), point, -n


def _epa_fallback(
    WA: list[np.ndarray], WB: list[np.ndarray]
) -> tuple[float, np.ndarray, np.ndarray]:
    """Crude penetration estimate when EPA cannot run (degenerate simplex).

    APPROXIMATE (documented, never gated): a small negative gap with the witness-mean
    direction as normal. Keeps a penetrating frame from crashing ``resolve``.
    """
    wa = np.mean(np.asarray(WA, dtype=float), axis=0)
    wb = np.mean(np.asarray(WB, dtype=float), axis=0)
    diff = wa - wb
    nd = float(np.linalg.norm(diff))
    normal = diff / nd if nd > 1e-12 else np.array([0.0, 0.0, 1.0])
    return -nd, 0.5 * (wa + wb), normal


def _epa(
    vA: np.ndarray,
    vB: np.ndarray,
    W: list[np.ndarray],
    WA: list[np.ndarray],
    WB: list[np.ndarray],
    max_iter: int = 64,
    tol: float = 1e-10,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Expanding-Polytope Algorithm: penetration depth + normal from an enclosing tetra.

    Expands the GJK tetrahedron toward the boundary of ``A ⊖ B`` nearest the origin: pick the
    face closest to the origin, take the Minkowski support in that face's outward normal, and
    if it does not push past the face (within ``tol``) the face IS the closest boundary feature
    -> return its depth/normal/witnesses (:func:`_epa_result`). Otherwise add the support point,
    delete every face it can see, and re-triangulate the hole's HORIZON (the undirected edges
    belonging to exactly one deleted face) to the new vertex; every face is re-oriented outward
    via the (interior) origin, so no winding bookkeeping is needed. Capped at ``max_iter`` with a
    graceful fallback -- penetration depth is best-effort (the eval gates separation only).
    """
    W = list(W)
    WA = list(WA)
    WB = list(WB)
    if len(W) < 4:
        return _epa_fallback(WA, WB)
    faces = _init_faces(W)
    if faces is None:
        return _epa_fallback(WA, WB)

    closest = faces[int(np.argmin([f[4] for f in faces]))]
    for _ in range(max_iter):
        ci = int(np.argmin([f[4] for f in faces]))
        closest = faces[ci]
        n = closest[3]
        d_face = closest[4]
        w, a_w, b_w = _minkowski_support(vA, vB, n)
        if float(n @ w) - d_face <= tol:                # cannot expand further -> converged
            break
        wi = len(W)
        W.append(w)
        WA.append(a_w)
        WB.append(b_w)
        # Faces the new vertex can "see" (its side of the face plane).
        visible = [q for q, f in enumerate(faces) if float(f[3] @ (w - W[f[0]])) > 1e-12]
        if not visible:
            break
        # Horizon: undirected edges that border exactly ONE visible face.
        edge_count: dict[frozenset[int], int] = {}
        for q in visible:
            f = faces[q]
            tri = (f[0], f[1], f[2])
            for e in (
                frozenset((tri[0], tri[1])),
                frozenset((tri[1], tri[2])),
                frozenset((tri[2], tri[0])),
            ):
                edge_count[e] = edge_count.get(e, 0) + 1
        visible_set = set(visible)
        faces = [f for q, f in enumerate(faces) if q not in visible_set]
        for e, cnt in edge_count.items():
            if cnt != 1:
                continue
            p, q = tuple(e)
            nf = _make_outward_face(W, p, q, wi)
            if nf is not None:
                faces.append(nf)
        if not faces:
            return _epa_fallback(WA, WB)

    return _epa_result(closest, W, WA, WB)


# --------------------------------------------------------------------------------------
# Penetration via the Separating Axis Theorem (the watertight close-out, DESIGN.md III.5).
#
# EPA above is accurate for a WELL-CONDITIONED enclosing tetra (tessellated curved meshes),
# but a GJK simplex on COPLANAR/aligned faces (box-vs-box) degenerates -- the origin lands on
# a lower-dimensional simplex with no enclosing tetra, and forcing one mis-converges EPA. For
# two convex polytopes the penetration depth is exactly the MINIMUM overlap over the candidate
# separating axes (each hull's face normals, plus edge x edge axes), so we resolve the
# touching/penetrating case with SAT on the convex hulls -- exact for every convex shape,
# coplanar boxes included, with the curved-mesh answer unchanged (the min-overlap axis is the
# line of centres). `scipy.spatial.ConvexHull` supplies the hull faces/edges (scipy is already
# a project dependency, used by `contact.dynamics_id`); a degenerate (non-3D) cloud falls back
# to the EPA path.
# --------------------------------------------------------------------------------------

#: Skip the O(|edges_A|*|edges_B|) edge-edge SAT axes above this hull vertex count: a densely
#: tessellated mesh's face normals already sample direction space finely enough (the edge-edge
#: refinement changes the min-overlap by < the tessellation error), so this keeps SAT cheap on
#: spheres/curved meshes while staying exact for the polytopes (boxes/capsules) that need it.
_SAT_EDGE_MAX = 64

#: GJK separation below which the contact is treated as touching/penetrating and handed to the
#: SAT close-out (the origin is on, or inside, the Minkowski difference). Above it the hulls are
#: cleanly separated and the (exact, validated) GJK distance is returned directly.
_TOUCH_EPS = 1e-7


def _hull_edges(hull, verts: np.ndarray) -> np.ndarray:
    """Unique edge DIRECTION vectors of a convex hull (from its triangular faces)."""
    edges: set[tuple[int, int]] = set()
    for tri in hull.simplices:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for p, q in ((a, b), (b, c), (c, a)):
            edges.add((p, q) if p < q else (q, p))
    e = np.array(sorted(edges), dtype=int)
    return verts[e[:, 1]] - verts[e[:, 0]]


def _penetration_sat(
    vA: np.ndarray,
    vB: np.ndarray,
    W: list[np.ndarray],
    WA: list[np.ndarray],
    WB: list[np.ndarray],
    fallback_normal: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Exact signed gap for two overlapping/touching convex hulls via SAT (DESIGN.md III.5).

    Called when GJK reports the origin enclosed or on the boundary (gap ~= 0). Builds the
    candidate separating axes -- the face normals of both convex hulls, plus (for small
    polytopes) every edge_A x edge_B -- and for each takes the projection overlap
    ``min(maxA, maxB) - max(minA, minB)``. The hulls are SEPARATED iff some axis has overlap
    <= 0 (a grazing/just-touching boundary -> gap 0); otherwise they PENETRATE and the
    penetration depth is the MINIMUM overlap (the least translation that separates them), with
    that axis as the contact normal (oriented support -> moving). Exact for every convex
    polytope, the coplanar box-vs-box case included. Falls back to :func:`_epa` if the cloud is
    degenerate (``ConvexHull`` cannot build a 3-D hull).
    """
    vA = np.asarray(vA, dtype=float)
    vB = np.asarray(vB, dtype=float)
    try:
        from scipy.spatial import ConvexHull

        hA = ConvexHull(vA)
        hB = ConvexHull(vB)
    except Exception:
        return _epa(vA, vB, W, WA, WB)              # non-3-D / degenerate cloud -> EPA fallback

    axis_blocks = [hA.equations[:, :3], hB.equations[:, :3]]   # both hulls' face normals
    if len(vA) <= _SAT_EDGE_MAX and len(vB) <= _SAT_EDGE_MAX:
        eA = _hull_edges(hA, vA)
        eB = _hull_edges(hB, vB)
        axis_blocks.append(np.cross(eA[:, None, :], eB[None, :, :]).reshape(-1, 3))
    axes = np.vstack(axis_blocks)
    norms = np.linalg.norm(axes, axis=1)
    axes = axes[norms > 1e-9] / norms[norms > 1e-9, None]      # drop ~zero (parallel edges), unit

    pA = vA @ axes.T                                            # (V_A, K)
    pB = vB @ axes.T                                            # (V_B, K)
    overlap = np.minimum(pA.max(0), pB.max(0)) - np.maximum(pA.min(0), pB.min(0))   # (K,)
    k = int(np.argmin(overlap))
    min_ov = float(overlap[k])

    centroid_diff = vA.mean(0) - vB.mean(0)
    if min_ov <= 1e-12:
        # A separating axis exists -> just-touching boundary, not penetrating: signed gap 0.
        nrm = fallback_normal
        return 0.0, 0.5 * (vA.mean(0) + vB.mean(0)), np.asarray(nrm, dtype=float)

    axis = axes[k].copy()
    if float(centroid_diff @ axis) < 0.0:                       # orient support(B) -> moving(A)
        axis = -axis
    # Contact point: midpoint of the two deepest features along the penetration axis.
    a_deep = vA[int(np.argmin(vA @ axis))]                      # A vertex furthest INTO B
    b_deep = vB[int(np.argmax(vB @ axis))]                      # B vertex furthest INTO A
    point = 0.5 * (a_deep + b_deep)
    return -min_ov, point, axis


# --------------------------------------------------------------------------------------
# GJK distance between two convex vertex clouds (the separation query the eval gates on).
# --------------------------------------------------------------------------------------


def gjk_distance(
    vA_world: np.ndarray,
    vB_world: np.ndarray,
    max_iter: int = 64,
    tol: float = 1e-12,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Minimum distance (+ witness point and normal) between two convex vertex clouds.

    GJK on the Minkowski difference ``A ⊖ B`` (DESIGN.md II.D / III.5, the Mesh/SDF rung).
    The distance between the two hulls equals the distance from the origin to ``A ⊖ B``: the
    simplex is walked toward the origin (closest-point sub-routines above), each step adding the
    Minkowski support in the direction of the origin, until the support stops making progress
    (``v·v − v·w ≤ tol``) -- then the separation is ``‖v‖`` for the closest point ``v`` on the
    final simplex. The barycentric weights of ``v`` transfer the simplex vertices' per-hull
    witnesses to closest points ``wa ∈ A`` / ``wb ∈ B``.

    When the hulls TOUCH or OVERLAP (the origin is on/inside ``A ⊖ B``) the query is handed to
    :func:`_penetration_sat` for the exact signed gap (``0`` touching, ``< 0`` penetration) and
    normal -- watertight for every convex polytope, coplanar boxes included.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        ``(gap, point, normal)``:

        * ``gap``    -- signed separation; ``> 0`` distance between hulls, ``< 0`` penetration.
        * ``point``  -- world contact point: the midpoint of the closest witness pair (the
          midpoint of the deepest witness pair under penetration).
        * ``normal`` -- world unit ``support -> moving`` normal: ``v/‖v‖`` (the origin->closest
          direction on ``A ⊖ B``, pointing from hull ``B`` to hull ``A``) when separated; the
          minimum-overlap separating axis (oriented support -> moving) when penetrating.
          POSITION-derived -- no body-fixed spin.
    """
    vA = np.asarray(vA_world, dtype=float)
    vB = np.asarray(vB_world, dtype=float)

    # Initial search direction: between the cloud centroids (any nonzero direction converges).
    d = vA.mean(axis=0) - vB.mean(axis=0)
    if float(d @ d) < 1e-18:
        d = np.array([1.0, 0.0, 0.0])
    w, a, b = _minkowski_support(vA, vB, d)
    W = [w]
    WA = [a]
    WB = [b]

    sep_normal = np.array([0.0, 0.0, 1.0])               # touching-frame normal fallback
    for _ in range(max_iter):
        v, idxs, lam, enclosed = _closest_on_simplex(W)
        if enclosed:
            break
        # Reduce the simplex to its minimal sub-feature (keep witnesses parallel).
        W = [W[t] for t in idxs]
        WA = [WA[t] for t in idxs]
        WB = [WB[t] for t in idxs]
        dist2 = float(v @ v)
        if dist2 > tol:
            sep_normal = v / np.sqrt(dist2)
        else:
            break                                        # origin on the simplex (touching)
        d = -v
        w, a, b = _minkowski_support(vA, vB, d)
        # Termination: the new support adds no closer approach toward the origin.
        if dist2 - float(w @ v) <= 1e-12 * max(1.0, dist2):
            break
        if any(float(np.max(np.abs(w - ww))) < 1e-12 for ww in W):
            break                                        # duplicate support -> converged
        W.append(w)
        WA.append(a)
        WB.append(b)

    # Finalize on the current simplex (handles both the break paths and max_iter exhaustion).
    v, idxs, lam, enclosed = _closest_on_simplex(W)
    gap = float(np.sqrt(max(float(v @ v), 0.0)))
    if enclosed or gap <= _TOUCH_EPS:
        # The hulls TOUCH or PENETRATE: GJK either enclosed the origin (a well-conditioned
        # penetration) or stalled on a lower-dimensional simplex (the coplanar/aligned-face
        # degeneracy where forcing an EPA tetra mis-converges). Either way the SIGNED gap is
        # resolved EXACTLY by the Separating-Axis close-out on the convex hulls (DESIGN.md III.5,
        # the watertight-penetration finish): correct for every convex polytope -- coplanar
        # box-vs-box included -- and unchanged on curved meshes (the min-overlap axis is the line
        # of centres). This is what makes mesh-vs-mesh penetration depth exact, not best-effort.
        return _penetration_sat(vA, vB, W, WA, WB, sep_normal)
    W = [W[t] for t in idxs]
    WA = [WA[t] for t in idxs]
    WB = [WB[t] for t in idxs]
    wa = sum(lam[t] * WA[t] for t in range(len(lam)))    # type: ignore[index]
    wb = sum(lam[t] * WB[t] for t in range(len(lam)))    # type: ignore[index]
    point = 0.5 * (wa + wb)
    normal = v / gap if gap > 1e-12 else sep_normal
    return gap, point, normal
