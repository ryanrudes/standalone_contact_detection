"""Contact-geometry resolvers: the fidelity ladder behind the `ContactGeometry` waist.

DESIGN.md PART III (III.1/III.2) and PART II section D. A resolver turns the pose streams
of a moving body and its (possibly moving) support into one world-frame
:data:`~contact.types.ContactFrame` per recorded frame — a ``(point, normal, gap)`` triple
(plus provenance sigmas) that :func:`contact.geometry.observe` consumes *before* running the
twist decomposition. Because the decomposition only ever sees this world-frame triple,
swapping resolvers (flat plane -> sphere -> mesh) leaves everything downstream untouched.

Phase 0 ships exactly one resolver, :class:`FlatRegion`, which wraps today's
``(surface, contact_point_local)`` spec and reproduces the world point/normal/gap arithmetic
of :func:`contact.geometry.observe` (its steps 1-3) bit-for-bit — the regression lock that
keeps every current demo byte-identical (DESIGN.md III.2 / §9 Phase 0).
"""

from __future__ import annotations

import numpy as np

# Reuse the EXACT same elementary helpers `observe` uses, so the arithmetic is identical
# (DESIGN.md III.2: FlatRegion must be bit-identical to the pre-refactor pipeline).
from .geometry import plane_gap, quat_rotate, quat_to_matrix
from .mesh_collision import convex_plane, gjk_distance
from .types import ContactFrame, ContactPoint, PoseTrajectory, SupportSurface

__all__ = [
    "FlatRegion",
    "SpherePlane",
    "SphereSphere",
    "BoxPlane",
    "MeshPlane",
    "MeshConvex",
]


class FlatRegion:
    """A flat planar support + a fixed tracked point on the moving body (DESIGN.md II.D).

    The default, mesh-free resolver: it is the configuration of today's
    ``(surface, contact_point_local)`` spec. Per frame it rotates the fixed *local* contact
    point by the moving body's quaternion into the world, carries the support plane into the
    world by the support's pose, and reports the signed plane gap — exactly the arithmetic of
    :func:`contact.geometry.observe` steps 1-3, so the resulting observations are bit-identical
    to the pre-refactor pipeline.

    The normal is glued to the support body frame, which is exact for a flat, *non-rotating*
    floor (the validated regime). A curved or rotating support needs a position-derived
    normal — that is a Phase-1 resolver (``SpherePlane`` / ``SphereSphere``), not this one.

    Provenance (DESIGN.md III.5): Phase 0 declares ``normal_sigma = gap_sigma = 0.0`` (no
    extra uncertainty), so the measurement-tempering path is untouched and the baseline stays
    bit-identical. No ``meas_cov`` is introduced here in Phase 0.
    """

    #: This resolver tracks a single FIXED material point on the moving body, so its world
    #: trajectory is smooth and :func:`contact.geometry.observe` recovers the moving-point
    #: velocity by DIFFERENTIATING it -- the bit-identical legacy path (DESIGN.md PART II.D).
    migrating = False

    def __init__(
        self, surface: SupportSurface, contact_point_local: np.ndarray = np.zeros(3)
    ) -> None:
        self.surface = surface
        self.contact_point_local = contact_point_local

    def resolve(
        self, moving: PoseTrajectory, support: PoseTrajectory
    ) -> list[ContactFrame]:
        """One single-point :data:`~contact.types.ContactFrame` per recorded frame (length T).

        Reproduces :func:`contact.geometry.observe` steps 1-3 verbatim — same helpers, same
        order of operations — so ``p``, ``normal_w`` and ``gap`` are bit-identical:

            p          = moving.position  + R(moving.quat)  @ contact_point_local
            plane_pt_w = support.position + R(support.quat) @ surface.point
            normal_w   = normalize(R(support.quat) @ surface.normal)
            gap        = plane_gap(p, plane_pt_w, normal_w)   ==  (p - plane_pt_w) . normal_w
        """
        mov_pos = np.asarray(moving.position, dtype=float)             # (T,3) world
        mov_quat = np.asarray(moving.quat, dtype=float)                # (T,4)
        sup_pos = np.asarray(support.position, dtype=float)            # (T,3) world
        sup_quat = np.asarray(support.quat, dtype=float)               # (T,4)
        cpl = np.asarray(self.contact_point_local, dtype=float)        # (3,)

        # Step 1: world contact point p(t) = moving origin + R(moving) @ contact_local.
        p = mov_pos + quat_rotate(mov_quat, cpl)                       # (T,3)

        # Step 2: carry the support plane into the world each frame. The plane *point* is a
        # body point -> rotate AND translate; the plane *normal* is a free direction ->
        # rotate only (no translation), then normalize to a unit world normal.
        plane_pt_w = sup_pos + quat_rotate(sup_quat, self.surface.point)  # (T,3) world point
        normal_w = quat_rotate(sup_quat, self.surface.normal)            # (T,3) world normal
        normal_w = normal_w / np.linalg.norm(normal_w, axis=-1, keepdims=True)

        # Step 3: support-relative gap = signed distance of p to the moving plane.
        gap = plane_gap(p, plane_pt_w, normal_w)                       # (T,)

        # One single-point ContactFrame per recorded frame. Phase-0 provenance sigmas are
        # 0.0 (no extra uncertainty) so the tempering path / baseline stay bit-identical.
        return [
            [
                ContactPoint(
                    point=p[i],
                    normal=normal_w[i],
                    gap=float(gap[i]),
                    normal_sigma=0.0,
                    gap_sigma=0.0,
                )
            ]
            for i in range(p.shape[0])
        ]


class SpherePlane:
    """A sphere of radius ``r_moving`` resting against a planar support (DESIGN.md II.D).

    The Phase-1 resolver for a sphere on a (possibly moving) plane. The plane is carried
    into the world by the support's pose exactly as :class:`FlatRegion` does — the world
    outward normal is the rotated local normal and the plane passes through
    ``sup_pos + R(sup_quat) @ surface.point``. The difference is the *sphere offset*: the
    tracked geometry point is the sphere CENTER (the moving body's origin, ``c``), and the
    contact lives one radius inboard along the normal, so the signed gap is the center's
    plane distance minus the radius and the reported point is the foot of the sphere
    (DESIGN.md II.D / III.3):

        normal_w   = normalize(R(sup_quat) @ surface.normal)
        plane_pt_w = sup_pos + R(sup_quat) @ surface.point
        c          = mov_pos                       # sphere center == body origin
        gap        = (c - plane_pt_w) . normal_w - r_moving
        point      = c - r_moving * normal_w       # the sphere's foot on the plane

    The normal here is still the (rotated) plane normal — exact for a flat support, whose
    surface direction is genuinely body-fixed. The position-derived normal that fixes the
    spinning-normal artifact is the *sphere-on-sphere* case (:class:`SphereSphere`); a plane
    has no such pathology. ``contact_point_local`` is accepted for signature symmetry with
    :class:`FlatRegion` (so resolvers are swappable), but a sphere tracks its center, i.e.
    the body origin, per DESIGN.md II.D.

    Provenance (DESIGN.md III.5): ``normal_sigma = gap_sigma = 0.0`` (exact primitive).
    """

    #: The tracked geometry point (the sphere's foot ``c - r*n``) rides rigidly with the
    #: sphere centre, so its trajectory is smooth and ``observe`` DIFFERENTIATES it -- the
    #: bit-identical path, not the migrating-corner analytic twist (DESIGN.md PART II.D).
    migrating = False

    def __init__(
        self,
        r_moving: float,
        surface: SupportSurface,
        contact_point_local: np.ndarray = np.zeros(3),
    ) -> None:
        self.r_moving = float(r_moving)
        self.surface = surface
        self.contact_point_local = contact_point_local

    def resolve(
        self, moving: PoseTrajectory, support: PoseTrajectory
    ) -> list[ContactFrame]:
        """One single-point :data:`~contact.types.ContactFrame` per recorded frame (length T)."""
        mov_pos = np.asarray(moving.position, dtype=float)            # (T,3) world (sphere center)
        sup_pos = np.asarray(support.position, dtype=float)           # (T,3) world
        sup_quat = np.asarray(support.quat, dtype=float)              # (T,4)

        # World plane: carry the support normal (rotate only -> normalize) and a point on the
        # plane (rotate AND translate) into the world, exactly as FlatRegion does.
        normal_w = quat_rotate(sup_quat, self.surface.normal)         # (T,3) world normal
        normal_w = normal_w / np.linalg.norm(normal_w, axis=-1, keepdims=True)
        plane_pt_w = sup_pos + quat_rotate(sup_quat, self.surface.point)  # (T,3) world point

        # Sphere center is the moving body origin; the gap is its signed plane distance minus
        # the radius, and the contact point is the sphere's foot one radius along the normal.
        c = mov_pos                                                   # (T,3)
        gap = plane_gap(c, plane_pt_w, normal_w) - self.r_moving      # (T,)
        point = c - self.r_moving * normal_w                          # (T,3)

        return [
            [
                ContactPoint(
                    point=point[i],
                    normal=normal_w[i],
                    gap=float(gap[i]),
                    normal_sigma=0.0,
                    gap_sigma=0.0,
                )
            ]
            for i in range(c.shape[0])
        ]


class SphereSphere:
    """Two spheres (radii ``r_moving``/``r_support``), the position-derived-normal resolver.

    The Phase-1 resolver that fixes the spinning-normal artifact on a ball<->ball edge
    (DESIGN.md II.D / III.5). Per frame the two sphere centers are the moving and support
    body world origins; the contact normal is derived from the *line of centers*, the gap
    is the centre distance minus the two radii, and the reported contact point is the moving
    sphere's surface point on that line:

        c1     = moving.position                   # moving sphere centre (world)
        c2     = support.position                  # support sphere centre (world)
        d      = c1 - c2 ;  dist = ||d||
        normal = d / dist                          # position-derived; [0,0,1] if dist < 1e-12
        gap    = dist - r_moving - r_support
        point  = c1 - r_moving * normal            # moving-sphere surface pt, on the line of centres

    CRITICAL (the whole point, DESIGN.md II.D): the normal comes from the POSITIONS
    (``c1 - c2``), never from a body-fixed local vector rotated by a quaternion. A spinning
    ball rotates any body-fixed direction, so a quat-carried normal would whirl with the
    spin and manufacture phantom closing/opening velocities (the "7 phantom impacts" on
    ``ballA<->ballB``). The line-of-centres normal is rotation-invariant, so the single real
    collision survives and the cradle's true closing velocities are recovered.

    NOTE on ``point`` (DESIGN.md II.D writes the geometric contact ``c2 + r_support*normal``):
    we report the MOVING sphere's surface point ``c1 - r_moving*normal`` instead. At contact
    (``dist == r_moving + r_support``) the two are the same location, but only the moving-pinned
    point has the right *trajectory*: :func:`contact.geometry.observe` recovers the moving
    body's velocity by differentiating this contact-point stream, so a support-pinned
    ``c2 + r_support*normal`` would track the SUPPORT and collapse the relative normal velocity
    to ~0 -- erasing the very closing velocity the impact detector needs (it would yield 0
    impulses, not the single real collision, and would NOT restore the cradle's closing
    velocities). This mirrors :class:`SpherePlane`'s own moving-pinned ``c - r_moving*normal``.

    Provenance (DESIGN.md III.5): ``normal_sigma = gap_sigma = 0.0`` (exact primitive).
    """

    #: The reported point is the moving sphere's surface point ``c1 - r1*n``, a point that
    #: rides rigidly with the moving sphere; its trajectory is smooth so ``observe``
    #: DIFFERENTIATES it (the bit-identical path), never the analytic migrating twist. The
    #: position-derived NORMAL fixes the spin artifact, but the point itself does not migrate
    #: across the body the way a box's nearest corner does (DESIGN.md PART II.D).
    migrating = False

    def __init__(self, r_moving: float, r_support: float) -> None:
        self.r_moving = float(r_moving)
        self.r_support = float(r_support)

    def resolve(
        self, moving: PoseTrajectory, support: PoseTrajectory
    ) -> list[ContactFrame]:
        """One single-point :data:`~contact.types.ContactFrame` per recorded frame (length T)."""
        c1 = np.asarray(moving.position, dtype=float)                 # (T,3) moving centre (world)
        c2 = np.asarray(support.position, dtype=float)                # (T,3) support centre (world)

        d = c1 - c2                                                   # (T,3) line of centres
        dist = np.linalg.norm(d, axis=-1)                             # (T,)

        # Position-derived unit normal (the spin-artifact fix). Default to +z, then overwrite
        # every well-separated frame with d/||d||; degenerate (coincident) frames keep [0,0,1].
        normal = np.tile(np.array([0.0, 0.0, 1.0]), (d.shape[0], 1))  # (T,3) fallback
        safe = dist >= 1e-12
        normal[safe] = d[safe] / dist[safe, None]

        gap = dist - self.r_moving - self.r_support                  # (T,)
        # Reported contact point = the MOVING sphere's surface point on the line of centres,
        # c1 - r_moving*normal. At contact this equals DESIGN.md II.D's c2 + r_support*normal,
        # but its TRAJECTORY rides with the moving body, which observe() differentiates to get
        # the moving body's velocity (v_moving_point = d/dt(point)). A support-pinned point
        # would instead track the support and zero out the relative normal velocity, erasing
        # the closing velocity the impact detector needs (the "restore closing velocities"
        # requirement of DESIGN.md III.3/III.5). See the class docstring NOTE.
        point = c1 - self.r_moving * normal                          # (T,3) moving sphere surface

        return [
            [
                ContactPoint(
                    point=point[i],
                    normal=normal[i],
                    gap=float(gap[i]),
                    normal_sigma=0.0,
                    gap_sigma=0.0,
                )
            ]
            for i in range(c1.shape[0])
        ]


class BoxPlane:
    """A box (8 corners) against a planar support — the *migrating-contact* resolver.

    The Phase-2 resolver that fixes the tumbling box (DESIGN.md PART II.D / III.5). Unlike
    :class:`FlatRegion`, which tracks one body-FIXED material point, a box bouncing across a
    plane touches with whichever corner is *currently lowest* — and that corner JUMPS from one
    bounce to the next. So the contact point is not a fixed material point; it MIGRATES across
    the body. Per frame we:

        * build the 8 box corners in the body-local frame (the sign combos of the half-extents);
        * place them in the world via the moving pose: ``corner_w = mov_pos + R(mov_quat) @ corner_local``;
        * carry the support plane into the world EXACTLY as :class:`FlatRegion` does
          (``n_w = normalize(R(sup_quat) @ surface.normal)``, ``p_w = sup_pos + R(sup_quat) @ surface.point``);
        * take each corner's signed plane distance ``dᵢ = (cornerᵢ − p_w) · n_w``;
        * the gap is ``min_i dᵢ`` and the contact points are the corner(s) with
          ``dᵢ ≤ gap + eps`` (1 = tipping on a corner, 2 = landing on an edge, 4 = a flat face),
          each a :class:`~contact.types.ContactPoint` carrying ``point=cornerᵢ``, ``normal=n_w``,
          ``gap=dᵢ``.

    The result is a possibly MULTI-POINT :data:`~contact.types.ContactFrame` per recorded frame
    (DESIGN.md PART II.C: the kinematic ``observe`` aggregates these to one representative for the
    mode; the full point list is what a force/structure layer would consume).

    ``migrating = True`` is the signal :func:`contact.geometry.observe` reads to compute the
    moving contact point's velocity ANALYTICALLY from the body twist
    (``v = v_com + omega × (point − com)``) instead of differentiating the point's world
    trajectory — because that trajectory teleports between corners and differentiating it would
    manufacture a velocity spike at every switch. The two formulas agree for a fixed material
    point; they differ exactly where the corner migrates, and the analytic one is the correct,
    spike-free choice there (DESIGN.md PART II.D).

    ``contact_point_local`` is accepted for signature symmetry with the other resolvers (so they
    stay swappable) but is unused: a box tracks its corners, not a single tracked point.

    Provenance (DESIGN.md III.5): ``normal_sigma = gap_sigma = 0.0`` (the corner geometry is
    exact for a box on a flat plane).
    """

    #: The nearest corner JUMPS between bounces, so the contact point migrates across the body.
    #: ``observe`` therefore recovers the moving-point velocity from the body twist analytically
    #: (``v = v_com + omega × r``), not by differentiating the teleporting point (DESIGN.md II.D).
    migrating = True

    #: Tolerance (m) for grouping corners into one contact: corners within ``eps`` of the lowest
    #: corner are all reported, so a near-flat face/edge yields the 2 or 4 simultaneous contacts.
    eps = 1e-3

    def __init__(
        self,
        half_extents: np.ndarray,
        surface: SupportSurface,
        contact_point_local: np.ndarray = np.zeros(3),
    ) -> None:
        self.half_extents = np.asarray(half_extents, dtype=float)  # (3,) box half-extents
        self.surface = surface
        self.contact_point_local = contact_point_local

    def resolve(
        self, moving: PoseTrajectory, support: PoseTrajectory
    ) -> list[ContactFrame]:
        """One (possibly MULTI-point) :data:`~contact.types.ContactFrame` per frame (length T)."""
        mov_pos = np.asarray(moving.position, dtype=float)            # (T,3) world box origin
        mov_quat = np.asarray(moving.quat, dtype=float)               # (T,4)
        sup_pos = np.asarray(support.position, dtype=float)           # (T,3) world
        sup_quat = np.asarray(support.quat, dtype=float)              # (T,4)
        he = np.asarray(self.half_extents, dtype=float)               # (3,)

        # The 8 box corners in the body-local frame: every sign combination of +/- half-extents.
        signs = np.array(
            [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
            dtype=float,
        )                                                             # (8,3)
        corners_local = signs * he                                   # (8,3)

        # Carry the support plane into the world each frame, EXACTLY as FlatRegion does: the
        # plane normal is a free direction (rotate only, then normalize) and the plane point is
        # a body point (rotate AND translate).
        normal_w = quat_rotate(sup_quat, self.surface.normal)        # (T,3) world normal
        normal_w = normal_w / np.linalg.norm(normal_w, axis=-1, keepdims=True)
        plane_pt_w = sup_pos + quat_rotate(sup_quat, self.surface.point)  # (T,3) world point

        # World position of each corner per frame: mov_pos + R(mov_quat) @ corner_local.
        corners_world = np.stack(
            [mov_pos + quat_rotate(mov_quat, corners_local[j]) for j in range(8)],
            axis=1,
        )                                                            # (T,8,3)

        # Signed plane distance of every corner: d[t,j] = (corner[t,j] - plane_pt[t]) . normal[t].
        d = np.einsum(
            "tjk,tk->tj", corners_world - plane_pt_w[:, None, :], normal_w
        )                                                            # (T,8)

        eps = float(self.eps)
        frames: list[ContactFrame] = []
        for i in range(corners_world.shape[0]):
            gap_i = float(d[i].min())                                # nearest-corner gap
            # Every corner within eps of the lowest one is a simultaneous contact (1/2/4 points).
            sel = np.nonzero(d[i] <= gap_i + eps)[0]
            frames.append(
                [
                    ContactPoint(
                        point=corners_world[i, j],
                        normal=normal_w[i],
                        gap=float(d[i, j]),
                        normal_sigma=0.0,
                        gap_sigma=0.0,
                    )
                    for j in sel
                ]
            )
        return frames


class MeshPlane:
    """A convex MESH (arbitrary vertex cloud) against a planar support — the Phase-3 plane resolver.

    The mesh generalization of :class:`BoxPlane` (DESIGN.md PART II.D / III.5, the Mesh/SDF
    rung). Where :class:`BoxPlane` knows its 8 corners in closed form, :class:`MeshPlane` takes
    an arbitrary convex vertex cloud ``vertices_local`` and runs the SAME per-vertex signed-plane
    arithmetic (:func:`contact.mesh_collision.convex_plane`). Per frame we:

        * place every vertex in the world via the moving pose:
          ``vert_w = mov_pos + R(mov_quat) @ vert_local``;
        * carry the support plane into the world EXACTLY as :class:`FlatRegion`/:class:`BoxPlane`
          do (``n_w = normalize(R(sup_quat) @ surface.normal)``,
          ``p_w = sup_pos + R(sup_quat) @ surface.point``);
        * take each vertex's signed plane distance ``dᵢ = (vertᵢ − p_w) · n_w``;
        * the gap is ``min_i dᵢ`` and the contact points are the vertices with ``dᵢ ≤ gap + eps``
          (1 = a single touching vertex, 2 = an edge, 3+ = a face), each a
          :class:`~contact.types.ContactPoint` carrying ``point=vertᵢ``, ``normal=n_w``, ``gap=dᵢ``.

    Because this is *exactly* :class:`BoxPlane`'s arithmetic lifted to a general cloud, feeding it
    the box's 8 corners reproduces :class:`BoxPlane` bit-for-bit (the same ``eps = 1e-3`` grouping,
    the same world normal, the same per-vertex gaps) — the Phase-3 acceptance check against the
    Phase-2 primitive.

    The result is a possibly MULTI-POINT :data:`~contact.types.ContactFrame` per recorded frame
    (DESIGN.md PART II.C); :func:`contact.geometry.observe` reduces it to the min-gap representative
    for the twist decomposition. ``contact_point_local`` is accepted for signature symmetry with the
    other resolvers (so they stay swappable) but is unused: a mesh tracks its vertices.

    Provenance (DESIGN.md III.5): ``normal_sigma = gap_sigma = 0.0`` (a convex mesh against a flat
    plane is an exact query).
    """

    #: The nearest vertex JUMPS between features as the body tumbles, so the contact point migrates
    #: across the body (exactly like :class:`BoxPlane`'s corner). ``observe`` therefore recovers the
    #: moving-point velocity from the body twist analytically (``v = v_com + omega × r``), not by
    #: differentiating the teleporting point (DESIGN.md PART II.D).
    migrating = True

    #: Tolerance (m) for grouping vertices into one contact, matching :class:`BoxPlane` so a box
    #: mesh's face/edge yields the same 2/3/4 simultaneous contacts bit-for-bit.
    eps = 1e-3

    def __init__(
        self,
        vertices_local: np.ndarray,
        surface: SupportSurface,
        contact_point_local: np.ndarray = np.zeros(3),
    ) -> None:
        self.vertices_local = np.asarray(vertices_local, dtype=float)  # (V,3) body-local cloud
        self.surface = surface
        self.contact_point_local = contact_point_local

    def resolve(
        self, moving: PoseTrajectory, support: PoseTrajectory
    ) -> list[ContactFrame]:
        """One (possibly MULTI-point) :data:`~contact.types.ContactFrame` per frame (length T)."""
        mov_pos = np.asarray(moving.position, dtype=float)            # (T,3) world body origin
        mov_quat = np.asarray(moving.quat, dtype=float)               # (T,4)
        sup_pos = np.asarray(support.position, dtype=float)           # (T,3) world
        sup_quat = np.asarray(support.quat, dtype=float)              # (T,4)
        verts_local = self.vertices_local                            # (V,3)

        # Carry the support plane into the world each frame, EXACTLY as FlatRegion/BoxPlane do:
        # the normal is a free direction (rotate only, then normalize), the point a body point
        # (rotate AND translate).
        normal_w = quat_rotate(sup_quat, self.surface.normal)        # (T,3) world normal
        normal_w = normal_w / np.linalg.norm(normal_w, axis=-1, keepdims=True)
        plane_pt_w = sup_pos + quat_rotate(sup_quat, self.surface.point)  # (T,3) world point

        # World position of every vertex per frame: mov_pos + R(mov_quat) @ vert_local. (Same
        # rotation/translation BoxPlane applies to each corner, vectorized over the cloud.)
        R = quat_to_matrix(mov_quat)                                  # (T,3,3)
        verts_world = mov_pos[:, None, :] + np.einsum(
            "tik,vk->tvi", R, verts_local
        )                                                            # (T,V,3)

        eps = float(self.eps)
        frames: list[ContactFrame] = []
        for i in range(verts_world.shape[0]):
            # Single-frame convex-cloud-vs-plane query: the min gap and the contacting vertices.
            gap_i, sel, n_i = convex_plane(verts_world[i], plane_pt_w[i], normal_w[i], eps)
            d_i = (verts_world[i] - plane_pt_w[i]) @ n_i              # (V,) per-vertex signed dist
            frames.append(
                [
                    ContactPoint(
                        point=verts_world[i, j],
                        normal=normal_w[i],
                        gap=float(d_i[j]),
                        normal_sigma=0.0,
                        gap_sigma=0.0,
                    )
                    for j in sel
                ]
            )
        return frames


class MeshConvex:
    """Two convex MESHES (arbitrary vertex clouds) — the GJK/EPA position-derived-normal resolver.

    The highest-fidelity rung (DESIGN.md PART II.D / III.5, Mesh/SDF). The moving body and its
    support are each an arbitrary convex vertex cloud; per frame both are placed in the world by
    their poses and the contact is the minimum-distance query between the two hulls via
    :func:`contact.mesh_collision.gjk_distance` (GJK for separation, EPA for penetration):

        world A = mov_pos + R(mov_quat) @ verts_moving_local
        world B = sup_pos + R(sup_quat) @ verts_support_local
        gap, point, normal = gjk_distance(world A, world B)

    This is the convex-mesh generalization of :class:`SphereSphere`: the contact ``normal`` is
    **position/geometry-derived** — the unit ``origin -> closest`` direction on the Minkowski
    difference ``A ⊖ B``, which points from the support hull to the moving hull
    (``support -> moving``) — and is NEVER a body-fixed vector rotated by a quaternion. A spinning
    body therefore injects no phantom closing/opening velocity (the spin-artifact fix of
    DESIGN.md PART II.D), exactly as the line-of-centres normal does for ``SphereSphere``, here
    for arbitrary convex shapes. The reported ``gap`` is the hull separation (``> 0``) or the EPA
    penetration depth (``< 0``); the ``point`` is the midpoint of the closest witness pair.

    Single-point :data:`~contact.types.ContactFrame` per frame (one closest feature), mirroring
    ``SphereSphere``. Provenance (DESIGN.md III.5): ``normal_sigma = gap_sigma = 0.0`` (an exact
    convex query, up to vertex tessellation).
    """

    #: The GJK closest feature (and its witness point) MIGRATES across the body's surface as the
    #: pose evolves — it is not a fixed material point — so ``observe`` recovers the moving-point
    #: velocity from the body twist analytically (``v = v_com + omega × r``) rather than by
    #: differentiating the migrating witness, avoiding a spike when the closest feature switches
    #: (DESIGN.md PART II.D, the same reason ``BoxPlane`` is migrating).
    migrating = True

    def __init__(
        self,
        verts_moving_local: np.ndarray,
        verts_support_local: np.ndarray,
    ) -> None:
        self.verts_moving_local = np.asarray(verts_moving_local, dtype=float)    # (VA,3)
        self.verts_support_local = np.asarray(verts_support_local, dtype=float)  # (VB,3)

    def resolve(
        self, moving: PoseTrajectory, support: PoseTrajectory
    ) -> list[ContactFrame]:
        """One single-point :data:`~contact.types.ContactFrame` per recorded frame (length T)."""
        mov_pos = np.asarray(moving.position, dtype=float)            # (T,3) world (moving origin)
        mov_quat = np.asarray(moving.quat, dtype=float)               # (T,4)
        sup_pos = np.asarray(support.position, dtype=float)           # (T,3) world (support origin)
        sup_quat = np.asarray(support.quat, dtype=float)              # (T,4)

        # Place both convex clouds in the world each frame: origin + R(quat) @ vert_local.
        RA = quat_to_matrix(mov_quat)                                 # (T,3,3)
        RB = quat_to_matrix(sup_quat)                                 # (T,3,3)
        world_a = mov_pos[:, None, :] + np.einsum(
            "tik,vk->tvi", RA, self.verts_moving_local
        )                                                            # (T,VA,3)
        world_b = sup_pos[:, None, :] + np.einsum(
            "tik,vk->tvi", RB, self.verts_support_local
        )                                                            # (T,VB,3)

        frames: list[ContactFrame] = []
        for i in range(world_a.shape[0]):
            # GJK distance (EPA on penetration): geometry-derived point/normal/gap for this frame.
            gap, point, normal = gjk_distance(world_a[i], world_b[i])
            frames.append(
                [
                    ContactPoint(
                        point=point,
                        normal=normal,
                        gap=float(gap),
                        normal_sigma=0.0,
                        gap_sigma=0.0,
                    )
                ]
            )
        return frames
