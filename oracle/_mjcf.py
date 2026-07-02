"""Shared MuJoCo scene-construction helpers for the truth factory and the demo builders.

A dependency-light leaf (``mujoco`` only) so both :mod:`oracle.factory` and the
``contact.demos_*`` builders it imports at the end can share these without an import cycle.
It replaces the byte-identical per-file copies that the old "self-contained, no
cross-import" rule used to force into every builder module.
"""

from __future__ import annotations

import mujoco


def options() -> str:
    """The MuJoCo ``<option>`` block shared by every scenario / scene / demo.

    Gravity -9.81 z, a fine 0.0005 s timestep (keeps a near-discontinuous impact crisp
    rather than smeared across frames, THEORY.md s.6), implicitfast integration, and a
    pyramidal friction cone (matching the rest of the package).
    """
    return (
        '<option timestep="0.0005" gravity="0 0 -9.81" '
        'integrator="implicitfast" cone="pyramidal"/>'
    )


def obj_id(model: mujoco.MjModel, objtype: int, name: str) -> int:
    """Resolve a named MuJoCo object to its integer id (raises if absent)."""
    i = mujoco.mj_name2id(model, objtype, name)
    if i < 0:
        raise KeyError(f"no {objtype!r} named {name!r} in model")
    return i


def body_id(model: mujoco.MjModel, name: str) -> int:
    """Resolve a named body to its integer id (raises if absent)."""
    return obj_id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def free_dofadr(model: mujoco.MjModel, bid: int) -> int:
    """Index into ``data.qvel`` of the first DOF of body ``bid``'s (free)joint.

    For a freejoint body the six dofs are [vx, vy, vz, wx, wy, wz] (linear then angular,
    world-frame), so ``qvel[adr:adr+3]`` is the COM linear velocity and ``qvel[adr+3:adr+6]``
    the angular velocity.
    """
    return int(model.jnt_dofadr[model.body_jntadr[bid]])


def body_dofadr(model: mujoco.MjModel, name: str) -> int:
    """First DOF address of a named body's joint (for writing initial qvel)."""
    return free_dofadr(model, body_id(model, name))
