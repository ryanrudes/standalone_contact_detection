"""An analytic truth factory — the canonical story with no physics simulator (THEORY.md §9).

MuJoCo is the repo's ground-truth oracle, but the *smallest* end-to-end run shouldn't need a
simulator at all. This module synthesizes the canonical story analytically — a box free-falls,
impacts a floor, rests, and is lifted off — with exact closed-form truth labels, so

    raw = synthetic_drop_rest_liftoff()
    obs = observe(raw.moving, raw.support, raw.surface, raw.contact_point_local)
    result = ContactDetector().detect(obs)

exercises FREE → IMPACT → STATIC → liftoff → FREE through the full pipeline in milliseconds.
Exactly like the MuJoCo factory, only the (noised) moving-body pose is "observable"; the truth
labels are withheld for scoring. Unlike it, the labels here are *constructed*, not measured —
which makes this the one scenario whose truth is beyond doubt, at the price of idealized
dynamics (the touchdown is a perfect arrest; the rest phase sits at a constant modeled
clearance bias, which is what the detector's EM gap-bias calibration (§7) must recover).
"""

from __future__ import annotations

import numpy as np

from contact.types import (
    FREE,
    IMPACT,
    STATIC,
    GroundTruth,
    PoseTrajectory,
    RawScenario,
    SupportSurface,
)


def synthetic_drop_rest_liftoff(noise_m: float = 3e-4, seed: int = 0) -> RawScenario:
    """Build the analytic drop→rest→liftoff clip as a fully labeled :class:`RawScenario`.

    A box (half-height ``h``) over a floor at z=0. It free-falls from rest, its bottom face
    arrests at the floor (a velocity step — an impact), rests with a small modeled clearance
    bias, then is lifted off under constant acceleration. ``contact_point_local`` is the
    bottom-face centre; ``surface`` is the world floor plane.

    noise_m: std-dev (m) of the Gaussian mocap noise added to the moving body's position.
    seed:    RNG seed for that noise.
    """
    g, h = 9.81, 0.10
    z0, bias = 0.45, 0.004                 # drop height (m); modeled resting clearance (m)
    t_lift, a_lift = 2.0, 5.0              # liftoff time (s); upward acceleration (m/s²)
    hz = 100.0
    t = np.arange(0.0, 3.0, 1.0 / hz)
    t_impact = float(np.sqrt(2.0 * (z0 - h) / g))   # bottom face (z−h) reaches 0
    z = np.empty_like(t)
    in_contact = np.zeros(t.shape, dtype=bool)
    mode = [FREE] * t.shape[0]
    for i, ti in enumerate(t):
        if ti < t_impact:                  # free fall: centre z = z0 − ½ g t²
            z[i] = z0 - 0.5 * g * ti * ti
        elif ti < t_lift:                  # rest on the floor with the clearance bias
            z[i] = h + bias
            in_contact[i] = True
            mode[i] = STATIC
        else:                              # lift off: rise under a_lift until clear
            z[i] = h + bias + 0.5 * a_lift * (ti - t_lift) ** 2
            if z[i] - h < 0.02:            # still essentially touching for the first instants
                in_contact[i] = True
                mode[i] = STATIC
    # The touchdown frame is an impact (the normal velocity is arrested across it).
    k_impact = int(np.searchsorted(t, t_impact))
    if 0 <= k_impact < len(mode):
        mode[k_impact] = IMPACT

    rng = np.random.default_rng(seed)
    position = np.zeros((t.shape[0], 3))
    position[:, 2] = z
    position = position + rng.normal(0.0, noise_m, size=position.shape)   # emulate mocap noise
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (t.shape[0], 1))       # no rotation
    moving = PoseTrajectory(t=t, position=position, quat=quat)

    ident = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (t.shape[0], 1))
    support = PoseTrajectory(t=t, position=np.zeros((t.shape[0], 3)), quat=ident)  # static world floor
    surface = SupportSurface(point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]))
    contact_point_local = np.array([0.0, 0.0, -h])                        # bottom-face centre
    truth = GroundTruth(
        t=t,
        in_contact=in_contact,
        mode=mode,
        normal_force=np.where(in_contact, 9.81, 0.0),
        penetration=np.zeros(t.shape[0]),
    )
    return RawScenario(
        name="synthetic_drop_rest_liftoff",
        moving=moving,
        support=support,
        surface=surface,
        contact_point_local=contact_point_local,
        truth=truth,
        meta={
            "note": (
                "Analytic truth (no simulator): free fall to t_impact, rest at a "
                f"{bias * 1e3:.1f} mm clearance bias, liftoff at t={t_lift:.1f} s."
            ),
            "t_impact": t_impact,
            "t_lift": t_lift,
            "resting_bias": bias,
        },
    )
