"""Synchronized, real-time, side-by-side visualizations of the whole pipeline.

For a scenario (or multi-body scene) this renders an animation with the *actual*
MuJoCo-rendered world on the left and the live signals + detections scrolling on the
right, all locked to a single moving playhead:

  [  rendered 3D scene  ] | gap / clearance .......|.....
                          | speeds (vn, vt, |w|) ...|.....
                          | contact posterior + mode ribbon + events
                          | (normal force, if compliance known)

The scene is re-simulated deterministically (same seed/hz as ``generate``) purely to
capture RGB frames; the signals and detections come from the detector run on the
(noisy) observed poses. Frame i of the render therefore lines up with frame i of every
trace. Plays at real time (video seconds == scene seconds).

Entry points: ``animate_scenario(name, ...)`` and ``animate_scene(name, ...)``; the
``viz.py`` CLI wraps both. Output is an .mp4 (ffmpeg) or .gif (pillow) by extension.
"""

from __future__ import annotations

import numpy as np

from oracle import factory, registry
from contact.config import DetectorConfig
from contact.geometry import observe
from contact.graph import detect_scene
from contact.model import ContactDetector
from contact.types import (
    FREE,
    IMPACT,
    PIVOTING,
    ROLLING,
    SLIDING,
    STATIC,
)

# Colour per contact mode (matplotlib colour specs).
MODE_COLORS: dict[str, str] = {
    FREE: "0.65",
    STATIC: "tab:green",
    SLIDING: "tab:orange",
    ROLLING: "tab:blue",
    PIVOTING: "tab:purple",
    IMPACT: "tab:red",
}


# --------------------------------------------------------------------------------------
# Scene rendering: re-simulate deterministically and capture RGB frames.
# --------------------------------------------------------------------------------------

# A vivid, distinct colour per body (planes always get the checker floor material).
_PALETTE = [
    (0.90, 0.32, 0.26), (0.22, 0.52, 0.85), (0.36, 0.72, 0.42),
    (0.92, 0.70, 0.20), (0.60, 0.40, 0.80), (0.30, 0.74, 0.74),
]


def _camera(mujoco, lookat: np.ndarray, distance: float, azimuth: float, elevation: float):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def _capture_xml(builder):
    """Run a registered scenario/scene builder, capturing the XML it compiles (so we can re-skin it).

    The builders compile via ``mujoco.MjModel.from_xml_string``; we briefly wrap it to grab
    the XML. Returns (xml_or_None, spec). Pure capture — no model is kept."""
    import mujoco

    cap: dict = {}
    orig = mujoco.MjModel.from_xml_string

    def grab(xml, *a, **k):
        cap.setdefault("xml", xml)
        return orig(xml, *a, **k)

    mujoco.MjModel.from_xml_string = grab
    try:
        build = builder()
    finally:
        mujoco.MjModel.from_xml_string = orig
    return cap.get("xml"), build


def _studio_model(mujoco, xml, width, height):
    """Re-skin a scenario's XML into a studio-quality scene via MjSpec, then compile.

    Adds (all VISUAL-only, so physics — and the tests — are unchanged): a gradient skybox,
    a checkered reflective floor, a vivid per-body colour palette, a shadow-casting
    directional light, antialiasing (offsamples) and crisp shadows. Returns the compiled
    pretty model. Raises on any incompatibility (caller falls back to the plain model)."""
    spec = mujoco.MjSpec.from_string(xml)

    spec.visual.global_.offwidth = max(int(width), 640)
    spec.visual.global_.offheight = max(int(height), 480)
    spec.visual.quality.shadowsize = 4096
    spec.visual.quality.offsamples = 8
    spec.visual.headlight.ambient = [0.35, 0.35, 0.38]
    spec.visual.headlight.diffuse = [0.40, 0.40, 0.40]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]

    sky = spec.add_texture()
    sky.name = "vz_sky"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1 = [0.45, 0.60, 0.80]
    sky.rgb2 = [0.09, 0.11, 0.16]
    sky.width = 512
    sky.height = 512

    grid = spec.add_texture()
    grid.name = "vz_grid"
    grid.type = mujoco.mjtTexture.mjTEXTURE_2D
    grid.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    grid.rgb1 = [0.27, 0.29, 0.33]
    grid.rgb2 = [0.20, 0.22, 0.26]
    grid.width = 300
    grid.height = 300
    gmat = spec.add_material()
    gmat.name = "vz_gridmat"
    gmat.texrepeat = [4, 4]
    gmat.texuniform = True
    gmat.reflectance = 0.12
    gmat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "vz_grid"

    palette = []
    for i, (r, g, b) in enumerate(_PALETTE):
        mm = spec.add_material()
        mm.name = f"vz_m{i}"
        mm.rgba = [r, g, b, 1.0]
        mm.specular = 0.3
        mm.shininess = 0.35
        mm.reflectance = 0.04
        palette.append(mm.name)

    # Focus materials for the "which two bodies?" highlight: when a contact is being shown,
    # its MOVING body is painted vz_focusA (teal), its SUPPORT body vz_focusB (orange), and
    # every OTHER body vz_faded (dark + translucent) so the tracked pair is unmistakable.
    fa = spec.add_material()
    fa.name = "vz_focusA"; fa.rgba = [0.10, 0.85, 0.85, 1.0]; fa.specular = 0.5; fa.shininess = 0.5
    fb = spec.add_material()
    fb.name = "vz_focusB"; fb.rgba = [1.00, 0.55, 0.10, 1.0]; fb.specular = 0.5; fb.shininess = 0.5
    fd = spec.add_material()
    fd.name = "vz_faded"; fd.rgba = [0.22, 0.22, 0.26, 0.22]; fd.specular = 0.0; fd.reflectance = 0.0

    # A bright accent for geoms named "*marker*" -- a contrasting spot so rotation is visible
    # (e.g. the spinning top's nub) even though a body's other geoms share one colour.
    acc = spec.add_material()
    acc.name = "vz_accent"
    acc.rgba = [0.97, 0.85, 0.12, 1.0]
    acc.specular = 0.5
    acc.shininess = 0.5

    plane = mujoco.mjtGeom.mjGEOM_PLANE
    color_map: dict[str, tuple] = {}   # body name -> its assigned RGB (for the legend)
    ci = 0
    for body in spec.bodies:
        is_world = body.name == "world"
        idx = 0 if is_world else ci % len(_PALETTE)
        if not is_world:
            color_map[body.name] = _PALETTE[idx]
            ci += 1
        for g in body.geoms:
            if g.type == plane:
                g.material = "vz_gridmat"
            elif "marker" in (g.name or ""):
                g.material = "vz_accent"
            else:
                g.material = palette[idx]

    light = spec.worldbody.add_light()
    light.pos = [1.5, 1.5, 4.0]
    light.dir = [-0.3, -0.3, -1.0]
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    light.castshadow = True
    light.diffuse = [0.75, 0.75, 0.72]
    light.specular = [0.25, 0.25, 0.25]

    return spec.compile(), color_map


def _scene_model(builder, width, height):
    """The studio-skinned model + {body: rgb} colour map, or the plain model as a fallback."""
    import mujoco

    xml, build = _capture_xml(builder)
    if xml is not None:
        try:
            model, color_map = _studio_model(mujoco, xml, width, height)
            return model, build, color_map
        except Exception:
            pass  # fall back to the plain model below
    build = builder()
    model = build.model
    model.vis.global_.offwidth = max(int(width), int(model.vis.global_.offwidth))
    model.vis.global_.offheight = max(int(height), int(model.vis.global_.offheight))
    model.vis.headlight.ambient[:] = [0.6, 0.6, 0.6]
    model.vis.headlight.diffuse[:] = [0.7, 0.7, 0.7]
    return model, build, {}


#: RGB of the focus materials (for legends/banners), matching vz_focusA / vz_focusB.
_FOCUS_A_RGB = (0.10, 0.85, 0.85)   # moving body  (teal)
_FOCUS_B_RGB = (1.00, 0.55, 0.10)   # support body (orange)


def _focus_setup(mujoco, model):
    """Precompute what _apply_focus needs: the base (cruise) material ids, the focus material
    ids, and per-geom (body name, is-plane). Returns None if the model has no focus materials
    (the plain-render fallback), so callers can no-op."""
    fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "vz_focusA")
    if fid < 0:
        return None
    base = model.geom_matid.copy()
    ids = {nm: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, nm)
           for nm in ("vz_focusA", "vz_focusB", "vz_faded")}
    bodyname = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g]))
                for g in range(model.ngeom)]
    is_plane = [int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
                for g in range(model.ngeom)]
    return base, ids, bodyname, is_plane


def _apply_focus(model, setup, a, b):
    """Paint moving body `a` teal, support body `b` orange, every OTHER body translucent-dark;
    planes and the world body keep their (grid/ground) material. b may be 'world'."""
    if setup is None or a is None:
        return
    base, ids, bodyname, is_plane = setup
    for g in range(model.ngeom):
        bn = bodyname[g]
        if is_plane[g] or bn == "world":
            model.geom_matid[g] = base[g]
        elif bn == a:
            model.geom_matid[g] = ids["vz_focusA"]
        elif bn == b:
            model.geom_matid[g] = ids["vz_focusB"]
        else:
            model.geom_matid[g] = ids["vz_faded"]


def _clear_focus(model, setup):
    """Restore the cruise (full-colour) materials."""
    if setup is not None:
        model.geom_matid[:] = setup[0]


def _quat_rotate(q, v):
    """Rotate vector v by scalar-first unit quaternion q."""
    w, x, y, z = [float(c) for c in q]
    v = np.asarray(v, float)
    u = np.array([x, y, z])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _event_camera_angle(normal, default_az):
    """Pick (azimuth, elevation) so the camera looks roughly PERPENDICULAR to the contact
    normal -- then the separation (gap opening/closing along the normal) is broadside to the
    view and the contact making/breaking is maximally legible.

    * ~vertical normal (floor/incline): a low side-on elevation, so a lift-off reads as a
      clear vertical gap; keep an oblique azimuth for depth.
    * ~horizontal normal (a wall): look along an azimuth perpendicular to the normal, so the
      horizontal gap opens left-right across the view.
    """
    n = np.asarray(normal, float)
    nn = float(np.linalg.norm(n))
    if nn < 1e-9:
        return default_az, -16.0
    n = n / nn
    if abs(n[2]) > 0.8:                       # vertical-ish normal -> side view, low elevation
        return default_az, -8.0
    az = np.degrees(np.arctan2(n[1], n[0])) + 90.0   # perpendicular to the normal's heading
    el = -6.0 - 22.0 * abs(n[2])              # tilt down a little if the normal has some +z
    return float(az), float(el)


def _lerp_angle(a0, a1, t):
    """Shortest-arc interpolation between two angles in degrees."""
    d = ((a1 - a0 + 180.0) % 360.0) - 180.0
    return a0 + d * t


def _step_cadence(model, build, hz):
    dt = float(model.opt.timestep)
    sub = max(1, int(round((1.0 / hz) / dt)))
    n_frames = int(round(build.duration * hz))
    return sub, n_frames


def _stepped_data(mujoco, model, build):
    """Build MjData and apply the SAME prelude as factory._simulate / _simulate_scene so
    the rendered motion matches the detected motion exactly: forward -> optional `settle`
    phase (stepped with `forcing`, clock reset) -> one-shot `init` -> one-shot `launch`.

    Missing `settle`/`launch` here is what made shoved scenes (dominoes, two_balls_collide,
    newtons_cradle, the skateboard's launch) render motionless — the shove never fired.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    forcing = build.forcing
    dt = float(model.opt.timestep)
    settle = float(getattr(build, "settle", 0.0))  # scenes only
    if settle > 0.0:
        for _ in range(int(round(settle / dt))):
            if forcing is not None:
                forcing(model, data)
            mujoco.mj_step(model, data)
        data.time = 0.0  # match _simulate_scene: recorded window starts at t=0 after settle
    init = getattr(build, "init", None)  # scenarios only
    if init is not None:
        init(model, data)
        mujoco.mj_forward(model, data)
    launch = getattr(build, "launch", None)  # scenes only
    if launch is not None:
        launch(model, data)
        mujoco.mj_forward(model, data)
    return data


def _frame_trajectory(mujoco, model, build, hz):
    """A pre-pass that steps the sim WITHOUT rendering to collect the world positions of
    every (non-world) body, so we can place a FIXED camera that frames the entire motion.

    A fixed camera (rather than one tracking the body) is what actually makes translation
    visible: a body sliding/rolling across a featureless floor looks frozen if the camera
    follows it, but obviously moves when the camera holds still and frames the whole path.
    """
    data = _stepped_data(mujoco, model, build)
    forcing = build.forcing
    sub, n_frames = _step_cadence(model, build, hz)
    pts = []
    for _ in range(n_frames):
        for _ in range(sub):
            if forcing is not None:
                forcing(model, data)
            mujoco.mj_step(model, data)
        pts.append(data.xpos[1:].copy())  # all bodies except the world (id 0)
    pts = np.asarray(pts).reshape(-1, 3)
    lo, hi = pts.min(0), pts.max(0)
    lookat = 0.5 * (lo + hi)
    extent = float(np.max(hi - lo))
    # Tight enough that the body is large and its travel fills a good fraction of the view,
    # loose enough that a large traverse stays in frame for the whole clip.
    distance = float(np.clip(extent * 1.25 + 0.8, 1.0, 8.0))
    return lookat, distance


def _render_run(builder, hz, width, height, distance, azimuth, elevation):
    """Studio-render a scenario/scene: re-skin its model, then step it exactly as
    ``factory._simulate`` does, capturing one RGB frame per recorded frame from a FIXED
    camera framed to the whole trajectory. ``distance`` overrides the auto-framed distance."""
    import mujoco

    model, build, color_map = _scene_model(builder, width, height)

    # Fixed camera framed to the entire motion.
    lookat, auto_distance = _frame_trajectory(mujoco, model, build, hz)
    cam = _camera(mujoco, lookat, distance or auto_distance, azimuth, elevation)

    data = _stepped_data(mujoco, model, build)
    forcing = build.forcing
    sub, n_frames = _step_cadence(model, build, hz)
    renderer = mujoco.Renderer(model, height, width)
    frames = []
    for _ in range(n_frames):
        for _ in range(sub):
            if forcing is not None:
                forcing(model, data)
            mujoco.mj_step(model, data)
        renderer.update_scene(data, cam)
        frames.append(renderer.render().copy())
    renderer.close()
    return np.asarray(frames, dtype=np.uint8), color_map


def render_scenario_frames(name, seed=0, hz=100.0, width=640, height=640,
                           distance=None, azimuth=120.0, elevation=-16.0):
    """(frames (N,H,W,3), body->rgb map) of a scenario, synced to ``generate(name, ...)``."""
    return _render_run(registry.SCENARIO_BUILDERS[name], hz, width, height, distance, azimuth, elevation)


def render_scene_frames(name, seed=0, hz=100.0, width=640, height=640,
                        distance=None, azimuth=120.0, elevation=-16.0):
    """(frames (N,H,W,3), body->rgb map) of a scene, synced to ``generate_scene(name, ...)``."""
    return _render_run(registry.SCENE_BUILDERS[name], hz, width, height, distance, azimuth, elevation)


# --------------------------------------------------------------------------------------
# Signal extraction (what scrolls on the right).
# --------------------------------------------------------------------------------------

def _feature_rows(obs, normal_force=None):
    """The raw per-frame feature channels the detector consumes (THEORY.md s.3), as
    (label, array) rows for the feature heatmap. Force is appended when known (s.7)."""
    rows = [
        ("gap", np.asarray(obs.gap, float)),
        ("|v$_n$|", np.abs(obs.v_normal)),
        ("|v$_t$|", np.linalg.norm(obs.v_tangent, axis=1)),
        ("|$\\omega_n$|", np.abs(obs.omega_normal)),
        ("|$\\omega_t$|", np.linalg.norm(obs.omega_tangent, axis=1)),
    ]
    if normal_force is not None:
        rows.append(("force", np.asarray(normal_force, float)))
    return rows


def _mode_runs(map_state):
    """List of (start, end_exclusive, mode) contiguous runs."""
    runs = []
    i = 0
    n = len(map_state)
    while i < n:
        j = i
        while j < n and map_state[j] == map_state[i]:
            j += 1
        runs.append((i, j, map_state[i]))
        i = j
    return runs


# --------------------------------------------------------------------------------------
# The animation.
# --------------------------------------------------------------------------------------

def _norm_rows(rows, n):
    """Stack (label, array) rows into a (R, n) matrix, each row min-max normalized to [0,1]
    so the heatmap shows each channel's RELATIVE activity over time."""
    labels, mat = [], []
    for lab, v in rows:
        v = np.asarray(v[:n], float)
        rng = float(np.ptp(v))
        mat.append((v - v.min()) / rng if rng > 1e-12 else np.zeros(n))
        labels.append(lab)
    return labels, np.asarray(mat)


def _heatmap_panel(ax, mat, labels, cmap: str, t0: float, t1: float, ylabel: str) -> None:
    """Draw one normalized (0..1) ``channels x time`` heatmap row -- the shared magma feature
    and viridis posterior panel used by the scenario and reel animations."""
    ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1,
              extent=[t0, t1, len(labels), 0], interpolation="nearest")
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)


def _save_anim(fig, update, n_frames: int, out_path, fps: int, bitrate: int = 3600):
    """Build the FuncAnimation, save by file extension (.gif -> Pillow, else FFmpeg), close.

    The single render/encode tail shared by every animation builder.
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000.0 / fps, blit=False)
    writer = (animation.PillowWriter(fps=fps) if str(out_path).lower().endswith(".gif")
              else animation.FFMpegWriter(fps=fps, bitrate=bitrate))
    anim.save(str(out_path), writer=writer, dpi=100)
    plt.close(fig)
    return out_path


def _build_animation(t, frames, feature_rows, state_post, state_labels, contact_post,
                     map_state, truth_in_contact, events, title, fps, out_path,
                     extra_strips=None, impulses=None, body_colors=None):
    """Assemble and save the synced animation: scene on the left; on the right a condensed
    FEATURE heatmap (every raw channel x time), the full STATE-POSTERIOR heatmap (the model's
    belief over all modes x time), and a contact-answer strip — all under one playhead."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation, gridspec

    n = min(len(t), len(frames))
    t = np.asarray(t[:n], float)
    frames = frames[:n]
    contact_post = np.asarray(contact_post[:n], float)
    map_state = list(map_state[:n])
    truth_in_contact = None if truth_in_contact is None else np.asarray(truth_in_contact[:n])
    feat_labels, feat_mat = _norm_rows(feature_rows, n)
    state_post = np.asarray(state_post)[:n]            # (n, S)
    t0, t1 = float(t[0]), float(t[-1])

    # Real-time playback: keep video seconds == scene seconds by striding to ~fps.
    scene_hz = 1.0 / np.median(np.diff(t)) if n > 1 else fps
    stride = max(1, int(round(scene_hz / fps)))
    idx = np.arange(0, n, stride)

    has_extra = extra_strips is not None
    nF, nS = len(feat_labels), len(state_labels)
    fig = plt.figure(figsize=(16, 8), facecolor="white")
    hr = [nF, nS, 2.2] + ([1.6] if has_extra else [])
    gs = gridspec.GridSpec(len(hr), 2, width_ratios=[1.0, 1.32], wspace=0.04, hspace=0.16,
                           height_ratios=hr)

    # --- left: the rendered scene (spans all rows) ---
    ax_scene = fig.add_subplot(gs[:, 0])
    ax_scene.axis("off")
    im = ax_scene.imshow(frames[0])
    mode_txt = ax_scene.set_title("", fontsize=15, fontweight="bold", pad=10)
    for spine in ax_scene.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(6)
    flash_txt = ax_scene.text(0.5, 0.035, "", transform=ax_scene.transAxes, ha="center",
                              va="bottom", fontsize=17, fontweight="bold", color="magenta")
    # Body colour legend so it is unmistakable which body is which (and thus which bodies a
    # detected contact is between -- the edge lanes name the pairs as "A<->B").
    if body_colors:
        for j, (bname, rgb) in enumerate(body_colors.items()):
            ax_scene.text(0.02, 0.97 - 0.05 * j, f"● {bname}", transform=ax_scene.transAxes,
                          ha="left", va="top", fontsize=9, fontweight="bold", color=rgb,
                          path_effects=None)

    axes = []
    # --- feature heatmap: every raw channel the detector consumes (THEORY.md s.3) ---
    ax_f = fig.add_subplot(gs[0, 1])
    _heatmap_panel(ax_f, feat_mat, feat_labels, "magma", t0, t1, "features")
    ax_f.set_title(title, fontsize=11, loc="left")
    axes.append(ax_f)

    # --- state-posterior heatmap: the model's full belief over modes (THEORY.md s.4/s.5) ---
    ax_s = fig.add_subplot(gs[1, 1], sharex=ax_f)
    _heatmap_panel(ax_s, state_post.T, state_labels, "viridis", t0, t1, "P(state)")
    axes.append(ax_s)

    # --- contact answer: P(contact) + true-contact + MAP mode ribbon ---
    ax_c = fig.add_subplot(gs[2, 1], sharex=ax_f)
    if truth_in_contact is not None:
        ax_c.fill_between(t, 0, 1, where=truth_in_contact, color="0.85", step="mid",
                          label="true")
    ax_c.plot(t, contact_post, color="black", lw=1.6, label="P(contact)")
    for s, e, m in _mode_runs(map_state):
        ax_c.axvspan(t[s], t[min(e, n - 1)], ymin=0.0, ymax=0.14, color=MODE_COLORS.get(m, "0.6"))
    ax_c.set_ylim(-0.02, 1.05)
    ax_c.set_ylabel("contact", fontsize=9)
    ax_c.legend(loc="center right", fontsize=7, framealpha=0.6)
    axes.append(ax_c)

    if has_extra:
        ax_e = fig.add_subplot(gs[3, 1], sharex=ax_f)
        extra_strips(ax_e, t, n)
        axes.append(ax_e)

    ax_f.set_xlim(t0, t1)
    axes[-1].set_xlabel("time (s)")
    for a in axes[:-1]:
        a.tick_params(labelbottom=False)

    for ev in events:
        for a in axes:
            a.axvline(ev.time, color=("lime" if ev.kind == "touchdown" else "red"),
                      lw=1.0, ls=":", alpha=0.85)
    impulses = impulses or []
    for imp in impulses:
        for a in axes:
            a.axvline(imp.time, color="cyan", lw=1.5, ls="-.", alpha=0.9, zorder=4)
    axes[-1].text(0.01, 0.93, "marks:  lime = touchdown   red = liftoff   cyan = impact (vₙ arrest)",
                  transform=axes[-1].transAxes, ha="left", va="top", fontsize=6.5, color="0.4")

    playheads = [a.axvline(t0, color="white", lw=1.4, alpha=0.95) for a in axes]
    playheads[-1].set_color("k")  # the contact strip has a white background

    def update(fi):
        i = int(idx[fi])
        im.set_array(frames[i])
        m = map_state[i]
        in_c = (m != FREE)
        col = MODE_COLORS.get(m, "0.6")
        mode_txt.set_text(f"t = {t[i]:5.2f}s    {'CONTACT' if in_c else 'FREE'} : {m}")
        mode_txt.set_color(col if in_c else "0.4")
        for spine in ax_scene.spines.values():
            spine.set_edgecolor(col if in_c else "0.8")
        near = [d for d in impulses if abs(t[i] - d.time) <= 0.06]
        flash_txt.set_text(f"IMPACT   e = {near[0].restitution:.2f}" if near else "")
        for ph in playheads:
            ph.set_xdata([t[i], t[i]])
        return [im, mode_txt, flash_txt, *playheads]

    return _save_anim(fig, update, len(idx), out_path, fps)


def animate_scenario(name, out_path, seed=0, hz=100.0, fps=50, config=None,
                     width=560, height=560, **cam):
    """Render a synced side-by-side animation of a single-pair scenario to ``out_path``."""
    config = config or DetectorConfig()
    raw = factory.generate(name, seed=seed, hz=hz)
    obs = observe(raw.moving, raw.support, raw.surface, raw.contact_point_local,
                  config.vel_smooth_time, geometry=getattr(raw, "geometry", None))
    result = ContactDetector(config).detect(obs)
    # Scenarios may pin a faster recording rate too (e.g. restitution_bounce 250 Hz) -- render
    # at that SAME rate or the scene and the detection panels run at different speeds / lengths.
    _b = registry.SCENARIO_BUILDERS[name]()
    rate = max(float(hz), float(_b.record_hz or hz))
    frames, color_map = render_scenario_frames(name, seed=seed, hz=rate, width=width, height=height, **cam)
    truth = raw.truth.in_contact
    return _build_animation(
        obs.t, frames, _feature_rows(obs, result.normal_force),
        result.state_posterior, result.states, result.contact_posterior, result.map_state,
        truth, result.events,
        title=f"contact detection — {name}", fps=fps, out_path=out_path,
        impulses=result.impulses, body_colors=color_map,
    )


def _config_for_scene(scene, config):
    """Return the base detector config for a scene.

    Emission velocity scaling is now done PER EDGE inside ``detect_scene`` (each edge fit to
    its own tangential motion), so a single scene-wide widening here is no longer needed -- and
    was in fact harmful: it let the fastest edge (e.g. a ball-ball surface slipping ~10 m/s)
    inflate the sliding scale for every slow edge, making their real sliding read FREE.
    """
    return config or DetectorConfig()


def animate_scene(name, out_path, seed=0, hz=100.0, fps=50, config=None,
                  width=620, height=620, **cam):
    """Render a synced animation of a multi-body scene with a per-edge active-set strip."""
    scene = factory.generate_scene(name, seed=seed, hz=hz)
    config = _config_for_scene(scene, config)
    result = detect_scene(scene, config)
    # Scenes may pin a faster recording rate (record_hz) to catch brief strikes, so
    # generate_scene records at max(hz, record_hz). Render at that SAME nominal rate (read from
    # the builder) or the scene frames and the detection panels run at different speeds and
    # drift apart. (Use the nominal rec_hz, not the median observed dt, so the physics-substep
    # rounding -- hence the exact frame count -- matches generate_scene's.)
    _build = registry.SCENE_BUILDERS[name]()
    rec_hz = max(float(hz), float(_build.record_hz or hz))
    frames, color_map = render_scene_frames(name, seed=seed, hz=rec_hz, width=width, height=height, **cam)

    # Use the first edge's signals for the scrolling traces; overlay all edges' posteriors.
    edges = result.edges
    e0 = edges[0]
    obs0 = result.per_edge[e0]
    # Re-derive obs for the first edge's signal traces.
    edge0 = next(e for e in scene.edges if e.edge_id == e0)
    sig_obs = observe(scene.bodies[edge0.moving_body],
                      _support_pose(scene, edge0), edge0.surface, edge0.contact_point_local,
                      config.vel_smooth_time)
    t = result.t

    # Human-readable "moving <-> support" labels so it is clear which bodies each edge joins.
    edge_label = {e.edge_id: f"{e.moving_body}↔{e.support_body}" for e in scene.edges}

    def extra_strips(ax, tt, n):
        # one coloured lane per edge: shaded where that edge is active (MAP), labelled A<->B.
        for k, eid in enumerate(edges):
            active = np.array([eid in s for s in result.map_active_set[:n]])
            ax.fill_between(tt, k, k + 0.8, where=active, step="mid",
                            color=f"C{k}", alpha=0.7)
            ax.text(tt[0], k + 0.4, f" {edge_label.get(eid, eid)}", fontsize=7.5,
                    va="center", ha="left", fontweight="bold")
        ax.set_ylim(-0.1, len(edges))
        ax.set_yticks([])
        ax.set_ylabel("active edges")

    return _build_animation(
        t, frames, _feature_rows(sig_obs, obs0.normal_force),
        obs0.state_posterior, obs0.states, obs0.contact_posterior, obs0.map_state,
        scene.truth[e0].in_contact, obs0.events,
        title=f"contact graph — {name}  (traces edge: {edge_label.get(e0, e0)})",
        fps=fps, out_path=out_path,
        extra_strips=extra_strips, impulses=obs0.impulses, body_colors=color_map,
    )


def _support_pose(scene, edge):
    """Resolve an edge's support PoseTrajectory, synthesizing world identity if needed."""
    from contact.types import PoseTrajectory

    if edge.support_body in scene.bodies:
        return scene.bodies[edge.support_body]
    any_body = next(iter(scene.bodies.values()))
    n = len(any_body.t)
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    return PoseTrajectory(any_body.t.copy(), np.zeros((n, 3)), quat)


# --------------------------------------------------------------------------------------
# Event-focused visualization: zoom to each detected contact event and play it in slow-mo,
# so you can SEE the contact happen and confirm the detection matches it. Two products:
# per-event clips (one short slow-mo zoom per event) and an event reel (overview + clips).
# --------------------------------------------------------------------------------------

def _render_window(builder, hz, width, height, lo, hi, lookat, distance, azimuth, elevation,
                   focus_a=None, focus_b=None):
    """Studio-render recorded frames [lo, hi) from a FIXED zoomed camera, with the tracked
    pair (focus_a, focus_b) highlighted and every other body faded throughout."""
    import mujoco

    model, build, _color_map = _scene_model(builder, width, height)
    setup = _focus_setup(mujoco, model)
    _apply_focus(model, setup, focus_a, focus_b)
    data = _stepped_data(mujoco, model, build)
    forcing = build.forcing
    sub, n_frames = _step_cadence(model, build, hz)
    cam = _camera(mujoco, np.asarray(lookat, float), distance, azimuth, elevation)
    renderer = mujoco.Renderer(model, height, width)
    frames = []
    hi = min(hi, n_frames)
    for k in range(hi):
        for _ in range(sub):
            if forcing is not None:
                forcing(model, data)
            mujoco.mj_step(model, data)
        if k >= lo:
            renderer.update_scene(data, cam)
            frames.append(renderer.render().copy())
    renderer.close()
    return np.asarray(frames, dtype=np.uint8)


def _slug(s):
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")


def _gather_events(name, seed, hz, config):
    """Return (builder, rate_hz, events). Each event: dict(time,index,label,pos(3,),post,map_state)."""
    if name in registry.SCENE_BUILDERS:
        scene = factory.generate_scene(name, seed=seed, hz=hz)
        config = _config_for_scene(scene, config)
        gr = detect_scene(scene, config)
        b = registry.SCENE_BUILDERS[name]()
        rate = max(float(hz), float(b.record_hz or hz))
        events = []
        for e in scene.edges:
            pe = gr.per_edge[e.edge_id]
            mov = scene.bodies[e.moving_body].position
            sq = scene.bodies[e.support_body].quat if e.support_body in scene.bodies else None

            def _norm(idx, sq=sq, e=e):
                q = sq[min(idx, len(sq) - 1)] if sq is not None else np.array([1.0, 0, 0, 0])
                return _quat_rotate(q, e.surface.normal)

            for ev in pe.events:
                events.append(dict(time=ev.time, index=ev.index, label=f"{e.edge_id} {ev.kind}",
                                   pos=mov[min(ev.index, len(mov) - 1)], post=pe.contact_posterior,
                                   map_state=pe.map_state, a=e.moving_body, b=e.support_body,
                                   normal=_norm(ev.index)))
            for im in pe.impulses:
                events.append(dict(time=im.time, index=im.index,
                                   label=f"{e.edge_id} impact e={im.restitution:.2f}",
                                   pos=mov[min(im.index, len(mov) - 1)], post=pe.contact_posterior,
                                   map_state=pe.map_state, a=e.moving_body, b=e.support_body,
                                   normal=_norm(im.index)))
        return registry.SCENE_BUILDERS[name], rate, events

    raw = factory.generate(name, seed=seed, hz=hz)
    cfg = config or DetectorConfig()
    obs = observe(raw.moving, raw.support, raw.surface, raw.contact_point_local, cfg.vel_smooth_time,
                  geometry=getattr(raw, "geometry", None))
    res = ContactDetector(cfg).detect(obs)
    bd = registry.SCENARIO_BUILDERS[name]()
    mov = raw.moving.position
    a, b, nrm = bd.moving_body, bd.support_body, np.asarray(raw.surface.normal, float)
    events = []
    for ev in res.events:
        events.append(dict(time=ev.time, index=ev.index, label=ev.kind,
                           pos=mov[min(ev.index, len(mov) - 1)], post=res.contact_posterior,
                           map_state=res.map_state, a=a, b=b, normal=nrm))
    for im in res.impulses:
        events.append(dict(time=im.time, index=im.index, label=f"impact e={im.restitution:.2f}",
                           pos=mov[min(im.index, len(mov) - 1)], post=res.contact_posterior,
                           map_state=res.map_state, a=a, b=b, normal=nrm))
    return registry.SCENARIO_BUILDERS[name], hz, sorted(events, key=lambda d: d["time"])


def _clip_animation(frames, lo, rate, ev, out_path, fps, slowmo):
    """A zoomed slow-mo clip of one event: scene (left) + contact posterior/mode (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation, gridspec

    n = len(frames)
    tw = (lo + np.arange(n)) / rate
    post = np.asarray(ev["post"])
    ms = list(ev["map_state"])
    fig = plt.figure(figsize=(11, 5), facecolor="white")
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.0], wspace=0.16)
    ax_s = fig.add_subplot(gs[0, 0]); ax_s.axis("off")
    im = ax_s.imshow(frames[0])
    ax_s.set_title(f"{ev['label']}   ({slowmo:.0f}x slow-mo)", fontsize=13, fontweight="bold",
                   color="magenta")

    ax_c = fig.add_subplot(gs[0, 1])
    seg = slice(max(0, lo), min(len(post), lo + n))
    ts = np.arange(seg.start, seg.stop) / rate
    ax_c.plot(ts, post[seg], color="black", lw=1.8, label="P(contact)")
    for s, e, m in _mode_runs(ms[seg]):
        ax_c.axvspan(ts[s], ts[min(e, len(ts) - 1)], ymin=0, ymax=0.14, color=MODE_COLORS.get(m, "0.6"))
    ax_c.axvline(ev["time"], color="magenta", lw=1.6, ls="-.", label=ev["label"].split()[-1])
    ax_c.set_ylim(-0.02, 1.05); ax_c.set_xlabel("time (s)"); ax_c.set_ylabel("contact")
    ax_c.set_title("detection at the event", fontsize=11, loc="left")
    ax_c.legend(loc="center right", fontsize=8, framealpha=0.6)
    head = ax_c.axvline(tw[0], color="0.2", lw=1.3)

    def update(i):
        im.set_array(frames[i])
        head.set_xdata([tw[i], tw[i]])
        return [im, head]

    _save_anim(fig, update, n, out_path, fps, bitrate=3200)


def animate_event_clips(name, outdir="media", seed=0, hz=100.0, config=None,
                        width=460, height=460, fps=16, pad_before=0.16, pad_after=0.30,
                        zoom=0.55):
    """Render one zoomed slow-mo clip per detected contact event. Returns the list of paths."""
    import os
    builder, rate, events = _gather_events(name, seed, hz, config)
    paths = []
    for ev in events:
        lo = max(0, int(ev["index"] - pad_before * rate))
        hi = int(ev["index"] + pad_after * rate)
        az, el = _event_camera_angle(ev.get("normal"), 118.0)   # angle to show the separation
        frames = _render_window(builder, rate, width, height, lo, hi, ev["pos"], zoom,
                                azimuth=az, elevation=el,
                                focus_a=ev.get("a"), focus_b=ev.get("b"))
        if len(frames) < 2:
            continue
        out = os.path.join(outdir, f"{name}__{_slug(ev['label'])}.mp4")
        _clip_animation(frames, lo, rate, ev, out, fps, slowmo=rate / fps)
        paths.append(out)
    return paths


def _build_reel_animation(frames, sched, panel, color_map, title, out_path, fps, persistent=False):
    """Scene (left, time-warped + zooming) + the heatmap timeline (right) with a playhead that
    tracks the scheduled sim frame, and a live/SLOW-MO banner. ``persistent`` keeps the
    tracked-pair banner/hint shown the whole time (a per-pair video)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation, gridspec

    t = np.asarray(panel["t"], float)
    t0, t1 = float(t[0]), float(t[-1])
    feat_labels, feat_mat = _norm_rows(panel["feat"], len(t))
    state_post = np.asarray(panel["post"])
    state_labels = panel["states"]
    cpost = np.asarray(panel["cpost"], float)
    mst = list(panel["mst"])
    nF, nS = len(feat_labels), len(state_labels)

    fig = plt.figure(figsize=(16, 8), facecolor="white")
    gs = gridspec.GridSpec(3, 2, width_ratios=[1.0, 1.32], wspace=0.04, hspace=0.16,
                           height_ratios=[nF, nS, 2.2])
    ax_scene = fig.add_subplot(gs[:, 0]); ax_scene.axis("off")
    im = ax_scene.imshow(frames[0])
    banner = ax_scene.set_title("", fontsize=14, fontweight="bold")
    for sp in ax_scene.spines.values():
        sp.set_visible(True); sp.set_linewidth(6)
    if color_map:
        for j, (bn, rgb) in enumerate(color_map.items()):
            ax_scene.text(0.02, 0.97 - 0.05 * j, f"● {bn}", transform=ax_scene.transAxes,
                          ha="left", va="top", fontsize=9, fontweight="bold", color=rgb)

    ax_f = fig.add_subplot(gs[0, 1])
    _heatmap_panel(ax_f, feat_mat, feat_labels, "magma", t0, t1, "features")
    ax_f.set_title(title, fontsize=11, loc="left")
    ax_s = fig.add_subplot(gs[1, 1], sharex=ax_f)
    _heatmap_panel(ax_s, state_post.T, state_labels, "viridis", t0, t1, "P(state)")
    ax_c = fig.add_subplot(gs[2, 1], sharex=ax_f)
    ax_c.plot(t, cpost, color="black", lw=1.6, label="P(contact)")
    for s, e, m in _mode_runs(mst):
        ax_c.axvspan(t[s], t[min(e, len(t) - 1)], ymin=0, ymax=0.14, color=MODE_COLORS.get(m, "0.6"))
    ax_c.set_ylim(-0.02, 1.05); ax_c.set_ylabel("contact", fontsize=9); ax_c.set_xlabel("time (s)")
    ax_c.set_xlim(t0, t1)
    axes = [ax_f, ax_s, ax_c]
    for a in axes[:-1]:
        a.tick_params(labelbottom=False)
    for imp in panel["imp"]:
        for a in axes:
            a.axvline(imp.time, color="cyan", lw=1.4, ls="-.", alpha=0.9, zorder=4)
    if panel["imp"]:
        axes[-1].text(0.01, 0.93, "cyan = impact event (normal-velocity arrest)",
                      transform=axes[-1].transAxes, ha="left", va="top", fontsize=7, color="0.4")
    playheads = [a.axvline(t0, color="white", lw=1.4) for a in axes]
    playheads[-1].set_color("k")
    # focus hint: shown only while zoomed -- which colour is the moving vs support body.
    hint = ax_scene.text(0.5, 0.035, "", transform=ax_scene.transAxes, ha="center", va="bottom",
                         fontsize=11, fontweight="bold")

    def update(k):
        entry = sched[k]
        idx, wt, lab = entry[0], entry[3], entry[4]
        a_body, b_body = entry[7], entry[8]
        idx = min(idx, len(t) - 1)
        im.set_array(frames[k])
        cur = mst[idx]
        if wt > 0.5:
            banner.set_text(f"◉ CONTACT EVENT (slow-mo) — {lab}")
            banner.set_color("magenta")
            for sp in ax_scene.spines.values():
                sp.set_edgecolor("magenta")
        elif persistent and a_body is not None:
            banner.set_text(f"▶ tracking {a_body}↔{b_body}   t = {t[idx]:5.2f}s   {cur}")
            banner.set_color(MODE_COLORS.get(cur, "0.5"))
            for sp in ax_scene.spines.values():
                sp.set_edgecolor("0.6")
        else:
            banner.set_text(f"▶ t = {t[idx]:5.2f}s   {cur}")
            banner.set_color(MODE_COLORS.get(cur, "0.4") if cur != FREE else "0.4")
            for sp in ax_scene.spines.values():
                sp.set_edgecolor("0.8")
        # the focus hint is persistent for a per-pair video, else shown only during a zoom
        if (persistent or wt > 0.5) and a_body is not None:
            hint.set_text(f"{a_body} (teal)  ↔  {b_body} (orange)   — others faded")
            hint.set_color("0.92")
        else:
            hint.set_text("")
        for ph in playheads:
            ph.set_xdata([t[idx], t[idx]])
        return [im, banner, hint, *playheads]

    return _save_anim(fig, update, len(sched), out_path, fps)


# --------------------------------------------------------------------------------------
# Per-PAIR contact videos: each video follows ONE fixed body pair (its two bodies coloured,
# every other body faded the whole time), cruising at real time and smoothly slowing +
# reorienting to the clearest angle at each of THAT pair's contact events. This is the clear,
# non-confusing structure: one recording == one contact relationship, one steady camera.
# --------------------------------------------------------------------------------------

def _smooth(x, k):
    """Moving-average smooth a (T,3) trajectory (so the camera tracks the pair without jitter)."""
    x = np.asarray(x, float)
    if k <= 1 or len(x) < 3:
        return x
    k = min(k, len(x) | 1)
    ker = np.ones(k) / k
    return np.stack([np.convolve(x[:, j], ker, mode="same") for j in range(x.shape[1])], axis=1)


def _pair_spec(name, seed, hz, config, use_force=False):
    """Return (builder, rate, pairs). Each pair: a, b (body names), snorm (local contact normal),
    squat (support quat trajectory or None), apos/bpos (world positions), events [(idx,label)],
    panel arrays, and a title.

    ``use_force`` feeds the per-edge FORCE channel (DESIGN.md PART II.A) using the simulator's
    true normal force as a stand-in for a real force sensor -- so force-mediated contacts that
    are invisible to kinematics (the Newton's-cradle clacks: ~0 relative velocity but a sharp
    force pulse, THEORY.md s.6-s.8) light up as IMPACT in the detection panels. A real sensor
    would supply the same stream; the truth force is used here only as the demonstration source.
    """
    import dataclasses

    tag = "   [+force sensor]" if use_force else ""
    pairs = []
    if name in registry.SCENE_BUILDERS:
        scene = factory.generate_scene(name, seed=seed, hz=hz)
        config = _config_for_scene(scene, config)
        # Feed each edge's force channel from its truth normal force (a stand-in sensor) when
        # requested; otherwise the kinematics-only detection (edge_forces=None) -- unchanged.
        edge_forces = (
            {e.edge_id: np.asarray(scene.truth[e.edge_id].normal_force, float) for e in scene.edges}
            if use_force else None
        )
        gr = detect_scene(scene, config, edge_forces=edge_forces)
        b = registry.SCENE_BUILDERS[name]()
        rate = max(float(hz), float(b.record_hz or hz))
        for e in scene.edges:
            pe = gr.per_edge[e.edge_id]
            sig = observe(scene.bodies[e.moving_body], _support_pose(scene, e), e.surface,
                          e.contact_point_local, config.vel_smooth_time)
            evs = [(ev.index, ev.kind) for ev in pe.events]
            evs += [(im.index, f"impact e={im.restitution:.2f}") for im in pe.impulses]
            pairs.append(dict(
                a=e.moving_body, b=e.support_body, snorm=np.asarray(e.surface.normal, float),
                squat=scene.bodies[e.support_body].quat if e.support_body in scene.bodies else None,
                apos=scene.bodies[e.moving_body].position,
                bpos=scene.bodies[e.support_body].position if e.support_body in scene.bodies else None,
                events=evs, title=f"{name}:  {e.moving_body} ↔ {e.support_body}{tag}",
                panel=dict(t=pe.t, feat=_feature_rows(sig, pe.normal_force), post=pe.state_posterior,
                           states=pe.states, cpost=pe.contact_posterior, mst=pe.map_state, imp=pe.impulses)))
        return registry.SCENE_BUILDERS[name], rate, pairs

    raw = factory.generate(name, seed=seed, hz=hz)
    cfg = config or DetectorConfig()
    obs = observe(raw.moving, raw.support, raw.surface, raw.contact_point_local, cfg.vel_smooth_time,
                  geometry=getattr(raw, "geometry", None))
    if use_force:
        obs = dataclasses.replace(obs, normal_force=np.asarray(raw.truth.normal_force, float)[:len(obs.t)])
    res = ContactDetector(cfg).detect(obs)
    b = registry.SCENARIO_BUILDERS[name]()
    rate = max(float(hz), float(b.record_hz or hz))
    evs = [(ev.index, ev.kind) for ev in res.events]
    evs += [(im.index, f"impact e={im.restitution:.2f}") for im in res.impulses]
    pairs.append(dict(
        a=b["moving_body"], b=b["support_body"], snorm=np.asarray(raw.surface.normal, float),
        squat=None, apos=raw.moving.position, bpos=None, events=evs,
        title=f"{name}:  {b['moving_body']} ↔ {b['support_body']}{tag}",
        panel=dict(t=obs.t, feat=_feature_rows(obs, res.normal_force), post=res.state_posterior,
                   states=res.states, cpost=res.contact_posterior, mst=res.map_state, imp=res.impulses)))
    return registry.SCENARIO_BUILDERS[name], rate, pairs


def animate_pair(name, pair, builder, rate, out_path, fps=30, width=600, height=600, zoom=0.5):
    """One per-pair contact video: persistent focus on (a, b), a steady camera that gently
    tracks the pair and smoothly zooms + reorients to the clearest angle at each of the pair's
    contact events, real-time cruise with ~13x slow-mo at events."""
    import mujoco

    panel = pair["panel"]
    t = np.asarray(panel["t"], float)
    n = len(t)
    a, b = pair["a"], pair["b"]
    apos = np.asarray(pair["apos"], float)[:n]
    center = apos.copy() if pair["bpos"] is None else 0.5 * (apos + np.asarray(pair["bpos"], float)[:n])
    center = _smooth(center, max(3, int(round(0.06 * rate))))

    if pair["squat"] is not None:
        sq = np.asarray(pair["squat"], float)
        nrm = np.array([_quat_rotate(sq[min(i, len(sq) - 1)], pair["snorm"]) for i in range(n)])
    else:
        nrm = np.tile(np.asarray(pair["snorm"], float), (n, 1))
    base_az, base_el = _event_camera_angle(np.median(nrm, axis=0), 120.0)
    ext = float(np.max(center.max(0) - center.min(0)))
    base_dist = float(np.clip(ext * 1.3 + 0.7, 0.9, 7.0))

    # event weight (trapezoid w/ generous ramp for smoothness) + per-event clearest angle
    w = np.zeros(n)
    eaz = [base_az] * n
    eel = [base_el] * n
    lab = [""] * n
    win = max(2, int(round(0.12 * rate)))
    ramp = max(3, int(round(0.22 * rate)))
    for idx, lname in pair["events"]:
        c = int(idx)
        lo, hi = max(0, c - win), min(n - 1, c + win)
        az, el = _event_camera_angle(nrm[min(c, n - 1)], base_az)
        for i in range(max(0, lo - ramp), min(n, hi + ramp + 1)):
            wi = 1.0 if lo <= i <= hi else max(0.0, 1.0 - (lo - i) / ramp) if i < lo \
                else max(0.0, 1.0 - (i - hi) / ramp)
            if wi > w[i]:
                w[i] = wi; eaz[i] = az; eel[i] = el; lab[i] = lname

    stride = max(1, int(round(rate / fps)))          # real-time cruise
    hold = max(1, int(round(13.0 * fps / rate)))      # ~13x slow-mo at events (frame holds)
    sched = []
    i = 0
    while i < n:
        wi = float(w[i])
        dist = base_dist * (1.0 - wi) + base_dist * zoom * wi
        az = _lerp_angle(base_az, eaz[i], wi)
        el = base_el * (1.0 - wi) + eel[i] * wi
        reps = 1 + int(round((hold - 1) * wi)) if wi > 0.25 else 1
        for _ in range(reps):
            sched.append((i, center[i], dist, wi, lab[i], float(az), float(el), a, b))
        i += 1 if wi > 0.25 else stride

    model, build, color_map = _scene_model(builder, width, height)
    setup = _focus_setup(mujoco, model)
    _apply_focus(model, setup, a, b)                  # persistent: the pair stays lit all video
    data = _stepped_data(mujoco, model, build)
    forcing = build.forcing
    sub, nf = _step_cadence(model, build, rate)
    cam_for = {}
    for (idx, la, dist, wi, lname, az, el, _a, _b) in sched:
        cam_for.setdefault(idx, (la, dist, az, el))
    renderer = mujoco.Renderer(model, height, width)
    rendered = {}
    for f in range(nf):
        for _ in range(sub):
            if forcing is not None:
                forcing(model, data)
            mujoco.mj_step(model, data)
        if f in cam_for:
            la, dist, az, el = cam_for[f]
            renderer.update_scene(data, _camera(mujoco, la, dist, az, el))
            rendered[f] = renderer.render().copy()
    renderer.close()
    frames = np.asarray([rendered[idx] for (idx, *_rest) in sched], dtype=np.uint8)
    return _build_reel_animation(frames, sched, panel, color_map, pair["title"], out_path, fps,
                                 persistent=True)


def animate_pairs(name, outdir="media/pairs", seed=0, hz=100.0, config=None, fps=30,
                  width=600, height=600, use_force=False):
    """Render one per-pair contact video for every contact pair in a demo. Returns the paths.

    ``use_force`` feeds the FORCE channel (truth force as a stand-in sensor) into the per-edge
    detection so force-mediated contacts kinematics cannot see -- the Newton's-cradle clacks --
    light up as IMPACT; outputs go to ``outdir`` with a ``__force`` suffix so they sit beside
    the kinematic versions.
    """
    import os
    builder, rate, pairs = _pair_spec(name, seed, hz, config, use_force=use_force)
    os.makedirs(outdir, exist_ok=True)
    suffix = "__force" if use_force else ""
    paths = []
    for pr in pairs:
        out = os.path.join(outdir, f"{name}__{_slug(pr['a'])}__{_slug(pr['b'])}{suffix}.mp4")
        animate_pair(name, pr, builder, rate, out, fps=fps, width=width, height=height)
        paths.append(out)
    return paths
