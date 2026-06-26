"""Scoring, text reporting, and plotting for detector output.

This is the validation/presentation layer of §9 of THEORY.md: we have run the
detector on the *observable* channel and now score its inferred posterior against
the *withheld* ground truth from the simulator, and render both for a human.

Nothing here feeds back into inference — it consumes a finished
`DetectionResult` plus the `GroundTruth` oracle and turns them into numbers,
text, and pictures. As such it imports only the shared data contracts
(`contact.types`); matplotlib is imported lazily inside `plot_result` so the rest
of the package stays usable in headless / plotting-free environments.

Conventions
-----------
* All internal distances are SI (metres); we convert to millimetres only at the
  presentation boundary (resting bias in the text report, gap on the plot axis).
* "Contact" everywhere means ``state != FREE`` — the active branch of the
  Signorini complementarity (THEORY.md §2). Both ``DetectionResult.in_contact``
  and ``GroundTruth.in_contact`` already encode exactly this, and we cross-check
  against the per-frame mode labels via the ``FREE`` sentinel.
"""

from __future__ import annotations

import numpy as np

from .types import (
    FREE,
    DetectionResult,
    GraphDetectionResult,
    GroundTruth,
    InverseDynamicsResult,
    MultiBodyScene,
)

# --------------------------------------------------------------------------------------
# Scoring (THEORY.md §9: score the inferred posterior against the withheld truth)
# --------------------------------------------------------------------------------------


def score(result: DetectionResult, truth: GroundTruth) -> dict:
    """Score a detection against ground truth, frame-for-frame.

    THEORY.md §9: the detector saw only the observable channel; here we compare
    its inferred contact decision and mode against the simulator's withheld labels
    on the *same* (frame-aligned, equal-length) time grid.

    Parameters
    ----------
    result : DetectionResult
        Detector output. ``result.in_contact`` (T,) and ``result.map_state``
        (length T) are used.
    truth : GroundTruth
        Oracle labels. ``truth.in_contact`` (T,) and ``truth.mode`` (length T).

    Returns
    -------
    dict
        Flat mapping of plain floats / ints:

        ``contact_iou``        Jaccard overlap of the two ``in_contact`` masks.
        ``contact_f1``         F1 of predicted contact vs. true contact (per frame).
        ``mode_accuracy``      Fraction of *truly-in-contact* frames whose MAP mode
                               equals the true mode. (Defined on true-contact frames
                               only, per the spec; ``nan`` if there are none.)
        ``contact_frames_true``  Number of true in-contact frames (int).
        ``contact_frames_pred``  Number of predicted in-contact frames (int).
    """
    pred = np.asarray(result.in_contact, dtype=bool)
    true = np.asarray(truth.in_contact, dtype=bool)

    if pred.shape != true.shape:
        raise ValueError(
            f"score(): frame-aligned masks must match in length; "
            f"got pred {pred.shape} vs truth {true.shape}"
        )

    # --- contact-existence agreement: IoU and F1 over the boolean masks ---
    intersection = int(np.count_nonzero(pred & true))
    union = int(np.count_nonzero(pred | true))
    # IoU is undefined when neither predicts nor truth has any contact; by
    # convention a perfect (empty == empty) agreement scores 1.0.
    contact_iou = 1.0 if union == 0 else intersection / union

    tp = intersection
    fp = int(np.count_nonzero(pred & ~true))
    fn = int(np.count_nonzero(~pred & true))
    denom = 2 * tp + fp + fn
    # F1 = 2TP / (2TP + FP + FN); empty-vs-empty is again a perfect 1.0.
    contact_f1 = 1.0 if denom == 0 else (2.0 * tp) / denom

    # --- mode agreement, restricted to truly-in-contact frames (spec) ---
    map_state = list(result.map_state)
    true_mode = list(truth.mode)
    n_true_contact = int(np.count_nonzero(true))
    if n_true_contact == 0:
        mode_accuracy = float("nan")
    else:
        idx = np.flatnonzero(true)
        matches = sum(1 for i in idx if map_state[i] == true_mode[i])
        mode_accuracy = matches / n_true_contact

    return {
        "contact_iou": float(contact_iou),
        "contact_f1": float(contact_f1),
        "mode_accuracy": float(mode_accuracy),
        "contact_frames_true": n_true_contact,
        "contact_frames_pred": int(np.count_nonzero(pred)),
    }


# --------------------------------------------------------------------------------------
# Text report helpers
# --------------------------------------------------------------------------------------


def _true_contact_intervals(truth: GroundTruth) -> list[tuple[float, float]]:
    """Contiguous (t_start, t_end) runs where ``truth.in_contact`` is True.

    The ground-truth analogue of ``DetectionResult.intervals`` — the contact
    *segments* of the active set (THEORY.md §2), recovered from the per-frame mask.
    """
    mask = np.asarray(truth.in_contact, dtype=bool)
    t = np.asarray(truth.t, dtype=float)
    runs: list[tuple[float, float]] = []
    if mask.size == 0:
        return runs
    # Edges where the mask flips; pad so leading/trailing runs are captured.
    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diffs == 1)
    ends = np.flatnonzero(diffs == -1) - 1  # inclusive last in-contact index
    for s, e in zip(starts, ends):
        runs.append((float(t[s]), float(t[e])))
    return runs


def _ascii_timeline(mask: np.ndarray, width: int = 72) -> str:
    """A compact one-line ASCII strip ('#' contact, '.' free) of a boolean mask.

    Down-samples T frames into ``width`` cells; a cell is '#' iff any frame mapped
    into it is in contact (so brief contacts stay visible). Purely for the eye —
    the aligned rows of predicted vs. truth make make/break disagreements (§6) and
    timing offsets pop out at a glance.
    """
    mask = np.asarray(mask, dtype=bool)
    n = mask.size
    if n == 0:
        return ""
    w = min(width, n)
    # Frame index -> cell index, then OR the frames within each cell.
    cell_of = (np.arange(n) * w) // n
    cells = np.zeros(w, dtype=bool)
    np.logical_or.at(cells, cell_of, mask)
    return "".join("#" if c else "." for c in cells)


def print_report(name: str, result: DetectionResult, truth: GroundTruth) -> None:
    """Print a human-readable terminal summary of one scenario's detection.

    Renders the §8 outputs the detector produced — detected contact intervals and
    their modes, make/break events with times (§6), the EM-recovered resting bias
    (§7) — alongside the ground-truth contact spans and the frame-level scores from
    :func:`score`, and closes with two aligned ASCII timelines (predicted over
    truth) so coverage and timing can be eyeballed.

    Parameters
    ----------
    name : str
        Scenario identifier, printed as the header.
    result : DetectionResult
        Detector output to summarise.
    truth : GroundTruth
        Oracle labels to compare against.
    """
    line = "=" * 72
    print(line)
    print(f"SCENARIO: {name}")
    print(line)

    # --- detected intervals (start, end, mode) ---
    print("Detected contact intervals (start, end, mode):")
    if result.intervals:
        for iv in result.intervals:
            print(f"  [{iv.t_start:8.3f}, {iv.t_end:8.3f}] s  {iv.mode}")
    else:
        print("  (none)")

    # --- true contact interval(s) ---
    true_runs = _true_contact_intervals(truth)
    print("True contact intervals (start, end):")
    if true_runs:
        for s, e in true_runs:
            print(f"  [{s:8.3f}, {e:8.3f}] s")
    else:
        print("  (none)")

    # --- scores ---
    scores = score(result, truth)
    print("Scores:")
    print(f"  contact_iou         : {scores['contact_iou']:.3f}")
    print(f"  contact_f1          : {scores['contact_f1']:.3f}")
    print(f"  mode_accuracy       : {scores['mode_accuracy']:.3f}")
    print(f"  contact_frames_true : {scores['contact_frames_true']}")
    print(f"  contact_frames_pred : {scores['contact_frames_pred']}")

    # --- events with times (THEORY.md §6) ---
    print("Events (kind, time, index):")
    if result.events:
        for ev in result.events:
            print(f"  {ev.kind:10s} t={ev.time:8.3f} s  (frame {ev.index})")
    else:
        print("  (none)")

    # --- impact atoms with their characterization (THEORY.md §6) ---
    # The impulses are the force-as-measure atoms: each is one velocity-step event with
    # its closing speed, measured restitution e (NaN if the bounce was not resolved),
    # and momentum-jump impulse (NaN when the moving body's mass is unknown -- s.7, the
    # atom magnitude is unobservable from kinematics alone). They are complementary to
    # the make/break events above: events mark *that* a transition happened; impulses
    # mark *how hard* it hit.
    print(f"Impact atoms ({len(result.impulses)}):")
    if result.impulses:
        for imp in result.impulses:
            e_str = "  e=  n/a" if np.isnan(imp.restitution) else f"  e={imp.restitution:5.3f}"
            j_str = (
                "  J=    n/a"
                if np.isnan(imp.normal_impulse)
                else f"  J={imp.normal_impulse:8.3f} N*s"
            )
            print(
                f"  t={imp.time:8.3f} s  (frame {imp.index})  "
                f"closing={imp.closing_speed:5.3f} m/s{e_str}{j_str}"
            )
    else:
        print("  (none)")

    # --- recovered resting bias, reported in mm (THEORY.md §7) ---
    print(f"Recovered resting bias: {result.resting_bias * 1e3:+.3f} mm")

    # --- normal force + stick/slip summary, only when the material was known (§7) ---
    if result.normal_force is not None:
        nf = np.asarray(result.normal_force, dtype=float)
        finite = nf[np.isfinite(nf)]
        if finite.size:
            print(
                f"Estimated normal force: peak {float(np.nanmax(finite)):.2f} N "
                f"(loaded-frame mean {float(np.mean(finite[finite > 0])) if np.any(finite > 0) else 0.0:.2f} N)"
            )
    if result.slip_state is not None:
        slip = list(result.slip_state)
        n_stick = sum(1 for s in slip if s == "stick")
        n_slip = sum(1 for s in slip if s == "slip")
        print(f"Friction state: {n_stick} stick frame(s), {n_slip} slip frame(s) (§7)")

    # --- aligned ASCII timelines ('#' contact, '.' free) ---
    pred_strip = _ascii_timeline(np.asarray(result.in_contact, dtype=bool))
    true_strip = _ascii_timeline(np.asarray(truth.in_contact, dtype=bool))
    print("Timeline ('#' = contact, '.' = free):")
    print(f"  pred : {pred_strip}")
    print(f"  true : {true_strip}")
    print(line)


# --------------------------------------------------------------------------------------
# Contact-implicit inverse dynamics reporting (THEORY.md s.8, the north star).
#
# The kinematic detector above asks "does the motion LOOK like contact?". The inverse-
# dynamics layer (`contact.dynamics_id`) asks the dual question: what physically-valid
# contact forces EXPLAIN the observed motion under Newton-Euler with Signorini
# complementarity (s.2) and the Coulomb cone (s.7)? Its `InverseDynamicsResult` carries
# recovered per-candidate forces, the implied active set, the summed normal force (which
# at rest must equal m*g), and the per-frame wrench residual. This helper scores that
# result against the simulator's withheld truth metadata (the same `raw.meta` the
# scenario emits): m*g, the per-frame MuJoCo summed corner force, and the per-frame
# Signorini active set. It is the s.8 analogue of `score`/`print_report`.
# --------------------------------------------------------------------------------------


def _settled_mask(truth_meta: dict, candidate_count: int) -> np.ndarray | None:
    """Per-frame mask of the SETTLED/REST frames (all candidates simultaneously active).

    THEORY.md s.7/s.8: the recovered total normal force is only meant to equal the static
    weight ``m*g`` at *rest* — during a touchdown impact the contact force vastly exceeds
    the weight (s.6, the force atom), and during partial-support phases only some corners
    carry load. The honest "rest" set is therefore the frames where the simulator's truth
    active set is the FULL candidate set (every corner closed and loaded): those are the
    statically-supported frames where ``sum f_n == m*g`` is the physical expectation.

    Returns a ``(T,)`` boolean mask, or ``None`` if the truth metadata lacks a per-frame
    active flag (then the caller falls back to a trajectory-tail window).
    """
    cand = (truth_meta or {}).get("candidates")
    if not isinstance(cand, dict) or "active" not in cand:
        return None
    act_kt = np.asarray(cand["active"], dtype=bool)  # (K, T)
    if act_kt.ndim != 2 or act_kt.shape[0] != candidate_count:
        return None
    return np.all(act_kt, axis=0)  # (T,) all candidates active => fully supported / at rest


def _id_active_set_strip(active_set: list[list[int]], width: int = 60) -> str:
    """A one-line ASCII strip of how many candidates the inverse dynamics found active.

    Down-samples the per-frame active-candidate *count* into ``width`` cells; each cell
    shows the max count over the frames mapped into it as a single hex-ish digit
    ('.' for 0, '1'..'9', '+' for >=10). Lets the recovered active-set timeline be
    eyeballed against the truth strip beneath it (THEORY.md s.8).
    """
    counts = np.array([len(a) for a in active_set], dtype=int)
    n = counts.size
    if n == 0:
        return ""
    w = min(width, n)
    cell_of = (np.arange(n) * w) // n
    cells = np.zeros(w, dtype=int)
    np.maximum.at(cells, cell_of, counts)

    def _ch(c: int) -> str:
        if c <= 0:
            return "."
        if c < 10:
            return str(c)
        return "+"

    return "".join(_ch(int(c)) for c in cells)


def print_inverse_dynamics(
    result: InverseDynamicsResult,
    truth_meta: dict,
    name: str | None = None,
) -> dict:
    """Print and return the inverse-dynamics scorecard against the withheld truth (s.8).

    THEORY.md s.8 (the north star) + s.9 (validation): the inverse-dynamics solver saw
    only the noisy observed pose; here we score what it recovered against the simulator's
    withheld physical truth carried in ``truth_meta`` (a scenario's ``raw.meta``). We
    report, all at the SETTLED/REST frames (where the truth active set is the full
    candidate set — see :func:`_settled_mask`):

    * **recovered total normal force** vs the analytic ``m*g`` and vs the MuJoCo summed
      per-corner force (``meta['candidates']['normal_force']``). At rest all three should
      agree; away from rest the MuJoCo sum spikes at impacts (s.6) and the recovered total
      tracks the *required* support, which is why we restrict to rest for the headline.
    * **active-set timeline** — recovered (count per frame) vs the Signorini truth
      (``meta['candidates']['active']``), as aligned ASCII strips and an IoU over the
      per-frame "any candidate active" masks.
    * **mean wrench residual** ``||G f - w||`` over all frames (how well the forces explain
      the motion) — the s.8 consistency check.

    The per-candidate load split is reported but flagged: in a statically-indeterminate
    configuration it is the regularizer's minimum-norm choice, NOT a measurement (s.7).

    Parameters
    ----------
    result : InverseDynamicsResult
        Output of :func:`contact.dynamics_id.contact_implicit_from_raw`.
    truth_meta : dict
        The scenario's ``raw.meta``; must carry ``inertial`` (mass) and ``gravity``, and
        ideally ``candidates`` (``normal_force`` (K,T), ``active`` (K,T)) for the truth
        comparison. Missing pieces degrade gracefully (the section is skipped, not fatal).
    name : str, optional
        Scenario id, printed in the header.

    Returns
    -------
    dict
        Flat scores: ``mg``, ``recovered_total_rest``, ``mujoco_total_rest``,
        ``mean_residual``, ``active_iou``, ``n_rest_frames``.
    """
    line = "=" * 72
    print(line)
    hdr = "INVERSE DYNAMICS (THEORY.md s.8: contact-implicit Newton-Euler)"
    print(f"{hdr}" + (f"  —  {name}" if name else ""))
    print(line)

    meta = truth_meta or {}
    K = int(np.asarray(result.candidate_points).reshape(-1, 3).shape[0])
    T = int(np.asarray(result.t).size)

    rec_total = np.asarray(result.total_normal_force, dtype=float)  # (T,)
    residual = np.asarray(result.wrench_residual, dtype=float)      # (T,)

    # --- analytic weight m*g (the static expectation) ---
    inert = meta.get("inertial", {})
    mass = float(inert.get("mass", float("nan"))) if isinstance(inert, dict) else float("nan")
    gravity = abs(float(meta.get("gravity", 9.81)))
    mg = mass * gravity

    # --- choose the SETTLED/REST frames (all candidates active in truth) ---
    settled = _settled_mask(meta, K)
    if settled is not None and settled.shape[0] == T and bool(np.any(settled)):
        rest_idx = np.flatnonzero(settled)
        rest_src = "frames with all candidates active (Signorini truth)"
    else:
        # Fallback: the last quarter of the trajectory (the quiet tail).
        tail0 = (3 * T) // 4
        rest_idx = np.arange(tail0, T)
        rest_src = "trajectory tail (no per-frame truth active flag)"
    n_rest = int(rest_idx.size)

    # Robust central value over the rest frames (median rejects the odd boundary spike).
    rec_rest = float(np.median(rec_total[rest_idx])) if n_rest else float("nan")

    cand = meta.get("candidates") if isinstance(meta.get("candidates"), dict) else None
    muj_rest = float("nan")
    if cand is not None and "normal_force" in cand:
        fn_kt = np.asarray(cand["normal_force"], dtype=float)  # (K, T)
        if fn_kt.ndim == 2 and fn_kt.shape == (K, T):
            muj_total = fn_kt.sum(axis=0)  # (T,) summed over corners
            muj_rest = float(np.median(muj_total[rest_idx])) if n_rest else float("nan")

    print(f"Settled/rest frames: {n_rest}/{T}  [{rest_src}]")
    print("Recovered total normal force at rest (THEORY.md s.7: sum f_n == m*g):")
    mg_s = "   n/a" if not np.isfinite(mg) else f"{mg:8.2f} N"
    print(f"  m*g (analytic)               : {mg_s}")
    print(f"  recovered  sum f_n  (median) : {rec_rest:8.2f} N")
    if np.isfinite(muj_rest):
        print(f"  MuJoCo summed corner force   : {muj_rest:8.2f} N")
    if np.isfinite(mg) and np.isfinite(rec_rest) and mg > 1e-9:
        err = 100.0 * (rec_rest - mg) / mg
        print(f"  recovered vs m*g             : {err:+6.1f} %")

    # --- per-candidate load split at rest (flagged unobservable in general, s.7) ---
    nf_tk = np.asarray(result.contact_normal_force, dtype=float)  # (T, K)
    if n_rest and nf_tk.shape == (T, K):
        split = np.median(nf_tk[rest_idx], axis=0)  # (K,) per-candidate median at rest
        split_s = ", ".join(f"{v:5.1f}" for v in split)
        print(f"  per-candidate split (median) : [{split_s}] N")
        if cand is not None and "normal_force" in cand:
            fn_kt = np.asarray(cand["normal_force"], dtype=float)
            if fn_kt.shape == (K, T):
                tsplit = np.median(fn_kt[:, rest_idx], axis=1)  # (K,)
                tsplit_s = ", ".join(f"{v:5.1f}" for v in tsplit)
                print(f"  truth per-corner    (median) : [{tsplit_s}] N")
        print(
            "  (NB: in an indeterminate config the split is the min-norm regularizer's "
            "choice, not a measurement — THEORY.md s.7.)"
        )

    # --- wrench residual (how well the forces explain the motion, s.8) ---
    mean_res = float(np.mean(residual)) if residual.size else float("nan")
    rest_res = float(np.mean(residual[rest_idx])) if n_rest else float("nan")
    print(f"Mean wrench residual ||G f - w||: {mean_res:.3e} N (all)  /  "
          f"{rest_res:.3e} N (rest)")

    # --- active-set timeline: recovered vs Signorini truth -----------------------------
    pred_any = np.array([len(a) > 0 for a in result.active_set], dtype=bool)  # (T,)
    active_iou = float("nan")
    if cand is not None and "active" in cand:
        act_kt = np.asarray(cand["active"], dtype=bool)
        if act_kt.shape == (K, T):
            true_any = np.any(act_kt, axis=0)  # (T,)
            inter = int(np.count_nonzero(pred_any & true_any))
            union = int(np.count_nonzero(pred_any | true_any))
            active_iou = 1.0 if union == 0 else inter / union
            print(f"Active-set existence IoU (any candidate active): {active_iou:.3f}")
            # Aligned strips: recovered active-count over the truth active-count.
            true_count_strip = _id_active_set_strip(
                [list(np.flatnonzero(act_kt[:, k])) for k in range(T)]
            )
            print("Active-set timeline (digit = # candidates active):")
            print(f"  recovered : {_id_active_set_strip(result.active_set)}")
            print(f"  truth     : {true_count_strip}")
    else:
        print("Recovered active-set timeline (digit = # candidates active):")
        print(f"  recovered : {_id_active_set_strip(result.active_set)}")
    print(line)

    return {
        "mg": float(mg),
        "recovered_total_rest": float(rec_rest),
        "mujoco_total_rest": float(muj_rest),
        "mean_residual": float(mean_res),
        "active_iou": float(active_iou),
        "n_rest_frames": int(n_rest),
    }


# --------------------------------------------------------------------------------------
# Plotting (lazy matplotlib import)
# --------------------------------------------------------------------------------------


def plot_result(obs, result: DetectionResult, truth: GroundTruth, out_path: str) -> None:
    """Render a stacked diagnostic figure and save it to ``out_path`` (dpi 120).

    Three x-aligned panels visualise the chain of THEORY.md reasoning on one
    scenario:

    1. **Gap (mm)** with the surface line at 0 — the support-relative signed
       distance of §1 whose zero-crossing is the make/break guard of §5.
    2. **Contact posterior** (the calibrated P(contact) of §4) with the true
       in-contact spans shaded behind it.
    3. **MAP mode vs. true mode** as aligned category step plots — the inferred
       twist-subspace mode of §3 against the oracle, with the per-frame stick/slip
       label (§7) shaded behind it when present.

    Detected make/break events (§6) are drawn as dotted vertical lines across all
    panels; impact atoms (§6) as solid vertical lines (one per arrest).

    matplotlib is imported lazily; if it is unavailable a note is printed and the
    function returns without writing a file. Empty/`None` impulses and slip_state are
    handled gracefully (nothing is drawn).

    Parameters
    ----------
    obs : ContactObservations
        Provides ``obs.t`` (s) and ``obs.gap`` (m) for the top panel.
    result : DetectionResult
        Detector output (posterior, MAP modes, events).
    truth : GroundTruth
        Oracle labels (in_contact spans, true modes).
    out_path : str
        Destination image path.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless-safe; we only save to file
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"plot_result: matplotlib unavailable ({exc!r}); skipping plot.")
        return

    t = np.asarray(obs.t, dtype=float)
    gap_mm = np.asarray(obs.gap, dtype=float) * 1e3
    posterior = np.asarray(result.contact_posterior, dtype=float)
    true_contact = np.asarray(truth.in_contact, dtype=bool)

    fig, (ax_gap, ax_post, ax_mode) = plt.subplots(
        3, 1, sharex=True, figsize=(10, 7)
    )

    # --- panel 1: gap (mm) with surface line ---
    ax_gap.plot(t, gap_mm, color="C0", lw=1.2, label="gap")
    ax_gap.axhline(0.0, color="k", lw=0.8, ls="--", label="surface")
    ax_gap.set_ylabel("gap (mm)")
    ax_gap.legend(loc="upper right", fontsize=8)
    ax_gap.set_title("contact detection diagnostics")

    # --- panel 2: contact posterior + true-contact shading ---
    ax_post.plot(t, posterior, color="C3", lw=1.4, label="P(contact)")
    ax_post.set_ylabel("P(contact)")
    ax_post.set_ylim(-0.05, 1.05)
    _shade_true_contact(ax_post, t, true_contact, label="true contact")
    ax_post.legend(loc="upper right", fontsize=8)

    # --- panel 3: MAP mode vs true mode as category step plots ---
    # Build a stable category axis from every label that actually appears, with
    # FREE pinned at the bottom (index 0) to match the canonical ALL_STATES order.
    map_state = list(result.map_state)
    true_mode = list(truth.mode)
    present: list[str] = [FREE]
    for lbl in map_state + true_mode:
        if lbl not in present:
            present.append(lbl)
    cat_index = {lbl: i for i, lbl in enumerate(present)}

    pred_y = np.array([cat_index[s] for s in map_state], dtype=float)
    true_y = np.array([cat_index[s] for s in true_mode], dtype=float)

    # --- per-frame stick/slip shading behind the modes (THEORY.md §7) ---
    # Shade the frames the friction layer labelled "slip" (a thin band along the
    # bottom of the mode panel), so the stick/slip prediction sits visually next to
    # the kinematic sliding mode it cross-checks. Absent/None slip_state draws nothing.
    if result.slip_state is not None:
        slip_mask = np.array(
            [s == "slip" for s in result.slip_state], dtype=bool
        )
        if slip_mask.size == t.size:
            _shade_mask(
                ax_mode, t, slip_mask, color="C5", alpha=0.20, label="slip (§7)"
            )

    ax_mode.step(t, true_y, where="post", color="0.6", lw=3.0, label="true mode")
    ax_mode.step(t, pred_y, where="post", color="C2", lw=1.4, label="MAP mode")
    ax_mode.set_yticks(range(len(present)))
    ax_mode.set_yticklabels(present)
    ax_mode.set_ylim(-0.5, len(present) - 0.5)
    ax_mode.set_ylabel("mode")
    ax_mode.set_xlabel("time (s)")
    ax_mode.legend(loc="upper right", fontsize=8)

    # --- mark make/break events with dotted vlines across all panels (THEORY.md §6) ---
    for ev in result.events:
        color = "C1" if ev.kind == "touchdown" else "C4"
        for ax in (ax_gap, ax_post, ax_mode):
            ax.axvline(ev.time, color=color, lw=0.9, ls=":", alpha=0.8)

    # --- mark impact atoms with solid vlines across all panels (THEORY.md §6) ---
    # One per arrest of the normal velocity; complementary to the make/break events.
    for imp in result.impulses:
        for ax in (ax_gap, ax_post, ax_mode):
            ax.axvline(imp.time, color="C6", lw=1.1, ls="-", alpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _shade_mask(
    ax, t: np.ndarray, mask: np.ndarray, *, color: str, alpha: float, label: str
) -> None:
    """Shade the contiguous spans where ``mask`` is True on ``ax`` (one legend entry).

    Generic span-shader used both for the true active set behind the posterior
    (THEORY.md §2) and for the per-frame stick/slip label behind the modes (§7). Only
    the first span carries the legend label so the legend stays a single entry.
    """
    runs: list[tuple[float, float]] = []
    if mask.size:
        padded = np.concatenate(([False], mask.astype(bool), [False]))
        diffs = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(diffs == 1)
        ends = np.flatnonzero(diffs == -1) - 1
        for s, e in zip(starts, ends):
            runs.append((float(t[s]), float(t[e])))
    for k, (s, e) in enumerate(runs):
        ax.axvspan(s, e, color=color, alpha=alpha, label=label if k == 0 else None)


def _shade_true_contact(ax, t: np.ndarray, mask: np.ndarray, label: str) -> None:
    """Shade the spans where ``mask`` is True on ``ax`` (one legend entry only).

    Visual stand-in for the true active set (THEORY.md §2) behind the posterior.
    """
    _shade_mask(ax, t, mask, color="C7", alpha=0.25, label=label)


# --------------------------------------------------------------------------------------
# Multi-body contact-graph reporting (THEORY.md §8: the joint active-set structure).
#
# The single-pair helpers above score and render ONE body-pair's DetectionResult against
# ONE GroundTruth. A scene (`MultiBodyScene`) is a whole contact graph: several candidate
# edges, each with its own per-edge DetectionResult (inside `GraphDetectionResult.per_edge`)
# and its own per-edge GroundTruth (inside `scene.truth`), fused into a joint active-set
# posterior. These helpers report that joint object: per-edge intervals/mode/score (reusing
# `score`), and the MAP active-set timeline (which edges are active over time) vs. truth.
# --------------------------------------------------------------------------------------


def _dominant_mode(result: DetectionResult) -> str:
    """The most frequent non-FREE MAP mode of one edge's detection (its overall type).

    THEORY.md §3: a contact's *type* is its twist-subspace mode. An edge's per-frame MAP
    labels may mix (e.g. a leading IMPACT then STATIC); the dominant non-FREE label names
    the edge's overall contact mode for the one-line summary. ``FREE`` if never in contact.
    """
    counts: dict[str, int] = {}
    for lbl in result.map_state:
        if lbl != FREE:
            counts[lbl] = counts.get(lbl, 0) + 1
    if not counts:
        return FREE
    return max(counts, key=lambda k: counts[k])


def _active_set_str(active: list[str]) -> str:
    """Render a set of active edge ids as ``{a, b}`` (``{}`` for the empty set)."""
    return "{" + ", ".join(active) + "}" if active else "{}"


def _true_active_set_per_frame(scene: MultiBodyScene, edges: list[str]) -> list[list[str]]:
    """Per-frame TRUE active set: the edge ids whose ground-truth ``in_contact`` is set.

    THEORY.md §8/§9: the withheld oracle for the joint structure. For each frame we collect
    the edge ids that the simulator labelled in-contact, in the column order of ``edges``,
    giving the truth analogue of ``GraphDetectionResult.map_active_set``.
    """
    masks: dict[str, np.ndarray] = {}
    T = 0
    for eid in edges:
        gt = scene.truth.get(eid)
        if gt is None:
            continue
        m = np.asarray(gt.in_contact, dtype=bool).ravel()
        masks[eid] = m
        T = max(T, m.shape[0])
    out: list[list[str]] = []
    for k in range(T):
        out.append([eid for eid in edges if eid in masks and k < masks[eid].shape[0] and masks[eid][k]])
    return out


def _active_set_runs(
    sets_per_frame: list[list[str]], t: np.ndarray
) -> list[tuple[float, float, list[str]]]:
    """Contiguous (t_start, t_end, active_set) runs of an identical active-set sequence.

    Collapses a per-frame list of active-edge-id lists into the segments over which the
    active set is constant — the structural timeline of THEORY.md §8 (the active set
    persists, then changes at the discrete guard instants of §5).
    """
    runs: list[tuple[float, float, list[str]]] = []
    n = len(sets_per_frame)
    if n == 0:
        return runs
    t = np.asarray(t, dtype=float).ravel()
    i = 0
    while i < n:
        cur = sorted(sets_per_frame[i])
        j = i + 1
        while j < n and sorted(sets_per_frame[j]) == cur:
            j += 1
        runs.append((float(t[i]), float(t[min(j - 1, t.shape[0] - 1)]), cur))
        i = j
    return runs


def print_graph_report(scene: MultiBodyScene, graph_result: GraphDetectionResult) -> None:
    """Print a terminal summary of a multi-body scene's joint contact-graph detection (§8).

    Renders, for the scene:

    * **Per edge** — the detected contact intervals with their dominant twist-subspace mode
      (§3), and the frame-level score of that edge's per-pair ``DetectionResult`` against the
      edge's withheld ``GroundTruth`` (reusing :func:`score` per edge, §9), each next to two
      aligned ASCII contact timelines (predicted over truth).
    * **Joint active set** — the MAP active-set timeline (which edges are simultaneously
      active over time, §8) as constant-active-set segments, alongside the ground-truth
      active-set segments, so a structural change (e.g. ``{a, b} -> {a}``) is visible.

    Parameters
    ----------
    scene : MultiBodyScene
        The scene (bodies, candidate edges, and per-edge ground truth).
    graph_result : GraphDetectionResult
        The output of :func:`contact.graph.detect_scene`.
    """
    line = "=" * 72
    edges = list(graph_result.edges)
    print(line)
    print(f"SCENE: {scene.name}   (edges: {len(edges)})")
    print(line)

    if not edges:
        print("(no candidate edges — empty contact graph)")
        print(line)
        return

    # --- per-edge detection + score vs that edge's ground truth (§9) -------------------
    for eid in edges:
        result = graph_result.per_edge.get(eid)
        truth = scene.truth.get(eid)
        print(f"EDGE {eid}:")
        if result is None:
            print("  (no detection result)")
            continue

        # detected intervals + dominant mode
        if result.intervals:
            dom = _dominant_mode(result)
            print(f"  Detected contact intervals (dominant mode: {dom}):")
            for iv in result.intervals:
                print(f"    [{iv.t_start:8.3f}, {iv.t_end:8.3f}] s  {iv.mode}")
        else:
            print("  Detected contact intervals: (none)")

        if truth is not None:
            true_runs = _true_contact_intervals(truth)
            if true_runs:
                print("  True contact intervals:")
                for s, e in true_runs:
                    print(f"    [{s:8.3f}, {e:8.3f}] s")
            else:
                print("  True contact intervals: (none)")

            sc = score(result, truth)
            macc = sc["mode_accuracy"]
            macc_s = "  n/a" if (isinstance(macc, float) and np.isnan(macc)) else f"{macc:.3f}"
            print(
                f"  Score vs truth: iou={sc['contact_iou']:.3f}  f1={sc['contact_f1']:.3f}  "
                f"mode_acc={macc_s}  (true {sc['contact_frames_true']} / "
                f"pred {sc['contact_frames_pred']} contact frames)"
            )
            pred_strip = _ascii_timeline(np.asarray(result.in_contact, dtype=bool))
            true_strip = _ascii_timeline(np.asarray(truth.in_contact, dtype=bool))
            print(f"    pred : {pred_strip}")
            print(f"    true : {true_strip}")
        else:
            print("  (no ground truth for this edge)")
        print()

    # --- joint MAP active-set timeline vs truth (the §8 structure) ---------------------
    t = np.asarray(graph_result.t, dtype=float).ravel()
    pred_runs = _active_set_runs(list(graph_result.map_active_set), t)
    true_runs = _active_set_runs(_true_active_set_per_frame(scene, edges), t)

    print("JOINT ACTIVE-SET timeline (MAP):")
    for s, e, active in pred_runs:
        print(f"  [{s:8.3f}, {e:8.3f}] s  {_active_set_str(active)}")
    print("JOINT ACTIVE-SET timeline (truth):")
    for s, e, active in true_runs:
        print(f"  [{s:8.3f}, {e:8.3f}] s  {_active_set_str(active)}")

    # --- joint-structure diagnostics from graph_result.meta ---------------------------
    meta = graph_result.meta or {}
    diag_bits = []
    if "num_subsets" in meta:
        diag_bits.append(f"subsets={meta['num_subsets']}")
    if "joint_loglik" in meta:
        diag_bits.append(f"joint_loglik={meta['joint_loglik']:.1f}")
    if meta.get("energy_prior_active"):
        diag_bits.append("energy-prior:on")
    if meta.get("balance_prior_active"):
        diag_bits.append("balance-prior:on")
    if diag_bits:
        print("Diagnostics: " + "  ".join(diag_bits))
    if scene.meta and scene.meta.get("active_set_change"):
        print(f"Truth note: {scene.meta['active_set_change']}")
    print(line)


def plot_graph(
    scene: MultiBodyScene, graph_result: GraphDetectionResult, out_path: str
) -> None:
    """Render the contact-graph diagnostic figure and save it to ``out_path`` (dpi 120).

    One row per edge plus a final active-set strip, x-aligned on a shared time axis:

    * **Per edge** — the support-relative gap (mm, §1) on a left y-axis with the surface
      line at 0, the calibrated per-edge contact posterior (§4) on a right y-axis, and the
      edge's true in-contact span shaded behind (§2/§9).
    * **Active-set strip** — a stacked image with one lane per edge, shaded where that edge
      is in the MAP active set per frame (§8); the truth active set is overlaid as a thin
      hatch lane so structural agreement/disagreement is visible at a glance.

    matplotlib is imported lazily; if unavailable a note is printed and no file is written
    (mirrors :func:`plot_result`). An empty scene (no edges) draws only the strip note.

    Parameters
    ----------
    scene : MultiBodyScene
    graph_result : GraphDetectionResult
        Output of :func:`contact.graph.detect_scene`.
    out_path : str
        Destination image path.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless-safe
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"plot_graph: matplotlib unavailable ({exc!r}); skipping plot.")
        return

    from . import geometry  # local import: keep report importable without geometry at top
    from .graph import _resolve_support  # the world-floor synthesizer (THEORY.md §1)

    edges = list(graph_result.edges)
    t = np.asarray(graph_result.t, dtype=float).ravel()
    edge_by_id = {e.edge_id: e for e in scene.edges}

    n_edge_rows = max(len(edges), 1)
    n_rows = n_edge_rows + 1  # + the active-set strip
    fig, axes = plt.subplots(
        n_rows, 1, sharex=True, figsize=(11, 2.1 * n_rows + 0.5)
    )
    if n_rows == 1:  # matplotlib returns a bare Axes for a single row
        axes = [axes]
    axes = list(np.atleast_1d(axes))

    axes[0].set_title(f"contact-graph detection: {scene.name}")

    # --- per-edge rows: gap (mm) + posterior + true-contact shading -------------------
    for row, eid in enumerate(edges):
        ax = axes[row]
        result = graph_result.per_edge.get(eid)
        edge = edge_by_id.get(eid)

        # Recover the support-relative gap for this edge (same observe path as detection),
        # synthesizing the implicit static world floor where the support is "world" (§1).
        gap_mm = None
        if edge is not None and edge.moving_body in scene.bodies:
            try:
                moving = scene.bodies[edge.moving_body]
                support = _resolve_support(scene, edge.support_body, moving)
                if support is not None:
                    obs = geometry.observe(
                        moving, support, edge.surface, edge.contact_point_local,
                        geometry=getattr(edge, "geometry", None),
                    )
                    gap_mm = np.asarray(obs.gap, dtype=float) * 1e3
            except Exception:
                gap_mm = None

        if gap_mm is not None:
            ax.plot(t[: gap_mm.shape[0]], gap_mm, color="C0", lw=1.1, label="gap (mm)")
            ax.axhline(0.0, color="k", lw=0.7, ls="--")
        ax.set_ylabel(f"{eid}\ngap (mm)", fontsize=8)

        ax_post = ax.twinx()
        if result is not None:
            post = np.asarray(result.contact_posterior, dtype=float)
            ax_post.plot(
                t[: post.shape[0]], post, color="C3", lw=1.3, label="P(contact)"
            )
        ax_post.set_ylim(-0.05, 1.05)
        ax_post.set_ylabel("P", fontsize=8, color="C3")

        truth = scene.truth.get(eid)
        if truth is not None:
            _shade_mask(
                ax,
                t,
                np.asarray(truth.in_contact, dtype=bool),
                color="C7",
                alpha=0.22,
                label="true contact",
            )
        # Single combined legend (gap + true-contact on ax; posterior label on ax_post).
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_post.get_legend_handles_labels()
        if h1 or h2:
            ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7)

    if not edges:
        axes[0].text(
            0.5, 0.5, "(empty contact graph — no edges)",
            ha="center", va="center", transform=axes[0].transAxes,
        )

    # --- active-set strip: one lane per edge, MAP (filled) vs truth (hatched) ----------
    ax_strip = axes[-1]
    pred_sets = list(graph_result.map_active_set)
    true_sets = _true_active_set_per_frame(scene, edges)
    T = t.shape[0]
    for lane, eid in enumerate(edges):
        pred_mask = np.array(
            [eid in (pred_sets[k] if k < len(pred_sets) else []) for k in range(T)],
            dtype=bool,
        )
        true_mask = np.array(
            [eid in (true_sets[k] if k < len(true_sets) else []) for k in range(T)],
            dtype=bool,
        )
        y = lane
        # MAP active: a solid filled band in this lane.
        _fill_lane(ax_strip, t, pred_mask, y, height=0.8, color=f"C{lane % 10}", alpha=0.55)
        # Truth active: a thinner hatched band overlaid (so disagreement shows).
        _fill_lane(
            ax_strip, t, true_mask, y, height=0.8, color="none",
            alpha=1.0, hatch="////", edgecolor="k",
        )
    ax_strip.set_yticks(range(max(len(edges), 1)))
    ax_strip.set_yticklabels(edges if edges else [""])
    ax_strip.set_ylim(-0.6, max(len(edges) - 0.4, 0.6))
    ax_strip.set_ylabel("active set", fontsize=8)
    ax_strip.set_xlabel("time (s)")
    ax_strip.set_title("MAP active set (solid) vs truth (hatched)", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _fill_lane(
    ax,
    t: np.ndarray,
    mask: np.ndarray,
    y: float,
    *,
    height: float,
    color: str,
    alpha: float,
    hatch: str | None = None,
    edgecolor=None,
) -> None:
    """Shade the contiguous True spans of ``mask`` as a horizontal band centred on ``y``.

    A horizontal analogue of :func:`_shade_mask` used by the active-set strip of
    :func:`plot_graph`: each contiguous run of frames where the edge is in the (predicted or
    true) active set becomes one rectangle in that edge's lane (THEORY.md §8).
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return
    t = np.asarray(t, dtype=float).ravel()
    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diffs == 1)
    ends = np.flatnonzero(diffs == -1) - 1
    y0 = y - height / 2.0
    for s, e in zip(starts, ends):
        _add_rect(
            ax, float(t[s]), float(t[min(e, t.shape[0] - 1)]), y0, height,
            color=color, alpha=alpha, hatch=hatch, edgecolor=edgecolor,
        )


def _add_rect(ax, x0: float, x1: float, y0: float, height: float, *, color, alpha, hatch, edgecolor):
    """Add a single time-bounded rectangle to ``ax`` (helper for :func:`_fill_lane`)."""
    import matplotlib.patches as mpatches

    width = max(x1 - x0, 1e-9)
    face = "none" if color == "none" else color
    rect = mpatches.Rectangle(
        (x0, y0), width, height,
        facecolor=face, alpha=alpha, hatch=hatch,
        edgecolor=(edgecolor if edgecolor is not None else "none"),
        linewidth=0.0 if edgecolor is None else 0.6,
    )
    ax.add_patch(rect)
