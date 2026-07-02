"""Tests for unsupervised contact-mode discovery (THEORY.md §8).

``contact.mode_discovery`` is the research-frontier rung of THEORY.md §8 that the
rest of the package sidesteps: instead of *presupposing* the canonical five modes of
§3 (the supervised emission bank of ``contact.emissions``), it fits a **sticky
HDP-HMM** that *discovers* the mode vocabulary from the raw per-frame twist feature --
learning *how many* modes a clip needs and which frames belong to each, with no access
to the canonical names. This suite stresses three claims of that module:

* **The feature is well-formed.** ``mode_feature_vector`` returns the ``(T, 5)`` twist
  signature ``[gap, |v_n|, |v_t|, |omega_n|, |omega_t|]`` of §3, finite everywhere.

* **Discovery rediscovers the regimes (synthetic).** On a hand-built 3-regime clip
  (rest -> slide -> spin, the §3 static/sliding/pivoting archetypes), the model uses
  >= 3 distinct modes, the dominant discovered mode in each regime window *aligns* to
  the expected canonical mode (``_align_signature``, validation-only), and -- thanks to
  the sticky self-transition prior of §5 -- the labels are piecewise-constant (a small
  number of switches, not a flickering segmentation). The same seed gives identical
  labels (the Gibbs sampler is seeded; §8 honesty note).

* **Discovery runs on MuJoCo truth (§9).** On ``push_to_slide`` (generate -> observe),
  discovery runs end to end and finds both a static-aligned and a sliding-aligned mode
  -- the static->sliding stick/slip regime of §7 that the scenario builds.

The discovery itself is label-free; the canonical-name ``alignment`` is the post-hoc
nearest-signature heuristic of ``_align_signature``, used here for validation only.

MuJoCo is required only for the scenario-backed test; if it is absent that test is
skipped while the synthetic tests of the same functions still run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the repo root importable regardless of how pytest is invoked.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from contact.config import InferenceParams
from contact.mode_discovery import discover_modes, mode_feature_vector
from contact.types import SLIDING, STATIC, ContactObservations

# A single fixed seed so every scenario-backed test is reproducible (the seed only
# drives the additive mocap noise in oracle.generate; the physics is deterministic).
SEED = 12345
HZ = 200.0

# Frames per regime in the synthetic clip (long enough that the sticky HDP-HMM dwells
# on one mode per window rather than fragmenting it).
_REGIME_FRAMES = 120


# --------------------------------------------------------------------------------------
# Synthetic 3-regime clip builder.
# --------------------------------------------------------------------------------------

def _three_regime_obs(seed: int = 0) -> ContactObservations:
    """Hand-build a rest -> slide -> spin clip as ``ContactObservations``.

    Three back-to-back contact regimes, each the §3 archetype of one canonical mode,
    built directly in the support-relative contact frame (so no ``observe`` round-trip
    is needed and the planted truth is exact):

    * **rest**  : every twist channel quiet, gap ~ 0          -> STATIC.
    * **slide** : tangential-linear speed excited (~0.25 m/s)  -> SLIDING.
    * **spin**  : spin about the normal excited (~1.5 rad/s)   -> PIVOTING.

    A little Gaussian jitter is added (well below the regime amplitudes) so the
    clusters have non-degenerate spread; the gap stays ~0 throughout so all three are
    *contact* regimes (the discovery clusters the twist channels, §3, not free/contact).
    """
    rng = np.random.default_rng(seed)
    n = _REGIME_FRAMES
    T = 3 * n
    dt = 1.0 / HZ
    t = np.arange(T) * dt

    gap = rng.normal(0.0, 5e-4, size=T)        # ~0 contact gap with tiny jitter
    v_normal = rng.normal(0.0, 5e-3, size=T)
    v_tangent = rng.normal(0.0, 1e-2, size=(T, 2))
    omega_normal = rng.normal(0.0, 2e-2, size=T)
    omega_tangent = rng.normal(0.0, 2e-2, size=(T, 2))

    # slide regime: a clear tangential-linear velocity along x.
    v_tangent[n:2 * n, 0] += 0.25
    # spin regime: a clear angular rate about the normal.
    omega_normal[2 * n:3 * n] += 1.5

    return ContactObservations(
        t=t,
        gap=gap,
        v_normal=v_normal,
        v_tangent=v_tangent,
        omega_normal=omega_normal,
        omega_tangent=omega_tangent,
    )


def _regime_windows() -> list[tuple[str, str, int, int]]:
    """``(regime_name, expected_canonical_mode, lo, hi)`` for the three windows."""
    n = _REGIME_FRAMES
    from contact.types import PIVOTING  # local import: only the windows need it

    return [
        ("rest", STATIC, 0, n),
        ("slide", SLIDING, n, 2 * n),
        ("spin", PIVOTING, 2 * n, 3 * n),
    ]


def _dominant_label(labels: np.ndarray, lo: int, hi: int) -> int:
    """The most frequent label id over the half-open window ``[lo, hi)``."""
    window = labels[lo:hi]
    vals, counts = np.unique(window, return_counts=True)
    return int(vals[int(np.argmax(counts))])


# --------------------------------------------------------------------------------------
# mode_feature_vector
# --------------------------------------------------------------------------------------

def test_mode_feature_vector_shape_and_finite():
    """``mode_feature_vector`` returns a finite ``(T, 5)`` twist signature (§3).

    The clustering feature is the 5-vector of channel magnitudes
    ``[gap, |v_n|, |v_t|, |omega_n|, |omega_t|]`` (gap kept signed -- its sign is the
    Signorini branch of §2). It must have one row per frame, five columns, and contain
    no NaN/inf (the downstream standardize/Gaussian math would otherwise poison the fit).
    """
    obs = _three_regime_obs(seed=0)
    feat = mode_feature_vector(obs)

    T = obs.t.shape[0]
    assert feat.shape == (T, 5)
    assert np.all(np.isfinite(feat))

    # The magnitude channels (columns 1..4) are non-negative by construction (they are
    # absolute values / Euclidean norms); the gap (column 0) is signed.
    assert np.all(feat[:, 1:] >= 0.0)


def test_mode_feature_vector_columns_track_planted_channels():
    """Each regime excites exactly its own feature column (a sanity anchor for §3).

    A cross-check that the feature columns mean what the test below relies on: the slide
    window has a large ``|v_t|`` (column 2) and the spin window a large ``|omega_n|``
    (column 3), each far above its value in the quiet rest window.
    """
    obs = _three_regime_obs(seed=0)
    feat = mode_feature_vector(obs)
    n = _REGIME_FRAMES

    rest_vt = feat[0:n, 2].mean()
    slide_vt = feat[n:2 * n, 2].mean()
    rest_wn = feat[0:n, 3].mean()
    spin_wn = feat[2 * n:3 * n, 3].mean()

    assert slide_vt > 0.15 and slide_vt > 10.0 * rest_vt
    assert spin_wn > 1.0 and spin_wn > 10.0 * rest_wn


# --------------------------------------------------------------------------------------
# discover_modes on the synthetic 3-regime clip
# --------------------------------------------------------------------------------------

def test_discover_modes_synthetic_three_regimes():
    """Discovery rediscovers static/sliding/pivoting on the 3-regime clip (§8).

    The sticky HDP-HMM, given only the raw twist feature (no canonical labels), should:

    * use **>= 3** distinct discovered modes (one per planted regime, possibly more);
    * have its **dominant** discovered mode in each regime window *align* (via the
      validation-only ``_align_signature``) to that regime's expected canonical mode --
      static for rest, sliding for slide, pivoting for spin (§3 archetypes);
    * produce **piecewise-constant** labels: the sticky self-transition prior (the §5
      dwell prior in nonparametric clothing) keeps a physical regime from fragmenting,
      so the number of label switches stays small (we allow a little slack above the two
      true regime boundaries, but reject a flickering segmentation).
    """
    obs = _three_regime_obs(seed=0)
    result = discover_modes(obs, InferenceParams(), seed=0)

    # >= 3 distinct modes were actually populated.
    assert result.n_modes >= 3
    assert len(set(result.labels.tolist())) >= 3

    # The dominant discovered mode in each regime window aligns to the expected canonical.
    for regime, expected, lo, hi in _regime_windows():
        dom = _dominant_label(result.labels, lo, hi)
        assert result.alignment[dom] == expected, (
            f"{regime} window: dominant discovered mode {dom} aligned to "
            f"{result.alignment[dom]!r}, expected {expected!r}"
        )

    # Piecewise-constant: few switches thanks to the sticky prior. The truth has exactly
    # two boundaries (rest->slide->spin); allow modest slack but forbid flicker.
    switches = int(np.sum(result.labels[1:] != result.labels[:-1]))
    assert switches <= 6, f"labels flicker ({switches} switches) despite the sticky prior"


def test_discover_modes_dominant_label_unique_per_regime():
    """The three regimes are carried by three *distinct* dominant modes (§8).

    Beyond aligning to the right canonical name, the dominant discovered id must differ
    across the three windows -- i.e. the model genuinely separated the regimes into
    different latent states rather than lumping two regimes under one mode that happens
    to align to both.
    """
    obs = _three_regime_obs(seed=0)
    result = discover_modes(obs, InferenceParams(), seed=0)

    dominants = [
        _dominant_label(result.labels, lo, hi)
        for _regime, _expected, lo, hi in _regime_windows()
    ]
    assert len(set(dominants)) == 3, f"regimes share a dominant mode: {dominants}"


def test_discover_modes_deterministic_same_seed():
    """Same input + same seed => identical labels (the seeded-Gibbs guarantee, §8).

    ``discover_modes`` seeds its blocked-Gibbs sampler from ``seed``; the module's
    docstring promises deterministic output for a fixed input and seed. Two independent
    calls must therefore agree exactly on the labels, the mode count, and the alignment.
    """
    obs = _three_regime_obs(seed=0)

    a = discover_modes(obs, InferenceParams(), seed=0)
    b = discover_modes(obs, InferenceParams(), seed=0)

    assert np.array_equal(a.labels, b.labels)
    assert a.n_modes == b.n_modes
    assert a.alignment == b.alignment


def test_discover_modes_single_frame_degenerate():
    """A one-frame clip is one mode (the documented degenerate short-circuit).

    ``discover_modes`` skips the sampler for ``T == 1`` and returns a single mode. This
    guards the boundary so the Gibbs machinery is never asked to segment a length-1 path.
    """
    t = np.array([0.0])
    obs = ContactObservations(
        t=t,
        gap=np.zeros(1),
        v_normal=np.zeros(1),
        v_tangent=np.zeros((1, 2)),
        omega_normal=np.zeros(1),
        omega_tangent=np.zeros((1, 2)),
    )
    result = discover_modes(obs, InferenceParams(), seed=0)
    assert result.n_modes == 1
    assert result.labels.shape == (1,)
    assert result.labels[0] == 0


# --------------------------------------------------------------------------------------
# Scenario-backed test (MuJoCo required)
# --------------------------------------------------------------------------------------

mujoco = pytest.importorskip("mujoco")

from contact import geometry
import oracle  # noqa: E402  (after the skip guard)


def test_discover_modes_push_to_slide_finds_static_and_sliding():
    """Discovery finds static- and sliding-aligned modes on push_to_slide (§8/§9).

    ``push_to_slide`` builds the stick->slip guard of §7: a box rests (STATIC) until a
    ramped horizontal force breaks friction, after which it SLIDES. Handed only the
    noisy observable channel (generate -> observe, §9), the unsupervised HDP-HMM should
    run end to end and discover *both* a static-aligned and a sliding-aligned mode --
    rediscovering the two physical regimes without being told they exist.

    This is the weakest-claim scenario test by design (§8 honesty: the diagonal-Gaussian
    emission on channel magnitudes cleanly separates static vs. sliding, which differ in
    exactly one channel), so we assert only that both aligned modes are present.
    """
    sc = oracle.generate("push_to_slide", seed=SEED, hz=HZ)
    obs = geometry.observe(sc.moving, sc.support, sc.surface, sc.contact_point_local,
                           geometry=getattr(sc, "geometry", None))

    result = discover_modes(obs, InferenceParams(), seed=0)

    aligned = set(result.alignment.values())
    assert STATIC in aligned, f"no static-aligned mode discovered: {result.alignment}"
    assert SLIDING in aligned, f"no sliding-aligned mode discovered: {result.alignment}"

    # Determinism carries over to the scenario feature too (same seed => same labels).
    again = discover_modes(obs, InferenceParams(), seed=0)
    assert np.array_equal(result.labels, again.labels)
