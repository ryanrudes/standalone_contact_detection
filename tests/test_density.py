"""Executable derivation for the emission channel densities (THEORY.md §3 & §4).

Each contact mode in ``contact.emissions`` is a composition of the encapsulated ``Density``
objects; this test verifies, WITHOUT running the detector, the two load-bearing properties
those densities must have for the cross-state likelihood ratio of §4 to stay
calibrated: (a) each is a PROPER density (integrates to 1 over its support), and (b) the
documented limit laws hold (e.g. the sliding ring collapses to the isotropic Gaussian as
speed -> 0). This turns the modules' prose normalization/limit claims into checks.
"""

import numpy as np
import pytest
from scipy import integrate

from contact.emissions import (
    IsoNormal2D,
    MixZero1D,
    MixZero2D,
    Normal1D,
    OffsetMagnitude1D,
    OffsetMagnitude2D,
    SplitNormalGap,
    UniformClearance,
)


def _mass_1d(d, lo, hi):
    val, _ = integrate.quad(lambda x: float(np.exp(d.logpdf(np.array([x]))[0])), lo, hi, limit=400)
    return val


def _mass_2d(d, hi):  # isotropic: logpdf depends only on r = |x|; integrate p(r) * 2*pi*r dr
    val, _ = integrate.quad(
        lambda r: float(np.exp(d.logpdf(np.array([[r, 0.0]]))[0])) * 2.0 * np.pi * r, 0.0, hi, limit=400
    )
    return val


@pytest.mark.parametrize(
    "density, lo, hi",
    [
        (Normal1D(0.0, 0.7), -10.0, 10.0),
        (SplitNormalGap(0.002, 0.0015, 0.006), -0.1, 0.1),
        (OffsetMagnitude1D(1.0, 0.3), -5.0, 5.0),
        (MixZero1D(0.3, 3.0, 0.25), -40.0, 40.0),
        (UniformClearance(2.0), 0.0, 2.0),
    ],
)
def test_density_1d_is_proper(density, lo, hi):
    assert _mass_1d(density, lo, hi) == pytest.approx(1.0, abs=3e-3)


@pytest.mark.parametrize(
    "density, hi",
    [
        (IsoNormal2D(0.5), 8.0),
        (OffsetMagnitude2D(0.15, 0.1), 3.0),
        (MixZero2D(0.3, 3.0, 0.25), 40.0),
    ],
)
def test_density_2d_is_proper(density, hi):
    assert _mass_2d(density, hi) == pytest.approx(1.0, abs=3e-3)


def test_ring_collapses_to_isotropic_as_speed_to_zero():
    x = np.array([[0.05, -0.2], [0.3, 0.1], [-0.4, 0.25]])
    assert np.allclose(OffsetMagnitude2D(0.0, 0.5).logpdf(x), IsoNormal2D(0.5).logpdf(x), atol=1e-12)


def test_offset1d_collapses_to_normal_as_speed_to_zero():
    x = np.array([-0.3, 0.0, 0.4, 1.1])
    assert np.allclose(OffsetMagnitude1D(0.0, 0.7).logpdf(x), Normal1D(0.0, 0.7).logpdf(x), atol=1e-12)


def test_split_normal_collapses_to_normal_when_sides_equal():
    x = np.array([-0.3, 0.0, 0.4, 1.1])
    assert np.allclose(SplitNormalGap(0.0, 0.5, 0.5).logpdf(x), Normal1D(0.0, 0.5).logpdf(x), atol=1e-12)


def test_mixture_collapses_to_tight_as_weight_to_zero():
    x = np.array([-0.3, 0.0, 0.4, 1.1])
    with np.errstate(divide="ignore"):  # w=0 -> log(0) by design; logaddexp handles it
        got = MixZero1D(0.4, 3.0, 0.0).logpdf(x)
    assert np.allclose(got, Normal1D(0.0, 0.4).logpdf(x), atol=1e-12)
