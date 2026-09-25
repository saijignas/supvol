"""Real pytest assertions covering the four synthetic validation cases:
self-intersection, a no-op sanity check, non-axis-aligned geometry, and
facet-to-facet footprint overlap. Ground truth for each is hand-computed in
supvol/fixtures.py, independent of the code under test.
"""

import pytest

from supvol.fixtures import (
    overlapping_overhangs,
    shelf_and_pillar,
    simple_overhang,
    tilted_overhang,
)
from supvol.raycast import compute_support_volume

REL_TOL = 0.02  # 2% -- generous enough to absorb grid-discretization error


def _relative_error(computed, truth):
    if truth == 0:
        return abs(computed)
    return abs(computed - truth) / truth


def test_shelf_and_pillar_self_intersection():
    """A shelf overhanging a pillar it's resting on -- the naive method should
    overestimate by exactly the pillar's own volume, and our method should not."""
    mesh, truth = shelf_and_pillar()
    naive, true, n = compute_support_volume(mesh, resolution=0.05)

    assert n > 0
    assert naive == pytest.approx(truth["naive_support_volume"], rel=1e-6)
    assert naive > true, "naive method must overestimate once self-intersection is ignored"
    assert _relative_error(true, truth["true_support_volume"]) < REL_TOL
    assert naive - true == pytest.approx(truth["pillar_volume"], rel=REL_TOL)


def test_simple_overhang_no_intersection():
    """Sanity check: with nothing else in the scene, our method must reduce
    exactly to the naive answer -- proves the fix doesn't 'invent' corrections
    where none are needed."""
    mesh, truth = simple_overhang()
    naive, true, n = compute_support_volume(mesh, resolution=0.05)

    assert n > 0
    assert naive == pytest.approx(true, rel=1e-9)
    assert _relative_error(true, truth["true_support_volume"]) < REL_TOL


def test_tilted_overhang_angle_math():
    """A non-axis-aligned overhang with no self-intersection: naive and true
    should still agree with each other (this tests the projected-area /
    column-integration math generalizes past flat horizontal facets, not the
    self-intersection correction itself)."""
    mesh, truth = tilted_overhang()
    naive, true, n = compute_support_volume(mesh, resolution=0.02)

    assert n > 0
    assert _relative_error(naive, truth["true_support_volume"]) < REL_TOL
    assert _relative_error(true, truth["true_support_volume"]) < REL_TOL


def test_overlapping_overhangs_facet_to_facet():
    """Two separate floating boxes with overlapping footprints but a real air
    gap between them -- the naive per-facet sum double-counts the shared
    region; our column-based method must not."""
    mesh, truth = overlapping_overhangs()
    naive, true, n = compute_support_volume(mesh, resolution=0.05)

    assert n > 0
    assert naive == pytest.approx(truth["naive_support_volume"], rel=1e-6)
    assert naive > true, "naive method must overestimate the overlapping region"
    assert _relative_error(true, truth["true_support_volume"]) < REL_TOL
