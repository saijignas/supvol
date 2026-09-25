"""Tests for the build_direction parameter -- added after discovering, by
manual testing rather than an automated test, that the flipped direction
(0,0,-1) silently returned 0 for everything. Root cause: both the ray
origin and the column-integration sign convention in supvol/raycast.py
still hard-coded a -Z-down assumption despite the parameter existing. This
file exists specifically so that regression can't happen silently again.
"""

import numpy as np
import pytest
import trimesh

from supvol import compute_support_volume
from supvol.fixtures import shelf_and_pillar, simple_overhang
from supvol.raycast import find_overhanging_facets


def test_non_z_axis_direction_raises_not_implemented():
    mesh, _ = simple_overhang()
    for bad_direction in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 1.0)]:
        with pytest.raises(NotImplementedError, match="Z axis"):
            compute_support_volume(mesh, build_direction=bad_direction)


def test_zero_build_direction_raises_value_error():
    mesh, _ = simple_overhang()
    with pytest.raises(ValueError, match="non-zero"):
        compute_support_volume(mesh, build_direction=(0.0, 0.0, 0.0))


def test_flipped_direction_matches_default_with_mirrored_ground():
    """Simple no-self-intersection case: a floating box's overhang direction
    flips with build_direction, and a z_ground placed the mirrored distance
    away on the other side must reproduce the same volume as the default
    orientation (5-unit gap either way)."""
    mesh, truth = simple_overhang()  # box z=5..7, default: bottom overhang, gap to z_ground=0 is 5

    default_result = compute_support_volume(mesh, resolution=0.05)
    # mirrored: top face becomes the overhang; plate placed 5 units above the
    # box's top (z=12) reproduces the same 5-unit gap in the opposite direction
    flipped_result = compute_support_volume(
        mesh, resolution=0.05, build_direction=(0.0, 0.0, -1.0), z_ground=12.0
    )

    assert flipped_result.naive_volume == pytest.approx(default_result.naive_volume, rel=1e-9)
    assert flipped_result.integrated_volume == pytest.approx(default_result.integrated_volume, rel=1e-9)
    assert flipped_result.integrated_volume == pytest.approx(truth["true_support_volume"], rel=0.02)


def _mirror_z(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Mirror a mesh through z=0. Negating one axis flips triangle winding
    (and therefore face normals), so faces must be re-wound to keep outward
    normals correct after the flip."""
    m = mesh.copy()
    m.vertices[:, 2] *= -1
    m.faces = m.faces[:, ::-1]
    m.process()
    return m


def test_flipped_direction_still_detects_self_intersection():
    """The harder case: mirror the shelf-and-pillar self-intersection fixture
    through z=0 (a pillar hanging down from an overhanging shelf above it
    becomes, after mirroring, a pillar rising up to an overhanging shelf
    below it -- i.e. exactly the flipped-build-direction version of the same
    physical self-intersection problem) and confirm the ray-cast integration
    still finds the same correction (640 -> 600) under build_direction=(0,0,-1),
    not just that it returns *some* nonzero, plausible-looking number.
    """
    mesh, truth = shelf_and_pillar()
    mirrored = _mirror_z(mesh)

    overhang_idx = find_overhanging_facets(mirrored, build_direction=(0.0, 0.0, -1.0))
    assert len(overhang_idx) > 0
    # after mirroring, the overhang facet (originally the shelf's bottom,
    # normal (0,0,-1)) now has normal (0,0,1)
    assert np.allclose(mirrored.face_normals[overhang_idx][:, 2], 1.0, atol=1e-6)

    result = compute_support_volume(
        mirrored, resolution=0.05, build_direction=(0.0, 0.0, -1.0), z_ground=0.0
    )

    assert result.naive_volume == pytest.approx(truth["naive_support_volume"], rel=1e-6)
    assert result.naive_volume > result.integrated_volume, (
        "must still detect self-intersection after mirroring + flipping build_direction"
    )
    rel_err = abs(result.integrated_volume - truth["true_support_volume"]) / truth["true_support_volume"]
    assert rel_err < 0.02, f"rel_err={rel_err:.2%}, integrated={result.integrated_volume}"
