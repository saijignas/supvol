"""Rotation tests: confirm the algorithm isn't accidentally relying on
axis-aligned footprints.

Rotating a part about the vertical (Z) axis re-orients its footprint in the
XY plane (triangles are no longer axis-aligned) without changing any
height/overhang relationship -- normals keep the same Z-component, and every
facet's height above the build plate is unchanged. Ground truth is therefore
identical to the un-rotated case, which makes this an analytically
understood case, not just a smoke test.
"""

import numpy as np
import pytest
import trimesh

from supvol import compute_support_volume
from supvol.fixtures import shelf_and_pillar, simple_overhang

REL_TOL = 0.02
YAW_ANGLES_DEG = [0.0, 15.0, 37.0, 90.0, 123.0]


def _yaw(mesh: trimesh.Trimesh, degrees: float) -> trimesh.Trimesh:
    """Rotate a mesh about the Z axis through its own centroid, in place on a copy."""
    m = mesh.copy()
    center = m.bounds.mean(axis=0)
    m.apply_translation(-center)
    R = trimesh.transformations.rotation_matrix(np.radians(degrees), [0, 0, 1])
    m.apply_transform(R)
    m.apply_translation(center)
    return m


@pytest.mark.parametrize("degrees", YAW_ANGLES_DEG)
def test_simple_overhang_invariant_under_yaw(degrees):
    mesh, truth = simple_overhang()
    rotated = _yaw(mesh, degrees)
    result = compute_support_volume(rotated, resolution=0.05)

    rel_err = abs(result.integrated_volume - truth["true_support_volume"]) / truth["true_support_volume"]
    assert rel_err < REL_TOL, f"yaw={degrees} deg: rel_err={rel_err:.2%}"


@pytest.mark.parametrize("degrees", YAW_ANGLES_DEG)
def test_shelf_and_pillar_self_intersection_invariant_under_yaw(degrees):
    """The harder case: self-intersection correction must still hold once the
    pillar's footprint is no longer axis-aligned -- this is the case most
    likely to expose an accidental axis-aligned assumption, since it depends
    on precisely which grid cells fall inside a now-rotated 2x2 square."""
    mesh, truth = shelf_and_pillar()
    rotated = _yaw(mesh, degrees)
    result = compute_support_volume(rotated, resolution=0.05)

    rel_err = abs(result.integrated_volume - truth["true_support_volume"]) / truth["true_support_volume"]
    assert result.naive_volume > result.integrated_volume, f"yaw={degrees}: must still detect self-intersection"
    assert rel_err < REL_TOL, f"yaw={degrees} deg: rel_err={rel_err:.2%}, integrated={result.integrated_volume}"
