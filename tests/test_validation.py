"""Input validation and mesh-diagnostics tests.

These check that structurally invalid inputs raise clear exceptions, that
mesh-quality issues warn rather than raise or silently repair, and that the
diagnostics themselves are computed correctly against known small meshes.
"""

import numpy as np
import pytest
import trimesh

from supvol import compute_support_volume
from supvol.fixtures import simple_overhang
from supvol.validation import compute_mesh_diagnostics, validate_mesh_inputs


def _empty_mesh():
    return trimesh.Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=int))


def _zero_face_mesh():
    # vertices but no faces
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return trimesh.Trimesh(vertices=verts, faces=np.empty((0, 3), dtype=int))


def _non_finite_mesh():
    mesh = simple_overhang()[0].copy()
    mesh.vertices[0, 0] = np.nan
    return mesh


def _mesh_with_zero_area_face():
    # a normal box, plus one degenerate (zero-area) triangle appended by hand
    mesh = simple_overhang()[0].copy()
    degenerate_vertex_idx = [0, 0, 0]  # a triangle with all three corners the same point -> zero area
    new_face = np.array([degenerate_vertex_idx])
    mesh.faces = np.vstack([mesh.faces, new_face])
    return mesh


def test_empty_mesh_raises():
    with pytest.raises(ValueError, match="no vertices"):
        validate_mesh_inputs(_empty_mesh(), resolution=0.1, angle_threshold_deg=50.0)


def test_zero_face_mesh_raises():
    with pytest.raises(ValueError, match="no faces"):
        validate_mesh_inputs(_zero_face_mesh(), resolution=0.1, angle_threshold_deg=50.0)


def test_non_positive_resolution_raises():
    mesh, _ = simple_overhang()
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="resolution"):
            validate_mesh_inputs(mesh, resolution=bad, angle_threshold_deg=50.0)


@pytest.mark.parametrize("bad_angle", [0.0, -10.0, 91.0, 180.0])
def test_invalid_angle_threshold_raises(bad_angle):
    mesh, _ = simple_overhang()
    with pytest.raises(ValueError, match="angle_threshold_deg"):
        validate_mesh_inputs(mesh, resolution=0.1, angle_threshold_deg=bad_angle)


def test_non_finite_vertices_raises():
    with pytest.raises(ValueError, match="non-finite"):
        validate_mesh_inputs(_non_finite_mesh(), resolution=0.1, angle_threshold_deg=50.0)


def test_zero_area_face_warns_not_raises():
    mesh = _mesh_with_zero_area_face()
    with pytest.warns(UserWarning, match="degenerate"):
        diagnostics = validate_mesh_inputs(mesh, resolution=0.1, angle_threshold_deg=50.0)
    assert diagnostics.n_zero_area_faces >= 1


def test_non_watertight_warns_not_raises():
    # simple_overhang is a single closed box -> watertight, so build a mesh
    # that isn't by dropping one face.
    mesh, _ = simple_overhang()
    mesh = mesh.copy()
    mesh.faces = mesh.faces[:-1]
    mesh.process()
    assert not mesh.is_watertight
    with pytest.warns(UserWarning, match="watertight"):
        validate_mesh_inputs(mesh, resolution=0.1, angle_threshold_deg=50.0)


def test_validation_does_not_mutate_mesh():
    """Explicitly documented behavior: no automatic/silent repair."""
    mesh, _ = simple_overhang()
    mesh = mesh.copy()
    v_before = mesh.vertices.copy()
    f_before = mesh.faces.copy()
    with pytest.warns(UserWarning) if not mesh.is_watertight else _no_warn_ctx():
        validate_mesh_inputs(mesh, resolution=0.1, angle_threshold_deg=50.0)
    np.testing.assert_array_equal(mesh.vertices, v_before)
    np.testing.assert_array_equal(mesh.faces, f_before)


class _no_warn_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_diagnostics_on_clean_box():
    """A trimesh-generated box should be watertight with no zero-area or
    non-manifold issues -- a known-good baseline for the diagnostics themselves."""
    mesh, _ = simple_overhang()
    diagnostics = compute_mesh_diagnostics(mesh)

    assert diagnostics.n_faces == len(mesh.faces)
    assert diagnostics.n_vertices == len(mesh.vertices)
    assert diagnostics.watertight is True
    assert diagnostics.n_zero_area_faces == 0
    assert diagnostics.n_nonmanifold_edges == 0
    assert diagnostics.n_boundary_edges == 0  # watertight => no boundary edges


def test_compute_support_volume_exposes_diagnostics():
    mesh, _ = simple_overhang()
    result = compute_support_volume(mesh, resolution=0.1)
    assert result.diagnostics.watertight is True
    assert result.diagnostics.n_faces == len(mesh.faces)


def test_validate_false_skips_warnings():
    mesh, _ = simple_overhang()
    mesh = mesh.copy()
    mesh.faces = mesh.faces[:-1]
    mesh.process()

    with warnings_should_not_fire():
        compute_support_volume(mesh, resolution=0.2, validate=False)


class warnings_should_not_fire:
    def __enter__(self):
        import warnings

        self._catch = warnings.catch_warnings(record=True)
        self._records = self._catch.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, *a):
        self._catch.__exit__(*a)
        assert len(self._records) == 0, f"expected no warnings, got {[str(w.message) for w in self._records]}"
        return False
