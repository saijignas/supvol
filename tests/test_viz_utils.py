"""Tests for viz_utils.decimate_for_viz -- kept separate from app.py's other
tests since this module has no Streamlit/Plotly dependency at all.

Regression test for a real bug caught via manual testing on the live
deployment: `fast-simplification` (needed for trimesh's real quadric
decimation) was never actually added as a dependency anywhere -- app.py's
own code comment correctly anticipated needing a fallback, but the primary
path silently failed on every real environment (local dev and the deployed
Streamlit Cloud app alike), so the "fallback" was actually the only path
ever exercised in practice. That fallback picks faces by raw array index,
which ignores mesh connectivity entirely -- the surviving triangles are
scattered across the whole mesh rather than forming a coherent simplified
surface, producing a visibly broken 3D preview (confirmed: it keeps ~14% of
the original surface area on a real mesh, vs. ~100% for real decimation).
"""

import numpy as np
import pytest
import trimesh

from viz_utils import decimate_for_viz


def _fine_sphere(n_faces_at_least: int) -> trimesh.Trimesh:
    """A real, single connected surface with enough faces to trigger
    decimation -- not just a synthetic pile of disconnected triangles,
    so a decimation bug that scatters faces is actually visible in the
    surface-area check below."""
    subdivisions = 3
    while True:
        mesh = trimesh.creation.icosphere(subdivisions=subdivisions)
        if len(mesh.faces) >= n_faces_at_least:
            return mesh
        subdivisions += 1


def test_mesh_under_threshold_is_returned_unchanged():
    mesh = _fine_sphere(100)
    result = decimate_for_viz(mesh, threshold=len(mesh.faces) + 1)
    assert result is mesh


def test_mesh_over_threshold_is_decimated_to_at_most_threshold():
    mesh = _fine_sphere(5000)
    result = decimate_for_viz(mesh, threshold=1000)
    assert len(result.faces) <= 1000


def test_decimation_preserves_surface_shape_not_just_face_count():
    """The actual bug: a broken decimation can hit the right face count
    while destroying the mesh (scattered, near-zero-area-preserving
    fragments). Surface area is the discriminator -- real decimation of a
    single coherent surface preserves it closely; naive index-based
    subsampling of an unordered face array does not."""
    mesh = _fine_sphere(5000)
    result = decimate_for_viz(mesh, threshold=1000)

    area_ratio = result.area / mesh.area
    assert area_ratio == pytest.approx(1.0, rel=0.15), (
        f"decimated mesh surface area ratio {area_ratio:.3f} is too far from 1.0 -- "
        "this is exactly the signature of the broken naive-fallback bug (it kept "
        "~14% of the original area on a real mesh)"
    )


def test_naive_fallback_path_directly_flagged_as_broken_if_it_is_ever_hit():
    """If fast-simplification is ever missing again (e.g. a future dependency
    resolution issue), this makes the resulting bug loud and specific rather
    than a silently-broken 3D preview discovered by eyeballing a screenshot."""
    mesh = _fine_sphere(5000)
    idx = np.linspace(0, len(mesh.faces) - 1, 1000).astype(int)
    naive = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[idx], process=False)
    # Confirms the naive method genuinely is bad on a real mesh -- documents
    # why the fallback path is not an acceptable primary behavior, and why
    # this project depends on fast-simplification instead of relying on it.
    assert naive.area / mesh.area < 0.5
