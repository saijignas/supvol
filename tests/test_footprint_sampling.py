"""Correctness check for the vectorized _footprint_sample_points against a
simple, obviously-correct-by-inspection reference (the original per-triangle
Python loop this replaced, for performance -- see the docstring in
supvol/raycast.py). Kept as a permanent regression test, not a one-off
manual check, specifically because this repo has already shipped one
optimization-shaped bug (the build_direction ray origin) that a manual
check alone didn't catch until poked at directly.
"""

import numpy as np
import trimesh

from supvol.fixtures import overlapping_overhangs, shelf_and_pillar, simple_overhang, tilted_overhang
from supvol.raycast import _footprint_sample_points as vectorized_sample_points
from supvol.raycast import _points_in_triangle, find_overhanging_facets


def _reference_footprint_sample_points(mesh, facet_idx, resolution):
    """The original, simple (slow) per-triangle-loop implementation. Not
    used by the library anymore -- exists only as a trusted-by-inspection
    ground truth for the vectorized version to be checked against."""
    tris_2d = mesh.vertices[mesh.faces[facet_idx]][:, :, :2]
    if len(tris_2d) == 0:
        return np.empty((0, 2))
    global_origin = tris_2d.reshape(-1, 2).min(axis=0)

    cell_set = set()
    for tri in tris_2d:
        t_min, t_max = tri.min(axis=0), tri.max(axis=0)
        ix_min = int(np.floor((t_min[0] - global_origin[0]) / resolution))
        ix_max = int(np.ceil((t_max[0] - global_origin[0]) / resolution))
        iy_min = int(np.floor((t_min[1] - global_origin[1]) / resolution))
        iy_max = int(np.ceil((t_max[1] - global_origin[1]) / resolution))

        ix, iy = np.arange(ix_min, ix_max + 1), np.arange(iy_min, iy_max + 1)
        if len(ix) == 0 or len(iy) == 0:
            continue
        gix, giy = np.meshgrid(ix, iy)
        cand_idx = np.column_stack([gix.ravel(), giy.ravel()])
        cand_xy = global_origin + (cand_idx + 0.5) * resolution

        inside = _points_in_triangle(cand_xy, tri)
        for cix, ciy in cand_idx[inside]:
            cell_set.add((int(cix), int(ciy)))

    if not cell_set:
        return np.empty((0, 2))
    cells = np.array(sorted(cell_set))
    return global_origin + (cells + 0.5) * resolution


def _assert_same_point_set(a: np.ndarray, b: np.ndarray, atol=1e-9):
    """Order doesn't matter (both build a set of grid cells); the actual
    (x, y) coordinates must match exactly, up to floating point noise."""
    assert len(a) == len(b), f"different number of sample points: {len(a)} vs {len(b)}"
    a_sorted = a[np.lexsort((a[:, 1], a[:, 0]))]
    b_sorted = b[np.lexsort((b[:, 1], b[:, 0]))]
    np.testing.assert_allclose(a_sorted, b_sorted, atol=atol)


def _yaw(mesh, degrees):
    m = mesh.copy()
    center = m.bounds.mean(axis=0)
    m.apply_translation(-center)
    R = trimesh.transformations.rotation_matrix(np.radians(degrees), [0, 0, 1])
    m.apply_transform(R)
    m.apply_translation(center)
    return m


CASES = {
    "shelf_and_pillar": lambda: shelf_and_pillar()[0],
    "simple_overhang": lambda: simple_overhang()[0],
    "tilted_overhang": lambda: tilted_overhang()[0],
    "overlapping_overhangs": lambda: overlapping_overhangs()[0],
    "shelf_and_pillar_rotated": lambda: _yaw(shelf_and_pillar()[0], 31.0),
}


def test_vectorized_matches_reference_per_case():
    for name, build_mesh in CASES.items():
        mesh = build_mesh()
        idx = find_overhanging_facets(mesh)
        for resolution in (0.2, 0.05):
            ref_points = _reference_footprint_sample_points(mesh, idx, resolution)
            vec_points, cell_area = vectorized_sample_points(mesh, idx, resolution)
            assert cell_area == resolution * resolution
            _assert_same_point_set(ref_points, vec_points)
        print(f"OK: {name}")  # noqa: T201


def test_empty_facet_set_returns_empty():
    mesh, _ = simple_overhang()
    empty_idx = np.array([], dtype=int)
    points, cell_area = vectorized_sample_points(mesh, empty_idx, 0.1)
    assert len(points) == 0
    assert cell_area == 0.1 * 0.1
