"""Core geometry: overhang-facet detection and the two support-volume estimators.

Units and conventions
----------------------
- No physical units are assumed; all lengths/areas/volumes are in whatever
  units the input mesh uses (mm is typical for STL files from slicers/CAD).
- The build direction is the direction material is added as printing
  progresses -- equivalently, "down" (the direction support must span) is
  the *opposite* of the build direction. The default build direction is
  ``(0, 0, 1)`` (printing proceeds in +Z), so "down" is ``(0, 0, -1)`` and
  the build plate is the horizontal plane ``z = z_ground``.
- **Current limitation:** only build directions parallel to the Z axis
  (``(0, 0, 1)`` or ``(0, 0, -1)``) are supported. The parameter exists so
  this assumption is explicit and visible in the API rather than silently
  hard-coded, and so that arbitrary-direction support can be added later
  without changing the public signature. See ``README.md`` for why this
  wasn't extended to arbitrary directions in this pass.

Two estimators are provided:

``compute_naive_support_volume``
    The reference baseline from Shonkwiler et al. (IDETC-2026): for every
    overhanging facet, ``centroid_height x horizontally_projected_area``,
    summed. This is what the paper's own PySLM-based pipeline computes, and
    it explicitly ignores any intersection between the support column and
    the part itself.

``compute_integrated_support_volume``
    This repository's contribution: sample a grid over the union of
    overhanging facets' footprints, cast one ray per column through the
    *entire* mesh, and integrate only the open-air gaps between the
    overhang surface and the build plate. This is a grid approximation,
    not an exact CSG computation -- see ``README.md`` for convergence
    evidence and known limitations. It is deliberately never called "the
    true volume" anywhere in this codebase, only "integrated" or
    "corrected", to avoid overclaiming exactness it doesn't have.
"""

from __future__ import annotations

import numpy as np
import trimesh

_Z_AXIS = np.array([0.0, 0.0, 1.0])


def _resolve_down_direction(build_direction: tuple[float, float, float]) -> np.ndarray:
    """Validate and normalize a build direction, returning the "down" unit vector.

    Raises NotImplementedError for anything not parallel to the Z axis --
    see the module docstring for why arbitrary directions aren't supported yet.
    """
    d = np.asarray(build_direction, dtype=float)
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        raise ValueError(f"build_direction must be non-zero, got {build_direction!r}")
    d = d / norm

    alignment = abs(float(d @ _Z_AXIS))
    if alignment < 1 - 1e-6:
        raise NotImplementedError(
            f"build_direction={build_direction!r} is not parallel to the Z axis. "
            "This implementation currently only supports a +Z or -Z build axis "
            "(see the module docstring in supvol/raycast.py). Arbitrary build "
            "directions are a documented future extension, not silently assumed."
        )
    return -d  # "down" = opposite of the build direction


def find_overhanging_facets(
    mesh: trimesh.Trimesh,
    angle_threshold_deg: float = 50.0,
    build_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Facet indices whose normal is within `angle_threshold_deg` of straight down.

    Mirrors the IDETC/NAMRC papers' own definition: a facet needs support if its
    outward normal points mostly downward (i.e. it's the underside of the part).
    """
    if not (0.0 < angle_threshold_deg <= 90.0):
        raise ValueError(f"angle_threshold_deg must be in (0, 90], got {angle_threshold_deg}")

    down = _resolve_down_direction(build_direction)
    cos_thresh = np.cos(np.radians(angle_threshold_deg))
    cos_angle = mesh.face_normals @ down
    return np.nonzero(cos_angle > cos_thresh)[0]


def _footprint_sample_points(mesh: trimesh.Trimesh, facet_idx: np.ndarray, resolution: float):
    """Grid-sample the union of the 2D (XY) projections of the given facets.

    Rasterizes each triangle only within its own local bounding box (not the
    full model's bounding box against every triangle) so this scales to real
    meshes with tens of thousands of overhang facets. All triangles snap to a
    shared global grid phase so results from different triangles can be
    deduplicated by integer cell index -- otherwise overlapping facets would
    double-count shared cells.

    Note the grid has a fixed phase anchored to this facet set's own bounding
    box, not to the mesh's absolute coordinates -- see
    ``tests/test_grid_sensitivity.py`` for how much this can matter and why.

    Implementation note: per-triangle candidate cells are built with a single
    batched point-in-triangle test rather than one Python-level call per
    triangle. On a real mesh (3DBenchy, 35,312 overhang facets) the typical
    triangle only spans ~4-6 grid cells -- a trivial amount of actual math --
    but the old per-triangle Python loop still took ~11s purely from
    interpreter/numpy-call overhead repeated 35,312 times; this version does
    the same total point-in-triangle work in one vectorized call. See
    ``tests/test_footprint_sampling.py`` for a direct correctness comparison
    against the original per-triangle-loop reference implementation, kept
    there specifically so this optimization can't silently drift from it.

    Returns an (N, 2) array of sample point centers and the cell area (resolution**2).
    """
    tris_2d = mesh.vertices[mesh.faces[facet_idx]][:, :, :2]  # (F, 3, 2)
    if len(tris_2d) == 0:
        return np.empty((0, 2)), resolution * resolution
    global_origin = tris_2d.reshape(-1, 2).min(axis=0)

    t_min = tris_2d.min(axis=1)  # (F, 2)
    t_max = tris_2d.max(axis=1)  # (F, 2)
    ix_min = np.floor((t_min[:, 0] - global_origin[0]) / resolution).astype(np.int64)
    ix_max = np.ceil((t_max[:, 0] - global_origin[0]) / resolution).astype(np.int64)
    iy_min = np.floor((t_min[:, 1] - global_origin[1]) / resolution).astype(np.int64)
    iy_max = np.ceil((t_max[:, 1] - global_origin[1]) / resolution).astype(np.int64)

    nx = ix_max - ix_min + 1  # (F,) cells spanned in x per triangle
    ny = iy_max - iy_min + 1  # (F,) cells spanned in y per triangle
    counts = nx * ny  # (F,) total candidate cells per triangle
    total = int(counts.sum())
    if total == 0:
        return np.empty((0, 2)), resolution * resolution

    # Build, for every (triangle, candidate cell) pair, which triangle it
    # belongs to and its position within that triangle's local nx-by-ny grid
    # -- a "ragged range" constructed via repeat rather than a Python loop.
    tri_of_cell = np.repeat(np.arange(len(tris_2d)), counts)
    cell_offset_in_tri = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    nx_per_cell = np.repeat(nx, counts)
    # x is the faster-varying (inner) axis, matching np.meshgrid(ix, iy)'s
    # default 'xy' indexing + .ravel() order in the reference implementation
    # (see tests/test_footprint_sampling.py) -- getting this backwards was a
    # real bug caught by that test: it silently dropped cells whenever a
    # triangle's bbox wasn't square (nx != ny).
    local_x = cell_offset_in_tri % nx_per_cell
    local_y = cell_offset_in_tri // nx_per_cell

    cand_ix = np.repeat(ix_min, counts) + local_x
    cand_iy = np.repeat(iy_min, counts) + local_y
    cand_xy = global_origin + (np.column_stack([cand_ix, cand_iy]) + 0.5) * resolution

    inside = _points_in_triangles_batched(cand_xy, tris_2d[tri_of_cell])

    if not np.any(inside):
        return np.empty((0, 2)), resolution * resolution

    cells = np.unique(np.column_stack([cand_ix[inside], cand_iy[inside]]), axis=0)
    points = global_origin + (cells + 0.5) * resolution
    return points, resolution * resolution


def _points_in_triangle(pts: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Vectorized point-in-triangle test: many points against one shared triangle.

    Kept as the simple reference implementation (used directly by
    ``tests/test_footprint_sampling.py`` as a ground truth to check the
    batched version below against), even though the hot path in
    ``_footprint_sample_points`` now uses ``_points_in_triangles_batched``
    instead for performance.
    """
    a, b, c = tri
    v0, v1 = c - a, b - a
    v2 = pts - a
    dot00 = v0 @ v0
    dot01 = v0 @ v1
    dot02 = v2 @ v0
    dot11 = v1 @ v1
    dot12 = v2 @ v1
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return np.zeros(len(pts), dtype=bool)
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)


def _points_in_triangles_batched(pts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Vectorized point-in-triangle test: each point against its own triangle.

    Same barycentric math as ``_points_in_triangle``, but ``tris`` is
    ``(N, 3, 2)`` -- one triangle per point in ``pts`` (``(N, 2)``) -- so an
    arbitrary number of (point, triangle) pairs can be tested in one call
    instead of one Python-level call per shared triangle. Degenerate
    (zero-area) triangles correctly test as containing no points, same as
    ``_points_in_triangle``.
    """
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    v0, v1 = c - a, b - a
    v2 = pts - a
    dot00 = np.einsum("ij,ij->i", v0, v0)
    dot01 = np.einsum("ij,ij->i", v0, v1)
    dot02 = np.einsum("ij,ij->i", v2, v0)
    dot11 = np.einsum("ij,ij->i", v1, v1)
    dot12 = np.einsum("ij,ij->i", v2, v1)
    denom = dot00 * dot11 - dot01 * dot01

    degenerate = np.abs(denom) < 1e-12
    safe_denom = np.where(degenerate, 1.0, denom)
    inv = 1.0 / safe_denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
    return inside & ~degenerate


def _column_needs_support(hit_z: np.ndarray, hit_nz: np.ndarray, z_ground: float = 0.0) -> float:
    """Self-intersection-aware support height for one vertical column.

    hit_z:  z-coordinates of every ray/mesh intersection along this column.
    hit_nz: z-component of each intersection's face normal (classifies the
            hit as the ray entering solid material, nz > 0, or exiting it,
            nz < 0, for a ray travelling in -z).

    Uses a solid-depth counter rather than a plain in/out boolean so that
    exactly-coincident surfaces (e.g. two parts touching flush) net out to
    zero instead of depending on tie-break ordering.
    """
    order = np.argsort(-hit_z)
    z, nz = hit_z[order], hit_nz[order]

    depth = 0
    support = 0.0
    gap_start = None

    for zi, n in zip(z, nz):
        was_outside = depth == 0
        depth += 1 if n > 0 else -1
        now_outside = depth == 0

        if was_outside and not now_outside and gap_start is not None:
            support += max(gap_start - zi, 0.0)
            gap_start = None
        elif now_outside and not was_outside:
            gap_start = zi

    if gap_start is not None:
        support += max(gap_start - z_ground, 0.0)
    return support


def compute_naive_support_volume(
    mesh: trimesh.Trimesh,
    angle_threshold_deg: float = 50.0,
    z_ground: float = 0.0,
    build_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    overhang_idx: np.ndarray | None = None,
) -> float:
    """The paper's reference baseline: sum of centroid_height x projected_area.

    This is exactly the calculation described in Shonkwiler et al. (IDETC-2026):
    it ignores any intersection between a support column and the part, so it
    is expected to over-estimate whenever the part's own geometry would
    already occupy part of that column (see ``compute_integrated_support_volume``).

    "Height" is measured from ``z_ground``, not from the coordinate origin --
    this matters whenever the build plate isn't literally at z=0.

    ``overhang_idx`` lets a caller reuse an already-computed facet selection
    (e.g. ``compute_support_volume`` does this) instead of recomputing it.
    """
    if overhang_idx is None:
        overhang_idx = find_overhanging_facets(mesh, angle_threshold_deg, build_direction)
    if len(overhang_idx) == 0:
        return 0.0

    down = _resolve_down_direction(build_direction)
    tri_areas_3d = mesh.area_faces[overhang_idx]
    normals = mesh.face_normals[overhang_idx]
    # projected area = 3D area * |cos(angle to build axis)|
    proj_areas = tri_areas_3d * np.abs(normals @ down)
    # Height above the build plate, measured along the "up" direction
    # (opposite of down). The build plate is always the horizontal plane
    # z=z_ground (only Z-axis builds are supported -- see module docstring),
    # but which side of that plane counts as "above" flips with build_direction:
    # z_ground - z when down is +Z (the flipped case), z - z_ground otherwise.
    up_sign = -1.0 if down[2] > 0 else 1.0
    heights = up_sign * (mesh.triangles_center[overhang_idx][:, 2] - z_ground)
    return float(np.sum(np.clip(heights, 0.0, None) * proj_areas))


def compute_integrated_support_volume(
    mesh: trimesh.Trimesh,
    angle_threshold_deg: float = 50.0,
    resolution: float = 0.05,
    z_ground: float = 0.0,
    build_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    overhang_idx: np.ndarray | None = None,
) -> tuple[float, int]:
    """This repository's ray-cast, column-integrated support volume estimate.

    Grid-samples the union footprint of overhanging facets, casts one
    vertical ray per sample column through the whole mesh, and sums the
    open-air gap length between the overhang surface and ``z_ground`` in
    each column, multiplied by cell area. This corrects both:

    - self-intersection: the part's own geometry occupying part of a column
      (the shelf/pillar case), and
    - facet-to-facet overlap: two separate overhangs projecting onto the
      same footprint (the naive per-facet sum double-counts this).

    Returns ``(integrated_volume, n_sample_columns)``. This is a grid
    approximation whose error shrinks with ``resolution`` -- see
    ``scripts/convergence_study.py`` and ``tests/test_convergence.py`` for
    quantified evidence, and do not treat this as an exact CSG result.
    """
    if resolution <= 0:
        raise ValueError(f"resolution must be > 0, got {resolution}")

    if overhang_idx is None:
        overhang_idx = find_overhanging_facets(mesh, angle_threshold_deg, build_direction)
    if len(overhang_idx) == 0:
        return 0.0, 0

    down = _resolve_down_direction(build_direction)
    # _column_needs_support's sort/gap arithmetic is written assuming rays
    # travel in -Z (i.e. "down" decreases z). When build_direction requests
    # the flipped case (down is +Z), we negate z throughout this function's
    # internal bookkeeping so that convention holds again -- the final
    # volume is unaffected by this axis flip, only the intermediate z values
    # are. (This bug -- silently returning 0 for the flipped direction,
    # because both the ray origin and the gap arithmetic below still assumed
    # -Z regardless of `down` -- shipped once already; see git history and
    # tests/test_build_direction.py, which exists specifically so it can't
    # regress silently again.)
    flip = down[2] > 0

    points_xy, cell_area = _footprint_sample_points(mesh, overhang_idx, resolution)
    if len(points_xy) == 0:
        return 0.0, 0

    # Rays must originate on the far side of the mesh from the direction of
    # travel: above the mesh when traveling -Z (the default), below it when
    # traveling +Z (the flipped case) -- getting this wrong means the ray
    # starts past the mesh already moving away from it and never hits anything.
    if not flip:
        origin_z = float(mesh.vertices[:, 2].max()) + 1.0
    else:
        origin_z = float(mesh.vertices[:, 2].min()) - 1.0
    ray_origins = np.column_stack([points_xy, np.full(len(points_xy), origin_z)])
    ray_directions = np.tile(down, (len(points_xy), 1))

    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins, ray_directions, multiple_hits=True
    )

    effective_z_ground = -z_ground if flip else z_ground

    integrated_volume = 0.0
    if len(locations) > 0:
        hit_normals_z = mesh.face_normals[index_tri][:, 2]
        hit_z_all = locations[:, 2]
        if flip:
            hit_z_all = -hit_z_all
            hit_normals_z = -hit_normals_z

        # Group hits by column via a single sort instead of one `index_ray ==
        # col` mask scan per column (the latter is O(n_columns x n_hits) --
        # this was the dominant cost on real meshes and fine synthetic grids;
        # see the performance note in README.md). _column_needs_support
        # itself is unchanged -- only how its inputs are grouped is different,
        # to keep this a low-risk optimization of an already-validated
        # algorithm rather than a rewrite of it.
        order = np.argsort(index_ray, kind="stable")
        sorted_ray_idx = index_ray[order]
        sorted_z = hit_z_all[order]
        sorted_nz = hit_normals_z[order]
        boundaries = np.nonzero(np.diff(sorted_ray_idx))[0] + 1
        for z_group, nz_group in zip(np.split(sorted_z, boundaries), np.split(sorted_nz, boundaries)):
            integrated_volume += _column_needs_support(z_group, nz_group, effective_z_ground)

    integrated_volume *= cell_area
    return integrated_volume, len(points_xy)
