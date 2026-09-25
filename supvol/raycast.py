"""Core algorithm: per-column ray-cast height-field support volume calculation.

Replaces the naive `centroid_height x projected_area` estimate (which assumes
a solid column of support material from every overhanging facet straight down
to the build plate) with a per-sample-column integration that correctly
subtracts any of the part's own solid material that already occupies part of
that column -- the self-intersection case the naive method ignores.
"""

import numpy as np
import trimesh

RAY_DIR = np.array([0.0, 0.0, -1.0])


def find_overhanging_facets(mesh: trimesh.Trimesh, angle_threshold_deg: float = 50.0):
    """Facet indices whose normal is within `angle_threshold_deg` of straight down.

    Mirrors the IDETC/NAMRC papers' own definition: a facet needs support if its
    outward normal points mostly downward (i.e. it's the underside of the part).
    """
    down = np.array([0.0, 0.0, -1.0])
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

    Returns an (N, 2) array of sample point centers and the cell area (resolution**2).
    """
    tris_2d = mesh.vertices[mesh.faces[facet_idx]][:, :, :2]  # (F, 3, 2)
    global_origin = tris_2d.reshape(-1, 2).min(axis=0)

    cell_set = set()
    for tri in tris_2d:
        t_min = tri.min(axis=0)
        t_max = tri.max(axis=0)
        ix_min = int(np.floor((t_min[0] - global_origin[0]) / resolution))
        ix_max = int(np.ceil((t_max[0] - global_origin[0]) / resolution))
        iy_min = int(np.floor((t_min[1] - global_origin[1]) / resolution))
        iy_max = int(np.ceil((t_max[1] - global_origin[1]) / resolution))

        ix = np.arange(ix_min, ix_max + 1)
        iy = np.arange(iy_min, iy_max + 1)
        if len(ix) == 0 or len(iy) == 0:
            continue
        gix, giy = np.meshgrid(ix, iy)
        cand_idx = np.column_stack([gix.ravel(), giy.ravel()])
        cand_xy = global_origin + (cand_idx + 0.5) * resolution

        inside = _points_in_triangle(cand_xy, tri)
        for cix, ciy in cand_idx[inside]:
            cell_set.add((int(cix), int(ciy)))

    if not cell_set:
        return np.empty((0, 2)), resolution * resolution

    cells = np.array(list(cell_set))
    points = global_origin + (cells + 0.5) * resolution
    return points, resolution * resolution


def _points_in_triangle(pts: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Vectorized point-in-triangle test via barycentric sign check."""
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


def _column_needs_support(hit_z: np.ndarray, hit_nz: np.ndarray, z_ground: float = 0.0) -> float:
    """True (self-intersection-aware) support height for one vertical column.

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


def compute_support_volume(
    mesh: trimesh.Trimesh,
    angle_threshold_deg: float = 50.0,
    resolution: float = 0.05,
    z_ground: float = 0.0,
):
    """Self-intersection-aware support volume, plus the naive estimate for comparison.

    Returns (naive_volume, true_volume, n_sample_points).
    """
    overhang_idx = find_overhanging_facets(mesh, angle_threshold_deg)
    if len(overhang_idx) == 0:
        return 0.0, 0.0, 0

    # naive estimate, exactly as the papers describe it: per-facet centroid
    # height x projected area, summed.
    tri_areas_3d = mesh.area_faces[overhang_idx]
    normals = mesh.face_normals[overhang_idx]
    # projected area = 3D area * |cos(angle to vertical)|
    proj_areas = tri_areas_3d * np.abs(normals[:, 2])
    centroid_z = mesh.triangles_center[overhang_idx][:, 2]
    naive_volume = float(np.sum(centroid_z * proj_areas))

    points_xy, cell_area = _footprint_sample_points(mesh, overhang_idx, resolution)
    if len(points_xy) == 0:
        return naive_volume, 0.0, 0

    ray_origins = np.column_stack([points_xy, np.full(len(points_xy), mesh.bounds[1, 2] + 1.0)])
    ray_directions = np.tile(RAY_DIR, (len(points_xy), 1))

    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins, ray_directions, multiple_hits=True
    )

    true_volume = 0.0
    if len(locations) > 0:
        hit_normals_z = mesh.face_normals[index_tri][:, 2]
        hit_z_all = locations[:, 2]
        for col in range(len(points_xy)):
            mask = index_ray == col
            if not np.any(mask):
                continue
            true_volume += _column_needs_support(hit_z_all[mask], hit_normals_z[mask], z_ground)

    true_volume *= cell_area
    return naive_volume, true_volume, len(points_xy)
