"""Input validation and mesh-quality diagnostics.

Deliberately does not attempt to repair anything automatically -- a caller
should decide whether and how to repair a mesh (e.g. via
``trimesh.repair.fill_holes``) rather than have this library silently alter
their geometry before measuring it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import trimesh

_ZERO_AREA_EPS = 1e-12


@dataclass(frozen=True)
class MeshDiagnostics:
    """Mesh-quality signals relevant to trusting a support-volume estimate.

    None values mean "not computed" (e.g. because a required optional
    dependency wasn't available), not "zero".
    """

    n_faces: int
    n_vertices: int
    watertight: bool
    n_zero_area_faces: int
    n_duplicate_faces: int | None
    n_nonmanifold_edges: int | None
    n_boundary_edges: int | None


def compute_mesh_diagnostics(mesh: trimesh.Trimesh) -> MeshDiagnostics:
    """Compute mesh-quality diagnostics without modifying the mesh."""
    n_zero_area = int(np.sum(mesh.area_faces < _ZERO_AREA_EPS))

    n_duplicate = None
    try:
        # rows of mesh.faces that are identical (same 3 vertex indices, any
        # winding) once sorted -- a cheap, dependency-free duplicate check.
        sorted_faces = np.sort(mesh.faces, axis=1)
        _, counts = np.unique(sorted_faces, axis=0, return_counts=True)
        n_duplicate = int(np.sum(counts - 1))  # extra copies beyond the first
    except Exception:
        n_duplicate = None

    n_nonmanifold = None
    n_boundary = None
    try:
        _, edge_counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
        n_boundary = int(np.sum(edge_counts == 1))
        n_nonmanifold = int(np.sum(edge_counts > 2))
    except Exception:
        pass

    return MeshDiagnostics(
        n_faces=len(mesh.faces),
        n_vertices=len(mesh.vertices),
        watertight=bool(mesh.is_watertight),
        n_zero_area_faces=n_zero_area,
        n_duplicate_faces=n_duplicate,
        n_nonmanifold_edges=n_nonmanifold,
        n_boundary_edges=n_boundary,
    )


def validate_mesh_inputs(
    mesh: trimesh.Trimesh,
    resolution: float,
    angle_threshold_deg: float,
) -> MeshDiagnostics:
    """Raise on structurally invalid inputs; warn (don't raise) on mesh-quality
    issues that don't prevent computing an answer, just make it less trustworthy.

    Returns the mesh's diagnostics so callers (e.g. ``compute_support_volume``)
    don't have to compute them twice.
    """
    if len(mesh.vertices) == 0:
        raise ValueError("mesh has no vertices (empty mesh)")
    if len(mesh.faces) == 0:
        raise ValueError("mesh has no faces (zero-face mesh)")
    if not np.all(np.isfinite(mesh.vertices)):
        raise ValueError("mesh contains non-finite vertex coordinates (NaN or Inf)")
    if resolution <= 0:
        raise ValueError(f"resolution must be > 0, got {resolution}")
    if not (0.0 < angle_threshold_deg <= 90.0):
        raise ValueError(f"angle_threshold_deg must be in (0, 90], got {angle_threshold_deg}")

    diagnostics = compute_mesh_diagnostics(mesh)

    if diagnostics.n_zero_area_faces > 0:
        warnings.warn(
            f"mesh has {diagnostics.n_zero_area_faces} degenerate (zero-area) "
            "face(s); these contribute no area to either estimate but may "
            "indicate a lower-quality source mesh.",
            stacklevel=3,
        )
    if not diagnostics.watertight:
        warnings.warn(
            "mesh is not watertight. The ray-cast integrated estimate assumes "
            "each vertical column's intersections alternate cleanly between "
            "entering and exiting solid material; a non-watertight mesh can "
            "violate that assumption in the affected region. Results were "
            "empirically stable on a non-watertight benchmark mesh (see "
            "README.md), but this is not a general guarantee -- inspect "
            "results near known holes/gaps before trusting them.",
            stacklevel=3,
        )
    if diagnostics.n_nonmanifold_edges:
        warnings.warn(
            f"mesh has {diagnostics.n_nonmanifold_edges} non-manifold edge(s), "
            "which can also violate the entering/exiting alternation assumption "
            "used by the ray-cast integrated estimate.",
            stacklevel=3,
        )

    return diagnostics
