"""Public, high-level API: run both estimators together and package a
structured, machine-readable result.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import trimesh

from .raycast import (
    compute_integrated_support_volume,
    compute_naive_support_volume,
    find_overhanging_facets,
)
from .validation import MeshDiagnostics, validate_mesh_inputs


@dataclass(frozen=True)
class SupportVolumeResult:
    """Everything needed to interpret and reproduce one support-volume run.

    ``naive_volume`` is the IDETC-2026 paper's own centroid-height x
    projected-area baseline. ``integrated_volume`` is this repository's
    ray-cast column-integration estimate -- deliberately not called "true" or
    "exact" anywhere, since it remains a grid approximation (see
    ``resolution`` and ``README.md``'s convergence evidence).
    """

    naive_volume: float
    integrated_volume: float
    resolution: float
    angle_threshold_deg: float
    z_ground: float
    build_direction: tuple[float, float, float]
    n_sample_columns: int
    runtime_seconds: float
    diagnostics: MeshDiagnostics

    @property
    def reduction_fraction(self) -> float:
        """Fractional reduction of integrated vs. naive (0 if naive is 0)."""
        if self.naive_volume == 0:
            return 0.0
        return (self.naive_volume - self.integrated_volume) / self.naive_volume

    def to_dict(self) -> dict:
        """A flat, JSON-serializable dict (diagnostics nested as a sub-dict)."""
        d = asdict(self)
        d["reduction_fraction"] = self.reduction_fraction
        return d


def compute_support_volume(
    mesh: trimesh.Trimesh,
    angle_threshold_deg: float = 50.0,
    resolution: float = 0.05,
    z_ground: float = 0.0,
    build_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    validate: bool = True,
) -> SupportVolumeResult:
    """Compute both the naive baseline and the integrated estimate together.

    This is the main entry point for most callers. Set ``validate=False``
    to skip mesh-quality checks (e.g. in a tight loop where the mesh has
    already been validated once) -- diagnostics are still computed either way,
    just without raising/warning on issues.
    """
    if validate:
        diagnostics = validate_mesh_inputs(mesh, resolution, angle_threshold_deg)
    else:
        from .validation import compute_mesh_diagnostics

        diagnostics = compute_mesh_diagnostics(mesh)

    t0 = time.perf_counter()
    overhang_idx = find_overhanging_facets(mesh, angle_threshold_deg, build_direction)
    naive_volume = compute_naive_support_volume(
        mesh, angle_threshold_deg, z_ground, build_direction, overhang_idx=overhang_idx
    )
    integrated_volume, n_columns = compute_integrated_support_volume(
        mesh, angle_threshold_deg, resolution, z_ground, build_direction, overhang_idx=overhang_idx
    )
    runtime = time.perf_counter() - t0

    return SupportVolumeResult(
        naive_volume=naive_volume,
        integrated_volume=integrated_volume,
        resolution=resolution,
        angle_threshold_deg=angle_threshold_deg,
        z_ground=z_ground,
        build_direction=tuple(build_direction),
        n_sample_columns=n_columns,
        runtime_seconds=runtime,
        diagnostics=diagnostics,
    )
