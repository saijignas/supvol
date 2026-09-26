"""Independent cross-check of SUPVOL's naive baseline against PySLM.

Shonkwiler et al. (IDETC-2026) state their ground-truth labels were computed
by "using PySLM [22] to identify overhanging facets... calculating the
truncated pyramid volume under each" -- exactly the formula this repo
reproduces as compute_naive_support_volume(). This script runs PySLM's own
getOverhangMesh() + approximateSupportMomentArea() on the same meshes this
repo already validates against (the synthetic fixtures and 3DBenchy) and
compares the result directly to compute_naive_support_volume(), as an
external check that the reproduction is faithful -- not just self-consistent
with this repo's own tests.

This is a standalone, validation-only script. PySLM is NOT a dependency of
the supvol package or app.py, and is never imported by either -- it is only
used here, offline, to double-check a claim already made in the README.

Requires:  pip install "PythonSLM[support]"
Run with:  python scripts/validate_against_pyslm.py
           python scripts/validate_against_pyslm.py --stl your_part.stl
"""

from __future__ import annotations

import argparse
import sys

import trimesh

from supvol.fixtures import overlapping_overhangs, shelf_and_pillar, simple_overhang, tilted_overhang
from supvol.raycast import compute_naive_support_volume

ANGLE_THRESHOLD_DEG = 50.0

FIXTURES = {
    "shelf_and_pillar": shelf_and_pillar,
    "simple_overhang": simple_overhang,
    "tilted_overhang": tilted_overhang,
    "overlapping_overhangs": overlapping_overhangs,
}


def _pyslm_naive_volume(mesh: trimesh.Trimesh, angle_threshold_deg: float) -> float:
    """PySLM's own reproduction of the paper's labeling method: getOverhangMesh
    (angle-threshold facet selection) + approximateSupportMomentArea (centroid
    height x projected area, summed) -- called exactly as the paper describes."""
    import pyslm
    import pyslm.support

    # setGeometryByMesh (not the read-only `.geometry` property) sets the mesh
    # without any implicit fixing/re-positioning; deliberately not calling
    # dropToPlatform() either, so PySLM analyzes the mesh in the exact same
    # coordinate frame our own z_ground=0.0 assumption uses -- otherwise an
    # auto-drop-to-platform could shift z-heights and produce a spurious
    # mismatch that has nothing to do with the actual algorithm.
    part = pyslm.Part("validation")
    part.setGeometryByMesh(mesh)

    return pyslm.support.approximateSupportMomentArea(part, angle_threshold_deg)


def _compare(label: str, mesh: trimesh.Trimesh) -> tuple[float, float]:
    ours = compute_naive_support_volume(mesh, angle_threshold_deg=ANGLE_THRESHOLD_DEG)
    theirs = _pyslm_naive_volume(mesh, ANGLE_THRESHOLD_DEG)
    rel_diff = abs(ours - theirs) / theirs if theirs else abs(ours - theirs)
    print(f"{label:24s}  ours={ours:12.4f}  pyslm={theirs:12.4f}  rel_diff={rel_diff:.6%}")
    return ours, theirs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl", type=str, default=None, help="Path to a real STL to also check")
    args = parser.parse_args()

    print(f"Overhang angle threshold: {ANGLE_THRESHOLD_DEG} deg\n")
    print(f"{'case':24s}  {'ours':>12s}  {'pyslm':>12s}  rel_diff")

    any_mismatch = False
    for name, factory in FIXTURES.items():
        mesh, _truth = factory()
        ours, theirs = _compare(name, mesh)
        if theirs and abs(ours - theirs) / theirs > 0.01:
            any_mismatch = True

    if args.stl:
        mesh = trimesh.load(args.stl, force="mesh")
        ours, theirs = _compare(args.stl, mesh)
        if theirs and abs(ours - theirs) / theirs > 0.01:
            any_mismatch = True

    if any_mismatch:
        print("\nWARNING: one or more cases disagree with PySLM by more than 1%.")
        return 1

    print("\nAll cases agree with PySLM's own overhang-detection + moment-area calculation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
