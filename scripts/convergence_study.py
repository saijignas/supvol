"""Reproducible grid-convergence experiment.

Runs the support-volume calculation at a list of resolutions against either
a real STL file or one of the built-in synthetic fixtures (which have known
analytical ground truth, so relative error can be reported directly).

Usage:
    python scripts/convergence_study.py --stl path/to/part.stl --resolutions 1.0 0.5 0.25
    python scripts/convergence_study.py --fixture shelf_and_pillar --resolutions 0.4 0.2 0.1 0.05
    python scripts/convergence_study.py --fixture overlapping_overhangs --out results.csv --json results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supvol import compute_support_volume  # noqa: E402
from supvol import fixtures as fixtures_module  # noqa: E402

FIXTURE_NAMES = ["shelf_and_pillar", "simple_overhang", "tilted_overhang", "overlapping_overhangs"]


def _load_target(args) -> tuple[trimesh.Trimesh, float | None, str]:
    if args.stl:
        mesh = trimesh.load(args.stl)
        return mesh, None, str(args.stl)
    fn = getattr(fixtures_module, args.fixture)
    mesh, truth = fn()
    return mesh, truth.get("true_support_volume"), f"fixture:{args.fixture}"


def run_convergence(mesh: trimesh.Trimesh, resolutions: list[float], ground_truth: float | None):
    rows = []
    for r in resolutions:
        t0 = time.perf_counter()
        result = compute_support_volume(mesh, resolution=r, validate=False)
        dt = time.perf_counter() - t0

        row = {
            "resolution": r,
            "naive_support_volume": result.naive_volume,
            "integrated_support_volume": result.integrated_volume,
            "relative_difference_naive_vs_integrated": result.reduction_fraction,
            "n_sample_columns": result.n_sample_columns,
            "runtime_seconds": dt,
        }
        if ground_truth is not None and ground_truth != 0:
            row["ground_truth"] = ground_truth
            row["abs_error_vs_ground_truth"] = abs(result.integrated_volume - ground_truth)
            row["rel_error_vs_ground_truth"] = abs(result.integrated_volume - ground_truth) / ground_truth
        rows.append(row)
    return rows


def print_table(rows: list[dict]):
    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c])) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = "  ".join(
            (f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c])).ljust(widths[c]) for c in cols
        )
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--stl", type=str, help="path to an STL file")
    src.add_argument("--fixture", choices=FIXTURE_NAMES, help="use a built-in synthetic fixture instead")
    parser.add_argument(
        "--resolutions", type=float, nargs="+", default=[0.4, 0.2, 0.1, 0.05, 0.025],
        help="list of grid resolutions to test, coarsest to finest",
    )
    parser.add_argument("--out", type=str, help="write results as CSV to this path")
    parser.add_argument("--json", type=str, help="write results as JSON to this path")
    args = parser.parse_args()

    mesh, ground_truth, label = _load_target(args)
    print(f"target: {label}  (faces={len(mesh.faces)}, watertight={mesh.is_watertight})")
    if ground_truth is not None:
        print(f"analytical ground truth: {ground_truth}")
    print()

    rows = run_convergence(mesh, sorted(args.resolutions, reverse=True), ground_truth)
    print_table(rows)

    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.out}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
