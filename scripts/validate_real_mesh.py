"""Real-mesh benchmark utility.

Takes an explicit STL path by default; the public-domain 3DBenchy benchmark
model is only downloaded if you explicitly ask for it with --download-benchy
(no network access is triggered otherwise, and this script is intentionally
NOT run in CI -- see .github/workflows/ci.yml -- since it needs a real file
and isn't a fast, deterministic pass/fail check the way the pytest suite is).

Usage:
    python scripts/validate_real_mesh.py --stl my_part.stl --resolution 0.5
    python scripts/validate_real_mesh.py --download-benchy --resolution 0.5 0.25
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supvol import compute_support_volume  # noqa: E402

BENCHY_URL = "https://raw.githubusercontent.com/CreativeTools/3DBenchy/master/Single-part/3DBenchy.stl"
DEFAULT_BENCHY_PATH = Path(__file__).resolve().parent.parent / "sample_models" / "3DBenchy.stl"


def download_benchy(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"Downloading public-domain 3DBenchy model from {BENCHY_URL} ...")
        urllib.request.urlretrieve(BENCHY_URL, dest)
    return dest


def benchmark(stl_path: Path, resolutions: list[float]) -> list[dict]:
    mesh = trimesh.load(stl_path)
    rows = []
    for r in resolutions:
        t0 = time.perf_counter()
        result = compute_support_volume(mesh, resolution=r, validate=False)
        dt = time.perf_counter() - t0
        rows.append(
            {
                "stl_path": str(stl_path),
                "n_faces": len(mesh.faces),
                "watertight": bool(mesh.is_watertight),
                "resolution": r,
                "n_sample_columns": result.n_sample_columns,
                "runtime_seconds": dt,
                "naive_support_volume": result.naive_volume,
                "integrated_support_volume": result.integrated_volume,
                "reduction_fraction": result.reduction_fraction,
            }
        )
    return rows


def print_table(rows: list[dict]):
    cols = ["resolution", "n_sample_columns", "runtime_seconds", "naive_support_volume",
            "integrated_support_volume", "reduction_fraction"]
    widths = {c: max(len(c), max(len(f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c])) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join((f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c])).ljust(widths[c]) for c in cols))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--stl", type=str, help="path to an existing STL file")
    src.add_argument("--download-benchy", action="store_true", help="download the public-domain 3DBenchy model")
    parser.add_argument("--resolution", type=float, nargs="+", default=[0.5], help="one or more grid resolutions")
    parser.add_argument("--json", type=str, help="write results as JSON to this path")
    args = parser.parse_args()

    stl_path = download_benchy(DEFAULT_BENCHY_PATH) if args.download_benchy else Path(args.stl)
    if not stl_path.exists():
        raise SystemExit(f"STL file not found: {stl_path}")

    rows = benchmark(stl_path, args.resolution)
    print(f"mesh: {stl_path}  (faces={rows[0]['n_faces']}, watertight={rows[0]['watertight']})\n")
    print_table(rows)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
