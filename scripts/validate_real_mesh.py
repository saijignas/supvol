"""Extended, manual validation against a real-world STL -- not run in CI
(the file is large and this isn't a deterministic pass/fail check the way the
synthetic tests are), but reproduces the numbers quoted in the README.

Usage:
    python scripts/validate_real_mesh.py
"""

import os
import time
import urllib.request

import trimesh

from supvol.raycast import compute_support_volume

BENCHY_URL = "https://raw.githubusercontent.com/CreativeTools/3DBenchy/master/Single-part/3DBenchy.stl"
LOCAL_PATH = os.path.join(os.path.dirname(__file__), "..", "sample_models", "3DBenchy.stl")


def ensure_model():
    os.makedirs(os.path.dirname(LOCAL_PATH), exist_ok=True)
    if not os.path.exists(LOCAL_PATH):
        print(f"Downloading 3DBenchy (public domain) from {BENCHY_URL} ...")
        urllib.request.urlretrieve(BENCHY_URL, LOCAL_PATH)
    return LOCAL_PATH


if __name__ == "__main__":
    path = ensure_model()
    mesh = trimesh.load(path)
    print(f"faces={len(mesh.faces)}, watertight={mesh.is_watertight}")

    for resolution in (0.5, 0.25):
        t0 = time.time()
        naive, true, n = compute_support_volume(mesh, resolution=resolution)
        dt = time.time() - t0
        reduction = (naive - true) / naive if naive else 0.0
        print(
            f"resolution={resolution}: columns={n}, naive={naive:.2f} mm3, "
            f"true={true:.2f} mm3, reduction={reduction:.1%}, time={dt:.1f}s"
        )
