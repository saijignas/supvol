"""Pure mesh-decimation helper for app.py's browser-side 3D preview.

Kept separate from app.py (which requires streamlit/plotly to even import)
specifically so it's directly unit-testable without a Streamlit runtime.
"""

from __future__ import annotations

import numpy as np
import trimesh

# Above this many faces, decimate purely for the *browser-side preview* --
# never affects the actual calculation, which always runs on the real mesh.
VIZ_DECIMATE_THRESHOLD = 30_000


def decimate_for_viz(mesh: trimesh.Trimesh, threshold: int = VIZ_DECIMATE_THRESHOLD) -> trimesh.Trimesh:
    if len(mesh.faces) <= threshold:
        return mesh
    target = threshold
    try:
        return mesh.simplify_quadric_decimation(face_count=target)
    except Exception:
        # simplify_quadric_decimation needs an optional dependency (fast-simplification);
        # fall back to a naive face subsample purely for the preview -- never used
        # for the actual calculation, so approximate-looking is fine here.
        idx = np.linspace(0, len(mesh.faces) - 1, target).astype(int)
        return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[idx], process=False)
