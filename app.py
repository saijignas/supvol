"""SUPVOL -- Self-Intersection-Aware Support Volume Estimator (Streamlit demo).

This is a thin UI over the real, tested supvol package -- it never
reimplements any part of the algorithm, and every number shown comes
directly from supvol.compute_support_volume() or supvol.validation.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import traceback

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import trimesh

from supvol import compute_support_volume
from supvol.fixtures import overlapping_overhangs, shelf_and_pillar, simple_overhang, tilted_overhang
from supvol.raycast import find_overhanging_facets
from supvol.validation import compute_mesh_diagnostics

REPO_URL = "https://github.com/saijignas/supvol"
PAPER_URL = "https://doi.org/10.1115/DETC2026-193146"

# Soft guard for "excessively large input" (item 12): above this, require an
# explicit acknowledgment before running the (potentially slow) calculation.
LARGE_MESH_FACE_WARNING = 300_000
# Above this many faces, decimate purely for the *browser-side preview* --
# never affects the actual calculation, which always runs on the real mesh.
VIZ_DECIMATE_THRESHOLD = 30_000

SAMPLE_MODELS = {
    "Shelf & pillar (self-intersection demo)": shelf_and_pillar,
    "Simple overhang (sanity check)": simple_overhang,
    "Tilted overhang (angle-math demo)": tilted_overhang,
    "Overlapping overhangs (footprint-overlap demo)": overlapping_overhangs,
}

BUILD_DIRECTIONS = {
    "+Z (prints upward, build plate at the bottom)": (0.0, 0.0, 1.0),
    "-Z (prints downward, build plate at the top)": (0.0, 0.0, -1.0),
}

UNIT_LABELS = ["mm", "cm", "m", "in"]


# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="SUPVOL", page_icon="🖨️", layout="wide")

st.title("SUPVOL")
st.caption("Self-Intersection-Aware Support Volume Estimator")
st.markdown(
    "A research prototype for self-intersection-aware support-volume estimation using "
    "ray-cast column integration. "
    f"&nbsp;|&nbsp; [Source on GitHub]({REPO_URL}) "
    f"&nbsp;|&nbsp; [Reference paper]({PAPER_URL})"
)
st.caption(
    "The reference paper's DOI was assigned at IDETC-CIE 2026 (Houston, Aug 2026); ASME's "
    "Digital Collection can take a few months after the conference to publish the proceedings "
    "and activate the DOI, so this link may not resolve yet."
)
st.divider()


# ----------------------------------------------------------------------------
# Cached mesh loading (avoid re-parsing on every widget interaction/rerun)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_stl_bytes(file_bytes: bytes) -> trimesh.Trimesh:
    """Load STL bytes into a Trimesh, raising ValueError for anything that
    isn't actually a usable mesh.

    trimesh.load does NOT raise on invalid/corrupt STL bytes -- it silently
    returns an empty trimesh.Scene instead (confirmed directly: feeding it
    arbitrary non-STL bytes returns a Scene with zero geometry, not an
    exception). Left unchecked, that Scene would reach later code expecting
    a Trimesh (e.g. `mesh.vertices`) and crash with a raw AttributeError past
    this function's own try/except in the caller -- exactly the kind of
    exposed stack trace item 12 rules out. Explicitly checked here instead.
    """
    loaded = trimesh.load(io.BytesIO(file_bytes), file_type="stl")
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError("File did not parse as a valid STL mesh.")
    return loaded


def _decimated_for_viz(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if len(mesh.faces) <= VIZ_DECIMATE_THRESHOLD:
        return mesh
    target = VIZ_DECIMATE_THRESHOLD
    try:
        return mesh.simplify_quadric_decimation(face_count=target)
    except Exception:
        # simplify_quadric_decimation needs an optional dependency (fast-simplification);
        # fall back to a naive face subsample purely for the preview -- never used
        # for the actual calculation, so approximate-looking is fine here.
        idx = np.linspace(0, len(mesh.faces) - 1, target).astype(int)
        return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[idx], process=False)


# ----------------------------------------------------------------------------
# 1 & 2. Header already rendered above; input panel now
# ----------------------------------------------------------------------------
st.subheader("1. Load a mesh")
col_upload, col_sample = st.columns(2)

with col_upload:
    upload = st.file_uploader("Upload an STL file", type=["stl"])

with col_sample:
    sample_choice = st.selectbox(
        "...or try a bundled sample model", ["(none)"] + list(SAMPLE_MODELS.keys())
    )

mesh: trimesh.Trimesh | None = None
mesh_label: str | None = None
load_error: str | None = None

if upload is not None:
    try:
        mesh = _load_stl_bytes(upload.getvalue())
        mesh_label = upload.name
    except Exception:
        load_error = "Could not read this file as an STL mesh. Please check the file and try again."
elif sample_choice != "(none)":
    mesh, _truth = SAMPLE_MODELS[sample_choice]()
    mesh_label = sample_choice

if load_error:
    st.error(load_error)

if mesh is not None and (len(mesh.vertices) == 0 or len(mesh.faces) == 0):
    st.error("This mesh has no geometry (0 vertices or 0 faces) and can't be analyzed.")
    mesh = None

# ----------------------------------------------------------------------------
# 3. Input validation / mesh diagnostics -- shown as soon as a mesh loads,
#    before any calculation runs
# ----------------------------------------------------------------------------
diagnostics = None
if mesh is not None:
    diagnostics = compute_mesh_diagnostics(mesh)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Faces", f"{diagnostics.n_faces:,}")
    d2.metric("Vertices", f"{diagnostics.n_vertices:,}")
    d3.metric("Watertight", "Yes" if diagnostics.watertight else "No")
    d4.metric("Zero-area faces", diagnostics.n_zero_area_faces)

    if not diagnostics.watertight:
        st.warning(
            "This mesh is not watertight. The ray-cast integrated estimate assumes each "
            "vertical column's intersections alternate cleanly between entering and exiting "
            "solid material; a non-watertight mesh can violate that assumption in the "
            "affected region. Results may still be reliable (see the real-mesh benchmark in "
            "the README), but treat this as a caveat, not a guarantee."
        )
    if diagnostics.n_zero_area_faces > 0:
        st.warning(f"{diagnostics.n_zero_area_faces} degenerate (zero-area) face(s) detected.")
    if diagnostics.n_nonmanifold_edges:
        st.warning(f"{diagnostics.n_nonmanifold_edges} non-manifold edge(s) detected.")

    st.info(
        "STL files do not encode physical units -- this is a plain geometric fact about "
        "the format, not something SUPVOL can detect from the file. Select the unit you "
        "know your model was authored in; it is used only to label results, not to convert "
        "anything.",
        icon="📏",
    )
    units = st.selectbox("Assumed units", UNIT_LABELS, index=0)

    if diagnostics.n_faces > LARGE_MESH_FACE_WARNING:
        st.warning(
            f"This mesh has {diagnostics.n_faces:,} faces, which may make the calculation "
            "slow, especially at fine grid resolutions."
        )
        proceed_anyway = st.checkbox("I understand this may be slow -- proceed anyway")
    else:
        proceed_anyway = True
else:
    units = "mm"
    proceed_anyway = False

st.divider()

# ----------------------------------------------------------------------------
# 2 (cont.). Calculation parameters
# ----------------------------------------------------------------------------
st.subheader("2. Set parameters")
p1, p2, p3 = st.columns(3)
with p1:
    build_direction_label = st.selectbox("Build direction", list(BUILD_DIRECTIONS.keys()))
    build_direction = BUILD_DIRECTIONS[build_direction_label]
with p2:
    angle_threshold_deg = st.number_input(
        "Overhang angle threshold (degrees)", min_value=0.1, max_value=90.0, value=50.0, step=1.0
    )
with p3:
    resolution = st.number_input(
        "Grid resolution", min_value=0.0001, value=0.5, step=0.1, format="%.4f"
    )

calculate_clicked = st.button(
    "Calculate", type="primary", disabled=(mesh is None or not proceed_anyway)
)

st.divider()

# ----------------------------------------------------------------------------
# Run the calculation ONLY on button press (item 11: no auto-calculate)
# ----------------------------------------------------------------------------
if calculate_clicked and mesh is not None:
    try:
        with st.spinner("Casting rays and integrating support volume..."):
            result = compute_support_volume(
                mesh,
                angle_threshold_deg=angle_threshold_deg,
                resolution=resolution,
                build_direction=build_direction,
                validate=False,  # diagnostics already shown above; avoid duplicate warnings
            )
        st.session_state["result"] = result
        st.session_state["result_meta"] = {
            "filename": mesh_label,
            "assumed_units": units,
            "build_direction": build_direction_label,
        }
    except (ValueError, NotImplementedError) as e:
        st.error(f"Could not run this calculation: {e}")
        st.session_state.pop("result", None)
    except Exception:
        st.error(
            "Something went wrong during the calculation. This has been recorded; "
            "please try a different mesh or parameters."
        )
        # Deliberately not shown to the user (item 12) -- surfaced only in the
        # server-side console for whoever is running the app locally.
        print("SUPVOL app: unexpected error during calculation:")  # noqa: T201
        print(traceback.format_exc())  # noqa: T201
        st.session_state.pop("result", None)

# ----------------------------------------------------------------------------
# 4, 6, 7. Results
# ----------------------------------------------------------------------------
result = st.session_state.get("result")
meta = st.session_state.get("result_meta", {})

if result is not None:
    st.subheader("3. Results")

    r1, r2, r3 = st.columns(3)
    r1.metric("Reference (naive) support volume", f"{result.naive_volume:,.2f} {units}³")
    r2.metric("SUPVOL integrated volume", f"{result.integrated_volume:,.2f} {units}³")
    abs_diff = result.naive_volume - result.integrated_volume
    # Near a case with ~zero true correction, grid-discretization noise can make
    # the integrated estimate marginally *larger* than naive (see the tilted_overhang
    # convergence note in the README) -- round-to-zero here instead of displaying a
    # confusing "-0.0%" / "-0 mm3" negative-zero artifact.
    reduction_pct = result.reduction_fraction * 100
    if round(reduction_pct, 1) == 0:
        reduction_pct = 0.0
    diff_display = 0.0 if round(abs_diff, 0) == 0 else abs_diff
    r3.metric("Reduction vs. reference", f"{reduction_pct:.1f}%")
    if diff_display >= 0:
        r3.caption(f"{diff_display:,.0f} {units}³ below reference")
    else:
        r3.caption(f"{abs(diff_display):,.0f} {units}³ above reference (grid-discretization noise)")

    r4, r5, r6 = st.columns(3)
    r4.metric("Overhang facets", f"{len(find_overhanging_facets(mesh, angle_threshold_deg, build_direction)):,}")
    r5.metric("Sampled columns", f"{result.n_sample_columns:,}")
    r6.metric("Runtime", f"{result.runtime_seconds:.2f} s")

    fig = go.Figure(
        data=[
            go.Bar(
                x=["Reference (naive)", "SUPVOL integrated"],
                y=[result.naive_volume, result.integrated_volume],
                marker_color=["#888888", "#2E86AB"],
                text=[f"{result.naive_volume:,.1f}", f"{result.integrated_volume:,.1f}"],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Reference vs. SUPVOL support volume",
        yaxis_title=f"Volume ({units}³)",
        height=350,
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig, width='stretch')

    with st.expander("Mesh diagnostics"):
        st.json(
            {
                "n_faces": result.diagnostics.n_faces,
                "n_vertices": result.diagnostics.n_vertices,
                "watertight": result.diagnostics.watertight,
                "n_zero_area_faces": result.diagnostics.n_zero_area_faces,
                "n_duplicate_faces": result.diagnostics.n_duplicate_faces,
                "n_nonmanifold_edges": result.diagnostics.n_nonmanifold_edges,
                "n_boundary_edges": result.diagnostics.n_boundary_edges,
            }
        )

    # ---- Download ----
    export = result.to_dict()
    export.update(
        {
            "input_filename": meta.get("filename"),
            "assumed_units": meta.get("assumed_units"),
            "build_direction_label": meta.get("build_direction"),
        }
    )
    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "Download results as JSON",
        data=json.dumps(export, indent=2, default=str),
        file_name="supvol_result.json",
        mime="application/json",
    )
    flat = {k: v for k, v in export.items() if k != "diagnostics"}
    flat.update({f"diagnostics.{k}": v for k, v in export.get("diagnostics", {}).items()})
    csv_text = "field,value\n" + "\n".join(f'"{k}","{v}"' for k, v in flat.items())
    dl2.download_button(
        "Download results as CSV", data=csv_text, file_name="supvol_result.csv", mime="text/csv"
    )

    st.divider()

    # ---- 5. Visualization (optional, item 5) ----
    st.subheader("4. 3D visualization (optional)")
    show_viz = st.checkbox("Show 3D mesh visualization", value=(diagnostics.n_faces <= VIZ_DECIMATE_THRESHOLD))
    if show_viz:
        viz_mesh = _decimated_for_viz(mesh)
        if len(viz_mesh.faces) != len(mesh.faces):
            st.caption(
                f"Preview decimated to {len(viz_mesh.faces):,} faces for browser performance "
                f"(original: {len(mesh.faces):,}); the calculation above always used the full mesh."
            )

        overhang_idx = find_overhanging_facets(viz_mesh, angle_threshold_deg, build_direction)
        face_colors = np.full(len(viz_mesh.faces), "#B0B0B0", dtype=object)
        face_colors[overhang_idx] = "#E63946"

        v = viz_mesh.vertices
        f = viz_mesh.faces
        mesh3d = go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            facecolor=face_colors,
            opacity=1.0,
            name="mesh",
        )

        center = v.mean(axis=0)
        scale = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0))) * 0.3
        bd = np.array(build_direction)
        cone = go.Cone(
            x=[center[0]], y=[center[1]], z=[v[:, 2].max() + scale],
            u=[bd[0] * scale], v=[bd[1] * scale], w=[bd[2] * scale],
            colorscale=[[0, "#1D3557"], [1, "#1D3557"]],
            showscale=False,
            sizemode="absolute",
            sizeref=scale * 0.5,
            name="build direction",
        )

        fig3d = go.Figure(data=[mesh3d, cone])
        fig3d.update_layout(
            height=600,
            margin=dict(t=20, b=20),
            scene=dict(aspectmode="data"),
            showlegend=False,
        )
        st.plotly_chart(fig3d, width='stretch')
        st.caption("Red faces are detected overhangs (support-needing facets); blue cone shows build direction.")

st.divider()

# ----------------------------------------------------------------------------
# 6. Method explanation
# ----------------------------------------------------------------------------
with st.expander("How SUPVOL works"):
    st.markdown(
        f"""
**Reference (published) method** -- from [Shonkwiler et al., IDETC-2026]({PAPER_URL}):
for every facet whose normal is within the angle threshold of straight down, compute
`facet_centroid_height x horizontally_projected_facet_area` and sum across all such
facets. This is exactly reproduced here as `compute_naive_support_volume()` -- it is
the published baseline, not a strawman -- and it explicitly ignores any place where the
support column would intersect the part's own geometry.

**SUPVOL's method** (this repository's contribution): grid-sample the union of all
overhanging facets' footprints, cast one vertical ray per sample column through the
*entire* mesh, and integrate only the open-air gap between the overhang surface and the
build plate in each column. This single integration corrects two things the reference
method does not:
- **self-intersection** -- where the part's own geometry already occupies part of a
  support column, so no material is needed there;
- **overlapping footprints** -- where two separate overhangs project onto the same
  area, which a naive per-facet sum would double-count.

SUPVOL's result is a **grid approximation**, not an exact boolean/CSG computation --
error shrinks with finer resolution but is never claimed to be exact. See the
[README]({REPO_URL}) for convergence evidence, rotation/grid-sensitivity tests, and a
real-mesh benchmark.
"""
    )

st.caption(
    "This is a scientific research demo, not a commercial slicer. It does not generate "
    "printable supports, does not model support density/pattern/interface layers, and "
    "does not reproduce any specific slicer's support strategy."
)
