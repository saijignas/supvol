# supvol

A research-quality, open-source implementation of a ray-cast, column-integrated support-volume estimator for additive manufacturing, built to study and address a specific limitation stated in a published support-volume prediction method.

**In one sentence:** this repository implements a ray-cast, column-integrated alternative intended to address the self-intersection limitation described in the paper below -- it is not a slicer, not an exact CSG solver, and not a claim of general novelty.

## Interactive demo (SUPVOL app)

A Streamlit research-demo UI lives at `app.py` -- upload an STL (or try a bundled sample), set the build direction/angle threshold/resolution, and see the reference baseline and SUPVOL's integrated estimate side by side, with an optional 3D visualization that colors detected overhang facets and shows the build direction. It is a thin UI layer only: every number comes directly from `supvol.compute_support_volume()`, nothing is reimplemented in the frontend.

```bash
pip install -e ".[app]"
streamlit run app.py
```

**What it demonstrates:** the reference-vs-SUPVOL comparison this whole repository is about, made explorable without writing any code -- upload your own part, or one of the four synthetic fixtures with known ground truth, and see the naive/integrated volumes, the mesh diagnostics, and (for smaller meshes) an interactive 3D view.

**Units**: STL files do not encode physical units -- the app states this explicitly and lets you *label* results with an assumed unit (mm by default); it never claims to detect units from the file, and the label has no effect on the actual numbers.

**Limitations specific to the app** (beyond the library's own, documented above): very large meshes (out of caution, anything over 300,000 faces) require an explicit acknowledgment before running the calculation, since runtime scales with mesh size and grid resolution; the 3D visualization decimates meshes above 30,000 faces purely for browser performance (the calculation itself always runs on the full, undecimated mesh); build direction in the UI is restricted to the same two Z-axis options the library itself supports, not a free-form vector, so the app can never hit the library's `NotImplementedError` path through normal use.

**Testing**: `tests/test_app.py` drives the app end to end via Streamlit's own official testing framework (`streamlit.testing.v1.AppTest`), including simulating a real file upload through the actual widget -- not just calling the underlying library functions app.py happens to use. This is also how a real bug was caught: `trimesh.load` doesn't raise on invalid/corrupt STL bytes, it silently returns an empty `Scene`; unhandled, that would have reached later code expecting a real mesh and crashed with a raw, unhandled exception. `test_invalid_file_shows_clean_error_not_a_crash` is a permanent regression test for exactly that. Skipped automatically (not failed) if the optional `app` extra isn't installed.

## A. Reference baseline (from the paper)

Shonkwiler et al.'s [IDETC-2026 paper](https://doi.org/10.1115/DETC2026-193146) on comparing 3D shape representations for ML prediction of LPBF support volume computes, for every facet whose normal is within 50 degrees of straight down:

```
support_volume = sum over overhanging facets(
    facet_centroid_height x horizontally_projected_facet_area
)
```

This is `compute_naive_support_volume()` in this repository -- a faithful reproduction of the paper's own method, not a strawman. The paper is explicit about its own limitation:

> "Intersections (i.e., where support material intersected with the part above the build platform) were ignored, and we assumed solid support that would fill the entire truncated pyramid volume... improvements to the approach we took here of generating the support volume labels (e.g., handling self intersections) would improve the value of the ML predictions themselves."

## B. Proposed method (this repository's contribution)

`compute_integrated_support_volume()` replaces the single per-facet centroid-height estimate with a per-column, ray-cast height-field integration:

1. Grid-sample the union of overhanging facets' XY footprints.
2. Cast one vertical ray per sample column through the *entire* mesh (not just the overhanging facet), collecting every intersection.
3. Classify each intersection as the ray entering or exiting solid material via the sign of `ray_direction . face_normal`.
4. Walk the intersections top-to-bottom with a solid-depth counter (robust to exactly-coincident/touching surfaces) and sum the height of every open-air gap between the overhang surface and the build plate.
5. Multiply by cell area and sum over all columns.

This single integration handles both of the following automatically, without separate special-case logic for either:

## C. What problem it addresses

- **Support/part self-intersection** -- where the part's own geometry (e.g. a feature below an overhang) already occupies part of a support column, so no material is needed there. Naive: assumes a solid column regardless. This repo: measures the actual open-air gap.
- **Overlapping projected facets** -- where two separate overhanging facets project onto overlapping XY footprints, which the naive per-facet sum double-counts. This repo: samples the *union* footprint once, so shared area is only integrated once.

Neither correction requires the part's own geometry or the facet layout to be known in advance; both fall out of doing full-mesh column analysis instead of independent per-facet summation.

## D. What this does NOT claim

- **Not a slicer.** It does not generate printable support structures, tree supports, or account for support density/pattern, breakaway force, or interface layers.
- **Not an exact CSG solution.** The correction is a grid approximation (see Validation below for quantified convergence), not a closed-form boolean-mesh computation. A true CSG approach was deliberately not used -- see the design-decision note below.
- **Does not reproduce every commercial slicer's support strategy.** Real slicers (Cura, PrusaSlicer, etc.) make different tradeoffs (support angle, pattern, density, interface gap) that this tool does not model.
- **Does not automatically produce a physically accurate printed support volume.** It estimates the *geometric* open-air volume beneath an overhang under this method's own assumptions (50-degree threshold, a single build direction), not the material actually consumed by a specific slicer/printer/support-style combination.
- **Grid discretization remains** -- results converge with finer resolution (see below) but are never treated as exact anywhere in this codebase; the integrated result is never called "true volume", only "integrated" or "corrected".

### Why ray-casting instead of full 3D boolean CSG

A textbook-exact solution to the self-intersection problem is full mesh boolean subtraction (support-column solid minus part solid, unioned across facets). This was deliberately not implemented: real-world STL files are frequently non-manifold or have inconsistent geometry (see the 3DBenchy benchmark below, which has 1,486 broken faces), and boolean mesh operations are notoriously fragile on such input without heavy mesh-repair preprocessing. Per-column ray-mesh intersection is a robust, well-supported operation by comparison. This is a deliberate engineering tradeoff, not an oversight -- see the git history for how this was scoped.

### Current limitation: build direction

The build direction (equivalently, "down", the direction support must span) is an explicit parameter (`build_direction`, default `(0, 0, 1)`, i.e. printing proceeds in +Z and support spans down to `z_ground`), not silently hard-coded -- but **only directions parallel to the Z axis are currently supported**; passing anything else raises `NotImplementedError` rather than silently giving a wrong answer. Supporting arbitrary build directions is possible (rotate the mesh to align the requested direction with -Z, run the existing pipeline, since volume is rotation-invariant) but was intentionally not implemented in this pass to avoid a half-tested feature; see the issue tracker for this as a scoped follow-up.

**A real bug this exposed**: the flipped direction (`(0, 0, -1)`) was added to the API but initially shipped silently broken -- both the ray origin and the column-integration sign convention still hard-coded a "-Z is down" assumption despite the parameter existing, so it returned 0 for everything instead of erroring or working. This was only caught by manually exercising the non-default branch after the fact, not by the test suite (which, like the code, only exercised the default direction). Fixed, and `tests/test_build_direction.py` now specifically covers both the trivial case and a mirrored version of the self-intersection fixture under the flipped direction, so this can't regress silently again. Kept here as a concrete illustration of why "the API accepts a parameter" and "the parameter works" are different claims.

## Validation

### Analytical synthetic cases

Four synthetic geometries with hand-computed ground truth (`supvol/fixtures.py`), each isolating one property:

| Case | Naive | Integrated | Ground truth | What it tests |
|---|---|---|---|---|
| `shelf_and_pillar` | 640 | 600 | 600 (exact) | self-intersection: a shelf resting on a pillar it overhangs beyond |
| `simple_overhang` | 80 | 80 | 80 (exact) | sanity check: no correction invented where none is needed |
| `tilted_overhang` | 93.53 | 93.58 | 93.53 | non-axis-aligned facet angle math (0.05% error, from grid discretization at a sloped boundary) |
| `overlapping_overhangs` | 144 | 124 | 124 (exact) | facet-to-facet double-counting: two separate overhangs with overlapping footprints |

Additional coverage beyond the four base cases:
- **Rotation** (`tests/test_rotation.py`): the sanity-check and self-intersection cases re-tested under 5 yaw angles (0-123 degrees) to confirm the algorithm isn't accidentally relying on axis-aligned footprints.
- **Grid-phase sensitivity** (`tests/test_grid_sensitivity.py`): a rigid whole-mesh translation is proven *exactly* invariant (the sampling grid's origin is anchored to the overhang footprint's own bounding box, so geometry and grid always move together). The practically meaningful question -- does the sub-resolution *relative* alignment between two separate solids change the answer -- is separately quantified: sweeping the pillar's position by several sub-resolution offsets against a fixed shelf produced a 0.17% spread, well within the resolution-justified 3% bound.
- **Input validation & mesh diagnostics** (`tests/test_validation.py`): empty/zero-face/non-finite-vertex meshes raise clear errors; invalid resolution/angle-threshold raise; degenerate faces and non-watertight/non-manifold meshes warn (never silently repaired); `MeshDiagnostics` exposes face/vertex counts, watertightness, zero-area face count, duplicate face count, and non-manifold/boundary edge counts.

### Convergence evidence

`tests/test_convergence.py` runs the self-intersection and overlapping-overhang cases at five resolutions (0.4 down to 0.025). One honest finding from building this: the *un-rotated*, axis-aligned fixtures hit exactly zero error at several "nice" resolutions purely because the grid phase happens to land exactly on their integer-coordinate boundaries -- a coincidence of the synthetic geometry, not a general guarantee. The representative evidence uses a yaw-rotated version of each case (ground truth unchanged by rotation, but the axis-alignment coincidence removed):

| Resolution | shelf_and_pillar (rotated) rel. error | overlapping_overhangs (rotated) rel. error |
|---|---|---|
| 0.4 | 0.53% | 1.16% |
| 0.2 | 0.27% | 0.23% |
| 0.1 | 0.10% | 0.27% |
| 0.05 | 0.02% | 0.15% |
| 0.025 | 0.00% | 0.01% |

Reproduce with: `python scripts/convergence_study.py --fixture shelf_and_pillar --resolutions 0.4 0.2 0.1 0.05 0.025`

Error bounds in the test suite are derived from geometry (quantization error near a boundary scales as `O(perimeter x resolution x height)`), not fitted to whatever number the code happened to produce.

### Real-mesh benchmark

Tested against [3DBenchy](https://www.3dbenchy.com/) (public-domain 3D-printing calibration model, 225,706 faces, genuinely non-watertight with 1,486 broken faces):

| Resolution | Naive | Integrated | Reduction | Runtime | Sample columns |
|---|---|---|---|---|---|
| 0.5 | 14,316 mm³ | 9,642 mm³ | 32.6% | ~12-15s | 3,000 |
| 0.25 | 14,316 mm³ | 9,491 mm³ | 33.7% | ~15s | 11,908 |

Result was stable before and after attempting `trimesh.repair.fill_holes` on the mesh (identical to 4 significant figures) -- the known non-watertight regions didn't happen to fall under a sampled overhang column for this part. This is evidence of robustness *on this mesh*, not a general guarantee for every possible mesh defect (see `MeshDiagnostics` and the validation warnings above).

Reproduce with: `python scripts/validate_real_mesh.py --download-benchy --resolution 0.5 0.25`

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```python
import trimesh
from supvol import compute_support_volume

mesh = trimesh.load("your_part.stl")
result = compute_support_volume(mesh, resolution=0.5)

print(result.naive_volume, result.integrated_volume, result.reduction_fraction)
print(result.diagnostics)       # MeshDiagnostics: watertight, zero-area faces, etc.
result.to_dict()                # JSON-serializable
```

The baseline and proposed methods are also exposed separately if you only need one:

```python
from supvol import compute_naive_support_volume, compute_integrated_support_volume
```

## Reproducible commands

```bash
# fast, deterministic test suite (this is what CI runs)
pytest tests/ -v

# grid-convergence experiment against a synthetic fixture (known ground truth)
python scripts/convergence_study.py --fixture shelf_and_pillar --resolutions 0.4 0.2 0.1 0.05 0.025

# grid-convergence experiment against your own STL
python scripts/convergence_study.py --stl your_part.stl --resolutions 1.0 0.5 0.25 --out results.csv

# real-mesh benchmark (downloads the public-domain 3DBenchy model)
python scripts/validate_real_mesh.py --download-benchy --resolution 0.5 0.25

# real-mesh benchmark against your own STL
python scripts/validate_real_mesh.py --stl your_part.stl --resolution 0.5

# interactive demo UI
pip install -e ".[app]"
streamlit run app.py
```

## Project status / next logical experiment

This estimator is validated on synthetic analytical cases (including rotation and grid-phase sensitivity) and one real benchmark mesh. It has not yet been validated against a second, independently-labeled real dataset, or against an existing slicer's own support-volume output as a cross-check. See the repository's issue tracker / commit history for the current scope boundary (Z-axis-only build direction, no PySLM integration, no ML, no GUI -- all deliberate, not oversights).

## License

MIT (see `LICENSE`).
