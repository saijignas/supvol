# supvol

A self-intersection-aware support-volume estimator for additive manufacturing (LPBF / FDM-style overhang support).

## Why

Shonkwiler et al.'s [IDETC-2026 paper](https://doi.org/10.1115/DETC2026-193146) on comparing 3D shape representations for ML prediction of LPBF support volume computes support volume as `centroid_height x projected_area` for each overhanging facet, and explicitly notes:

> "Intersections (i.e., where support material intersected with the part above the build platform) were ignored, and we assumed solid support that would fill the entire truncated pyramid volume... improvements to the approach we took here of generating the support volume labels (e.g., handling self intersections) would improve the value of the ML predictions themselves."

This project closes that gap: it replaces the single per-facet centroid-height estimate with a per-column, ray-cast height-field integration that correctly subtracts:
1. **Self-intersection** -- where the part's own geometry (e.g. a lower feature) already occupies part of an overhang's support column, so no support material is needed there.
2. **Facet-to-facet overlap** -- where two separate overhanging facets project onto overlapping footprints, which a naive per-facet sum double-counts.

## Results

| Test case | Naive | Self-intersection-aware | Notes |
|---|---|---|---|
| Synthetic: shelf over a supporting pillar | 640 | 600 | exact match to hand-computed truth (difference = pillar's own volume) |
| Synthetic: isolated overhang (sanity check) | 80 | 80 | confirms the method doesn't invent corrections where none exist |
| Synthetic: tilted (non-axis-aligned) overhang | 93.53 | 93.58 | 0.05% error, from grid discretization at a sloped boundary -- expected and documented, not a bug |
| Synthetic: two separate overlapping overhangs | 144 | 124 | exact match to hand-computed truth |
| **Real mesh: [3DBenchy](https://www.3dbenchy.com/)** (225,706 faces, public-domain 3D-printing benchmark model) | 14,316 mm³ | 9,642 mm³ | **32.6% reduction**, consistent at finer grid resolution (33.7% at 0.25mm vs 0.5mm), and stable before/after mesh repair |

## Method

For each grid column sampled inside the union footprint of overhanging facets (facets whose normal is within 50 degrees of straight down, matching the source paper's own threshold):

1. Cast a single ray straight down through the *entire* mesh, collecting every intersection.
2. Classify each hit as the ray entering or exiting solid material via the sign of `ray_direction . face_normal`.
3. Walk the hits top-to-bottom with a solid-depth counter (robust to exactly-coincident/touching surfaces, unlike a plain in/out boolean) and sum the height of every open-air gap between the topmost surface and the build plate.
4. Multiply by cell area and sum over all columns.

This single per-column integration handles both self-intersection and facet-to-facet overlap automatically, without any special-case logic for either -- both fall out of doing full-mesh column analysis instead of independent per-facet summation.

## Known limitations

- Grid-based, not exact CSG: results converge with finer resolution but are an approximation, same tradeoff the source papers themselves make with voxel/point-cloud representations.
- Assumes a reasonably well-formed mesh. Tested against a real, non-watertight STL (1,486 broken faces) with stable results, but a differently-broken mesh could corrupt a column's hit parity if the defect falls directly under a sampled overhang. No automatic repair or malformed-parity detection is implemented yet.

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```python
import trimesh
from supvol import compute_support_volume

mesh = trimesh.load("your_part.stl")
naive_volume, true_volume, n_columns_sampled = compute_support_volume(mesh, resolution=0.5)
```

## Test

```bash
pytest tests/ -v
```
