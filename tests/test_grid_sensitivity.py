"""Grid-phase / translation sensitivity.

Two distinct questions are worth separating here:

1. Does *rigidly* translating the whole part in world coordinates change the
   answer? By construction, no: the sampling grid's origin is anchored to
   the overhang footprint's own bounding box (see
   ``supvol/raycast.py::_footprint_sample_points``), so a rigid translation
   moves the grid and the geometry together and the relative alignment
   between them -- which is all that matters for point-in-triangle and
   column-integration math -- never changes. This is a real, provable
   property, and it's checked directly below rather than just asserted.

2. Does the *relative* sub-resolution alignment between two separate solids
   in the same part (here, the pillar vs. the shelf whose footprint anchors
   the grid) change the answer? This is the practically meaningful
   discretization-sensitivity question -- e.g. if the pillar sits at a
   slightly different sub-millimeter position for manufacturing/design
   reasons, does the computed correction jump around non-physically? Ground
   truth doesn't depend on the pillar's exact position (as long as it stays
   within the shelf's footprint), so this is directly measurable rather than
   just plausible.
"""

import numpy as np

from supvol import compute_support_volume
from supvol.fixtures import shelf_and_pillar

RESOLUTION = 0.05
TRUE_VOLUME = 600.0


def test_rigid_whole_mesh_translation_is_exactly_invariant():
    mesh, _ = shelf_and_pillar()
    offsets = [(0.0, 0.0), (0.5, -0.3), (1.234, 5.678), (-10.0, 3.0), (0.5 * RESOLUTION, 0.5 * RESOLUTION)]

    volumes = []
    for dx, dy in offsets:
        m = mesh.copy()
        m.apply_translation([dx, dy, 0.0])
        result = compute_support_volume(m, resolution=RESOLUTION)
        volumes.append(result.integrated_volume)

    volumes = np.array(volumes)
    spread = volumes.max() - volumes.min()
    # "exactly" up to floating point noise from translating coordinates, not
    # from any grid-phase effect
    assert spread < 1e-6, (
        f"expected exact invariance under rigid translation, got spread={spread} "
        f"across volumes={volumes.tolist()}"
    )


def test_pillar_subresolution_offset_sensitivity():
    """Shift the pillar by several sub-resolution amounts relative to the
    shelf (which anchors the sampling grid) and quantify how much the
    integrated volume moves. Ground truth (600) is unchanged by construction
    for all of these offsets, so any spread here is pure discretization
    sensitivity, not a real geometric effect.
    """
    offsets = [
        (0.0, 0.0),
        (0.5 * RESOLUTION, 0.0),
        (0.0, 0.5 * RESOLUTION),
        (0.25 * RESOLUTION, 0.75 * RESOLUTION),
        (-0.4 * RESOLUTION, 0.2 * RESOLUTION),
        (0.9 * RESOLUTION, -0.9 * RESOLUTION),
    ]

    volumes = []
    for dx, dy in offsets:
        mesh, _ = shelf_and_pillar(pillar_offset=(dx, dy))
        result = compute_support_volume(mesh, resolution=RESOLUTION)
        volumes.append(result.integrated_volume)

    volumes = np.array(volumes)
    spread = volumes.max() - volumes.min()
    mean = volumes.mean()
    rel_spread = spread / TRUE_VOLUME

    print(
        f"\npillar sub-resolution offset sensitivity: "
        f"volumes={np.round(volumes, 3).tolist()}, "
        f"max={volumes.max():.3f}, min={volumes.min():.3f}, mean={mean:.3f}, "
        f"spread={spread:.3f} ({rel_spread:.2%} of true volume)"
    )

    # Justified bound: quantization error near a boundary is O(perimeter x
    # resolution x height). The pillar's perimeter is 8, height ~10, so at
    # resolution=0.05 the expected error scale is ~8*0.05*10 = 4, i.e. under
    # ~1% of the true 600 -- allow a bit of headroom (3%) rather than tuning
    # this to whatever the code happens to produce.
    assert rel_spread < 0.03, f"grid-phase sensitivity too large: {rel_spread:.2%} of true volume"

    # And every individual value should still be a reasonable estimate of
    # the true answer on its own, not just self-consistent.
    for v in volumes:
        assert abs(v - TRUE_VOLUME) / TRUE_VOLUME < 0.03
