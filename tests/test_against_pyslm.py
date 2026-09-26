"""Independent cross-check of compute_naive_support_volume() against PySLM.

Shonkwiler et al. (IDETC-2026) state their labels were computed by "using
PySLM [22] to identify overhanging facets... calculating the truncated
pyramid volume under each" -- exactly the formula compute_naive_support_volume()
implements. This confirms that reproduction directly against PySLM's own
getOverhangMesh() + approximateSupportMomentArea(), not just this repo's own
hand-computed fixtures.

PySLM is NOT a dependency of the supvol package or app.py -- it is only used
here, and only if already installed (pip install "PythonSLM[support]"),
skipped entirely (not failed) otherwise. See scripts/validate_against_pyslm.py
for a standalone runnable version, including against a real STL.
"""

import pytest

pyslm = pytest.importorskip("pyslm")
pyslm_support = pytest.importorskip("pyslm.support")

from supvol.fixtures import overlapping_overhangs, shelf_and_pillar, simple_overhang, tilted_overhang  # noqa: E402
from supvol.raycast import compute_naive_support_volume  # noqa: E402

ANGLE_THRESHOLD_DEG = 50.0


def _pyslm_naive_volume(mesh) -> float:
    # Deliberately not calling part.dropToPlatform(): several fixtures float
    # above z=0 on purpose (to test height is measured from the real build
    # plate, not the mesh's own bounding box) -- dropping to platform would
    # shift them and produce a mismatch that has nothing to do with the
    # actual algorithm.
    part = pyslm.Part("validation")
    part.setGeometryByMesh(mesh)
    return pyslm_support.approximateSupportMomentArea(part, ANGLE_THRESHOLD_DEG)


@pytest.mark.parametrize(
    "factory", [shelf_and_pillar, simple_overhang, tilted_overhang, overlapping_overhangs]
)
def test_naive_volume_matches_pyslm(factory):
    mesh, _truth = factory()
    ours = compute_naive_support_volume(mesh, angle_threshold_deg=ANGLE_THRESHOLD_DEG)
    theirs = _pyslm_naive_volume(mesh)
    assert ours == pytest.approx(theirs, rel=1e-6)
