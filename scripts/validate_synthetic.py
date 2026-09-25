"""Quick validation script: run the self-intersection-aware algorithm against
the synthetic fixtures where the correct answer is known by hand, and check
that it beats the naive estimate and converges to ground truth."""

from supvol.fixtures import shelf_and_pillar, simple_overhang, tilted_overhang, overlapping_overhangs
from supvol.raycast import compute_support_volume


def report(name, mesh, truth, resolution=0.05):
    naive, true, n_pts = compute_support_volume(mesh, resolution=resolution)
    print(f"\n=== {name} (grid resolution={resolution}, {n_pts} sample columns) ===")
    print(f"  naive volume (paper's method):        {naive:.3f}")
    if "naive_support_volume" in truth:
        print(f"  naive volume (hand-computed truth):   {truth['naive_support_volume']:.3f}")
    print(f"  computed self-intersection-aware:     {true:.3f}")
    print(f"  hand-computed true answer:             {truth['true_support_volume']:.3f}")
    err = abs(true - truth["true_support_volume"])
    rel_err = err / truth["true_support_volume"] if truth["true_support_volume"] else 0
    print(f"  absolute error: {err:.3f}   relative error: {rel_err:.2%}")
    return rel_err


if __name__ == "__main__":
    mesh, truth = shelf_and_pillar()
    err1 = report("shelf_and_pillar (self-intersection case)", mesh, truth, resolution=0.05)

    mesh2, truth2 = simple_overhang()
    err2 = report("simple_overhang (sanity check, no intersection)", mesh2, truth2, resolution=0.05)

    mesh3, truth3 = tilted_overhang()
    print(f"\n  (tilted_overhang: {truth3['n_overhang_facets']} facets flagged as overhanging -- should be 2, the split triangles of the single sloped bottom face)")
    err3 = report("tilted_overhang (angle math check)", mesh3, truth3, resolution=0.02)

    mesh4, truth4 = overlapping_overhangs()
    err4 = report("overlapping_overhangs (facet-to-facet overlap)", mesh4, truth4, resolution=0.05)

    print("\n--- summary ---")
    errs = {"shelf_and_pillar": err1, "simple_overhang": err2, "tilted_overhang": err3, "overlapping_overhangs": err4}
    ok = all(e < 0.02 for e in errs.values())
    print("PASS" if ok else "FAIL", errs)
