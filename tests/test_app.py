"""End-to-end tests for the Streamlit app (app.py), driven through Streamlit's
own official testing framework (streamlit.testing.v1.AppTest).

This directly closes a gap flagged in the frontend's own commit history: the
built-in browser automation available at the time couldn't drive the actual
file-upload widget (OS-level file pickers aren't scriptable that way), so
the upload path was only verified manually. AppTest's file_uploader.upload()
simulates a real upload without needing a browser at all -- these tests
exercise the *exact* upload -> Calculate -> results flow a real user drives,
not just the underlying library functions app.py happens to call.

Skipped entirely (not failed) if the optional "app" extra isn't installed,
so the core test suite doesn't require streamlit/plotly.
"""

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
pytest.importorskip("plotly")

from supvol.fixtures import shelf_and_pillar, simple_overhang, tilted_overhang  # noqa: E402

AppTest = st_testing.AppTest
TIMEOUT = 30


def _upload_and_calculate(at, mesh, filename="test.stl"):
    stl_bytes = mesh.export(file_type="stl")
    at.file_uploader[0].upload(filename=filename, content=stl_bytes, mime_type="application/octet-stream")
    at.run(timeout=TIMEOUT)
    assert not at.exception, f"unexpected exception after upload: {at.exception}"

    calculate_button = next(b for b in at.button if b.label == "Calculate")
    assert not calculate_button.disabled, "Calculate should be enabled once a valid mesh is loaded"
    calculate_button.click()
    at.run(timeout=TIMEOUT)
    assert not at.exception, f"unexpected exception after Calculate: {at.exception}"
    return at


def _metric_value(at, label):
    for m in at.metric:
        if m.label == label:
            return m.value
    raise AssertionError(f"no metric with label {label!r} found; metrics present: {[m.label for m in at.metric]}")


def test_upload_and_calculate_matches_ground_truth():
    """The full real-user flow: upload a real STL through the actual widget,
    click Calculate, and check the displayed numbers -- not just that the
    library function returns the right thing, but that the app correctly
    wires the upload through to a result."""
    mesh, truth = simple_overhang()

    at = AppTest.from_file("../app.py", default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception

    _upload_and_calculate(at, mesh)

    assert _metric_value(at, "Faces") == str(len(mesh.faces))
    assert _metric_value(at, "Watertight") == "Yes"
    assert f"{truth['true_support_volume']:.2f}" in _metric_value(at, "Reference (naive) support volume")
    assert f"{truth['true_support_volume']:.2f}" in _metric_value(at, "SUPVOL integrated volume")


def test_upload_self_intersection_case_shows_correction():
    """The harder case: confirm the app actually surfaces the self-intersection
    correction end to end (naive > integrated), not just that it runs."""
    mesh, truth = shelf_and_pillar()

    at = AppTest.from_file("../app.py", default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    _upload_and_calculate(at, mesh)

    naive_str = _metric_value(at, "Reference (naive) support volume")
    integrated_str = _metric_value(at, "SUPVOL integrated volume")
    naive_val = float(naive_str.split()[0].replace(",", ""))
    integrated_val = float(integrated_str.split()[0].replace(",", ""))

    assert naive_val == pytest.approx(truth["naive_support_volume"], rel=1e-6)
    assert integrated_val == pytest.approx(truth["true_support_volume"], rel=0.02)
    assert naive_val > integrated_val


def test_invalid_file_shows_clean_error_not_a_crash():
    """Regression test for the real bug caught during manual testing:
    trimesh.load doesn't raise on invalid STL bytes (it silently returns an
    empty Scene), which previously would have reached later code expecting
    a Trimesh and crashed with a raw, unhandled AttributeError. Confirms the
    app shows a clean error instead of raising past Streamlit's own handling."""
    at = AppTest.from_file("../app.py", default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)

    at.file_uploader[0].upload(
        filename="not_really_stl.stl", content=b"this is not a real STL file", mime_type="application/octet-stream"
    )
    at.run(timeout=TIMEOUT)

    assert not at.exception, (
        f"an invalid file must produce a clean st.error(), not an unhandled exception: {at.exception}"
    )
    assert len(at.error) > 0, "expected a clean st.error() message for an invalid file"


def test_near_zero_correction_does_not_show_negative_zero():
    """Regression test for a real display bug caught via manual testing on the
    live deployment: tilted_overhang has ~zero true correction, and at a fine
    grid resolution, discretization noise can make the integrated estimate
    marginally *larger* than naive -- this produced a confusing "-0.0%" /
    "-0 mm3 below reference" negative-zero artifact. Exercises the actual
    near-zero-or-negative branch, not just the normal positive-reduction path
    the other tests cover. Doesn't assume which sign the noise takes (that can
    vary across platforms -- see the coincident-surface CI lesson), only that
    whichever sign it is, it's displayed cleanly and worded correctly."""
    mesh, truth = tilted_overhang()
    at = AppTest.from_file("../app.py", default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)

    resolution_input = next(n for n in at.number_input if n.label == "Grid resolution")
    resolution_input.set_value(0.02).run(timeout=TIMEOUT)

    _upload_and_calculate(at, mesh)

    pct_str = _metric_value(at, "Reduction vs. reference")
    assert pct_str != "-0.0%", f"negative-zero display artifact: {pct_str!r}"

    diff_caption = next(c.value for c in at.caption if "below reference" in c.value or "above reference" in c.value)
    assert not diff_caption.startswith("-0 "), f"negative-zero display artifact: {diff_caption!r}"

    # Sign-vs-wording must agree, but only when the *displayed* magnitude is
    # actually non-zero -- when it rounds to display-zero, "below"/"above" is
    # moot (0 mm3 either way) and either wording is fine.
    displayed_magnitude = float(diff_caption.split()[0].replace(",", ""))
    if displayed_magnitude != 0:
        naive_val = float(_metric_value(at, "Reference (naive) support volume").split()[0].replace(",", ""))
        integrated_val = float(_metric_value(at, "SUPVOL integrated volume").split()[0].replace(",", ""))
        if integrated_val > naive_val:
            assert "above reference" in diff_caption, f"integrated > naive but caption says: {diff_caption!r}"
        elif integrated_val < naive_val:
            assert "below reference" in diff_caption, f"integrated < naive but caption says: {diff_caption!r}"


def test_empty_state_calculate_button_disabled():
    """Before anything is loaded, Calculate must be disabled -- not just
    'the user shouldn't click it', an actual disabled widget state."""
    at = AppTest.from_file("../app.py", default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    calculate_button = next(b for b in at.button if b.label == "Calculate")
    assert calculate_button.disabled
