# Contributing

## The one rule that actually matters here

**A new parameter's non-default value needs its own test, not just a test that the default still works.**

This has caused two real bugs in this repo's history, not a hypothetical concern:

1. A synthetic test fixture had two surfaces touching at an exactly coincident plane. It passed locally on Windows and failed in CI on Linux/Python 3.12, because the two platforms' ray intersectors classified the exact seam differently.
2. `build_direction=(0,0,-1)` was added as an explicit, validated API parameter, but every test (like the manual usage that motivated adding it) only ever exercised the default `(0,0,1)`. It shipped silently returning 0 for everything -- not erroring, not obviously wrong, just quietly incorrect -- because two places still hard-coded a "-Z is down" assumption despite the parameter existing to make that configurable.

Neither of these was caught by review or by "does it still pass" -- both were caught by deliberately exercising the untested branch. Before adding a parameter, a code path, or an optimization that's supposed to produce the same answer a different way:

- Write a test for the non-default / new-code-path case specifically, not just a smoke test that nothing else broke.
- If you're optimizing an existing, already-validated implementation (see the footprint-sampling vectorization in `supvol/raycast.py` for an example), keep the original simple version around as a reference and add a test comparing the new version's output against it directly, rather than trusting "the existing test suite still passes" -- the existing suite was written against the old implementation's behavior, not as an independent check of the new one.

## Otherwise

- Keep `pytest tests/ -v` and `ruff check supvol/ tests/ scripts/` passing.
- Ground truth in `supvol/fixtures.py` should be hand-computable and stated in the docstring, not just asserted.
- Don't loosen a test tolerance to make it pass -- figure out whether the tolerance was wrong or the code was.
