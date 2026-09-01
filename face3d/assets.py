"""Locating the FLAME model, without hard-coding a path into every script.

None of the FLAME variants may be committed (non-commercial licence, and 2023
Open is CC-BY but still 51 MB), so tests cannot assume the model is present.
Two consequences are handled here:

  * `model_path()` searches the usual places and returns None rather than
    raising, so callers can skip cleanly instead of failing a build for a
    missing licensed file.
  * FACE3D_MODEL overrides the search. CI points it at the synthetic fixture
    from tests/fixtures/make_fixture.py, which shares FLAME's pickle structure but
    contains no MPI data — so the rasteriser, LBS, gradient and interface tests
    all run on a public runner.
"""

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

CANDIDATES = [
    "FLAME2020/generic_model.pkl",
    "FLAME2023Open/flame2023_Open.pkl",
    "tests/fixtures/tiny_head.pkl",
]


def model_path(variant=None):
    """Path to a FLAME-format model, or None if none is available."""
    env = os.environ.get("FACE3D_MODEL")
    if env:
        p = pathlib.Path(env)
        return p if p.exists() else None
    if variant:
        p = ROOT / variant
        return p if p.exists() else None
    for c in CANDIDATES:
        p = ROOT / c
        if p.exists():
            return p
    return None


def model_path_or_skip(variant=None):
    """Path, or exit(0) with a SKIP message.

    Exiting 0 is deliberate: an absent licensed asset is not a test failure, and
    a red build for it would train everyone to ignore red builds.
    """
    p = model_path(variant)
    if p is None:
        print(f"SKIP — no FLAME model found. Looked for {CANDIDATES}, "
              f"and FACE3D_MODEL is unset.\n"
              f"      Run `python tests/fixtures/make_fixture.py` for a synthetic "
              f"stand-in, or place a FLAME model in the project root.")
        sys.exit(0)
    return p


def is_fixture(p) -> bool:
    return pathlib.Path(p).name == "tiny_head.pkl"
