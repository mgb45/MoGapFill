from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def gait_raw_csv_path() -> Path:
    return DATA_DIR / "gait-raw.csv"


@pytest.fixture
def golden_output() -> np.ndarray:
    return np.load(DATA_DIR / "golden_output_seed0.npy")


@pytest.fixture
def synthetic_c3d_path(tmp_path) -> Path:
    """A small, valid, synthetic C3D file for exercising the C3D I/O
    functions. There's no real binary C3D sample data in this repo (the
    once-included `gait-raw.c3d` turned out to actually be CSV text with
    the wrong extension, not a real C3D file), so this generates a minimal
    one on the fly instead of relying on a fabricated committed fixture.
    """
    ezc3d = pytest.importorskip("ezc3d")

    rng = np.random.default_rng(0)
    marker_names = ["M0", "M1", "M2", "M3"]
    n_frames = 30

    c3d = ezc3d.c3d()
    c3d["parameters"]["POINT"]["RATE"]["value"] = [100.0]
    c3d["parameters"]["POINT"]["LABELS"]["value"] = marker_names

    points = np.zeros((4, len(marker_names), n_frames))
    points[:3, :, :] = rng.normal(size=(3, len(marker_names), n_frames)).cumsum(axis=2)
    points[3, :, :] = 0.0  # residual: 0 = valid
    # mark a gap: marker 1 occluded for frames 10-14. NaN coordinates are
    # what actually survives an ezc3d write/read round trip reliably (the
    # residual channel is not preserved as written), so mark both.
    points[:3, 1, 10:15] = np.nan
    points[3, 1, 10:15] = -1.0

    c3d["data"]["points"] = points
    path = tmp_path / "synthetic.c3d"
    c3d.write(str(path))
    return path
