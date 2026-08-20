"""Proves the ported algorithm's reconstruction is numerically equivalent
to the original (pre-port) implementation for the code paths that weren't
intentionally changed. `tests/data/golden_output_seed0.npy` was generated
by running the original algorithm (with only its `print` statements fixed
to Python 3 syntax -- no other change) against `gait-raw.csv` with a fixed
random seed, before any of the porting/refactoring work began.
"""

import numpy as np

from mogapfill import fill_gaps


def test_matches_original_algorithm_output(gait_raw_csv_path, golden_output):
    data = np.genfromtxt(gait_raw_csv_path, delimiter=",")
    result = fill_gaps(data, 0.0025, 1e-3, random_state=0)
    np.testing.assert_allclose(result, golden_output, rtol=1e-10, atol=1e-12)
