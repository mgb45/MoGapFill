import numpy as np
import pytest

from mogapfill import fill_gaps, read_c3d, read_csv, write_c3d, write_csv


def test_read_csv_shape_and_nans(gait_raw_csv_path):
    data = read_csv(gait_raw_csv_path)
    assert data.shape == (142, 39)
    assert np.isnan(data).any()


def test_csv_round_trip(tmp_path, gait_raw_csv_path):
    data = read_csv(gait_raw_csv_path)
    out_path = tmp_path / "roundtrip.csv"
    write_csv(out_path, data)
    reloaded = read_csv(out_path)
    np.testing.assert_allclose(reloaded, data, equal_nan=True)


def test_read_c3d(synthetic_c3d_path):
    data, marker_names = read_c3d(synthetic_c3d_path)
    assert marker_names == ["M0", "M1", "M2", "M3"]
    assert data.shape == (30, 12)
    # marker M1 was marked occluded (negative residual) for frames 10-14
    assert np.isnan(data[10:15, 3:6]).all()
    assert not np.isnan(data[:10, 3:6]).any()


def test_c3d_roundtrip_through_fill_gaps(tmp_path, synthetic_c3d_path):
    data, marker_names = read_c3d(synthetic_c3d_path)
    filled = fill_gaps(data, tol=0.1, random_state=0)
    assert not np.isnan(filled).any()

    out_path = tmp_path / "filled.c3d"
    write_c3d(synthetic_c3d_path, out_path, filled)

    reloaded, reloaded_names = read_c3d(out_path)
    assert reloaded_names == marker_names
    assert not np.isnan(reloaded).any()
    np.testing.assert_allclose(reloaded, filled, rtol=1e-4, atol=1e-4)


def test_from_dataframe_to_dataframe_round_trip():
    pytest.importorskip("pandas")
    data = np.arange(2 * 6, dtype=float).reshape(2, 6)
    marker_names = ["a", "b"]
    from mogapfill.io import from_dataframe, to_dataframe

    df = to_dataframe(data, marker_names)
    assert list(df.columns) == ["a_x", "a_y", "a_z", "b_x", "b_y", "b_z"]

    round_tripped, names = from_dataframe(df)
    assert names == marker_names
    np.testing.assert_allclose(round_tripped, data)
