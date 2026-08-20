import numpy as np
import pytest

from mogapfill import InsufficientObservationsError, fill_gaps


def _synthetic(n_frames=30, n_markers=4, seed=0):
    rng = np.random.default_rng(seed)
    # low-rank (2 latent dims) trajectory so the model has something to learn.
    latent = rng.normal(size=(n_frames, 2)).cumsum(axis=0)
    basis = rng.normal(size=(2, 3 * n_markers))
    return latent @ basis


def test_shape_preserved(gait_raw_csv_path):
    data = np.genfromtxt(gait_raw_csv_path, delimiter=",")
    out = fill_gaps(data, random_state=0)
    assert out.shape == data.shape


def test_keep_original_passes_through_observed_values(gait_raw_csv_path):
    data = np.genfromtxt(gait_raw_csv_path, delimiter=",")
    out = fill_gaps(data, random_state=0, keep_original=True)
    observed = ~np.isnan(data)
    assert np.array_equal(out[observed], data[observed])


def test_no_nans_in_output(gait_raw_csv_path):
    data = np.genfromtxt(gait_raw_csv_path, delimiter=",")
    out = fill_gaps(data, random_state=0)
    assert not np.isnan(out).any()


def test_raises_when_no_frame_fully_observed():
    data = np.full((5, 6), np.nan)
    with pytest.raises(InsufficientObservationsError):
        fill_gaps(data)


def test_default_coasts_through_fully_missing_interior_frame():
    data = _synthetic()
    data[10, :] = np.nan
    out = fill_gaps(data, tol=0.1, random_state=0)
    assert not np.isnan(out).any()


def test_strict_raises_on_fully_missing_interior_frame():
    data = _synthetic()
    data[10, :] = np.nan
    with pytest.raises(InsufficientObservationsError):
        fill_gaps(data, tol=0.1, random_state=0, strict=True)


def test_return_uncertainty_shape_and_validity():
    data = _synthetic()
    data[10, :] = np.nan
    data[5, 0:3] = np.nan
    _filled, var = fill_gaps(data, tol=0.1, random_state=0, return_uncertainty=True)
    assert var.shape == data.shape
    assert (var >= 0).all()
    observed = ~np.isnan(data)
    assert np.allclose(var[observed], 0.0)
    # uncertainty over the fully-missing frame should be materially higher
    # than at an observed marker elsewhere.
    assert var[10, :].mean() > var[0, :].mean()


def test_d_overrides_tol():
    data = _synthetic()
    out = fill_gaps(data, d=2, random_state=0)
    assert out.shape == data.shape
    assert not np.isnan(out).any()


def test_process_noise_and_initial_cov_scale_affect_output():
    data = _synthetic()
    data[10, :] = np.nan
    base = fill_gaps(data, tol=0.1, random_state=0)
    scaled = fill_gaps(
        data, tol=0.1, random_state=0, process_noise_scale=50.0, initial_cov_scale=1.0
    )
    assert not np.allclose(base, scaled)


def test_recovers_known_values_on_low_rank_trajectory():
    data = _synthetic(n_frames=50, seed=1)
    true = data.copy()
    gapped = data.copy()
    gapped[20, 0] = np.nan
    out = fill_gaps(gapped, tol=0.05, random_state=0)
    assert abs(out[20, 0] - true[20, 0]) < 1.0
