# MoGapFill

Low-dimensional Kalman smoother that fills gaps in motion capture marker trajectories.

Frames where every marker is observed are used to fit a low-dimensional
PCA/SVD subspace of the motion; a Kalman filter forward pass and RTS
smoother backward pass then track the trajectory through that subspace,
naturally handling frames with arbitrary per-marker gaps.

If you use this, please cite:

> Burke, M. and Lasenby, J. (2016). Estimating missing marker positions
> using low dimensional Kalman smoothing. *Journal of Biomechanics*.
> https://doi.org/10.1016/j.jbiomech.2016.04.016

## Installation

Not published on PyPI — install directly from GitHub:

```bash
pip install git+https://github.com/mgb45/MoGapFill
# with pandas DataFrame support:
pip install "mogapfill[pandas] @ git+https://github.com/mgb45/MoGapFill"
```

## Quickstart

CSV:

```python
from mogapfill import read_csv, write_csv, fill_gaps

data = read_csv("markers.csv")        # (frames, 3 * n_markers), NaN for gaps
filled = fill_gaps(data)
write_csv("filled.csv", filled)
```

C3D:

```python
from mogapfill import read_c3d, write_c3d, fill_gaps

data, marker_names = read_c3d("trial.c3d")
filled = fill_gaps(data)
write_c3d("trial.c3d", "trial_filled.c3d", filled)
```

## API

- `fill_gaps(rawdata, tol=0.0025, sig_r=1e-3, keep_original=True, *, d=None, process_noise_scale=1.0, initial_cov_scale=1e12, random_state=None, strict=False, return_uncertainty=False)` — the core algorithm.
- `read_csv` / `write_csv` — flat marker array in/out of a CSV file.
- `read_c3d` / `write_c3d` — flat marker array + marker names in/out of a C3D file (via [`ezc3d`](https://github.com/pyomeca/ezc3d)).
- `mogapfill.io.from_dataframe` / `mogapfill.io.to_dataframe` — flat marker array in/out of a `pandas.DataFrame` with `<marker>_x/y/z` columns (requires the `pandas` extra).

### Missing data

By default, a frame with no markers observed at all is handled gracefully:
the Kalman update is skipped for that frame and the state coasts forward on
the process model alone, then gets smoothed by the backward pass using
neighboring frames, same as any other partially-observed frame. Pass
`strict=True` to instead raise `InsufficientObservationsError` for any such
frame. The one case that always raises `InsufficientObservationsError`,
regardless of `strict`, is when *no* frame has every marker observed —
there's no complete-case data to fit the low-dimensional subspace against.

### Uncertainty

Pass `return_uncertainty=True` to also get a per-coordinate variance
estimate for every frame (`filled, uncertainty = fill_gaps(..., return_uncertainty=True)`)
— near zero at observed values, larger over long/interior gaps.

### Hyperparameters

- `tol` — fraction of cumulative singular-value energy discarded when
  choosing the latent dimensionality (smaller = higher-dimensional, more
  expressive but noisier subspace). Ignored if `d` is given.
- `sig_r` — assumed observation noise variance; larger values trust the
  smoothed trajectory more relative to the raw observations.
- `d` — fix the latent dimensionality directly instead of selecting it from `tol`.
- `process_noise_scale` — multiplier on the process noise, independent of `sig_r`.
- `initial_cov_scale` — how uninformative the filter's initial belief is.
- `random_state` — seeds the (statistically irrelevant but non-deterministic
  by default) random initial state, for reproducible output.

## Data convention

`fill_gaps` itself is marker-name-agnostic: it takes a flat 2D array, one
row per frame, columns `[m0x, m0y, m0z, m1x, m1y, m1z, ...]`, with `NaN`
for missing values. The `read_*`/`write_*` I/O helpers translate to and
from this convention for named formats.

## License

MIT — see [LICENSE](LICENSE).
