"""I/O adapters translating common mocap formats to/from the flat marker
array convention used by :func:`mogapfill.core.fill_gaps`: one row per
frame, columns ``[m0x, m0y, m0z, m1x, m1y, m1z, ...]``, NaN for gaps.

The core algorithm stays name-agnostic; this module is the only place
marker identity/order is handled, via a single column-slice helper, so
each format's read/write functions can't independently drift or
reintroduce indexing bugs.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

__all__ = [
    "from_dataframe",
    "read_c3d",
    "read_csv",
    "to_dataframe",
    "write_c3d",
    "write_csv",
]


def _marker_slice(i: int) -> slice:
    return slice(3 * i, 3 * i + 3)


def read_csv(path: str | Path, has_header: bool = False) -> np.ndarray:
    """Read a marker-position CSV into a flat ``(frames, 3*n_markers)`` array.

    Missing values must be the literal token ``NaN``.
    """
    return np.genfromtxt(path, delimiter=",", skip_header=1 if has_header else 0)


def write_csv(
    path: str | Path,
    data: np.ndarray,
    header: Sequence[str] | None = None,
) -> None:
    """Write a flat marker array to CSV."""
    kwargs = {}
    if header is not None:
        kwargs["header"] = ",".join(header)
        kwargs["comments"] = ""
    np.savetxt(path, data, delimiter=",", **kwargs)


def read_c3d(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Read a C3D file's point data into a flat array + marker names.

    Points whose residual is negative (the C3D convention for an occluded
    / not-tracked marker in that frame) are treated as missing (NaN),
    alongside any point that is already NaN in the file.
    """
    import ezc3d

    c3d = ezc3d.c3d(str(path))
    marker_names = list(c3d["parameters"]["POINT"]["LABELS"]["value"])
    points = c3d["data"]["points"]  # shape (4, n_markers, n_frames): x, y, z, residual

    xyz = points[:3, :, :]
    residual = points[3, :, :]
    missing = (residual < 0) | np.isnan(xyz).any(axis=0)
    xyz = np.where(missing[np.newaxis, :, :], np.nan, xyz)

    n_markers, n_frames = xyz.shape[1], xyz.shape[2]
    data = np.empty((n_frames, 3 * n_markers))
    for i in range(n_markers):
        data[:, _marker_slice(i)] = xyz[:, i, :].T
    return data, marker_names


def write_c3d(path_in: str | Path, path_out: str | Path, filled: np.ndarray) -> None:
    """Write `filled` marker data back into a copy of the C3D at `path_in`.

    All other C3D metadata (rates, units, analog channels, ...) is
    preserved unchanged from `path_in`; only the point trajectories and
    their residuals (marked valid, since the output has no gaps) are
    overwritten.
    """
    import ezc3d

    c3d = ezc3d.c3d(str(path_in))
    n_markers = len(c3d["parameters"]["POINT"]["LABELS"]["value"])
    n_frames = filled.shape[0]
    if filled.shape[1] != 3 * n_markers:
        raise ValueError(
            f"filled has {filled.shape[1]} columns, expected {3 * n_markers} "
            f"for {n_markers} markers in {path_in}"
        )

    points = np.empty((4, n_markers, n_frames))
    for i in range(n_markers):
        points[:3, i, :] = filled[:, _marker_slice(i)].T
    points[3, :, :] = 0.0  # mark all points valid

    c3d["data"]["points"] = points
    c3d.write(str(path_out))


def from_dataframe(df) -> tuple[np.ndarray, list[str]]:
    """Convert a wide DataFrame with columns named ``'<marker>_x/y/z'``
    into a flat array + ordered marker name list. Requires the optional
    ``pandas`` extra.
    """
    marker_names = []
    seen = set()
    for col in df.columns:
        name = col.rsplit("_", 1)[0]
        if name not in seen:
            marker_names.append(name)
            seen.add(name)

    data = np.empty((len(df), 3 * len(marker_names)))
    for i, name in enumerate(marker_names):
        data[:, _marker_slice(i)] = df[[f"{name}_x", f"{name}_y", f"{name}_z"]].to_numpy()
    return data, marker_names


def to_dataframe(data: np.ndarray, marker_names: Sequence[str]):
    """Inverse of :func:`from_dataframe`. Requires the optional ``pandas`` extra."""
    import pandas as pd

    columns = {}
    for i, name in enumerate(marker_names):
        block = data[:, _marker_slice(i)]
        columns[f"{name}_x"] = block[:, 0]
        columns[f"{name}_y"] = block[:, 1]
        columns[f"{name}_z"] = block[:, 2]
    return pd.DataFrame(columns)
