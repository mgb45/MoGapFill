"""Low-dimensional Kalman smoother for filling gaps in motion capture data.

Implements the method of Burke & Lasenby, "Estimating missing marker
positions using low dimensional Kalman smoothing", Journal of Biomechanics
2016 (https://doi.org/10.1016/j.jbiomech.2016.04.016): frames with all
markers observed are used to fit a low-dimensional PCA/SVD subspace of the
motion, and a Kalman filter + RTS smoother tracks the trajectory through
that subspace, naturally handling frames with arbitrary per-marker gaps.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["InsufficientObservationsError", "fill_gaps"]


class InsufficientObservationsError(ValueError):
    """Raised when there is not enough observed data to fit the model.

    Specifically, when no frame in the input has every marker observed,
    there is no complete-case data to fit the low-dimensional subspace
    against, and the algorithm cannot proceed.
    """


def _observation(row: np.ndarray, v_d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the observed sub-vector `z` and selection/observation matrices
    `h`, `ht = h @ v_d.T` for one frame, dropping unobserved markers.
    """
    observed = ~np.isnan(row)
    z = row[observed]
    h = np.diag(observed)
    h = h[~np.all(h == 0, axis=1)]
    return z, h, h @ v_d.T


def fill_gaps(
    rawdata: np.ndarray,
    tol: float = 0.0025,
    sig_r: float = 1e-3,
    keep_original: bool = True,
    *,
    d: int | None = None,
    process_noise_scale: float = 1.0,
    initial_cov_scale: float = 1e12,
    random_state: int | np.random.Generator | None = None,
    strict: bool = False,
    return_uncertainty: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Fill gaps in motion capture marker trajectories.

    Fits a low-dimensional PCA/SVD subspace of the motion from the frames
    where every marker is observed, then runs a Kalman filter forward pass
    and an RTS smoother backward pass through that subspace to estimate
    every frame, including ones with missing markers.

    Parameters
    ----------
    rawdata : np.ndarray, shape (n_frames, 3 * n_markers)
        One row per frame, columns are marker coordinates flattened as
        ``[m0x, m0y, m0z, m1x, m1y, m1z, ...]``. Missing values are ``NaN``.
    tol : float, default 0.0025
        Fraction of cumulative singular-value energy allowed to be
        discarded when choosing the latent dimensionality. Smaller values
        keep a higher-dimensional, more expressive but noisier latent
        subspace. Ignored if `d` is given.
    sig_r : float, default 1e-3
        Assumed observation noise variance (Kalman measurement noise
        ``R = sig_r * I``). Larger values trust the smoothed trajectory
        more relative to the raw per-frame observations.
    keep_original : bool, default True
        If True, originally observed (non-NaN) values are pasted back over
        the model's reconstruction, so only genuinely missing entries are
        filled by the model.
    d : int, optional
        Fix the latent subspace dimensionality directly instead of
        selecting it from `tol`.
    process_noise_scale : float, default 1.0
        Multiplier on the process noise covariance `Q`, independent of
        `sig_r`. Larger values let the latent state change more freely
        between frames (trust the observations more, the motion model
        less).
    initial_cov_scale : float, default 1e12
        Initial state covariance, i.e. how uninformative the filter's
        starting belief is. The default is effectively "no prior".
    random_state : int, np.random.Generator, or None
        Seeds the (statistically irrelevant, but non-deterministic by
        default) random initial state estimate, for reproducible output.
    strict : bool, default False
        If True, raise `InsufficientObservationsError` for any frame with
        no observed markers at all, matching the original implementation's
        behavior. If False (default), such frames are handled gracefully:
        the Kalman update is skipped and the state coasts forward on the
        process model alone, then gets smoothed by the backward pass using
        neighboring frames, same as any other partially-observed frame.
    return_uncertainty : bool, default False
        If True, also return a per-coordinate variance estimate for every
        frame (same shape as the output). This is computed by a separate,
        numerically-stabilized (Joseph-form) Kalman/RTS covariance
        recursion, decoupled from the position reconstruction above, since
        the naive covariance formula used for the reconstruction can lose
        positive-semi-definiteness under typical default parameters
        (particularly the very large default `initial_cov_scale`). It is
        near zero at observed values and larger over long/interior gaps.

    Returns
    -------
    filled : np.ndarray, shape (n_frames, 3 * n_markers)
        The gap-filled trajectories.
    uncertainty : np.ndarray, shape (n_frames, 3 * n_markers)
        Only returned if `return_uncertainty` is True.

    Raises
    ------
    InsufficientObservationsError
        If no frame has every marker observed (nothing to fit the
        subspace against), or if `strict=True` and some frame has no
        markers observed at all.
    """
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    complete = ~np.isnan(rawdata).any(axis=1)
    x = rawdata[complete]
    if x.shape[0] == 0:
        raise InsufficientObservationsError(
            "no frame has every marker observed; cannot fit the low-dimensional "
            "subspace against complete-case data"
        )

    fully_missing = np.nonzero(np.isnan(rawdata).all(axis=1))[0]
    if strict and fully_missing.size:
        raise InsufficientObservationsError(
            f"frame(s) {fully_missing.tolist()} have no observed markers"
        )

    m = np.mean(x, axis=0)

    logger.debug("Computing SVD...")
    _u, s, v = np.linalg.svd(x - m)
    logger.debug("done")

    if d is None:
        d = int(np.nonzero(np.cumsum(s) / np.sum(s) > (1 - tol))[0][0])
    v_d = v[0:d, :]

    q = process_noise_scale * (v_d @ np.diag(np.std(np.diff(x, axis=0), axis=0)) @ v_d.T)

    n_frames = rawdata.shape[0]
    state = [rng.normal(0.0, 1.0, d)]
    state_pred = [rng.normal(0.0, 1.0, d)]
    cov = [initial_cov_scale * np.eye(d)]
    cov_pred = [initial_cov_scale * np.eye(d)]

    logger.debug("Forward Pass")
    for i in range(1, n_frames + 1):
        z, h, ht = _observation(rawdata[i - 1, :], v_d)

        state_pred.append(state[i - 1])
        cov_pred.append(cov[i - 1] + q)

        if h.shape[0] == 0:
            # No markers observed this frame: skip the update, coast on
            # the process-model prediction alone.
            state.append(state_pred[i])
            cov.append(cov_pred[i])
            continue

        r = sig_r * np.eye(h.shape[0])
        k = cov_pred[i] @ ht.T @ np.linalg.inv(ht @ cov_pred[i] @ ht.T + r)

        state.append(state_pred[i] + k @ (z - (ht @ state_pred[i] + h @ m)))
        cov.append((np.eye(d) - k @ ht) @ cov_pred[i])

    logger.debug("Backward Pass")
    y = np.zeros(rawdata.shape)
    y[-1, :] = v_d.T @ state[-1] + m
    for i in range(len(state) - 2, 0, -1):
        gain = cov[i] @ np.linalg.inv(cov_pred[i])
        state[i] = state[i] + gain @ (state[i + 1] - state_pred[i + 1])
        cov[i] = cov[i] + gain @ (cov[i + 1] - cov_pred[i + 1]) @ cov[i]

        y[i - 1, :] = v_d.T @ state[i] + m

    if keep_original:
        y[~np.isnan(rawdata)] = rawdata[~np.isnan(rawdata)]

    if not return_uncertainty:
        return y

    var = _uncertainty(rawdata, v_d, q, sig_r, initial_cov_scale)
    if keep_original:
        var[~np.isnan(rawdata)] = 0.0
    return y, var


def _uncertainty(
    rawdata: np.ndarray,
    v_d: np.ndarray,
    q: np.ndarray,
    sig_r: float,
    initial_cov_scale: float,
) -> np.ndarray:
    """Numerically-stable Kalman/RTS covariance recursion, decoupled from
    the position reconstruction, projected to per-coordinate variance.

    Uses Joseph-form covariance updates (guaranteed positive semi-definite
    under floating point round-off, unlike the simpler ``(I - KH) P``
    form used for the reconstruction above) so it stays valid even where
    the reconstruction's own covariance bookkeeping would not.
    """
    d = v_d.shape[0]
    eye_d = np.eye(d)
    n_frames = rawdata.shape[0]

    cov = [initial_cov_scale * eye_d]
    cov_pred = [initial_cov_scale * eye_d]
    for i in range(1, n_frames + 1):
        _z, h, ht = _observation(rawdata[i - 1, :], v_d)
        cp = cov[i - 1] + q
        cov_pred.append(cp)
        if h.shape[0] == 0:
            cov.append(cp)
            continue
        r = sig_r * np.eye(h.shape[0])
        k = cp @ ht.T @ np.linalg.inv(ht @ cp @ ht.T + r)
        a = eye_d - k @ ht
        cov.append(a @ cp @ a.T + k @ r @ k.T)

    var = np.zeros(rawdata.shape)
    var[-1, :] = np.diag(v_d.T @ cov[-1] @ v_d)
    for i in range(len(cov) - 2, 0, -1):
        gain = cov[i] @ np.linalg.inv(cov_pred[i + 1])
        cov[i] = cov[i] + gain @ (cov[i + 1] - cov_pred[i + 1]) @ gain.T
        var[i - 1, :] = np.diag(v_d.T @ cov[i] @ v_d)
    return var
