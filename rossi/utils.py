
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

def mmult(x: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
    """
    MATLAB mmult.m
    Multiplies a vector x by each column of y (elementwise), returning an (n,k) array.

    Parameters
    ----------
    x : (n,) or (n,1)
    y : (n,k)

    Returns
    -------
    (n,k) float array
    """
    x_arr = np.asarray(x, dtype=float).reshape(-1, 1)
    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"mmult: x and y must have same number of rows; got {x_arr.shape[0]} vs {y_arr.shape[0]}")
    return x_arr * y_arr

def differ(x: ArrayLike, k: int) -> NDArray[np.float64]:
    """
    MATLAB differ.m
    k-th difference with MATLAB-style NaN padding at top.

    If k==0 returns x.
    """
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    if k == 0:
        return x_arr.copy()
    if k < 0:
        raise ValueError("differ: k must be >= 0")
    n, m = x_arr.shape
    out = np.full((n, m), np.nan, dtype=float)
    if k >= n:
        return out
    out[k:, :] = x_arr[k:, :] - x_arr[:-k, :]
    return out

def clean_nan_rows(x: ArrayLike) -> NDArray[np.float64]:
    """
    MATLAB cleanNaN.m behavior: drop rows that contain ANY NaN.
    """
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    mask = ~np.isnan(x_arr).any(axis=1)
    return x_arr[mask, :]

def filter_nan(w: ArrayLike, x: ArrayLike) -> NDArray[np.float64]:
    """
    MATLAB filterNAN.m behavior:
    y = filter(w, 1, x); then set first n=max(len(w),1) rows to NaN.

    Uses scipy.signal.lfilter if available; otherwise falls back to convolution.
    """
    w_arr = np.asarray(w, dtype=float).ravel()
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    try:
        from scipy.signal import lfilter
        y = np.column_stack([lfilter(w_arr, [1.0], x_arr[:, j]) for j in range(x_arr.shape[1])])
    except Exception:
        # Fallback: causal FIR filter via convolution
        y = np.full_like(x_arr, np.nan, dtype=float)
        for j in range(x_arr.shape[1]):
            col = x_arr[:, j]
            out = np.convolve(col, w_arr, mode="full")[: len(col)]
            y[:, j] = out
    n = int(max(w_arr.size, 1))
    y[:n, :] = np.nan
    return y
