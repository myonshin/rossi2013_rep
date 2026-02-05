
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

def _as2d(a: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr

def gmmbeta(y: ArrayLike, z: ArrayLike, x: ArrayLike, heter: int = 1) -> NDArray[np.float64]:
    """
    MATLAB gmmbeta.m
    Two-step GMM (Hayashi ch.3), with diagonal E[e^2] weighting for heter==1.
    If heter==0, returns the 1st-stage (homoskedastic) estimator behavior matches MATLAB.

    Parameters
    ----------
    y : (T,1)
    z : (T,l) regressors
    x : (T,k) instruments
    heter : 0 or 1
    """
    y = _as2d(y)
    z = _as2d(z)
    x = _as2d(x)
    T = y.shape[0]
    k = x.shape[1]
    l = z.shape[1]
    if k < l:
        raise ValueError("gmmbeta: order condition violated (num instruments < num regressors)")

    Sxx = (x.T @ x) / T
    Sxz = (x.T @ z) / T
    Sxy = (x.T @ y) / T

    W1 = np.linalg.inv(Sxx)
    b1 = np.linalg.inv(Sxz.T @ W1 @ Sxz) @ (Sxz.T @ W1 @ Sxy)
    e = y - z @ b1

    # Heteroskedastic weighting matrix
    if heter == 1:
        S = (x.T @ (e**2 * x)) / T   # x' diag(e^2) x / T, vectorized
        W2 = np.linalg.inv(S)
        b2 = np.linalg.inv(Sxz.T @ W2 @ Sxz) @ (Sxz.T @ W2 @ Sxy)
        return b2
    else:
        # MATLAB's gmmbeta always computes b2 but returns b2; however their gmmres chooses b1 for heter==0.
        S = np.linalg.inv(Sxx) * float((e.T @ e) / T)
        W2 = np.linalg.inv(S)
        b2 = np.linalg.inv(Sxz.T @ W2 @ Sxz) @ (Sxz.T @ W2 @ Sxy)
        return b2

def gmmvar(y: ArrayLike, z: ArrayLike, x: ArrayLike, heter: int = 1) -> NDArray[np.float64]:
    """
    MATLAB gmmvar.m
    Variance matrix for the two-step GMM estimator.
    """
    y = _as2d(y); z = _as2d(z); x = _as2d(x)
    T = y.shape[0]
    k = x.shape[1]
    l = z.shape[1]
    if k < l:
        raise ValueError("gmmvar: order condition violated (num instruments < num regressors)")

    Sxx = (x.T @ x) / T
    Sxz = (x.T @ z) / T
    Sxy = (x.T @ y) / T

    W1 = np.linalg.inv(Sxx)
    b1 = np.linalg.inv(Sxz.T @ W1 @ Sxz) @ (Sxz.T @ W1 @ Sxy)
    e = y - z @ b1

    if heter == 1:
        S = (x.T @ (e**2 * x)) / T
        W2 = np.linalg.inv(S)
        # avarb = inv(Sxz' W2 Sxz)
        avarb = np.linalg.inv(Sxz.T @ W2 @ Sxz)
        return avarb
    else:
        # Homoskedastic variant in the MATLAB file
        S = np.linalg.inv(Sxx) * float((e.T @ e) / T)
        W2 = np.linalg.inv(S)
        b2 = np.linalg.inv(Sxz.T @ W2 @ Sxz) @ (Sxz.T @ W2 @ Sxy)
        resid = y - z @ b2
        avarb = np.linalg.inv(Sxx) * float((resid.T @ resid) / T)
        return avarb

def gmmres(y: ArrayLike, z: ArrayLike, x: ArrayLike, heter: int = 1) -> NDArray[np.float64]:
    """
    MATLAB gmmres.m
    Residuals y - z*b, where b is b2 if heter==1 else b1.
    """
    y = _as2d(y); z = _as2d(z); x = _as2d(x)
    T = y.shape[0]
    k = x.shape[1]
    l = z.shape[1]
    if k < l:
        raise ValueError("gmmres: order condition violated (num instruments < num regressors)")

    Sxx = (x.T @ x) / T
    Sxz = (x.T @ z) / T
    Sxy = (x.T @ y) / T

    W1 = np.linalg.inv(Sxx)
    b1 = np.linalg.inv(Sxz.T @ W1 @ Sxz) @ (Sxz.T @ W1 @ Sxy)
    e = y - z @ b1

    # Build S the MATLAB loop version (equivalent to x' diag(e^2) x / T)
    S = (x.T @ (e**2 * x)) / T
    W2 = np.linalg.inv(S)
    b2 = np.linalg.inv(Sxz.T @ W2 @ Sxz) @ (Sxz.T @ W2 @ Sxy)

    b = b2 if heter == 1 else b1
    return y - z @ b
