
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .gmm import gmmbeta, gmmres, gmmvar

def _as2d(a: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr

def cdfchic(x: float, k: int) -> float:
    """
    MATLAB cdfchic.m (modern portion): upper-tail p-value P(Chi^2_k >= x).
    """
    from scipy.stats import chi2
    if x < 0:
        raise ValueError("x must be nonnegative for chi-square")
    if k <= 0 or int(k) != k:
        raise ValueError("k must be a positive integer")
    return float(chi2.sf(x, k))

def pvcalc(osser: float, tavola: NDArray[np.float64], kk: int) -> float:
    """
    MATLAB pvcalc.m
    Interpolates p-value from a simulated critical value table.

    tavola is assumed to be (34, 1+K) where column 0 is p-values grid
    and columns 2.. correspond to stats for different # regressors.
    """
    tavola = np.asarray(tavola, dtype=float)
    col = kk + 1  # MATLAB indexing: kk+2 -> 0-based kk+1
    if col >= tavola.shape[1]:
        raise ValueError(f"pvcalc: kk={kk} too large for table with {tavola.shape[1]} cols")

    if osser <= tavola[0, col]:
        return 1.0
    if osser >= tavola[-1, col]:
        return 0.0

    # find last row where tavola[:,col] <= osser
    idxs = np.where(tavola[:, col] <= osser)[0]
    i = int(idxs.max())
    # interpolate between i and i+1 using p-values in column 0
    p_hi, x_hi = tavola[i, 0], tavola[i, col]
    p_lo, x_lo = tavola[i + 1, 0], tavola[i + 1, col]
    # MATLAB formula: pv = sel(2,1) + (sel(2,2)-osser)*(sel(1,1)-sel(2,1))/(sel(2,2)-sel(1,2))
    # where sel(2,:) is (p_lo, x_lo) and sel(1,:) is (p_hi, x_hi)
    pv = p_lo + (x_lo - osser) * (p_hi - p_lo) / (x_lo - x_hi)
    return float(pv)

def chowgmm(y: ArrayLike, x: ArrayLike, z: ArrayLike | int, w: ArrayLike, t: int, heter: int = 1) -> float:
    """
    MATLAB chowgmm.m
    Andrews-style fixed-break Chow statistic via GMM.
    z can be 0 (no time-invariant regressors).
    """
    y = _as2d(y); x = _as2d(x); w = _as2d(w)
    T, p = x.shape
    pi = t / T
    k = w.shape[1]

    if isinstance(z, (int, float)) and z == 0:
        y1, x1, w1 = y[:t, :], x[:t, :], w[:t, :]
        y2, x2, w2 = y[t:, :], x[t:, :], w[t:, :]

        X = np.block([
            [x1, np.zeros((t, p))],
            [np.zeros((T - t, p)), x2],
        ])
        W = np.block([
            [w1, np.zeros((t, k))],
            [np.zeros((T - t, k)), w2],
        ])
        b = gmmbeta(y, X, W, heter)
        b1, b2 = b[:p, :], b[p:2*p, :]
        e = gmmres(y, X, W, heter)

        M1 = (w1.T @ x1) / t
        M2 = (w2.T @ x2) / (T - t)

        if heter == 1:
            S1 = (w1.T @ (e[:t, :]**2 * w1)) / t
            S2 = (w2.T @ (e[t:, :]**2 * w2)) / (T - t)
        else:
            s2 = float((e.T @ e) / T)
            S1 = s2 * (w1.T @ w1) / t
            S2 = s2 * (w2.T @ w2) / (T - t)

        V1 = np.linalg.inv(M1.T @ np.linalg.inv(S1) @ M1)
        V2 = np.linalg.inv(M2.T @ np.linalg.inv(S2) @ M2)
        wald = T * float((b1 - b2).T @ np.linalg.inv(V1 / pi + V2 / (1 - pi)) @ (b1 - b2))
        return wald

    # with time-invariant regressors z
    z = _as2d(z)
    y1, x1, z1, w1 = y[:t, :], x[:t, :], z[:t, :], w[:t, :]
    y2, x2, z2, w2 = y[t:, :], x[t:, :], z[t:, :], w[t:, :]

    X = np.block([
        [x1, np.zeros((t, p)), z1],
        [np.zeros((T - t, p)), x2, z2],
    ])
    W = np.block([
        [w1, np.zeros((t, k))],
        [np.zeros((T - t, k)), w2],
    ])
    b = gmmbeta(y, X, W, heter)
    b1, b2 = b[:p, :], b[p:2*p, :]
    e = gmmres(y, X, W, heter)

    M1 = (w1.T @ x1) / t
    M2 = (w2.T @ x2) / (T - t)

    if heter == 1:
        S1 = (w1.T @ (e[:t, :]**2 * w1)) / t
        S2 = (w2.T @ (e[t:, :]**2 * w2)) / (T - t)
    else:
        s2 = float((e.T @ e) / T)
        S1 = s2 * (w1.T @ w1) / t
        S2 = s2 * (w2.T @ w2) / (T - t)

    V1 = np.linalg.inv(M1.T @ np.linalg.inv(S1) @ M1)
    V2 = np.linalg.inv(M2.T @ np.linalg.inv(S2) @ M2)
    wald = T * float((b1 - b2).T @ np.linalg.inv(V1 / pi + V2 / (1 - pi)) @ (b1 - b2))
    return wald

def chowgmmstar(y: ArrayLike, x: ArrayLike, z: ArrayLike | int, w: ArrayLike, t: int, heter: int = 1) -> float:
    """
    MATLAB chowgmmstar.m
    Chow* (optimal) test statistic for a fixed break.
    """
    y = _as2d(y); x = _as2d(x); w = _as2d(w)
    T, p = x.shape
    pi = t / T
    k = w.shape[1]

    if isinstance(z, (int, float)) and z == 0:
        y1, x1, w1 = y[:t, :], x[:t, :], w[:t, :]
        y2, x2, w2 = y[t:, :], x[t:, :], w[t:, :]

        b1 = gmmbeta(y1, x1, w1, heter)
        b2 = gmmbeta(y2, x2, w2, heter)
        e1 = gmmres(y1, x1, w1, heter)
        e2 = gmmres(y2, x2, w2, heter)

        if heter == 1:
            Sigma1 = (w1.T @ (e1**2 * w1)) / t
            Sigma2 = (w2.T @ (e2**2 * w2)) / (T - t)
        else:
            s2 = float(((e1.T @ e1) + (e2.T @ e2)) / T)
            Sigma1 = s2 * (w1.T @ w1) / t
            Sigma2 = s2 * (w2.T @ w2) / (T - t)

        # Gamma = inv([pi*Sigma1, 0; 0, (1-pi)*Sigma2])
        Gamma = np.linalg.inv(np.block([
            [pi * Sigma1, np.zeros((k, k))],
            [np.zeros((k, k)), (1 - pi) * Sigma2],
        ]))
        Swx1 = (w1.T @ x1) / T
        Swx2 = (w2.T @ x2) / T
        M = np.block([
            [Swx1, np.zeros((k, p))],
            [np.zeros((k, p)), Swx2],
        ])
        R = np.block([
            [np.eye(p), -np.eye(p)],
            [pi * np.eye(p), (1 - pi) * np.eye(p)],
        ])
        VRb = R @ np.linalg.inv(M.T @ Gamma @ M) @ R.T
        vec = np.vstack([b1 - b2, pi * b1 + (1 - pi) * b2])
        wald = T * float(vec.T @ np.linalg.inv(VRb) @ vec)
        return wald

    # z present (time-invariant regressors)
    z = _as2d(z)
    q = z.shape[1]
    y1, x1, z1, w1 = y[:t, :], x[:t, :], z[:t, :], w[:t, :]
    y2, x2, z2, w2 = y[t:, :], x[t:, :], z[t:, :], w[t:, :]

    X = np.block([
        [x1, np.zeros((t, p)), z1],
        [np.zeros((T - t, p)), x2, z2],
    ])
    W = np.block([
        [w1, np.zeros((t, k))],
        [np.zeros((T - t, k)), w2],
    ])
    b = gmmbeta(y, X, W, heter)
    b1, b2 = b[:p, :], b[p:2*p, :]
    e = gmmres(y, X, W, heter)

    if heter == 1:
        Sigma1 = (w1.T @ (e[:t, :]**2 * w1)) / t
        Sigma2 = (w2.T @ (e[t:, :]**2 * w2)) / (T - t)
    else:
        s2 = float((e.T @ e) / T)
        Sigma1 = s2 * (w1.T @ w1) / t
        Sigma2 = s2 * (w2.T @ w2) / (T - t)

    Gamma = np.linalg.inv(np.block([
        [pi * Sigma1, np.zeros((k, k))],
        [np.zeros((k, k)), (1 - pi) * Sigma2],
    ]))
    Swx1 = (w1.T @ x1) / T
    Swx2 = (w2.T @ x2) / T
    Swz1 = (w1.T @ z1) / T
    Swz2 = (w2.T @ z2) / T
    M = np.block([
        [Swx1, np.zeros((k, p)), Swz1],
        [np.zeros((k, p)), Swx2, Swz2],
    ])
    # R: picks constraints on time-varying parameters only (p)
    R = np.block([
        [np.eye(p), -np.eye(p), np.zeros((p, q))],
        [pi * np.eye(p), (1 - pi) * np.eye(p), np.zeros((p, q))],
        [np.zeros((q, p)), np.zeros((q, p)), np.eye(q)],
    ])
    VRb = R @ np.linalg.inv(M.T @ Gamma @ M) @ R.T
    vec = np.vstack([b1 - b2, pi * b1 + (1 - pi) * b2, b[2*p:, :]])
    wald = T * float(vec.T @ np.linalg.inv(VRb) @ vec)
    return wald

def nyblomstar(y: ArrayLike, x: ArrayLike, z: ArrayLike | int, w: ArrayLike, heter: int, b0: ArrayLike) -> float:
    """
    MATLAB nyblomstar.m
    Optimal Nyblom test. If z==0 uses simple form; else uses projection adjustment.
    """
    y = _as2d(y); x = _as2d(x); w = _as2d(w); b0 = _as2d(b0)

    if isinstance(z, (int, float)) and z == 0:
        e = y - x @ b0
        T = y.shape[0]

        if heter == 1:
            Sigma = (w.T @ (e**2 * w)) / T
        else:
            s2 = float((e.T @ e) / T)
            Sigma = s2 * (w.T @ w) / T

        we = w * e  # elementwise by row (broadcast)
        es = np.cumsum(we, axis=0) / np.sqrt(T)

        invSigma = np.linalg.inv(Sigma)
        stat = 0.0
        for i in range(T):
            stat += float(es[i:i+1, :] @ invSigma @ es[i:i+1, :].T) / T
        return stat

    # z present
    z = _as2d(z)
    # residuals from gmmres(y - x*b0, z, w, heter) per MATLAB
    e = gmmres(y - x @ b0, z, w, heter)
    T = y.shape[0]
    k = z.shape[1]

    if heter == 1:
        Sigma = (w.T @ (e**2 * w)) / T
    else:
        e2 = np.cumsum(e**2, axis=0)
        s2 = float(e2[-1, 0] / (T - k))
        Sigma = s2 * (w.T @ w) / T

    from scipy.linalg import sqrtm
    inv_sqrtSigma = np.linalg.inv(sqrtm(Sigma))

    Mbetabar = inv_sqrtSigma @ (-(w.T @ x) / T)
    Mdeltabar = inv_sqrtSigma @ (-(w.T @ z) / T)

    Pbardelta = Mdeltabar @ np.linalg.inv(Mdeltabar.T @ Mdeltabar) @ Mdeltabar.T
    omegaN = Mbetabar.T @ (np.eye(Pbardelta.shape[0]) - Pbardelta) @ Mbetabar

    # GradQ(i,:) = w(i,:)*e(i,:)*inv(sqrtm(Sigma))*Mbetabar  (row vector)
    # Here w(i,:)*e(i) gives (1,m), then @ inv_sqrtSigma @ Mbetabar gives (1,p)
    GradQ = np.zeros((T, x.shape[1]), dtype=float)
    for i in range(T):
        GradQ[i, :] = (w[i:i+1, :] * e[i, 0]) @ inv_sqrtSigma @ Mbetabar

    GradQpiT = np.cumsum(GradQ, axis=0) / np.sqrt(T)
    invOmega = np.linalg.inv(omegaN)

    stat = 0.0
    for i in range(T):
        g = GradQpiT[i:i+1, :]
        stat += float(g @ invOmega @ g.T) / T
    return stat
