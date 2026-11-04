"""Tools for large permanent-gauge / DST records.

* wavelet_denoise     Daubechies-4 soft-threshold denoising (no external wavelet library needed)
* smart_reduce        keep the points that matter: pressure steps, log-time spacing, rate changes
* detect_events       find rate changes / shut-ins from the pressure signal alone
* rate_from_pressure  reconstruct a piecewise-constant rate history from pressure with a reservoir model
* temperature tools   relaxation fit and pressure-temperature cross-plot data
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls, curve_fit

from . import models as M

# Daubechies-4 analysis filters
_S3 = np.sqrt(3.0)
_H = np.array([1 + _S3, 3 + _S3, 3 - _S3, 1 - _S3]) / (4 * np.sqrt(2.0))
_G = np.array([_H[3], -_H[2], _H[1], -_H[0]])


def _dwt_step(a):
    n = len(a)
    ext = np.concatenate([a, a[:3]])
    lo = np.array([np.dot(_H, ext[i:i + 4]) for i in range(0, n, 2)])
    hi = np.array([np.dot(_G, ext[i:i + 4]) for i in range(0, n, 2)])
    return lo, hi


def _idwt_step(lo, hi):
    n = 2 * len(lo)
    out = np.zeros(n + 3)
    for k in range(len(lo)):
        out[2 * k:2 * k + 4] += lo[k] * _H + hi[k] * _G
    out[:3] += out[n:n + 3]
    return out[:n]


def wavelet_denoise(y, levels=None, strength=1.0):
    """Soft-threshold DWT (db4) with the universal threshold; returns (denoised, sigma_estimate)."""
    y = np.asarray(y, float)
    n0 = len(y)
    n = 1 << int(np.ceil(np.log2(max(n0, 16))))
    pad = n - n0
    yp = np.concatenate([y, y[::-1][:pad]]) if pad <= n0 else np.concatenate([y, np.full(pad, y[-1])])
    levels = levels or max(1, int(np.log2(n)) - 4)
    a, details = yp.copy(), []
    for _ in range(levels):
        a, d = _dwt_step(a)
        details.append(d)
    sigma = np.median(np.abs(details[0])) / 0.6745
    thr = strength * sigma * np.sqrt(2 * np.log(n))
    for i, d in enumerate(details):
        details[i] = np.sign(d) * np.maximum(np.abs(d) - thr, 0.0)
    for d in reversed(details):
        a = _idwt_step(a, d)
    return a[:n0], float(sigma)


def smart_reduce(t, p, dp_tol=0.02, log_step=0.03, keep_times=(), keep_window=0.0):
    """Reduce a record keeping first/last point, every pressure move >= dp_tol since the last kept
    point, a minimum log-time density (ratio 1+log_step) and the neighbourhood of keep_times."""
    t, p = np.asarray(t, float), np.asarray(p, float)
    keep = np.zeros(len(t), bool)
    keep[0] = keep[-1] = True
    last = 0
    for i in range(1, len(t)):
        if abs(p[i] - p[last]) >= dp_tol or t[i] - t[last] >= (log_step * max(t[last] - t[0], 1e-9)) * 1.0:
            keep[i] = True
            last = i
    for kt in keep_times:
        keep |= np.abs(t - kt) <= max(keep_window, 0.0) if keep_window > 0 else (np.abs(t - kt) == np.abs(t - kt).min())
    idx = np.where(keep)[0]
    return t[idx], p[idx], idx


def detect_events(t, p, k=6.0, min_sep=0.0):
    """Candidate rate changes: points where the local pressure slope changes abruptly (robust z-score
    of the second difference on a wavelet-smoothed series)."""
    t, p = np.asarray(t, float), np.asarray(p, float)
    ps, _ = wavelet_denoise(p, strength=0.7)
    d1 = np.gradient(ps, t)
    d2 = np.gradient(d1, t)
    z = np.abs(d2 - np.median(d2)) / (1.4826 * np.median(np.abs(d2 - np.median(d2))) + 1e-30)
    cand = np.where(z > k)[0]
    events, lastt = [], -np.inf
    for i in cand[np.argsort(-z[cand])]:
        if all(abs(t[i] - e) > max(min_sep, 5 * np.median(np.diff(t))) for e in events):
            events.append(float(t[i]))
    return sorted(events)


def rate_from_pressure(p_params: M.WellTestParams, t, pwf, edges, smooth=0.05):
    """Piecewise-constant rate history (stb/d) in the intervals `edges` reproducing pwf(t).

    Solves  pi - pwf(t_n) = sum_j q_j [Ps(t_n - e_j) - Ps(t_n - e_{j+1})]  by regularised NNLS using
    the full reservoir response (storage and skin included) as influence function.
    """
    t, pwf = np.asarray(t, float), np.asarray(pwf, float)
    edges = np.asarray(edges, float)
    tau = np.unique(np.concatenate([[0.0], np.geomspace(1e-6, max(t.max(), edges.max()) * 1.2, 300)]))
    dp, _ = M.unit_rate_response(p_params, tau[1:])
    dpi = np.concatenate([[0.0], dp])

    def Ps(x):
        return np.interp(np.maximum(x, 0.0), tau, dpi)

    nj = len(edges) - 1
    A = np.zeros((len(t), nj))
    for j in range(nj):
        A[:, j] = Ps(t - edges[j]) * (t > edges[j]) - Ps(t - edges[j + 1]) * (t > edges[j + 1])
    b = p_params.pi - pwf
    sc = np.abs(b).max() + 1e-9
    # second-difference regularisation (rates are expected to change in steps, penalise wiggles weakly)
    D = np.zeros((nj - 2, nj)) if nj > 2 else np.zeros((0, nj))
    for i in range(nj - 2):
        D[i, i:i + 3] = [1, -2, 1]
    qref = max(np.abs(np.linalg.lstsq(A, b, rcond=None)[0]).max(), 1.0)
    lam = smooth * np.linalg.norm(A, 2) / max(qref, 1.0)
    Aaug = np.vstack([A / sc, lam * D / sc]) if len(D) else A / sc
    baug = np.concatenate([b / sc, np.zeros(len(D))]) if len(D) else b / sc
    q, _ = nnls(Aaug, baug)
    return q, float(np.sqrt(np.mean((A @ q - b) ** 2)))


def temperature_relaxation(t, T):
    """Fit T(t) = Tf + (T0 - Tf) exp(-t/tau); returns dict (T0, Tf, tau) and fit residual."""
    t, T = np.asarray(t, float), np.asarray(T, float)

    def f(x, T0, Tf, tau):
        return Tf + (T0 - Tf) * np.exp(-x / tau)

    p0 = [T[0], T[-1], max((t[-1] - t[0]) / 3, 1e-6)]
    popt, _ = curve_fit(f, t - t[0], T, p0=p0, maxfev=4000)
    res = float(np.sqrt(np.mean((f(t - t[0], *popt) - T) ** 2)))
    return {"T0": float(popt[0]), "Tf": float(popt[1]), "tau": float(abs(popt[2]))}, res


def pt_crossplot(t, p, T, window=None):
    """Data for a P-T cross-plot and a log-time temperature derivative (dT/dln t)."""
    t, p, T = (np.asarray(a, float) for a in (t, p, T))
    tt = np.maximum(t - t[0], 1e-9)
    dTdlnt = np.gradient(T, np.log(tt + 1e-12))
    return {"p": p, "T": T, "dT_dlnt": dTdlnt}
