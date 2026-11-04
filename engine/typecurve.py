"""Gringarten-style type curves for a homogeneous reservoir with storage and skin.

Curves are parameterised by G = CD*exp(2S) and plotted against tD/CD.
G alone is a late-time similarity group, not an exact finite-radius identity.
For G < 100 a zero-skin effective-radius reference (CD=G) avoids the unstable
negative lumped-skin finite-radius solution; otherwise CD=100 is retained.
Math only.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from . import laplace as L

DEFAULT_G = [0.1, 1.0, 10.0, 1e2, 1e3, 1e5, 1e8, 1e12, 1e20, 1e30, 1e40, 1e60]
_CD_REF = 100.0


def _response(G, x):
    if not np.isfinite(G) or G <= 0:
        raise ValueError("The type-curve group CD exp(2S) must be finite and positive.")
    x = np.asarray(x, float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or np.any(x <= 0):
        raise ValueError("Type-curve times must be a non-empty array of finite positive values.")
    # Negative lumped skin has a positive-real Laplace pole at early time.
    CD = min(_CD_REF, G)
    S = 0.5 * np.log(G / CD)
    return L.unit_response(lambda u: L.pws_radial(u, L.f_homogeneous, CD, S), x * CD)


def family(G_values=None, x_grid=None):
    """Return dict G -> (x = tD/CD, pD, tD*dpD/dtD)."""
    G_values = DEFAULT_G if G_values is None else G_values
    x = np.logspace(-3, 7, 240) if x_grid is None else np.asarray(x_grid, float)
    out = {}
    for G in G_values:
        pD, dD = _response(G, x)
        out[G] = (x, pD, dD)
    return out


def _curve_logs(G, x_eval):
    pD, dD = _response(G, x_eval)
    return np.log(np.maximum(pD, 1e-12)), np.log(np.maximum(dD, 1e-12))


def auto_match(t, dp, der, G_grid=None):
    """Find (G, time shift, pressure shift) minimising log-space misfit.

    Free variables are ln G, ln sx, ln sy (all continuous); G_grid only supplies
    starting values.  x = sx * t ;  pD = sy * dp.
    Returns dict with sx, sy, G, rms.
    """
    t = np.asarray(t, float)
    dp = np.asarray(dp, float)
    der = np.asarray(der, float)
    ok = np.isfinite(der) & (der > 0) & (dp > 0) & (t > 0)
    t, dp, der = t[ok], dp[ok], der[ok]
    G_grid = [1e1, 1e4, 1e8, 1e15, 1e30] if G_grid is None else G_grid
    lo_x, hi_x = np.log(1e-4), np.log(1e8)

    def res(v):
        G, sx, sy = np.exp(v)
        lx = np.log(sx * t)
        xe = np.exp(np.clip(lx, lo_x, hi_x))
        lp, ld = _curve_logs(G, xe)
        r1 = lp - np.log(sy * dp)
        r2 = ld - np.log(sy * der)
        out_of_range = np.maximum(lo_x - lx, 0) + np.maximum(lx - hi_x, 0)
        return np.concatenate([r1 + out_of_range, r2 + out_of_range])

    sy0 = 0.5 / der[-1]
    lb = np.log([0.1, 1e-6, 1e-8])
    ub = np.log([1e60, 1e8, 1e4])
    best = None
    for G0 in G_grid:
        for sx0 in (1e-2, 1.0, 1e2):
            v0 = np.clip(np.log([G0, sx0, sy0]), lb + 1e-6, ub - 1e-6)
            try:
                sol = least_squares(res, v0, bounds=(lb, ub), max_nfev=80, x_scale="jac")
            except Exception:
                continue
            rms = float(np.sqrt(np.mean(sol.fun ** 2)))
            if best is None or rms < best["rms"]:
                G, sx, sy = np.exp(sol.x)
                best = {"G": float(G), "sx": float(sx), "sy": float(sy), "rms": rms}
    return best


def interpret_match(sx, sy, G, q, B, mu, h, phi, ct, rw):
    """Convert match factors to k, C, CD, S (field units)."""
    k = 141.2 * q * B * mu * sy / h                      # pD/dp match
    C = (0.0002637 * k * h / (0.8936 * mu)) / sx         # (tD/CD)/t match
    CD = 0.8936 * C / (phi * ct * h * rw ** 2)
    S = 0.5 * np.log(G / CD)
    return {"k": k, "C": C, "CD": CD, "S": S}
