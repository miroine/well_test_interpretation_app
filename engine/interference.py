"""Interference and pulse tests: pressure at an observation well caused by one or more active wells.

Homogeneous or dual-porosity infinite reservoir; superposition in time over each active well's
rate schedule and in space over all active wells.  For boundaries, faults or heterogeneity use the
2D areal model (engine/areal.py), which handles several wells natively.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from . import laplace as L
from . import models as M


@dataclass
class ActiveWell:
    x: float = 0.0
    y: float = 0.0
    C: float = 0.01
    S: float = 0.0
    schedule: list = field(default_factory=lambda: [(0.0, 1000.0)])


def _unit(p: M.WellTestParams, r_ft, t_hr, C, S):
    """Observation-well drawdown [psi] per 1 stb/d from one active well at distance r_ft."""
    t_hr = np.asarray(t_hr, float)
    kern = M._kernel(p)
    CD = 0.8936 * C / (p.phi * p.ct * p.h * p.rw ** 2)
    rD = max(r_ft / p.rw, 1.0)
    tD = 0.0002637 * p.k * t_hr / (p.phi * p.mu * p.ct * p.rw ** 2)
    out = np.zeros_like(tD)
    ok = tD > 0
    if ok.any():
        out[ok] = L.invert(lambda u: L.pws_interference(u, kern, rD, CD, S), tD[ok])
    return out * M.p_scale(p)


def dp_observed(p: M.WellTestParams, obs_xy, wells, t_hr):
    t_hr = np.asarray(t_hr, float)
    total = np.zeros_like(t_hr)
    for w in wells:
        r = float(np.hypot(w.x - obs_xy[0], w.y - obs_xy[1]))
        prev = 0.0
        for ts, q in w.schedule:
            dq = q - prev
            prev = q
            if dq == 0:
                continue
            tt = t_hr - ts
            total += dq * _unit(p, r, np.where(tt > 0, tt, 0.0), w.C, w.S)
    return total


def pulse_metrics(t, dp, pulse_start, pulse_end):
    """Time lag and amplitude of the response to one pulse (rate on at pulse_start, off at pulse_end).

    lag_peak     time from the end of the pulse to the maximum observed pressure change
    lag_onset    time from the start of the pulse until the pressure change exceeds 10 % of its maximum
    amplitude    peak dp minus the dp that would have continued from the trend before the pulse
                 (approximated by the value at the start of the pulse response).
    """
    t, dp = np.asarray(t, float), np.asarray(dp, float)
    m = (t >= pulse_start)
    tt, dd = t[m], dp[m]
    if len(tt) < 3:
        return {"lag_peak": np.nan, "lag_onset": np.nan, "amplitude": np.nan}
    base = dd[0]
    i = int(np.argmax(dd))
    amp = dd[i] - base
    thr = base + 0.1 * amp
    j = int(np.argmax(dd >= thr)) if amp > 0 else 0
    return {"lag_peak": float(tt[i] - pulse_end), "lag_onset": float(tt[j] - pulse_start), "amplitude": float(amp)}


def fit_interference(p0: M.WellTestParams, obs_xy, wells, t, dp_obs, free=("k", "phi"), max_nfev=60):
    """Fit k and/or phi (or ct) to observed interference pressure changes (psi, positive = drawdown)."""
    free = list(free)
    t = np.asarray(t, float)
    dp_obs = np.asarray(dp_obs, float)
    scale = max(np.abs(dp_obs).max(), 1e-6)

    def mk(v):
        p = M.WellTestParams(**p0.to_dict())
        for n, x in zip(free, v):
            setattr(p, n, 10 ** x)
        return p

    def resid(v):
        return (dp_observed(mk(v), obs_xy, wells, t) - dp_obs) / scale

    v0 = np.log10([getattr(p0, n) for n in free])
    sol = least_squares(resid, v0, max_nfev=max_nfev, x_scale=1.0,
                        bounds=(v0 - 2.0, v0 + 2.0))
    pf = mk(sol.x)
    rms = float(np.sqrt(np.mean(sol.fun ** 2)) * scale)
    ci = {}
    try:
        J = sol.jac
        dof = max(len(sol.fun) - len(free), 1)
        s2 = float((sol.fun ** 2).sum() / dof)
        cov = np.linalg.inv(J.T @ J) * s2
        for i, n in enumerate(free):
            sd = np.sqrt(max(cov[i, i], 0.0))
            ci[n] = {"lo": getattr(pf, n) * 10 ** (-1.96 * sd), "hi": getattr(pf, n) * 10 ** (1.96 * sd)}
    except np.linalg.LinAlgError:
        pass
    return pf, {"rms_psi": rms, "ci": ci, "nfev": int(sol.nfev)}
