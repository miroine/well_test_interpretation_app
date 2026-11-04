"""Wellbore and near-well effects that break Laplace-space superposition.

* changing wellbore storage (Hegeman-style exponential change, restarts at every rate change)
* rate-dependent skin  S_eff = S + D |q|   (non-Darcy / turbulence)
* stress-sensitive permeability  k(p) = ki exp(-gamma (pi - p))  (Pedrosa transformation)

The reservoir sandface response (no storage, no skin) comes from the analytical model; the well is
then marched in time by discrete convolution, solving for the sandface rate in every step.
Stress sensitivity is treated as reversible (permeability recovers on build-up); irreversible
loss is not represented.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from . import models as M


@dataclass
class WellboreEffects:
    C_i: float = 0.01           # bbl/psi at the start of each rate period
    C_f: float = 0.01           # bbl/psi long-time value
    alpha: float = 1.0          # hr, time constant of the change
    D: float = 0.0              # skin per (stb/d):  S_eff = S + D*|q_sf|
    gamma: float = 0.0          # 1/psi, permeability modulus


def _sandface_unit(p: M.WellTestParams, tmax):
    """Return Ps(tau) [psi per stb/d] for the reservoir alone (C = S = 0), log-interpolated."""
    q = M.WellTestParams(**p.to_dict())
    q.C, q.S = 0.0, 0.0
    tau = np.geomspace(1e-7, max(tmax * 1.2, 1e-3), 220)
    ps, _ = M.unit_rate_response(q, tau)
    ps = np.maximum(ps, 1e-12)
    lt, lp = np.log(tau), np.log(ps)

    def Ps(x):
        x = np.asarray(x, float)
        out = np.zeros_like(x)
        ok = x > tau[0]
        out[ok] = np.exp(np.interp(np.log(x[ok]), lt, lp))
        small = (~ok) & (x > 0)
        out[small] = ps[0] * x[small] / tau[0]          # early linear fade-out
        return out
    return Ps


def march(p: M.WellTestParams, fx: WellboreEffects, schedule, t_out, per_decade=24):
    """Pressure history with wellbore effects. schedule: [(start_hr, stb/d)], starts at 0."""
    sched = sorted((float(a), float(b)) for a, b in schedule)
    t_out = np.asarray(t_out, float)
    t_end = float(t_out.max())
    events = sorted({0.0} | {s for s, _ in sched if s < t_end} | {t_end})
    grid = [0.0]
    for a, b in zip(events[:-1], events[1:]):
        dt = min(1e-4, (b - a) / 20)
        t = a
        while t < b - 1e-12:
            t = min(b, t + dt)
            grid.append(t)
            dt = min(dt * 10 ** (1.0 / per_decade), (b - a) / 12)
    grid = np.unique(np.concatenate([grid, t_out[t_out > 0]]))
    Ps = _sandface_unit(p, t_end)
    sk = M.p_scale(p)
    qs = np.array([_rate_of(sched, 0.5 * (a + b)) for a, b in zip(grid[:-1], grid[1:])])
    starts = np.array([_period_start(sched, 0.5 * (a + b)) for a, b in zip(grid[:-1], grid[1:])])
    N = len(qs)
    qsf = np.zeros(N)
    dpw = np.zeros(N + 1)
    gam = fx.gamma

    def pedrosa(x):
        if gam <= 0:
            return x
        arg = 1.0 - gam * x
        return -np.log(max(arg, 1e-9)) / gam

    for n in range(1, N + 1):
        tn = grid[n]
        dt = grid[n] - grid[n - 1]
        j = np.arange(1, n)
        hist = 0.0
        if n > 1:
            hist = float(np.sum(qsf[j - 1] * (Ps(tn - grid[j - 1]) - Ps(tn - grid[j]))))
        step = float(Ps(np.array([dt]))[0])
        tau_c = 0.5 * (grid[n - 1] + tn) - starts[n - 1]
        C = fx.C_f + (fx.C_i - fx.C_f) * np.exp(-tau_c / max(fx.alpha, 1e-9))

        def dpwf(q):
            res = pedrosa(hist + q * step)
            return res + sk * q * (p.S + fx.D * abs(q))

        def R(q):
            return q + (C / p.B) * 24.0 / dt * (dpwf(q) - dpw[n - 1]) - qs[n - 1]

        span = 3.0 * max(abs(qs).max(), 1.0)
        try:
            q_n = brentq(R, -span, span, xtol=1e-10 * span, maxiter=200)
        except ValueError:
            q_n = qs[n - 1]
        qsf[n - 1] = q_n
        dpw[n] = dpwf(q_n)
    pw = np.interp(t_out, grid, p.pi - dpw)
    pw = np.where(t_out <= 0, p.pi, pw)
    qsf_out = np.interp(t_out, 0.5 * (grid[:-1] + grid[1:]), qsf)
    return {"t": t_out, "pwf": pw, "qsf": qsf_out, "grid": grid, "dp": dpw}


def _rate_of(sched, t):
    r = 0.0
    for s, q in sched:
        if t >= s - 1e-12:
            r = q
    return r


def _period_start(sched, t):
    st = 0.0
    for s, _ in sched:
        if t >= s - 1e-12:
            st = s
    return st
