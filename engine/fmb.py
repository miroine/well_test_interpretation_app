"""Flowing material balance by tank-model regression (oil and gas).

For each observation (cumulative Gp or Np, rate q, flowing pwf) the average
reservoir pressure is obtained from the material balance for a trial G (or N);
the pseudo-steady inflow q = J (p_avg - pwf) then gives a residual.  G (N) and
J are the two fitted unknowns, so no shut-in survey is needed - the 'flowing'
in flowing material balance.

Gas uses normalized pseudo-pressure.  Constant J is assumed (boundary-dominated
flow); see Notes in the UI.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from . import gas as GS


class GasTank:
    def __init__(self, pi, T_F, gamma_g, n=300):
        self.pi, self.T_F, self.gamma_g = pi, T_F, gamma_g
        self.npp = GS.NormalizedPseudoPressure(pi, T_F, gamma_g, p_min=14.7, n=n)
        grid = np.linspace(14.7, pi * 1.02, 250)
        self.pz = np.array([p / GS.z_factor(p, T_F, gamma_g) for p in grid])
        self.grid = grid
        self.pz_i = pi / GS.z_factor(pi, T_F, gamma_g)

    def p_avg(self, Gp, G):
        target = self.pz_i * np.clip(1.0 - np.asarray(Gp, float) / G, 1e-6, 1.0)
        return np.interp(target, self.pz, self.grid)


def simulate_gas(tank: GasTank, G, J, pwf_fn, t_end, dt=5.0):
    """Reference generator: dGp/dt = J (m(p_avg) - m(pwf)). Returns t, q, pwf, Gp."""
    t, Gp = 0.0, 0.0
    T, Q, P, GP = [], [], [], []
    while t < t_end:
        pw = pwf_fn(t)
        pa = float(tank.p_avg(Gp, G))
        q = max(J * (float(tank.npp.forward(pa)) - float(tank.npp.forward(pw))), 0.0)
        T.append(t); Q.append(q); P.append(pw); GP.append(Gp)
        Gp += q * dt
        t += dt
    return np.array(T), np.array(Q), np.array(P), np.array(GP)


def fit_gas_fmb(tank: GasTank, Gp, q, pwf, G0, J0=None):
    """Fit G [Mscf] and J [Mscf/d/psi]. Returns dict with G, J, rms, residual array."""
    Gp, q, pwf = (np.asarray(a, float) for a in (Gp, q, pwf))
    ok = q > 0
    Gp, q, pwf = Gp[ok], q[ok], pwf[ok]
    mw = tank.npp.forward(pwf)
    if J0 is None:
        J0 = float(np.median(q / np.maximum(tank.pi - mw, 1.0)))
    qs = np.median(q)

    def res(v):
        G, J = np.exp(v)
        if G <= Gp.max() * 1.0001:
            return np.full_like(q, 1e3)
        pa = tank.p_avg(Gp, G)
        return (q - J * (tank.npp.forward(pa) - mw)) / qs

    best = None
    for gf in (1.5, 3.0, 10.0):
        v0 = np.log([max(G0, Gp.max() * 1.2) * gf / 1.5, J0])
        sol = least_squares(res, v0, x_scale="jac", max_nfev=200)
        if best is None or sol.cost < best.cost:
            best = sol
    G, J = np.exp(best.x)
    return {"G": float(G), "J": float(J), "rms": float(np.sqrt(2 * best.cost / len(q))),
            "resid": res(best.x)}


def gas_fmb_line(tank: GasTank, Gp, q, pwf):
    """Classical flowing-MB plot data: normalised rate q/(m(pi)-m(pwf)) vs Gp and its
    linear extrapolation to zero rate (a quick, approximate G)."""
    Gp, q, pwf = (np.asarray(a, float) for a in (Gp, q, pwf))
    ok = q > 0
    qn = q[ok] / np.maximum(tank.pi - tank.npp.forward(pwf[ok]), 1e-6)
    m, b = np.polyfit(Gp[ok], qn, 1)
    return {"Gp": Gp[ok], "qn": qn, "slope": float(m), "intercept": float(b),
            "G_line": float(-b / m) if m < 0 else float("nan")}


# ------------------------------------------------------------------- oil
def simulate_oil(N, J, ceff, Boi, Bo, pi, pwf_fn, t_end, dt=5.0):
    t, Np = 0.0, 0.0
    T, Q, P, NP = [], [], [], []
    while t < t_end:
        pw = pwf_fn(t)
        pa = pi - Np * Bo / (N * ceff * Boi)
        q = max(J * (pa - pw), 0.0)
        T.append(t); Q.append(q); P.append(pw); NP.append(Np)
        Np += q * dt
        t += dt
    return np.array(T), np.array(Q), np.array(P), np.array(NP)


def fit_oil_fmb(pi, ceff, Boi, Bo, Np, q, pwf, N0, J0=None):
    """Undersaturated oil tank: p_avg = pi - Np Bo / (N ceff Boi); q = J (p_avg - pwf)."""
    Np, q, pwf = (np.asarray(a, float) for a in (Np, q, pwf))
    ok = q > 0
    Np, q, pwf = Np[ok], q[ok], pwf[ok]
    if J0 is None:
        J0 = float(np.median(q / np.maximum(pi - pwf, 1.0)))
    qs = np.median(q)

    def res(v):
        N, J = np.exp(v)
        pa = pi - Np * Bo / (N * ceff * Boi)
        return (q - J * (pa - pwf)) / qs

    best = None
    for nf in (0.5, 1.0, 3.0, 10.0):
        sol = least_squares(res, np.log([max(N0, Np.max() * 2) * nf, J0]), x_scale="jac", max_nfev=200)
        if best is None or sol.cost < best.cost:
            best = sol
    N, J = np.exp(best.x)
    return {"N": float(N), "J": float(J), "rms": float(np.sqrt(2 * best.cost / len(q))), "resid": res(best.x)}
