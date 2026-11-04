"""Regularised pressure-rate deconvolution.

Unknown: the derivative d(s) = dg/d(ln t) >= 0 of the unit-rate response g(t),
piecewise constant on log-spaced bins, plus (optionally) the initial pressure.
The observation operator is *linear* in d (superposition), so the problem is a
bounded linear least-squares with a curvature penalty on d.

This is not von Schroeter's nonlinear formulation; it is a simpler linear
non-negative-derivative variant, appropriate for a first-pass deconvolved
derivative.  Always compare with the model/type-curve response.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from . import multirate as MR


def deconvolve(schedule, t, pwf, pi=None, n_bins=60, t_min=None, lam=0.5, pi_prior=None):
    """Returns dict: t (bin centres), deriv (psi per unit rate per ln-cycle),
    g (unit-rate drawdown at bin upper edges), pi_est, fit (model pwf), rms."""
    t = np.asarray(t, float)
    pwf = np.asarray(pwf, float)
    m = t > 0
    t, pwf = t[m], pwf[m]
    ts, dq = MR.rate_changes(schedule)
    t_min = 0.5 * np.min(np.diff(np.concatenate([[0.0], t])).clip(1e-9)) if t_min is None else t_min
    s_edges = np.linspace(np.log(max(t_min, 1e-6)), np.log(t.max()), n_bins + 1)
    lo, hi = s_edges[:-1], s_edges[1:]

    def row(tau):
        ls = np.log(tau)
        return np.clip(np.minimum(hi, ls) - lo, 0.0, None)

    A_rows = np.zeros((len(t), n_bins))
    for i, (t0, d) in enumerate(zip(ts, dq)):
        mask = t > t0
        if not mask.any():
            continue
        lt = np.log(t[mask] - t0)
        A_rows[mask] += d * np.clip(np.minimum(hi[None, :], lt[:, None]) - lo[None, :], 0.0, None)
    scale = float(np.max(np.abs(pwf - pwf[0])) or 1.0)
    if pi is None:
        # pwf = pi - A d  ->  columns [-A, 1]
        X = np.hstack([-A_rows, np.ones((len(t), 1))])
    else:
        X = -A_rows
    y = pwf if pi is None else pwf - pi
    # curvature penalty on d (second differences)
    D2 = np.zeros((n_bins - 2, X.shape[1]))
    for j in range(n_bins - 2):
        D2[j, j], D2[j, j + 1], D2[j, j + 2] = 1.0, -2.0, 1.0
    qref = max(np.max(np.abs(dq)), 1e-9)
    reg = lam * scale / qref * 0.05
    Xs = np.vstack([X, reg * D2])
    ys = np.concatenate([y, np.zeros(n_bins - 2)])
    lb = np.concatenate([np.zeros(n_bins), [-np.inf] if pi is None else []])
    ub = np.full(X.shape[1], np.inf)
    sol = lsq_linear(Xs, ys, bounds=(lb, ub), method="bvls", max_iter=500)
    d = sol.x[:n_bins]
    pi_est = float(sol.x[-1]) if pi is None else float(pi)
    fit = X @ sol.x
    if pi is not None:
        fit = fit + pi
    centres = np.exp(0.5 * (lo + hi))
    g = np.cumsum(d * (hi - lo))
    return {"t": centres, "deriv": d, "g": g, "pi_est": pi_est, "fit": fit,
            "rms": float(np.sqrt(np.mean((fit - pwf) ** 2)))}


# ------------------------------------------------------------------ nonlinear
def _g_eval(tau, nodes, z, g0):
    """Unit-rate drawdown g(tau) for z = ln(dg/d ln t) piecewise-linear in ln t.

    Below the first node the response is taken as g0 * tau / t_min (unit-slope storage tail).
    """
    tau = np.asarray(tau, float)
    out = np.zeros_like(tau)
    pos = tau > 0
    s = np.log(np.where(pos, tau, 1.0))
    dsig = np.diff(nodes)
    ez = np.exp(z)
    dz = np.diff(z)
    slope = dz / dsig
    seg = np.where(np.abs(dz) > 1e-9, (ez[1:] - ez[:-1]) / np.where(np.abs(slope) > 1e-12, slope, 1.0),
                   ez[:-1] * dsig)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    t_min = np.exp(nodes[0])
    early = pos & (s < nodes[0])
    out[early] = g0 * tau[early] / t_min
    inside = pos & (s >= nodes[0])
    ss = np.clip(s[inside], nodes[0], nodes[-1])
    j = np.clip(np.searchsorted(nodes, ss, side="right") - 1, 0, len(nodes) - 2)
    zloc = z[j] + slope[j] * (ss - nodes[j])
    part = np.where(np.abs(dz[j]) > 1e-9,
                    (np.exp(zloc) - ez[j]) / np.where(np.abs(slope[j]) > 1e-12, slope[j], 1.0),
                    ez[j] * (ss - nodes[j]))
    gin = g0 + cum[j] + part
    # beyond the last node: extend with the last derivative level
    extra = np.maximum(s[inside] - nodes[-1], 0.0) * ez[-1]
    out[inside] = gin + extra
    return out


def deconvolve_nonlinear(schedule, t, pwf, pi=None, n_nodes=40, lam=0.5, t_min=None, noise=None,
                         max_nfev=120):
    """von Schroeter-type deconvolution: unknowns z = ln(dg/d ln t) on log-spaced nodes, the
    early offset g0 and (optionally) the initial pressure. Curvature-regularised nonlinear
    least squares. Guarantees a strictly positive derivative by construction.
    """
    from scipy.optimize import least_squares
    from . import qc

    t = np.asarray(t, float)
    pwf = np.asarray(pwf, float)
    m = t > 0
    t, pwf = t[m], pwf[m]
    ts, dq = MR.rate_changes(schedule)
    t_min = 0.5 * np.min(np.diff(np.concatenate([[0.0], t])).clip(1e-9)) if t_min is None else t_min
    nodes = np.linspace(np.log(max(t_min, 1e-6)), np.log(t.max()), n_nodes)
    lin = deconvolve(schedule, t, pwf, pi=pi, n_bins=max(n_nodes, 30), t_min=t_min, lam=1.0)
    der0 = np.interp(nodes, np.log(lin["t"]), np.maximum(lin["deriv"], 1e-6 * np.max(lin["deriv"])))
    z0 = np.log(der0)
    g00 = max(float(lin["g"][0]), 1e-6)
    sigma = float(noise if noise is not None else max(qc.noise_estimate(pwf), 0.02))
    N = len(t)
    w_reg = lam * 0.25 * np.sqrt(N) / max(sigma, 1e-9) * 0.05 * max(np.max(np.abs(pwf - pwf[0])), 1.0) ** 0.0

    def unpack(v):
        z = v[:n_nodes]
        g0 = np.exp(v[n_nodes])
        p_i = v[n_nodes + 1] if pi is None else pi
        return z, g0, p_i

    def model(v):
        z, g0, p_i = unpack(v)
        out = np.full(N, p_i)
        for t0, d in zip(ts, dq):
            mask = t > t0
            if mask.any():
                out[mask] -= d * _g_eval(t[mask] - t0, nodes, z, g0)
        return out

    def resid(v):
        z = v[:n_nodes]
        curv = z[:-2] - 2 * z[1:-1] + z[2:]
        return np.concatenate([(model(v) - pwf) / sigma, w_reg * curv])

    v0 = np.concatenate([z0, [np.log(g00)], [] if pi is not None else [lin["pi_est"]]])
    sol = least_squares(resid, v0, max_nfev=max_nfev, x_scale="jac")
    z, g0, p_i = unpack(sol.x)
    fit = model(sol.x)
    t_nodes = np.exp(nodes)
    return {"t": t_nodes, "deriv": np.exp(z), "g": _g_eval(t_nodes, nodes, z, g0), "pi_est": float(p_i),
            "fit": fit, "rms": float(np.sqrt(np.mean((fit - pwf) ** 2))), "method": "nonlinear", "nfev": int(sol.nfev)}
