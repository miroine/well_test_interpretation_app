"""Multi-rate histories: schedule handling, superposition diagnostics, regression."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from . import analysis as A
from . import models as M


def schedule_from_series(t, q, rel_tol: float = 1e-6, min_len: int = 1):
    """Piecewise-constant schedule [(start_hr, rate)] from a rate time series.
    Times are shifted so the first sample is t = 0."""
    t = np.asarray(t, float)
    q = np.asarray(q, float)
    t0 = t[0]
    sched = [(0.0, float(q[0]))]
    for i in range(1, len(t)):
        ref = max(abs(sched[-1][1]), 1e-12)
        if abs(q[i] - sched[-1][1]) > rel_tol * ref + 1e-12:
            sched.append((float(t[i] - t0), float(q[i])))
    return sched


def clean_schedule(schedule):
    """Sort, merge equal-rate neighbours, force first start to 0."""
    s = sorted((float(a), float(b)) for a, b in schedule)
    out = []
    for st_, q in s:
        if out and abs(out[-1][1] - q) < 1e-12:
            continue
        out.append((st_, q))
    if out and out[0][0] != 0.0:
        out[0] = (0.0, out[0][1])
    return out


def rate_changes(schedule):
    ts = np.array([s[0] for s in schedule], float)
    qs = np.array([s[1] for s in schedule], float)
    return ts, np.diff(np.concatenate([[0.0], qs]))


def period_diagnostics(schedule, t, pwf, pi, period, L=0.1, start_pressure=None):
    """Signed local pressure change and Bourdet derivative on elapsed period time.

    This is an observed period overlay, not removal of earlier rate responses.
    Missing boundary pressure is linearly interpolated from recorded readings.
    """
    t, pwf = np.asarray(t, float), np.asarray(pwf, float)
    if (t.ndim != 1 or t.shape != pwf.shape or len(t) < 2 or
            not np.isfinite(t).all() or not np.isfinite(pwf).all() or np.any(np.diff(t) <= 0)):
        raise ValueError("Period diagnostics require finite, increasing time and matching pressure readings.")
    ts, dq = rate_changes(schedule)
    if not 0 <= period < len(ts) or abs(dq[period]) < 1e-12:
        raise ValueError("Choose a period with a non-zero rate change.")
    start = ts[period]
    stop = ts[period + 1] if period + 1 < len(ts) else t[-1]
    mask = (t > start) & (t <= stop)
    if start_pressure is None:
        if period == 0:
            start_pressure = pi
        elif start < t[0] or start > t[-1]:
            raise ValueError("The period start is outside the recorded pressure range.")
        else:
            start_pressure = np.interp(start, t, pwf)
    elapsed = t[mask] - start
    change = np.sign(dq[period]) * (float(start_pressure) - pwf[mask])
    return {"dt": elapsed, "dp": change, "der": A.bourdet_derivative(elapsed, change, L),
            "dq": float(dq[period]), "start_pressure": float(start_pressure)}


def superposition_diagnostics(schedule, t, pwf, pi, L: float = 0.1, period=None):
    """Rate-normalised pressure and derivative in a chosen flow period.

    g(t) = (pi - pwf) / dq_n ;  spt = sum_i (dq_i / dq_n) ln(t - t_{i-1})
    The derivative dg/d(spt) -> 0.5 * 141.2 B mu/(k h) in radial flow.
    Returns dict(dt, g, spt, der, dq) for points in the period (default: last
    period with a non-zero rate change).
    """
    t = np.asarray(t, float)
    pwf = np.asarray(pwf, float)
    ts, dq = rate_changes(schedule)
    n = (len(ts) - 1) if period is None else int(period)
    while n > 0 and abs(dq[n]) < 1e-12:
        n -= 1
    t_end = ts[n + 1] if n + 1 < len(ts) else np.inf
    m = (t > ts[n]) & (t <= t_end)
    tt, pp = t[m], pwf[m]
    spt = np.zeros_like(tt)
    for i in range(n + 1):
        spt += (dq[i] / dq[n]) * np.log(tt - ts[i])
    g = (pi - pp) / dq[n]
    der = A.bourdet_derivative(spt, g, L, log_x=False) if len(tt) > 3 else np.full_like(tt, np.nan)
    return {"dt": tt - ts[n], "g": g, "spt": spt, "der": der, "dq": float(dq[n]), "n": n}


def _plateau_derivative(x, der, w=5):
    ok = np.isfinite(der) & (der > 0)
    if ok.sum() < w + 3:
        return None
    xx, dd = x[ok], der[ok]
    s = np.gradient(np.log(dd), np.log(xx))
    best, bi = np.inf, None
    for i in range(len(xx) - w):
        v = np.mean(np.abs(s[i:i + w]))
        if v < best:
            best, bi = v, i
    return float(np.median(dd[bi:bi + w]))


def fit_multirate(p0: M.WellTestParams, schedule, t, pwf, free, multistart=True,
                  weight_deriv=1.0, max_nfev=80, *, progress=None):
    """Regression of a multi-rate pressure record via superposition."""
    if progress:
        progress(0, 0, "Preparing rate history and starting values")
    t = np.asarray(t, float)
    pwf = np.asarray(pwf, float)
    ok_t = t > 0
    t, pwf = t[ok_t], pwf[ok_t]
    span = float(np.ptp(pwf)) or 1.0
    diag = superposition_diagnostics(schedule, t, pwf, p0.pi)
    base = M.WellTestParams(**p0.to_dict())
    if "k" in free:
        pl = _plateau_derivative(diag["dt"], diag["der"])
        if pl:
            base.k = float(np.clip(base.B * base.mu * 70.6 / (base.h * pl), 1e-3, 1e5))
    lo = np.array([np.log10(A.FITTABLE[n][0]) if n in A.LOG_PARAMS else A.FITTABLE[n][0] for n in free])
    hi = np.array([np.log10(A.FITTABLE[n][1]) if n in A.LOG_PARAMS else A.FITTABLE[n][1] for n in free])

    def unpack(v, ref):
        p = M.WellTestParams(**ref.to_dict())
        for n, x in zip(free, v):
            setattr(p, n, 10 ** x if n in A.LOG_PARAMS else x)
        return p

    def pack(p):
        return np.array([np.log10(getattr(p, n)) if n in A.LOG_PARAMS else getattr(p, n) for n in free])

    ref_diag_ok = np.isfinite(diag["der"]) & (diag["der"] > 0)
    evaluate = M.prepare_pwf_history(t, schedule)

    def resid(v, ref):
        p = unpack(v, ref)
        mod = evaluate(p)
        r1 = (mod - pwf) / (0.02 * span)
        r = [r1]
        if weight_deriv > 0 and ref_diag_ok.any():
            md = superposition_diagnostics(schedule, t, mod, p.pi)
            ok = ref_diag_ok & np.isfinite(md["der"]) & (md["der"] > 0)
            r.append(weight_deriv * (np.log(md["der"][ok]) - np.log(diag["der"][ok])))
        return np.concatenate(r)

    starts = [base] + (A._start_grid(base, free, cap=10) if multistart else [])
    starts = A._unique_starts(starts, free)
    best = None
    failed, last_error = 0, None
    for index, s in enumerate(starts):
        if progress:
            progress(index, len(starts), f"Fitting attempt {index + 1} of {len(starts)}")
        v0 = np.clip(pack(s), lo + 1e-9, hi - 1e-9)
        try:
            sol = least_squares(resid, v0, bounds=(lo, hi), args=(s,), max_nfev=max_nfev, x_scale="jac")
        except Exception as exc:
            failed += 1
            last_error = exc
            if progress:
                progress(index + 1, len(starts), f"Attempt {index + 1} failed: {exc}")
            continue
        if best is None or sol.cost < best[1].cost:
            best = (unpack(sol.x, s), sol)
        if progress:
            progress(index + 1, len(starts), f"Completed {index + 1}/{len(starts)} attempts; {failed} failed")
    if best is None:
        raise ValueError(f"No multi-rate fitting attempt succeeded ({failed} failed). Last error: {last_error}") from last_error
    p_fit, sol = best
    return p_fit, {"cost": float(sol.cost), "nfev": int(sol.nfev), "status": int(sol.status),
                   "rms": float(np.sqrt(2 * sol.cost / len(sol.fun))), "failed_starts": failed}


def synthetic_multirate(p: M.WellTestParams, schedule, t_end, n=260, noise_psi=0.0, seed=4):
    """Log-ish spaced samples after every rate change over [0, t_end]."""
    ts = [s[0] for s in schedule] + [t_end]
    pts = []
    for a, b in zip(ts[:-1], ts[1:]):
        pts.append(a + np.logspace(-3, np.log10(b - a), max(n // len(schedule), 20)))
    t = np.unique(np.concatenate(pts))
    t = t[t <= t_end]
    pw = M.pwf_history(p, t, schedule)
    if noise_psi > 0:
        pw = pw + np.random.default_rng(seed).normal(0, noise_psi, len(pw))
    return t, pw
