"""Derivative, semilog analysis and automatic matching. Math only."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from . import models as M


def bourdet_derivative(x, y, L: float = 0.1, log_x: bool = True):
    """Bourdet 3-point derivative dy/d(ln x) with smoothing window L (ln units)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    lx = np.log(x) if log_x else x
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(1, n - 1):
        j = i - 1
        while j > 0 and lx[i] - lx[j] < L:
            j -= 1
        k = i + 1
        while k < n - 1 and lx[k] - lx[i] < L:
            k += 1
        if lx[i] - lx[j] < 1e-12 or lx[k] - lx[i] < 1e-12:
            continue
        d1 = (y[i] - y[j]) / (lx[i] - lx[j])
        d2 = (y[k] - y[i]) / (lx[k] - lx[i])
        w1, w2 = lx[k] - lx[i], lx[i] - lx[j]
        out[i] = (d1 * w1 + d2 * w2) / (w1 + w2)
    return out


def agarwal_time(tp, dt):
    dt = np.asarray(dt, float)
    return tp * dt / (tp + dt)


def horner_time(tp, dt):
    dt = np.asarray(dt, float)
    return (tp + dt) / dt


def semilog_radial(x, y, window):
    """Fit y = m*log10(x) + b on x within (lo, hi). Returns slope, intercept."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = (x >= window[0]) & (x <= window[1])
    if mask.sum() < 3:
        raise ValueError("fewer than 3 points in the selected window")
    m, b = np.polyfit(np.log10(x[mask]), y[mask], 1)
    return float(m), float(b)


def permeability_from_slope(m_psi_per_cycle, q, B, mu, h):
    """Horner slope m (psi/log cycle, magnitude) -> k [md]."""
    return 162.6 * q * B * mu / (abs(m_psi_per_cycle) * h)


def skin_from_horner(p1hr, pwf_shutin, m, k, phi, mu, ct, rw):
    m = abs(m)
    return 1.1513 * ((p1hr - pwf_shutin) / m
                     - np.log10(k / (phi * mu * ct * rw ** 2)) + 3.2275)


# ------------------------------------------------------------------ fitting
FITTABLE = {
    "k": (1e-3, 1e5), "S": (-6.0, 60.0), "C": (1e-5, 10.0),
    "omega": (1e-3, 0.99), "lam": (1e-10, 1e-1), "xf": (1.0, 2000.0),
    "d1": (20.0, 20000.0), "d2": (20.0, 20000.0),
    "r_inner": (5.0, 5000.0), "M": (0.05, 100.0), "D": (0.05, 100.0),
    "pi": (1.0, 2e4),
    "Lw": (20.0, 20000.0), "FcD": (0.05, 1e4), "re": (30.0, 1e6), "xe": (50.0, 1e6), "ye": (50.0, 1e6), "r_mid": (10.0, 1e5), "M3": (0.05, 100.0), "D3": (0.05, 100.0), "f_space": (10.0, 5000.0), "kv_kh": (1e-4, 1.0), "zw_h": (0.02, 0.98),
}
LOG_PARAMS = {"k", "C", "lam", "xf", "d1", "d2", "r_inner", "M", "D", "omega", "Lw", "kv_kh", "FcD", "f_space", "re", "xe", "ye", "r_mid", "M3", "D3"}


SLOW_MODELS = ("horizontal", "mfhw", "fracture_fc")


def _fit_once(p0, q, tp, dt, dp_obs, der_obs, ok, ta, free, weight_deriv, max_nfev):
    sched = M.simple_buildup_schedule(q, tp)
    t_shut = tp * (1 - 1e-9)
    evaluate = M.prepare_pwf_history(np.concatenate([[t_shut], tp + dt]), sched)

    def pack(p):
        return np.array([np.log10(getattr(p, n)) if n in LOG_PARAMS else getattr(p, n)
                         for n in free])

    def unpack(v):
        p = M.WellTestParams(**p0.to_dict())
        for n, val in zip(free, v):
            setattr(p, n, 10 ** val if n in LOG_PARAMS else val)
        return p

    lo = np.array([np.log10(FITTABLE[n][0]) if n in LOG_PARAMS else FITTABLE[n][0]
                   for n in free])
    hi = np.array([np.log10(FITTABLE[n][1]) if n in LOG_PARAMS else FITTABLE[n][1]
                   for n in free])
    if p0.model in M.INFINITE_FRACTURE_MODELS and "S" in free:
        lo[free.index("S")] = 0.

    def resid(v):
        p = unpack(v)
        mod = evaluate(p)
        dp_mod = mod[1:] - mod[0]
        r1 = (dp_mod[ok] - dp_obs[ok]) / np.maximum(np.abs(dp_obs[ok]), 1.0)
        dm = bourdet_derivative(ta, dp_mod)
        r2 = weight_deriv * (np.log(np.maximum(dm[ok], 1e-6))
                             - np.log(np.maximum(der_obs[ok], 1e-6)))
        return np.concatenate([r1, r2])

    v0 = np.clip(pack(p0), lo + 1e-9, hi - 1e-9)
    if p0.model in M.INFINITE_FRACTURE_MODELS and "S" in free:
        # A near-zero bounded seed can give TRF a vanishing initial trust region.
        v0[free.index("S")] = max(v0[free.index("S")], .01)
    sol = least_squares(resid, v0, bounds=(lo, hi), max_nfev=max_nfev, x_scale="jac")
    return unpack(sol.x), sol


_MULT = {"k": (0.5, 1.0, 2.0), "d1": (0.4, 1.0, 2.5), "d2": (0.4, 1.0, 2.5), "xf": (0.5, 2.0),
         "r_inner": (0.4, 2.5), "M": (0.3, 3.0), "Lw": (0.5, 2.0), "FcD": (0.2, 5.0), "re": (0.5, 2.0), "xe": (0.6, 1.7), "ye": (0.6, 1.7), "r_mid": (0.5, 2.0), "M3": (0.3, 3.0), "f_space": (0.5, 2.0), "kv_kh": (0.2, 5.0)}
_ABS = {"S": (0.0, 5.0), "omega": (0.02, 0.2), "lam": (1e-7, 1e-5)}


def _start_grid(p0, free, cap=30, seed=0):
    """Starting points for multi-start: product of per-parameter grids, capped."""
    import itertools
    axes = []
    for n in free:
        if n in _MULT:
            axes.append([(n, getattr(p0, n) * f) for f in _MULT[n]])
        elif n in _ABS:
            axes.append([(n, v) for v in _ABS[n]])
    if not axes:
        return []
    combos = list(itertools.product(*axes))
    if len(combos) > cap:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in rng.choice(len(combos), cap, replace=False)]
    out = []
    for combo in combos:
        s = M.WellTestParams(**p0.to_dict())
        for n, v in combo:
            setattr(s, n, float(v))
        out.append(s)
    return out


def _unique_starts(starts, free):
    """Keep distinct optimizer starting vectors after applying its bounds."""
    seen, unique = set(), []
    for p in starts:
        vector = []
        for name in free:
            low, high = FITTABLE[name]
            value = getattr(p, name)
            if name in LOG_PARAMS:
                low, high, value = np.log10([low, high, value])
            vector.append(float(np.clip(value, low + 1e-9, high - 1e-9)))
        key = tuple(vector)
        if key not in seen:
            unique.append(p)
            seen.add(key)
    return unique


def auto_initial(p0, q, tp, dt, dp, der, ta):
    """Analytic starting values: k from the derivative plateau, C from the early
    unit-slope points, skin from the plateau pressure/derivative ratio.
    Returns dict of whichever of k, C, S could be estimated."""
    ok = np.isfinite(der) & (der > 0) & (dp > 0)
    if ok.sum() < 8:
        return {}
    t, d, y = ta[ok], der[ok], dp[ok]
    slope = np.gradient(np.log(d), np.log(t))
    w = 5
    best_i, best = None, np.inf
    for i in range(0, len(t) - w):
        # skip the early storage region: require dp/der well above 1
        if np.median(y[i:i + w] / d[i:i + w]) < 1.5:
            continue
        s = np.mean(np.abs(slope[i:i + w]))
        if s < best:
            best, best_i = s, i
    out = {}
    dd = np.sort(dt[ok])[:3]
    qB = q * p0.B
    out["C"] = float(np.clip(np.median(qB * dt[ok][:3] / (24.0 * dp[ok][:3])), 1e-5, 10.0))
    if best_i is not None:
        sl = slice(best_i, best_i + w)
        der_p = float(np.median(d[sl]))
        k0 = 70.6 * q * p0.B * p0.mu / (p0.h * der_p)
        tw = float(np.median(t[sl]))
        S0 = 0.5 * (float(np.median(y[sl] / d[sl])) - np.log(k0 * tw / (p0.phi * p0.mu * p0.ct * p0.rw ** 2)) + 7.43)
        out["k"] = float(np.clip(k0, 1e-3, 1e5))
        out["S"] = float(np.clip(S0, -6.0, 60.0))
    return out


def _screen_dual(p0, q, tp, dt, dp_obs, der_obs, ok, ta, free, keep=3):
    """Coarse screening for dual-porosity (omega, lambda): the dip depth gives omega,
    its time gives lambda; try a grid and return the best few starting points."""
    sched = M.simple_buildup_schedule(q, tp)
    t_shut = tp * (1 - 1e-9)
    idx = np.where(ok)[0]
    if len(idx) < 12:
        return []
    dd = der_obs[idx]
    levels = [float(np.nanpercentile(dd, 90)), float(np.nanmedian(dd)), float(np.nanpercentile(dd, 40))]
    k_grid = sorted({float(np.clip(70.6 * q * p0.B * p0.mu / (p0.h * lv), 1e-3, 1e5)) for lv in levels if lv > 0})
    times = np.concatenate([[t_shut], tp + dt])
    evaluate = M.prepare_pwf_history(times, sched)

    def cost(p):
        try:
            mod = evaluate(p)
        except Exception:
            return np.inf
        dpm = mod[1:] - mod[0]
        r1 = (dpm[ok] - dp_obs[ok]) / np.maximum(np.abs(dp_obs[ok]), 1.0)
        dm = bourdet_derivative(ta, dpm)
        r2 = np.log(np.maximum(dm[ok], 1e-6)) - np.log(np.maximum(der_obs[ok], 1e-6))
        return float(np.sum(r1 ** 2) + np.sum(r2[np.isfinite(r2)] ** 2))

    scored = []
    for om in (0.01, 0.03, 0.1, 0.3):
        for lam in np.logspace(-9, -3, 13):
            for kk in (k_grid if "k" in free else [p0.k]):
                for sk in (-2.0, 1.0):
                    s = M.WellTestParams(**p0.to_dict())
                    s.omega, s.lam, s.k = om, float(lam), kk
                    if "S" in free:
                        s.S = sk
                    scored.append((cost(s), s))
    scored.sort(key=lambda x: x[0])
    return [s for c, s in scored[:keep] if np.isfinite(c)]


def confidence_intervals(sol, free, p_fit, level=1.96):
    """Linearised 95 % intervals and correlations from the least-squares Jacobian.

    Computed in the packed space (log10 for LOG_PARAMS). The residual variance is scaled by the
    actual misfit, so a poor model widens the intervals. Treat them as indicative: the derivative
    residuals are correlated and the problem is nonlinear.
    """
    J, m, n = sol.jac, len(sol.fun), len(free)
    dof = max(m - n, 1)
    s2 = 2.0 * sol.cost / dof
    try:
        cov = s2 * np.linalg.pinv(J.T @ J)
    except np.linalg.LinAlgError:
        return {"ci": {}, "corr": None}
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    ci = {}
    for i, nme in enumerate(free):
        v = getattr(p_fit, nme)
        if nme in LOG_PARAMS:
            lo, hi = v * 10 ** (-level * sd[i]), v * 10 ** (level * sd[i])
            rel = float(np.log(10) * sd[i])
        else:
            lo, hi = v - level * sd[i], v + level * sd[i]
            rel = float(sd[i] / abs(v)) if v else float("nan")
        ci[nme] = {"value": float(v), "lo": float(lo), "hi": float(hi), "rel_sd": rel}
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(sd, sd)
    return {"ci": ci, "corr": np.nan_to_num(corr).tolist()}


def sensitivity_curves(p: M.WellTestParams, q, tp, dt, names, factors=(0.5, 2.0)):
    """Model dp / derivative for each parameter scaled by each factor (others fixed)."""
    from . import viz
    base = viz.model_curves(p, q, tp, float(np.max(dt)), n=120, dt_min=float(np.min(dt)))
    out = {"base": base, "runs": []}
    for nme in names:
        for fct in factors:
            pp = M.WellTestParams(**p.to_dict())
            lo, hi = FITTABLE.get(nme, (-np.inf, np.inf))
            setattr(pp, nme, float(np.clip(getattr(p, nme) * fct, lo, hi)) if getattr(p, nme) != 0 else fct)
            try:
                out["runs"].append({"name": nme, "factor": fct,
                                    "curves": viz.model_curves(pp, q, tp, float(np.max(dt)), n=120,
                                                               dt_min=float(np.min(dt)))})
            except Exception:
                continue
    return out


def monte_carlo(p_fit: M.WellTestParams, q, tp, dt, pws, free, pwf_shutin=None, n=20, seed=0,
                max_nfev=40):
    """Refit n synthetic data sets (fitted model + Gaussian noise of the observed misfit)."""
    rng = np.random.default_rng(seed)
    dt = np.asarray(dt, float)
    pws = np.asarray(pws, float)
    sched = M.simple_buildup_schedule(q, tp)
    pw0 = float(pws[0] if pwf_shutin is None else pwf_shutin)
    times = np.concatenate([[tp * (1 - 1e-9)], tp + dt])
    mod = M.pwf_history(p_fit, times, sched)
    model_p = pw0 + (mod[1:] - mod[0])
    sigma = float(np.std(pws - model_p))
    samples = {nme: [] for nme in free}
    for _ in range(n):
        syn = model_p + rng.normal(0.0, sigma, len(dt))
        try:
            pf, _ = fit_buildup(p_fit, q, tp, dt, syn, free, pwf_shutin=pw0, max_nfev=max_nfev, multistart=False)
        except Exception:
            continue
        for nme in free:
            samples[nme].append(getattr(pf, nme))
    summ = {}
    for nme, v in samples.items():
        if len(v) >= 3:
            a = np.asarray(v)
            summ[nme] = {"p10": float(np.percentile(a, 10)), "p50": float(np.percentile(a, 50)),
                         "p90": float(np.percentile(a, 90)), "n": int(len(a))}
    return {"samples": {k: list(map(float, v)) for k, v in samples.items()}, "summary": summ, "sigma": sigma}


def fit_buildup(p0: M.WellTestParams, q, tp, dt, pws, free, pwf_shutin=None,
                weight_deriv=1.0, max_nfev=80, multistart=True, *, progress=None):
    """Least-squares match of pressure change and derivative (log space).

    pwf_shutin  last flowing pressure before shut-in (defaults to pws[0]).
    multistart  also tries a small grid of starting values for skin / storage /
                k and keeps the lowest-cost solution (guards against the k-S
                trade-off that traps single-start fits).
    """
    if progress:
        progress(0, 0, "Preparing data and starting values")
    if p0.model in SLOW_MODELS:           # expensive forward models: single start, capped iterations
        multistart = False
        max_nfev = min(max_nfev, 60)
    dt = np.asarray(dt, float)
    pws = np.asarray(pws, float)
    pw0 = float(pws[0] if pwf_shutin is None else pwf_shutin)
    dp_obs = pws - pw0
    ta = agarwal_time(tp, dt)
    der_obs = bourdet_derivative(ta, dp_obs)
    ok = np.isfinite(der_obs) & (dp_obs > 0)

    starts = [p0]
    guess = auto_initial(p0, q, tp, dt, dp_obs, der_obs, ta)
    centre = M.WellTestParams(**p0.to_dict())
    for n, v in guess.items():
        if n in free:
            setattr(centre, n, v)
    if guess:
        starts.insert(0, centre)
    if multistart and p0.model in ("dual_pss", "dual_slab") and "omega" in free and "lam" in free:
        if progress:
            progress(0, 0, "Screening dual-porosity starting values")
        for s in _screen_dual(centre, q, tp, dt, dp_obs, der_obs, ok, ta, free):
            for n, v in guess.items():
                if n in free and n not in ("k", "S"):
                    setattr(s, n, v)
            starts.insert(0, s)
    if multistart:
        starts += _start_grid(centre, free, cap=14)
    starts = _unique_starts(starts, free)
    best = None
    failed, last_error = 0, None
    for index, s in enumerate(starts):
        if progress:
            progress(index, len(starts), f"Fitting attempt {index + 1} of {len(starts)}")
        try:
            p_fit, sol = _fit_once(s, q, tp, dt, dp_obs, der_obs, ok, ta, free,
                                   weight_deriv, max_nfev)
        except Exception as exc:
            failed += 1
            last_error = exc
            if progress:
                progress(index + 1, len(starts), f"Attempt {index + 1} failed: {exc}")
            continue
        if best is None or sol.cost < best[1].cost:
            best = (p_fit, sol)
        if progress:
            progress(index + 1, len(starts), f"Completed {index + 1}/{len(starts)} attempts; {failed} failed")
    if best is None:
        raise ValueError(f"No fitting attempt succeeded ({failed} failed). Last error: {last_error}") from last_error
    p_fit, sol = best
    info = {"cost": float(sol.cost), "nfev": int(sol.nfev), "status": int(sol.status),
            "rms": float(np.sqrt(2 * sol.cost / len(sol.fun))), "failed_starts": failed}
    info.update(confidence_intervals(sol, free, p_fit))
    return p_fit, info


def classify_regimes(t, deriv, tol=0.12):
    """Label derivative slope regimes (very light heuristic, used by animation)."""
    lt, ld = np.log10(t), np.log10(np.maximum(deriv, 1e-12))
    s = np.gradient(ld, lt)
    labels = []
    for si in s:
        if abs(si - 1.0) < tol:
            labels.append("wellbore storage")
        elif abs(si - 0.5) < tol:
            labels.append("linear flow")
        elif abs(si - 0.25) < tol:
            labels.append("bilinear flow")
        elif abs(si) < tol:
            labels.append("radial flow")
        elif si < -tol:
            labels.append("transition (dual-porosity / pressure support)")
        else:
            labels.append("transition")
    return labels
