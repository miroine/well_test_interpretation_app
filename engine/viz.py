"""Plot-ready data builders (JSON-friendly). Math only, no UI imports."""
from __future__ import annotations

import numpy as np

from . import analysis as A
from . import models as M


def diagnostic_from_data(tp, dt, pws, pwf_shutin=None, L=0.1):
    """Buildup diagnostics: Delta p and Bourdet derivative vs Agarwal time."""
    dt = np.asarray(dt, float)
    pws = np.asarray(pws, float)
    pw0 = float(pws[0] if pwf_shutin is None else pwf_shutin)
    dp = pws - pw0
    ta = A.agarwal_time(tp, dt)
    der = A.bourdet_derivative(ta, dp, L)
    return {"dt": dt, "ta": ta, "dp": dp, "der": der, "ht": A.horner_time(tp, dt)}


def model_curves(p: M.WellTestParams, q, tp, dt_end, n=140, dt_min=1e-3):
    dt = np.logspace(np.log10(dt_min), np.log10(dt_end), n)
    sched = M.simple_buildup_schedule(q, tp)
    t_shut = tp * (1 - 1e-9)
    pw = M.pwf_history(p, np.concatenate([[t_shut], tp + dt]), sched)
    dp = pw[1:] - pw[0]
    ta = A.agarwal_time(tp, dt)
    return {"dt": dt, "ta": ta, "dp": dp, "der": A.bourdet_derivative(ta, dp, 0.1),
            "pws": pw[1:], "ht": A.horner_time(tp, dt), "pwf_shutin": pw[0]}


def designed_test(p: M.WellTestParams, q, tp, dt_end, n=140):
    """Chronological pressure/rate and diagnostics for one flow/shut-in cycle.

    Drawdown uses elapsed flow time and the exact model derivative. Buildup uses
    either elapsed shut-in time or Agarwal time, with its derivative taken with
    respect to the corresponding log-time coordinate.
    """
    if not all(np.isfinite(v) for v in (q, tp, dt_end)) or q < 0 or tp <= 0 or dt_end <= 0:
        raise ValueError("Rate must be finite and non-negative; both durations must be finite and positive.")
    if not isinstance(n, int) or n < 3:
        raise ValueError("At least three samples per period are required.")
    flow_t = np.unique(np.concatenate([
        np.geomspace(min(1e-3, tp / 1000), tp, n),
        np.linspace(tp / n, tp, n // 2),
    ]))
    dp, der = M.unit_rate_response(p, flow_t)
    dd = {"t": flow_t, "dp": q * dp, "der": q * der, "pressure": p.pi - q * dp}
    bu = model_curves(p, q, tp, dt_end, n=n, dt_min=min(1e-3, dt_end / 1000))
    bu["der_elapsed"] = A.bourdet_derivative(bu["dt"], bu["dp"], 0.1)
    timeline = {
        "t": np.concatenate([[0.0], flow_t, tp + bu["dt"]]),
        "pressure": np.concatenate([[p.pi], dd["pressure"], bu["pws"]]),
        "rate_t": np.array([0.0, tp, tp, tp + dt_end]),
        "rate": np.array([q, q, 0.0, 0.0]),
    }
    return {"drawdown": dd, "buildup": bu, "timeline": timeline}


def design_summary(p: M.WellTestParams, q, tp, dt_end):
    """Concrete recommendations for test design (hours)."""
    CD = M.CD_of(p)
    t_wbs_end = 60.0 * CD * np.exp(0.14 * p.S) / (0.0002637 * p.k / (p.phi * p.mu * p.ct * M._scale_length(p) ** 2))
    t_wbs_end = max(t_wbs_end, 0.0)
    ri_end = float(M.radius_of_investigation(p, dt_end))
    out = {"t_wbs_end_hr": float(t_wbs_end), "ri_end_ft": ri_end,
           "ri_tp_ft": float(M.radius_of_investigation(p, tp)),
           "pressure_scale_psi": float(M.p_scale(p) * q)}
    if p.boundary != "none":
        d = p.d1
        out["t_boundary_hr"] = float(948.0 * p.phi * p.mu * p.ct * (2 * d) ** 2 / p.k)
        out["boundary_distance_ft"] = d
    # recommended shut-in length: 10x end of storage for radial flow, or time
    # to reach twice the nearest boundary distance if one is specified
    rec = 10.0 * t_wbs_end
    if "t_boundary_hr" in out:
        rec = max(rec, 2.0 * out["t_boundary_hr"])
    out["recommended_shutin_hr"] = float(max(rec, 1.0))
    return out


def monte_carlo_envelope(p, q, tp, dt, samples):
    """Pointwise response percentiles from paired Monte Carlo parameter rows."""
    dt = np.asarray(dt, float)
    if dt.ndim != 1 or len(dt) < 3 or not np.isfinite(dt).all() or np.any(dt <= 0) or np.any(np.diff(dt) <= 0):
        raise ValueError("Envelope times must be finite, positive and strictly increasing.")
    if not samples or any(name not in p.to_dict() for name in samples):
        raise ValueError("Monte Carlo samples must contain valid model parameters.")
    columns = {name: np.asarray(values, float) for name, values in samples.items()}
    lengths = {len(values) for values in columns.values() if values.ndim == 1}
    if len(lengths) != 1 or any(values.ndim != 1 for values in columns.values()):
        raise ValueError("Monte Carlo parameter columns must be paired and have equal lengths.")
    count = lengths.pop()
    if count < 3 or any(not np.isfinite(values).all() for values in columns.values()):
        raise ValueError("At least three finite, paired Monte Carlo parameter sets are required.")
    flow_t = np.geomspace(min(1e-3, tp / 1000), tp * (1 - 1e-9), 80)
    times = np.concatenate([[0.], flow_t, tp + dt])
    ta = A.agarwal_time(tp, dt)
    curves = {"pressure": [], "dp": [], "der": []}
    for i in range(count):
        member = M.WellTestParams(**{**p.to_dict(), **{name: float(values[i]) for name, values in columns.items()}})
        pressure = M.pwf_history(member, times, M.simple_buildup_schedule(q, tp))
        dp = pressure[1 + len(flow_t):] - pressure[len(flow_t)]
        derivative = A.bourdet_derivative(ta, dp, 0.1)
        if not np.isfinite(pressure).all() or not np.isfinite(dp).all():
            raise ValueError(f"Monte Carlo parameter set {i + 1} produces non-finite pressure.")
        curves["pressure"].append(pressure)
        curves["dp"].append(dp)
        curves["der"].append(derivative)
    result = {"t": times, "ta": ta, "count": count}
    for name, rows in curves.items():
        values = np.asarray(rows)
        valid = np.isfinite(values).all(axis=0)
        if name == "der" and valid.sum() < 3:
            raise ValueError("Too few finite derivative locations for the Monte Carlo envelope.")
        bounds = np.full((3, values.shape[1]), np.nan)
        bounds[:, valid] = np.percentile(values[:, valid], [10, 50, 90], axis=0)
        result[name] = dict(zip(("p10", "p50", "p90"), bounds))
    return result


def animation_payload(p: M.WellTestParams, q, tp, dt_end, n_frames=121, n_radii=56):
    nt = 2 * (n_frames - 1) + 1
    dt = np.logspace(-3, np.log10(dt_end), nt)
    sched = M.simple_buildup_schedule(q, tp)
    t_shut = tp * (1 - 1e-9)
    pw = M.pwf_history(p, np.concatenate([[t_shut], tp + dt]), sched)
    dp = pw[1:] - pw[0]
    ta = A.agarwal_time(tp, dt)
    der = A.bourdet_derivative(ta, dp, 0.1)
    labels = A.classify_regimes(ta, np.where(np.isfinite(der), der, np.nan_to_num(der, nan=1e-9)))
    ri = M.radius_of_investigation(p, dt)
    rmax = 1.25 * float(max(M.radius_of_investigation(p, tp + dt_end),
                            1.6 * p.d1 if p.boundary != "none" else 0.0))
    rmax = max(rmax, 50.0)
    r = np.logspace(np.log10(max(p.rw, 1e-3)), np.log10(rmax), n_radii)
    idx = np.arange(0, nt, 2)
    prof = [np.round(M.radial_profile(p, r, tp + dt[i], sched), 3).tolist() for i in idx]
    return {
        "pi": p.pi, "tp": tp, "q": q, "rw": p.rw, "rmax": rmax,
        "r": np.round(r, 3).tolist(),
        "frame_dt": np.round(dt[idx], 6).tolist(),
        "frame_idx": idx.tolist(),
        "prof": prof,
        "ri": np.round(ri, 3).tolist(),
        "dt": np.round(dt, 6).tolist(),
        "pws": np.round(pw[1:], 3).tolist(),
        "dp": np.round(dp, 4).tolist(),
        "ta": np.round(ta, 6).tolist(),
        "der": [None if not np.isfinite(x) else round(float(x), 5) for x in der],
        "labels": labels,
        "pwf_shutin": float(pw[0]),
        "boundary": {"type": p.boundary, "d1": p.d1, "d2": p.d2},
        "title": f"{p.model.replace('_', ' ')} | k={p.k:g} md | S={p.S:g}",
    }
