"""Gauge-data QC: cleaning, outlier detection, noise estimate. Math only."""
from __future__ import annotations

import numpy as np
import pandas as pd


def clean_gauge(df: pd.DataFrame, tcol: str, pcol: str):
    """Sort, drop NaN / duplicate times. Returns (clean_df, issues[list[str]])."""
    if tcol == pcol:
        raise ValueError("Choose different columns for time and pressure.")
    issues = []
    d = df[[tcol, pcol]].copy()
    d.columns = ["t", "p"]
    n0 = len(d)
    d = d.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < n0:
        issues.append(f"{n0 - len(d)} non-numeric, empty or non-finite rows removed")
    if not d["t"].is_monotonic_increasing:
        issues.append("time column was not sorted - sorted ascending")
        d = d.sort_values("t")
    dup = d["t"].duplicated().sum()
    if dup:
        issues.append(f"{dup} duplicate time stamps - averaged")
        d = d.groupby("t", as_index=False).mean()
    return d.reset_index(drop=True), issues


def validate_buildup(t, p):
    """Reject records that cannot support buildup diagnostics."""
    t, p = np.asarray(t, float), np.asarray(p, float)
    if t.ndim != 1 or p.ndim != 1 or t.shape != p.shape:
        raise ValueError("Time and pressure must be one-dimensional arrays of equal length.")
    if t.size < 3:
        raise ValueError("At least three valid positive-time readings are required.")
    if not np.isfinite(t).all() or not np.isfinite(p).all():
        raise ValueError("Time and pressure must contain only finite values.")
    if np.any(t <= 0) or np.any(np.diff(t) <= 0):
        raise ValueError("Shut-in times must be positive, unique and increasing.")


def clean_history(df, tcol, pcol, qcol):
    """Keep synchronous pressure/rate readings; never average away rate events."""
    if len({tcol, pcol, qcol}) != 3:
        raise ValueError("Choose different columns for time, pressure and rate.")
    d = df[[tcol, pcol, qcol]].copy()
    d.columns = ["t", "p", "rate"]
    n = len(d)
    d = d.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    issues = [f"{n - len(d)} non-numeric, empty or non-finite rows removed"] if len(d) < n else []
    if not d["t"].is_monotonic_increasing:
        issues.append("time column was not sorted - sorted ascending, preserving rate-event order")
    d = d.sort_values("t", kind="stable").reset_index(drop=True)
    if len(d) < 3 or (d["t"] < 0).any():
        raise ValueError("History requires at least three readings with non-negative elapsed times.")
    return {k: d[k].to_numpy() for k in ("t", "p", "rate")}, issues


def response_source(dataset):
    """Return original readings, including excluded points, in engine units."""
    source = dataset.get("qc_source", dataset)
    if source.get("history") is not None:
        return {k: np.asarray(v, float) for k, v in source["history"].items()}
    dt = np.asarray(source["dt"], float)
    return {"t": dataset["tp"] + dt, "p": np.asarray(source["pws"], float),
            "rate": np.zeros_like(dt)}


def filter_response(dataset, excluded, start, end, drop=False, ppc=0, pressure_range=None):
    """Apply QC atomically from original rows, retaining a reversible audit trail."""
    history = response_source(dataset)
    t, p = history["t"], history["p"]
    if not np.isfinite([start, end]).all() or start >= end:
        raise ValueError("The kept time range must have a finite start earlier than its end.")
    excluded = np.asarray(excluded, int)
    if np.any((excluded < 0) | (excluded >= len(t))):
        raise ValueError("Selected point IDs are outside the original response.")
    keep = (t >= start) & (t <= end)
    keep[excluded] = False
    if pressure_range is not None:
        low, high = pressure_range
        if not np.isfinite([low, high]).all() or low > high:
            raise ValueError("The pressure range must have finite bounds in increasing order.")
        keep &= (p >= low) & (p <= high)
    buildup = t > dataset["tp"]
    # Detect outliers only within buildup, never across a rate discontinuity.
    if drop:
        indices = np.flatnonzero(buildup)
        keep[indices[hampel_outliers(p[indices])]] = False
    use = keep & buildup
    if np.any(np.abs(history["rate"][use]) > 1e-10):
        raise ValueError("The selected buildup contains non-zero rates. Select a shut-in period; use Multi-rate for flowing data.")
    dt, pws = t[use] - dataset["tp"], p[use]
    validate_buildup(dt, pws)
    if ppc:
        dt, pws = smooth_log_time(dt, pws, ppc)
        validate_buildup(dt, pws)
    source = dataset.get("qc_source")
    if source is None:
        source = {k: dataset[k] for k in ("dt", "pws", "history") if k in dataset}
    return {**dataset, "qc_source": source, "qc_excluded": np.flatnonzero(~keep).tolist(),
            "qc_window": [float(start), float(end)], "dt": dt, "pws": pws}


def restore_response(dataset):
    restored = {**dataset, **dataset["qc_source"]}
    for key in ("qc_source", "qc_excluded", "qc_window"):
        restored.pop(key, None)
    return restored


def hampel_outliers(y, window: int = 7, n_sigma: float = 4.0):
    """Boolean mask of outliers using a rolling median / MAD filter."""
    y = np.asarray(y, float)
    s = pd.Series(y)
    med = s.rolling(window, center=True, min_periods=3).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=3).median()
    sigma = 1.4826 * mad.replace(0, np.nan)
    z = ((s - med).abs() / sigma).fillna(0.0)
    return (z > n_sigma).to_numpy()


def noise_estimate(y):
    """Robust point-to-point noise (psi) from second differences."""
    y = np.asarray(y, float)
    if len(y) < 5:
        return 0.0
    d2 = y[2:] - 2 * y[1:-1] + y[:-2]
    return float(1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6.0))


def resolution_flags(t, p, min_psi_step=0.0):
    """Return warnings about gauge resolution / sampling gaps."""
    warns = []
    t = np.asarray(t, float)
    p = np.asarray(p, float)
    dt = np.diff(t)
    if len(dt) and dt.max() > 20 * np.median(dt):
        warns.append(f"large sampling gap: {dt.max():.3g} h vs median {np.median(dt):.3g} h")
    dp = np.abs(np.diff(p))
    nz = dp[dp > 0]
    if len(nz) and np.median(nz) > 0:
        quant = nz.min()
        if quant > 0.05 * np.ptp(p) / max(len(p), 1) * 5:
            warns.append(f"coarse gauge quantisation ~ {quant:.3g}")
    return warns


def smooth_log_time(t, y, points_per_cycle: int = 12):
    """Down-sample to roughly log-uniform spacing by averaging within bins."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = t > 0
    lt = np.log10(t[m])
    edges = np.arange(lt.min(), lt.max() + 1.0 / points_per_cycle, 1.0 / points_per_cycle)
    idx = np.digitize(lt, edges)
    tt, yy = [], []
    for i in np.unique(idx):
        sel = idx == i
        tt.append(np.exp(np.mean(np.log(t[m][sel]))))
        yy.append(np.mean(y[m][sel]))
    return np.array(tt), np.array(yy)
