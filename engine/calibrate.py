"""Automatic calibration of the black-oil simulator to measured well data."""
from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .blackoil import BlackOilSim, SimConfig, RelPerm, PVTConfig, NewtonFailure

# name -> (where, attribute, lo, hi, log-scale)
CAL_PARAMS = {
    "k":        ("cfg", "k", 1.0, 5000.0, True),
    "skin":     ("cfg", "skin", -3.0, 40.0, False),
    "kv_kh":    ("cfg", "kv_kh", 0.005, 2.0, True),
    "C":        ("cfg", "C", 1e-4, 5.0, True),
    "phi":      ("cfg", "phi", 0.03, 0.4, False),
    "krg_max":  ("rp", "krg_max", 0.1, 1.0, False),
    "krw_max":  ("rp", "krw_max", 0.05, 1.0, False),
    "ng":       ("rp", "ng", 1.0, 5.0, False),
    "outer_pv_mult": ("cfg", "outer_pv_mult", 1.0, 1e4, True),
}


def _get(cfg, rp, name):
    where, attr = CAL_PARAMS[name][:2]
    return getattr(cfg if where == "cfg" else rp, attr)


def _set(cfg, rp, name, val):
    where, attr = CAL_PARAMS[name][:2]
    setattr(cfg if where == "cfg" else rp, attr, float(val))


def simulate(cfg, pvt, rp, schedule, t_end, **kw):
    return BlackOilSim(cfg, pvt, rp).run(schedule, t_end, **kw)


def _series(res, key, t_obs):
    """Interpolate a simulated series at t_obs, holding the last flowing value across shut-in gaps."""
    a = pd.Series(np.asarray(res[key], float)).ffill().bfill().to_numpy()
    return np.interp(t_obs, res["t"], a)


def synthetic_observations(res, t_obs, noise_psi=0.0, seed=0, noise_gor=0.03, noise_wc=0.005):
    """Pressure, GOR and water cut sampled from a simulation. GOR / WC are NaN while shut in."""
    rng = np.random.default_rng(seed)
    t_obs = np.asarray(t_obs, float)
    pw = np.interp(t_obs, res["t"], res["pwf"]) + rng.normal(0, noise_psi, len(t_obs))
    flowing = np.interp(t_obs, res["t"], np.asarray(res["qset"], float)) > 0
    gor = _series(res, "GOR", t_obs) * (1 + rng.normal(0, noise_gor, len(t_obs)))
    wc = np.clip(_series(res, "WC", t_obs) + rng.normal(0, noise_wc, len(t_obs)), 0, 1)
    gor = np.where(flowing, gor, np.nan)
    wc = np.where(flowing, wc, np.nan)
    return {"t": t_obs, "pwf": pw, "GOR": gor, "WC": wc}


def calibrate(cfg: SimConfig, pvt: PVTConfig | None, rp: RelPerm, schedule, obs: dict, free,
              use=("pwf",), max_nfev=25, sigma_psi=1.0, log=None):
    """Least-squares match of simulated pwf (and optionally GOR / WC) to observations.

    obs      dict with 't' [hr] and any of 'pwf' [psi], 'GOR' [scf/stb], 'WC' [-]
    free     names from CAL_PARAMS
    Returns (cfg_fit, rp_fit, info)
    """
    free = list(free)
    t_obs = np.asarray(obs["t"], float)
    t_end = float(t_obs.max())
    lo = np.array([np.log10(CAL_PARAMS[n][2]) if CAL_PARAMS[n][4] else CAL_PARAMS[n][2] for n in free])
    hi = np.array([np.log10(CAL_PARAMS[n][3]) if CAL_PARAMS[n][4] else CAL_PARAMS[n][3] for n in free])
    v0 = np.array([np.log10(_get(cfg, rp, n)) if CAL_PARAMS[n][4] else _get(cfg, rp, n) for n in free])
    v0 = np.clip(v0, lo + 1e-9, hi - 1e-9)
    trace = []

    span = np.maximum(hi - lo, 1e-9)
    OFF = 10.0                                   # normalised variables live in [10, 11] so the
                                                 # relative finite-difference step is well defined

    def build(w):
        c, r = copy.deepcopy(cfg), copy.deepcopy(rp)
        v = lo + (np.asarray(w) - OFF) * span
        for n, x in zip(free, v):
            _set(c, r, n, 10 ** x if CAL_PARAMS[n][4] else x)
        return c, r

    masks = {u: np.isfinite(np.asarray(obs[u], float)) for u in use}
    n_res = int(sum(m.sum() for m in masks.values()))

    def resid(v):
        c, r = build(v)
        try:
            res = simulate(c, pvt, r, schedule, t_end)
        except (NewtonFailure, ValueError):
            return np.full(n_res, 1e3)
        out = []
        if "pwf" in use:
            m = masks["pwf"]
            out.append((np.interp(t_obs, res["t"], res["pwf"])[m] - np.asarray(obs["pwf"], float)[m]) / sigma_psi)
        if "GOR" in use:
            m = masks["GOR"]
            g = _series(res, "GOR", t_obs)[m]
            out.append(5.0 * (np.log(np.maximum(g, 1.0)) - np.log(np.maximum(np.asarray(obs["GOR"], float)[m], 1.0))))
        if "WC" in use:
            m = masks["WC"]
            out.append(20.0 * (_series(res, "WC", t_obs)[m] - np.asarray(obs["WC"], float)[m]))
        r_ = np.concatenate(out)
        trace.append(float(np.sqrt(np.mean(r_ ** 2))))
        if log:
            log(trace[-1])
        return r_

    w0 = OFF + (v0 - lo) / span
    sol = least_squares(resid, w0, bounds=(np.full(len(free), OFF), np.full(len(free), OFF + 1.0)),
                        max_nfev=max_nfev, diff_step=0.002, x_scale=0.1)
    c, r = build(sol.x)
    info = {"free": free, "start": {n: _get(cfg, rp, n) for n in free},
            "fitted": {n: _get(c, r, n) for n in free},
            "rms": float(np.sqrt(np.mean(sol.fun ** 2))) * sigma_psi if tuple(use) == ("pwf",) else float(np.sqrt(np.mean(sol.fun ** 2))),
            "use": list(use),
            "nfev": int(sol.nfev), "trace": trace}
    return c, r, info
