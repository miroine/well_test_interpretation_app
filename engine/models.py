"""Dimensional well-test model layer (field units: psi, ft, md, cp, hr, stb/d).

Math only - no Streamlit imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from functools import lru_cache
from typing import Optional

import numpy as np

from . import laplace as L

MODEL_TYPES = ("homogeneous", "dual_pss", "dual_slab", "composite",
               "fracture", "limited_entry", "horizontal", "fracture_fc", "mfhw", "composite3")
BOUNDARY_TYPES = ("none", "fault", "const_p_fault", "channel", "corner", "circle", "circle_cp", "rectangle")
CLOSED_BOUNDARIES = ("circle", "circle_cp", "rectangle")


@dataclass
class WellTestParams:
    # reservoir / fluid
    k: float = 50.0          # md
    h: float = 50.0          # ft
    phi: float = 0.18
    mu: float = 1.0          # cp
    ct: float = 1.5e-5       # 1/psi
    B: float = 1.2           # rb/stb
    rw: float = 0.354        # ft
    pi: float = 4000.0       # psi (initial pressure)
    # near-well
    C: float = 0.01          # bbl/psi
    S: float = 2.0
    # model
    model: str = "homogeneous"
    omega: float = 0.05
    lam: float = 1e-6
    xf: float = 100.0        # ft  (fracture half-length)
    r_inner: float = 200.0   # ft  (composite)
    M: float = 2.0
    D: float = 2.0
    kv_kh: float = 0.1       # limited entry
    perf_frac: float = 0.3
    perf_pos: float = 0.5
    FcD: float = 10.0        # fracture conductivity kf wf / (k xf)
    n_frac: int = 4          # multi-fractured horizontal well
    f_space: float = 250.0   # ft  fracture spacing
    r_mid: float = 1000.0    # ft  (three-zone composite: outer radius of the middle zone)
    M3: float = 1.0          # mobility ratio inner/outer zone (three-zone)
    D3: float = 1.0          # diffusivity ratio inner/outer zone (three-zone)
    re: float = 2000.0       # ft  (circular reservoir radius)
    xe: float = 3000.0       # ft  (rectangle, x extent)
    ye: float = 1500.0       # ft  (rectangle, y extent)
    xw: float = 1500.0       # ft  (well position in the rectangle)
    yw: float = 750.0
    Lw: float = 1000.0       # ft  (horizontal well length)
    zw_h: float = 0.5        # horizontal well depth / h
    top_bc: str = "sealed"   # sealed | const_p (gas cap)
    bot_bc: str = "sealed"   # sealed | const_p (aquifer)
    # boundary
    boundary: str = "none"
    d1: float = 500.0        # ft
    d2: float = 800.0        # ft

    def to_dict(self):
        return asdict(self)


# ------------------------------------------------------------------ helpers
def _scale_length(p: WellTestParams) -> float:
    return p.xf if p.model in ("fracture", "fracture_fc", "mfhw") else p.rw


def tD_of(p: WellTestParams, t_hr):
    Ls = _scale_length(p)
    return 0.0002637 * p.k * np.asarray(t_hr, float) / (p.phi * p.mu * p.ct * Ls ** 2)


def CD_of(p: WellTestParams) -> float:
    Ls = _scale_length(p)
    return 0.8936 * p.C / (p.phi * p.ct * p.h * Ls ** 2)


def p_scale(p: WellTestParams) -> float:
    """psi per (stb/d) per unit pD."""
    return 141.2 * p.B * p.mu / (p.k * p.h)


def radius_of_investigation(p: WellTestParams, t_hr) -> np.ndarray:
    return np.sqrt(p.k * np.asarray(t_hr, float) / (948.0 * p.phi * p.mu * p.ct))


def _boundary_dict(p: WellTestParams):
    Ls = _scale_length(p)
    if p.boundary == "none" or p.boundary in CLOSED_BOUNDARIES:
        return None
    if p.boundary == "fault":
        return {"type": "fault", "d": p.d1 / Ls, "sealing": True}
    if p.boundary == "const_p_fault":
        return {"type": "fault", "d": p.d1 / Ls, "sealing": False}
    if p.boundary in ("channel", "corner"):
        return {"type": p.boundary, "d1": p.d1 / Ls, "d2": p.d2 / Ls}
    raise ValueError(p.boundary)


def _kernel(p: WellTestParams):
    if p.model == "dual_pss":
        return lambda u: L.f_dual_pss(u, p.omega, p.lam)
    if p.model == "dual_slab":
        return lambda u: L.f_dual_slab(u, p.omega, p.lam)
    return L.f_homogeneous


def boundary_warning(p: WellTestParams):
    """Text when a boundary is too close for the composite-image approximation, else None."""
    if p.model == "composite" and p.boundary != "none":
        near = min(x for x in (p.d1, p.d2 if p.boundary in ("channel", "corner") else p.d1))
        if near <= 1.2 * p.r_inner:
            return ("The boundary is within 1.2 x the inner-zone radius; the image approximation for "
                    "the composite model is not reliable there.")
    if p.boundary in CLOSED_BOUNDARIES and p.model not in ("homogeneous", "dual_pss", "dual_slab"):
        return "Closed-reservoir boundaries apply to homogeneous and dual-porosity models; the boundary is ignored here."
    return None


def laplace_function(p: WellTestParams):
    """Return pws(u) for the chosen model (skin and storage included)."""
    kern, CD, S, bd = _kernel(p), CD_of(p), p.S, _boundary_dict(p)
    if p.boundary in CLOSED_BOUNDARIES and p.model in ("homogeneous", "dual_pss", "dual_slab"):
        rw = p.rw
        if p.boundary == "circle":
            return lambda u: L.pws_circle(u, kern, CD, S, p.re / rw, "sealed")
        if p.boundary == "circle_cp":
            return lambda u: L.pws_circle(u, kern, CD, S, p.re / rw, "const_p")
        return lambda u: L.pws_rectangle(u, kern, CD, S, p.xe / rw, p.ye / rw, p.xw / rw, p.yw / rw)
    if p.model == "composite3":
        return lambda u: L.pws_composite3(u, kern, CD, S, p.r_inner / p.rw, p.r_mid / p.rw,
                                          p.M, p.D, p.M3, p.D3)
    if p.model in ("homogeneous", "dual_pss", "dual_slab"):
        return lambda u: L.pws_radial(u, kern, CD, S, bd)
    if p.model == "composite":
        rD = p.r_inner / p.rw
        return lambda u: L.pws_composite(u, kern, CD, S, rD, p.M, p.D, bd)
    if p.model == "fracture":
        return lambda u: L.pws_fracture(u, kern, CD, S, 0.732, bd)
    if p.model == "fracture_fc":
        return lambda u: L.pws_fracture_fc(u, kern, CD, S, p.FcD, bd)
    if p.model == "mfhw":
        return lambda u: L.pws_mfhw(u, kern, CD, S, p.FcD, int(p.n_frac), p.f_space / p.xf, bd)
    if p.model == "limited_entry":
        hD = (p.h / p.rw) * np.sqrt(1.0 / p.kv_kh)
        return lambda u: L.pws_limited_entry(u, kern, CD, S, hD, p.perf_frac,
                                             p.perf_pos, bd)
    if p.model == "horizontal":
        hD = (p.h / p.rw) * np.sqrt(1.0 / p.kv_kh)
        return lambda u: L.pws_horizontal(u, kern, CD, S, p.Lw / p.rw, hD, p.zw_h,
                                          p.top_bc, p.bot_bc)
    raise ValueError(p.model)


# --------------------------------------------------------------- responses
def unit_rate_response(p: WellTestParams, t_hr):
    """Drawdown response for 1 stb/d: (dp [psi], derivative t*dp/dt [psi])."""
    t_hr = np.asarray(t_hr, float)
    pD, dD = L.unit_response(laplace_function(p), tD_of(p, t_hr))
    s = p_scale(p)
    return pD * s, dD * s


def rate_schedule_arrays(schedule):
    """schedule: list of (start_hr, rate_stbd). Must start at t=0."""
    ts = np.array([s[0] for s in schedule], float)
    qs = np.array([s[1] for s in schedule], float)
    return ts, qs


def superposition_dp(p: WellTestParams, t_hr, schedule):
    """Pressure drop (pi - pwf) at times t_hr for a multi-rate history."""
    t_hr = np.asarray(t_hr, float)
    ts, qs = rate_schedule_arrays(schedule)
    dq = np.diff(np.concatenate([[0.0], qs]))
    dp = np.zeros_like(t_hr)
    for t0, d in zip(ts, dq):
        if d == 0:
            continue
        m = t_hr > t0
        if m.any():
            f, _ = unit_rate_response(p, t_hr[m] - t0)
            dp[m] += d * f
    return dp


def pwf_history(p: WellTestParams, t_hr, schedule):
    return p.pi - superposition_dp(p, t_hr, schedule)


def prepare_pwf_history(t_hr, schedule):
    """Prepare repeated pressure evaluations for one fixed record and schedule.

    Cache two reservoir responses so skin/storage/initial-pressure perturbations
    in a numerical Jacobian reuse the costly reservoir solve. Storage and skin
    are still applied in Laplace space before inversion, not to time-domain data.
    The bounded cache belongs to this evaluator only, never to a user session.
    """
    times = np.asarray(t_hr, float).copy()
    ts, qs = rate_schedule_arrays(schedule)
    dq = np.diff(np.concatenate([[0.0], qs]))
    steps = []
    for start, change in zip(ts, dq):
        mask = times > start
        if change != 0 and mask.any():
            steps.append((mask, times[mask] - start, change))
    coefficients = L.stehfest_coeffs(12)
    ln2 = np.log(2.0)

    @lru_cache(maxsize=2)
    def reservoir(key):
        base = WellTestParams(**dict(key), C=0., S=0., pi=0.)
        kernel = laplace_function(base)
        responses = []
        for mask, lag, change in steps:
            td = tD_of(base, lag)
            nodes = [(i * ln2 / td) for i in range(1, len(coefficients) + 1)]
            responses.append((td, [(u, kernel(u)) for u in nodes]))
        return responses

    def evaluate(p):
        key = tuple((name, value) for name, value in p.to_dict().items()
                    if name not in ("C", "S", "pi"))
        responses = reservoir(key)
        drop = np.zeros_like(times)
        cd, scale = CD_of(p), p_scale(p)
        for (mask, lag, change), (td, nodes) in zip(steps, responses):
            pressure = np.zeros_like(td)
            for coefficient, (u, value) in zip(coefficients, nodes):
                pressure += coefficient * L._add_storage_skin(value, u, cd, p.S)
            drop[mask] += change * (pressure * ln2 / td * scale)
        return p.pi - drop

    return evaluate


def simple_buildup_schedule(q: float, tp_hr: float):
    return [(0.0, q), (tp_hr, 0.0)]


def synthetic_buildup(p: WellTestParams, q: float, tp_hr: float, dt_end_hr: float,
                      n: int = 120, noise_psi: float = 0.0, seed: Optional[int] = 1):
    """Log-spaced buildup following a flow period tp. Returns dict of arrays."""
    dt = np.logspace(-3, np.log10(dt_end_hr), n)
    sched = simple_buildup_schedule(q, tp_hr)
    pws = pwf_history(p, tp_hr + dt, sched)
    if noise_psi > 0:
        rng = np.random.default_rng(seed)
        pws = pws + rng.normal(0.0, noise_psi, size=pws.shape)
    pwf0 = float(pwf_history(p, np.array([tp_hr * (1 - 1e-9)]), sched)[0])
    return {"dt": dt, "pws": pws, "pwf_at_shutin": pwf0, "q": q, "tp": tp_hr}


# ------------------------------------------------ radial pressure profiles
def radial_profile(p: WellTestParams, r_ft, t_hr, schedule):
    """Approximate p(r, t) used by the animation.

    The sandface rate comes from the chosen model (storage + skin included);
    the radial decay uses the homogeneous / dual-porosity kernel, so for
    composite, fractured and limited-entry wells the near-well shape is
    schematic while the far-field diffusion front is right.
    """
    r_ft = np.asarray(r_ft, float)
    Ls = _scale_length(p)
    rD = np.maximum(r_ft / Ls, 1.0)
    kern, CD = _kernel(p), CD_of(p)
    pws_u = laplace_function(p)
    ts, qs = rate_schedule_arrays(schedule)
    dq = np.diff(np.concatenate([[0.0], qs]))
    V = L.stehfest_coeffs(12)
    ln2 = np.log(2.0)
    dp = np.zeros_like(rD)
    for t0, d in zip(ts, dq):
        te = float(t_hr) - t0
        if d == 0 or te <= 0:
            continue
        tD = float(tD_of(p, te))
        acc = np.zeros_like(rD)
        for i, Vi in enumerate(V, start=1):
            u = np.array([i * ln2 / tD])
            z = np.sqrt(u * kern(u))[0]
            qsf = (1.0 / (u * (1.0 + CD * u * u * pws_u(u))))[0]
            acc += Vi * qsf * L.k0e(z * rD) / (z * L.k1e(z)) * np.exp(-z * (rD - 1.0))
        dp += d * (acc * ln2 / tD) * p_scale(p)
    return p.pi - dp
