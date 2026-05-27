"""
modules/engine.py
-----------------
Pure-math pressure transient analysis engine. No Streamlit imports — keeps
the math layer independent of UI for testability and reuse.

Supports liquid (oil/water) AND gas wells:
  - For oil/water: works directly in ΔP and Δt.
  - For gas:       works in Δm(p) (pseudo-pressure) and Δtₐ (pseudo-time).
                   Math is identical — same line-source, same Bourdet,
                   same IARF regression — only the variables and slope
                   constant differ. See modules.pvt for the transforms.

Implements:
  - Radial line-source (Theis / Ei-function) drawdown solution
  - Fair–Schulz wellbore-storage transition model
  - Bourdet derivative with adjustable log-time smoothing window L
  - Event (flow-period) detection from rate signal
  - Point decimation for high-frequency gauge data
  - Regression helpers for IARF (radial) and linear-flow (sqrt-time) regimes
  - Wellbore storage clearance rule-of-thumb
  - Gas wrappers: simulate_drawdown_gas, regress_radial_gas
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import exp1   # E1(x) exponential integral
from dataclasses import dataclass
from enum import Enum


# =====================================================================
#  Fluid type
# =====================================================================
class FluidType(str, Enum):
    OIL   = "Oil"
    GAS   = "Gas"
    WATER = "Water"   # behaves like oil for engine purposes (slightly compressible)


# =====================================================================
#  Field-units constants
#  Liquid (oil/water) PTA, q [STB/D], B [rb/STB], μ [cp], k [mD], h [ft],
#  t [hr], P [psi], rw [ft], ct [1/psi], C [bbl/psi]
# =====================================================================
PSEUDO_PRESSURE_CONST = 162.6      # liquid semilog slope: m = 162.6 q B μ /(k h)
LINESRC_CONST = 70.6               # ΔP = 70.6 qBμ/(kh) · [-Ei(-x)]
ETA_CONST = 0.0002637              # k·t / (φμct·rw²) dimensionless time prefactor

# Gas constants — Lee/Wattenbarger Chapter 6
# Gas semilog slope on m(p) plot:  m_g = 1637 · qg · T / (k · h)
#   qg  in Mscf/D
#   T   in °R
#   k   in mD
#   h   in ft
#   m_g in psi²/cp per cycle
GAS_SEMILOG_CONST = 1637.0
# Gas line-source pseudo-pressure prefactor (analogue of 70.6 for oil):
#   Δm(p) = 711 · qg·T/(kh) · [-Ei(-x)]  → 1637 / ln(10) ≈ 711
GAS_LINESRC_CONST = GAS_SEMILOG_CONST / np.log(10.0)   # ≈ 711.0


# =====================================================================
#  Inputs container
# =====================================================================
@dataclass
class WellInputs:
    k: float          # permeability,  mD
    h: float          # net pay,       ft
    q: float          # rate,          STB/D
    s: float          # skin,          dimensionless
    C: float          # wellbore storage coefficient, bbl/psi
    mu: float = 1.0   # viscosity,     cp
    B: float = 1.2    # FVF,           rb/STB
    phi: float = 0.15 # porosity
    ct: float = 1.0e-5  # total compressibility, 1/psi
    rw: float = 0.354   # wellbore radius, ft
    pi: float = 5000.0  # initial pressure, psi


# =====================================================================
#  Drawdown — Radial line-source + Fair–Schulz wellbore storage
# =====================================================================
def line_source_dp(t_hr: np.ndarray, w: WellInputs) -> np.ndarray:
    """
    Radial line-source (Ei-function) infinite-acting drawdown solution.

        ΔP(t) = (70.6 q B μ / k h) * [ -Ei(-x) ]
        where x = 948 φ μ ct rw² / (k t)   (rw is reservoir-distance argument)

    Includes the skin pressure drop ΔP_skin = (141.2 qBμ/kh) * s, applied as
    a step contribution at the wellbore.
    """
    t = np.asarray(t_hr, dtype=float)
    t = np.clip(t, 1e-6, None)

    arg = 948.0 * w.phi * w.mu * w.ct * (w.rw ** 2) / (w.k * t)
    # exp1(arg) == -Ei(-arg) for arg > 0, which is what we want here
    ei_term = exp1(arg)

    dp_reservoir = (LINESRC_CONST * w.q * w.B * w.mu / (w.k * w.h)) * ei_term
    dp_skin = (2.0 * LINESRC_CONST * w.q * w.B * w.mu / (w.k * w.h)) * w.s
    return dp_reservoir + dp_skin


def fair_schulz_transition(t_hr: np.ndarray, w: WellInputs) -> np.ndarray:
    """
    Fair–Schulz transition smoothly blends pure wellbore storage (early)
    into the formation response (late) via a weighting based on the
    dimensionless wellbore-storage time tD/CD.

    Pure WBS limit:   ΔP = (q B / 24 C) * t      (linear in t, unit slope on log-log)
    Late time:        line-source solution
    Weight:           w = 1 - exp( -(t / τ)^2 ),   τ ≈ 200 C / kh  (empirical)

    When C ≤ 0, returns the pure line-source response (no WBS).
    """
    t = np.asarray(t_hr, dtype=float)
    t = np.clip(t, 1e-6, None)

    # Reservoir line-source — always computed
    dp_res = line_source_dp(t, w)

    if w.C <= 0:
        return dp_res

    # Pure wellbore-storage contribution
    dp_wbs = (w.q * w.B / (24.0 * w.C)) * t

    # Characteristic transition time (hr) — empirical Fair–Schulz scaling
    tau = max(200.0 * w.C / (w.k * w.h), 1e-3)
    weight = 1.0 - np.exp(-((t / tau) ** 2))

    return (1.0 - weight) * dp_wbs + weight * dp_res


def simulate_drawdown(duration_hr: float, n_points: int, w: WellInputs,
                      re_ft: float | None = None) -> pd.DataFrame:
    """
    Simulate a synthetic drawdown test on a log-time grid.

    If ``re_ft`` is provided, a closed circular drainage area of radius re
    is assumed. The PSS contribution is added on top of the infinite-acting
    response. This is the standard textbook approximation:

        Infinite-acting:   ΔP_inf  =  line-source · ei_term + skin
        PSS contribution:  ΔP_pss  =  m* · t                   (linear in t)
                           m*       =  0.234 · q·B / (φ ct h · π re²)

    The crossover occurs near tD ≈ 0.1·re_D². Before that, the line-source
    answer dominates; after, PSS dominates and the derivative trends with
    unit slope on log-log.

    Returns a DataFrame with columns:
        t              elapsed time (hr)
        dp             ΔP (psi)
        tdp_dt         t·dP/dt — canonical Bourdet derivative (psi)
        pressure       bottom-hole pressure (psi)
    """
    t = np.logspace(-3, np.log10(max(duration_hr, 1e-2)), n_points)
    dp = fair_schulz_transition(t, w)

    if re_ft is not None and re_ft > 0:
        # Material-balance depletion slope (Lee Eq. 3.78 simplified)
        # m* = 0.234 q B / (φ ct h A),   A = π re²
        area = np.pi * re_ft * re_ft
        m_star = 0.234 * abs(w.q) * w.B / (w.phi * w.ct * w.h * area)
        # Only add PSS contribution AFTER the boundary is felt:
        # tD_pss_onset ≈ 0.1·re_D² in dimensionless time; in real time:
        # t_pss = 0.1·re²·φ μ ct / (k · ETA_CONST)
        t_pss_onset = 0.1 * re_ft * re_ft * w.phi * w.mu * w.ct / (w.k * ETA_CONST)
        # Smooth turn-on with a logistic so the derivative is differentiable
        switch = 1.0 / (1.0 + np.exp(-3.0 * (np.log(t) - np.log(t_pss_onset))))
        dp_pss = m_star * t * switch
        dp = dp + dp_pss

    tdp_dt = bourdet_derivative(t, dp, L=0.1)
    pressure = w.pi - dp
    return pd.DataFrame({"t": t, "dp": dp, "tdp_dt": tdp_dt, "pressure": pressure})


# =====================================================================
#  Bourdet derivative with log-time smoothing window L
# =====================================================================
def bourdet_derivative(t: np.ndarray, dp: np.ndarray, L: float = 0.1) -> np.ndarray:
    """
    Three-point Bourdet derivative on a log-time grid (fully vectorized).

    Because differences are taken w.r.t. ln(t), this returns the canonical
            t · dP/dt   ≡   d(ΔP) / d(ln t)
    which is what's plotted on the log-log diagnostic.

    For each point i, find neighbors j (left) and k (right) such that
        ln(t[i]) - ln(t[j]) ≥ L     and     ln(t[k]) - ln(t[i]) ≥ L
    Then:
        t·dP/dt[i] = ( m_left·Δln_right + m_right·Δln_left )
                     / (Δln_left + Δln_right)
    with the two-sided log-time slopes m_left, m_right.

    Implementation: `np.searchsorted` locates the first index in [0..i-1] whose
    ln(t) is ≥ ln(t[i]) - L (→ j candidate) and the first index in [i+1..n-1]
    whose ln(t) is ≥ ln(t[i]) + L (→ k candidate). Edge points fall back to a
    two-point log-time slope. Runtime: O(n log n) vs the previous O(n·m).
    """
    t = np.asarray(t, dtype=float)
    dp = np.asarray(dp, dtype=float)
    n = len(t)
    if n < 2:
        return np.zeros(n)

    lnt = np.log(np.clip(t, 1e-12, None))

    # Left neighbor j: largest index j < i with lnt[j] <= lnt[i] - L.
    # searchsorted with 'right' on (lnt[i] - L) gives first index strictly greater,
    # so j = that_index - 1.
    target_left = lnt - L
    j_idx = np.searchsorted(lnt, target_left, side="right") - 1
    j_idx = np.clip(j_idx, 0, n - 1)

    # Right neighbor k: smallest index k > i with lnt[k] >= lnt[i] + L.
    target_right = lnt + L
    k_idx = np.searchsorted(lnt, target_right, side="left")
    k_idx = np.clip(k_idx, 0, n - 1)

    # Where the L window can't be satisfied (neighbor collapses to i itself or
    # crosses to the wrong side), fall back to two-point log-time slopes.
    valid = (j_idx < np.arange(n)) & (k_idx > np.arange(n))

    dl_left = lnt - lnt[j_idx]
    dl_right = lnt[k_idx] - lnt
    # avoid div-by-zero (only used where invalid; result is overwritten)
    safe_dl_left  = np.where(dl_left  > 0, dl_left,  1.0)
    safe_dl_right = np.where(dl_right > 0, dl_right, 1.0)
    m_left  = (dp - dp[j_idx]) / safe_dl_left
    m_right = (dp[k_idx] - dp) / safe_dl_right

    deriv = (m_left * dl_right + m_right * dl_left) / (dl_left + dl_right + 1e-30)

    # Edge / fallback two-point slope: use immediate neighbors
    if n >= 2:
        edge = np.zeros(n)
        # interior centered diff
        edge[1:-1] = (dp[2:] - dp[:-2]) / np.where(
            (lnt[2:] - lnt[:-2]) > 0, lnt[2:] - lnt[:-2], 1.0
        )
        # endpoints
        edge[0]  = (dp[1] - dp[0])   / max(lnt[1] - lnt[0],   1e-12)
        edge[-1] = (dp[-1] - dp[-2]) / max(lnt[-1] - lnt[-2], 1e-12)
        deriv = np.where(valid, deriv, edge)

    # `deriv` is d(ΔP)/d(ln t) ≡ t·dP/dt — the canonical Bourdet quantity.
    return deriv


# =====================================================================
#  Diagnostic utilities
# =====================================================================
def wbs_clearance_time(w: WellInputs) -> float:
    """
    Rule-of-thumb time required to clear wellbore storage and reach IARF:
        t_wbs ≈ (200000 + 12000 s) * C / (kh / μ)        [hr]
    Conservative variant of the (50 + 3.5 s)*CD relation transformed to field units.
    """
    return (200000.0 + 12000.0 * max(w.s, 0.0)) * w.C / max(w.k * w.h / w.mu, 1e-6)


# =====================================================================
#  Data ingestion utilities
# =====================================================================
def decimate(df: pd.DataFrame, n_keep: int) -> pd.DataFrame:
    """Keep every Nth point. Used to tame high-frequency gauge data."""
    if n_keep <= 1:
        return df.reset_index(drop=True)
    return df.iloc[::n_keep, :].reset_index(drop=True)


def detect_flow_periods(time: np.ndarray, rate: np.ndarray,
                        rate_threshold: float = 1.0) -> list[dict]:
    """
    Detect flow periods from a rate signal. A new period starts whenever
    rate changes by more than `rate_threshold` between consecutive samples.

    Returns a list of dicts:
       [{"i_start", "i_end", "t_start", "t_end", "rate", "type"}, ...]
    type is 'flow' if rate > 0 else 'buildup'.
    """
    t = np.asarray(time, dtype=float)
    q = np.asarray(rate, dtype=float)
    if len(t) < 2:
        return []

    # change points
    dq = np.abs(np.diff(q))
    change_idx = np.where(dq > rate_threshold)[0] + 1
    boundaries = np.concatenate(([0], change_idx, [len(t)]))

    periods = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if b - a < 3:        # too short to be meaningful
            continue
        seg_rate = float(np.median(q[a:b]))
        periods.append({
            "i_start": int(a),
            "i_end": int(b - 1),
            "t_start": float(t[a]),
            "t_end": float(t[b - 1]),
            "rate": seg_rate,
            "type": "buildup" if abs(seg_rate) < rate_threshold else "flow",
        })
    return periods


def build_buildup_derivatives(df: pd.DataFrame, periods: list[dict],
                              L: float = 0.1) -> list[dict]:
    """
    For each detected buildup period, build the (Δt, ΔP, ΔP') triplet
    plus the producing time `t_p` (duration of the preceding flow period)
    needed for Horner-time superposition plots.
    """
    out = []
    for idx, p in enumerate(periods):
        if p["type"] != "buildup":
            continue
        prev_flow = None
        for q in reversed(periods[:idx]):
            if q["type"] == "flow":
                prev_flow = q
                break
        if prev_flow is None:
            continue

        a, b = p["i_start"], p["i_end"]
        t_seg = df["time"].iloc[a:b + 1].to_numpy()
        p_seg = df["pressure"].iloc[a:b + 1].to_numpy()

        dt = t_seg - t_seg[0]
        dp = p_seg - p_seg[0]
        mask = dt > 1e-6
        if mask.sum() < 5:
            continue

        dt_v = dt[mask]
        dp_v = dp[mask]
        dpd = bourdet_derivative(dt_v, dp_v, L=L)

        # Producing time t_p for Horner: time from start of flow to start of buildup
        t_p = float(prev_flow["t_end"] - prev_flow["t_start"])

        out.append({
            "label": f"Buildup #{len(out) + 1}  (q_ref={prev_flow['rate']:.1f})",
            "q_ref": prev_flow["rate"],
            "dt": dt_v,
            "dp": np.abs(dp_v),
            # bourdet_derivative already returns t·dP/dt — no further multiplication
            "tdp_dt": np.abs(dpd),
            "t_p": t_p,
            "p0": float(p_seg[0]),       # BHP at start of buildup
        })
    return out


def horner_time(dt: np.ndarray, t_p: float) -> np.ndarray:
    """
    Horner buildup time (dimensionless, log-scale):
        t_H = (t_p + Δt) / Δt
    Plotted on a log-x axis, IARF appears as a straight line whose slope
    gives kh via m = 162.6·qBμ/(kh).
    """
    dt = np.asarray(dt, dtype=float)
    return (t_p + dt) / np.clip(dt, 1e-12, None)


def superposition_time(dt: np.ndarray, t_p: float) -> np.ndarray:
    """
    Single-flow superposition time:  T = log10((t_p + Δt) / Δt).
    Same content as Horner-time but rendered on a LINEAR axis — this is
    what most PTA software plots as "Superposition(T)".
    """
    return np.log10(horner_time(dt, t_p))


# =====================================================================
#  Regression — radial flow (IARF) and linear flow (sqrt-time)
# =====================================================================
def regress_radial(dt: np.ndarray, dp_deriv_tdpdt: np.ndarray,
                   q: float, B: float, mu: float, h: float,
                   t_min: float, t_max: float,
                   dp: np.ndarray | None = None,
                   phi: float | None = None,
                   ct: float | None = None,
                   rw: float | None = None) -> dict:
    """
    Within a user-selected window [t_min, t_max], assume IARF and compute:

        plateau   = geom-mean( t·dP/dt | window )                 # psi
        m         = ln(10) * plateau                              # psi/cycle
        k         = 162.6 · q · B · μ / (m · h)                   # mD

    If ΔP, φ, ctᵼ, rw are provided, ALSO compute the Horner-style skin from
    the buildup ΔP at Δt = 1 hr (interpolated within the window):

        s = 1.151 · [ ΔP(1 hr) / m  −  log10(k / (φ μ cₜ rw²))  +  3.23 ]

    A "flatness" diagnostic is the geometric standard deviation of the
    derivative inside the window — values close to 1.0 indicate a clean
    plateau, values ≳ 1.4 suggest the window is not in pure IARF.
    """
    mask = (dt >= t_min) & (dt <= t_max) & (dp_deriv_tdpdt > 0)
    if mask.sum() < 3:
        return {"ok": False, "message": "Not enough points in the window."}

    tdpdt_win = dp_deriv_tdpdt[mask]
    ln_tdpdt = np.log(tdpdt_win)
    plateau = float(np.exp(np.mean(ln_tdpdt)))                       # geom mean
    flatness = float(np.exp(np.std(ln_tdpdt)))                       # geom std
    m = np.log(10.0) * plateau                                       # semilog slope
    k = PSEUDO_PRESSURE_CONST * abs(q) * B * mu / (m * h) if (m * h) > 0 else np.nan

    result = {
        "ok": True,
        "regime": "Radial Flow (IARF)",
        "plateau_tdpdt": plateau,
        "semilog_slope_m": m,
        "k_mD": k,
        "flatness_geom_std": flatness,
        "n_points": int(mask.sum()),
        "t_min": t_min, "t_max": t_max,
    }

    # Optional skin calculation
    if dp is not None and phi is not None and ct is not None and rw is not None \
            and k > 0 and m > 0:
        dt_win = dt[mask]
        dp_win = dp[mask]
        # Interpolate ΔP at Δt = 1 hr (or use nearest if window misses 1 hr)
        if dt_win.min() <= 1.0 <= dt_win.max():
            dp_1hr = float(np.interp(1.0, dt_win, dp_win))
        else:
            idx = np.argmin(np.abs(dt_win - 1.0))
            dp_1hr = float(dp_win[idx])
        try:
            term = np.log10(k / (phi * mu * ct * rw * rw))
            # Standard Horner skin formula (Earlougher):
            #   s = 1.151 * [ ΔP(1hr) / m  −  log10(k / (φ μ cₜ rw²))  +  3.23 ]
            s_horner = 1.151 * (dp_1hr / m - term + 3.23)
            result["skin"] = float(s_horner)
            result["dp_1hr"] = dp_1hr
        except Exception:
            pass

    return result


def regress_linear(dt: np.ndarray, dp: np.ndarray,
                   q: float, B: float, mu: float, h: float, ct: float, phi: float,
                   t_min: float, t_max: float) -> dict:
    """
    Square-root time plot: ΔP vs √t. Linear flow slope m_L gives the
    fracture-half-length capacity xf·√k:

        m_L = 4.064 q B / (h) * √(μ / (φ ct k)) / xf
      ⇒ xf·√k = 4.064 q B / (m_L · h) · √(μ / (φ ct))
    """
    mask = (dt >= t_min) & (dt <= t_max)
    if mask.sum() < 3:
        return {"ok": False, "message": "Not enough points in the window."}

    sqrt_t = np.sqrt(dt[mask])
    y = dp[mask]
    A = np.vstack([sqrt_t, np.ones_like(sqrt_t)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    if slope <= 0:
        return {"ok": False, "message": "Non-physical (negative) linear-flow slope."}

    xf_sqrtk = 4.064 * abs(q) * B / (slope * h) * np.sqrt(mu / (phi * ct))
    return {
        "ok": True,
        "regime": "Linear Flow (Fracture)",
        "slope": float(slope),
        "intercept": float(intercept),
        "xf_sqrtk": float(xf_sqrtk),
        "n_points": int(mask.sum()),
        "t_min": t_min, "t_max": t_max,
    }


# =====================================================================
#  Gas-well wrappers
#
#  Strategy: gas PTA in pseudo-variables looks IDENTICAL to oil PTA in
#  ΔP/Δt. So we transform inputs (p → m(p), t → tₐ) and outputs once at
#  the boundary, then reuse the existing simulation and regression
#  machinery untouched. The slope constant changes (1637 vs 162.6) so we
#  recompute k from the pseudo-pressure plateau directly.
# =====================================================================
@dataclass
class GasWellInputs:
    """Gas-well inputs. All in oilfield units.
        k  [mD], h [ft], qg [Mscf/D], s [-], C [bbl/psi],
        T_R [°R], pi [psia], phi [-], rw [ft].
    Viscosity, FVF, ct, m(p), tₐ all come from the PVT table.
    """
    k: float
    h: float
    qg: float
    s: float
    C: float
    T_R: float
    pi: float = 5000.0
    phi: float = 0.15
    rw: float = 0.354


def simulate_gas_drawdown(duration_hr: float, n_points: int,
                          gw: GasWellInputs, pvt_table: dict,
                          re_ft: float | None = None) -> pd.DataFrame:
    """
    Simulate a gas drawdown using the line-source solution applied in
    pseudo-pressure space.

        Δm(p)(t) = 711·qg·T/(kh) · [-Ei(-x)] + 2·711·qg·T/(kh)·s
        x = 948·φ·μ(pi)·ct(pi)·rw² / (k·t_a)      (μ, ct at reference pressure pi)

    If ``re_ft`` is provided, a closed circular boundary at radius re is
    added via a PSS contribution m_g* · t (in pseudo-pressure space).

    Returns DataFrame columns:
        t       elapsed time [hr]
        t_a     pseudo-time  [psi·hr/cp]   (single-pressure approximation at pi)
        p       BHP [psia]                 (inverted from m(pwf) = m(pi) − Δm)
        dm      Δm(p) [psi²/cp]
        tdmdt   t · dm/dt   (Bourdet derivative on pseudo-pressure)
    """
    from pvt import pseudo_pressure_table, pseudo_pressure

    # Pseudo-pressure table for the chosen PVT
    p_tab, m_tab = pseudo_pressure_table(pvt_table)

    # μ, ct at the reference pressure (initial pressure) for pseudo-time
    mu_ref = float(np.interp(gw.pi, pvt_table["p"], pvt_table["mu"]))
    cg_ref = float(np.interp(gw.pi, pvt_table["p"], pvt_table["cg"]))
    # cₜ_total ≈ cg for dry gas; could add cw·Sw + cf if user provides them
    ct_ref = cg_ref

    # Time grid (log-spaced)
    t = np.logspace(-3, np.log10(max(duration_hr, 1e-2)), n_points)
    t_a = t / (mu_ref * ct_ref)            # constant-pressure pseudo-time

    # Apply line-source / Fair-Schulz in pseudo-space using "fake" liquid inputs:
    # treat Δm(p) and tₐ as the physics variables; the gas slope constant is
    # 1637 vs 162.6 for oil. We compute the Ei argument with μ_ref·ct_ref
    # already absorbed into tₐ, so just use t_a directly with k.
    arg = 948.0 * gw.phi * (gw.rw ** 2) / (gw.k * np.clip(t_a, 1e-12, None))
    ei_term = exp1(arg)
    A_lscoeff = GAS_LINESRC_CONST * gw.qg * gw.T_R / (gw.k * gw.h)
    dm_reservoir = A_lscoeff * ei_term
    dm_skin = 2.0 * A_lscoeff * gw.s
    dm_pure = dm_reservoir + dm_skin

    # Pure wellbore-storage contribution: treat as a linear ramp in REAL time
    #   Δm(p) for pure WBS ≈ (qg · Bg / 24 · C) — but Bg is pressure-dependent.
    # Use Bg at pi as a representative value; this is an approximation.
    Bg_ref = float(np.interp(gw.pi, pvt_table["p"], pvt_table["Bg"]))   # rb/Mscf
    # Convert Bg [rb/Mscf] · qg [Mscf/D] → rb/D, then to bbl/hr via /24
    dm_wbs_dp = (gw.qg * Bg_ref / (24.0 * gw.C)) * t                    # psi (rough)
    # To express that in pseudo-pressure space: convert dp → dm using local slope
    # dm/dp ≈ 2 p / (μ Z). At p = pi:
    Z_ref = float(np.interp(gw.pi, pvt_table["p"], pvt_table["Z"]))
    dmdp_ref = 2.0 * gw.pi / (mu_ref * Z_ref)
    dm_wbs = dm_wbs_dp * dmdp_ref

    # Fair-Schulz weighting in pseudo-time space
    tau = max(200.0 * gw.C / (gw.k * gw.h), 1e-3)   # hours
    weight = 1.0 - np.exp(-((t / tau) ** 2))
    dm = (1.0 - weight) * dm_wbs + weight * dm_pure

    # Closed-boundary PSS contribution (in pseudo-pressure space)
    if re_ft is not None and re_ft > 0:
        area = np.pi * re_ft * re_ft
        m_g_star = 2.356e6 * abs(gw.qg) * gw.T_R / (gw.phi * ct_ref * gw.h * area)
        t_pss_onset = (0.1 * re_ft * re_ft * gw.phi * mu_ref * ct_ref
                       / (gw.k * ETA_CONST))
        switch = 1.0 / (1.0 + np.exp(-3.0 * (np.log(t) - np.log(t_pss_onset))))
        dm_pss = m_g_star * t * switch
        dm = dm + dm_pss

    # Invert m → p so the user can see a pressure trace as well
    m_pi = pseudo_pressure(gw.pi, p_tab, m_tab)
    m_pwf = m_pi - dm
    # Clip to table range before inverting
    m_pwf_clip = np.clip(m_pwf, m_tab[0], m_tab[-1])
    p_wf = np.interp(m_pwf_clip, m_tab, p_tab)

    # Bourdet derivative in pseudo-pressure space — uses the already-vectorized helper
    tdmdt = bourdet_derivative(t, dm, L=0.1)

    return pd.DataFrame({
        "t": t, "t_a": t_a, "p": p_wf, "dm": dm, "tdmdt": tdmdt,
    })


def wbs_clearance_time_gas(gw: GasWellInputs, pvt_table: dict) -> float:
    """
    Rule-of-thumb clearance time for a gas well, using μ at reference pressure.
        t_wbs ≈ (200000 + 12000·s) · C / (k·h / μ_ref)        [hr]
    """
    mu_ref = float(np.interp(gw.pi, pvt_table["p"], pvt_table["mu"]))
    return ((200000.0 + 12000.0 * max(gw.s, 0.0))
            * gw.C / max(gw.k * gw.h / mu_ref, 1e-6))


def regress_radial_gas(dt: np.ndarray, dm_deriv_tdmdt: np.ndarray,
                       qg: float, T_R: float, h: float,
                       t_min: float, t_max: float,
                       dm: np.ndarray | None = None,
                       phi: float | None = None,
                       mu_ref: float | None = None,
                       ct_ref: float | None = None,
                       k_guess: float | None = None,
                       rw: float | None = None) -> dict:
    """
    Radial-flow regression on a gas buildup (or drawdown converted to dm).
    Replaces ΔP with Δm(p) and applies the gas slope constant:

        plateau   = geom-mean( t·dm/dt | window )                 [psi²/cp]
        m_g       = ln(10) · plateau                              [psi²/cp per cycle]
        k         = 1637 · qg · T / (m_g · h)                     [mD]

    Skin (optional, if ΔP_at_1hr extras provided):
        s = 1.151 · [ Δm(1hr) / m_g  −  log10(k / (φ μ_ref cₜ_ref rw²)) + 3.23 ]
    """
    mask = (dt >= t_min) & (dt <= t_max) & (dm_deriv_tdmdt > 0)
    if mask.sum() < 3:
        return {"ok": False, "message": "Not enough points in the window."}

    tdmdt_win = dm_deriv_tdmdt[mask]
    plateau = float(np.exp(np.mean(np.log(tdmdt_win))))
    flatness = float(np.exp(np.std(np.log(tdmdt_win))))
    m_g = np.log(10.0) * plateau
    k = GAS_SEMILOG_CONST * abs(qg) * T_R / (m_g * h) if (m_g * h) > 0 else np.nan

    result = {
        "ok": True,
        "regime": "Radial Flow (IARF, gas pseudo-pressure)",
        "plateau_tdmdt": plateau,
        "semilog_slope_m_g": m_g,
        "k_mD": k,
        "flatness_geom_std": flatness,
        "n_points": int(mask.sum()),
        "t_min": t_min, "t_max": t_max,
    }

    # Skin requires Δm at 1 hr plus PVT-derived μ_ref, ct_ref, φ, rw
    if (dm is not None and phi is not None and mu_ref is not None
            and ct_ref is not None and rw is not None and k > 0 and m_g > 0):
        dt_win = dt[mask]
        dm_win = dm[mask]
        if dt_win.min() <= 1.0 <= dt_win.max():
            dm_1hr = float(np.interp(1.0, dt_win, dm_win))
        else:
            idx = np.argmin(np.abs(dt_win - 1.0))
            dm_1hr = float(dm_win[idx])
        try:
            term = np.log10(k / (phi * mu_ref * ct_ref * rw * rw))
            s_horner = 1.151 * (dm_1hr / m_g - term + 3.23)
            result["skin"] = float(s_horner)
            result["dm_1hr"] = dm_1hr
        except Exception:
            pass

    return result


# =====================================================================
#  Linear flow for gas (fracture-dominated, infinite conductivity)
#
#  Same √t analysis as oil, but Δm(p) vs √Δt and gas constant 40.92:
#       Δm(p) = m_L · √t                                 (psi²/cp)
#       xf · √k  =  40.92 · qg · T / (h · m_L) · √(1 / (φ μ_ref ct_ref))
#
#  Derivation: liquid linear-flow slope formula is
#       m_L_liq = 4.064 · q·B/h · √(μ/(φ ct k)) / xf
#  In pseudo-pressure, the corresponding pressure-squared analogue gives
#  the prefactor 40.92 (= 4.064 · √(141.2/1.4...)·... see Lee-Wattenbarger
#  Eq. 6.108). We follow the standard textbook form.
# =====================================================================
def regress_linear_gas(dt: np.ndarray, dm: np.ndarray,
                       qg: float, T_R: float, h: float,
                       phi: float, mu_ref: float, ct_ref: float,
                       t_min: float, t_max: float) -> dict:
    """
    Square-root-time regression on gas Δm(p) (pseudo-pressure linear flow).

        Δm(p) = m_L · √t                                 [psi²/cp]
        xf · √k = 40.92 · qg · T_R / (m_L · h)
                  · √(1 / (φ μ_ref ct_ref))             [ft · √mD]

    The μ_ref, ct_ref are evaluated at a representative pressure (typically
    the initial reservoir pressure pi) from the cached PVT table.
    """
    mask = (dt >= t_min) & (dt <= t_max)
    if mask.sum() < 3:
        return {"ok": False, "message": "Not enough points in the window."}

    sqrt_t = np.sqrt(dt[mask])
    y = dm[mask]
    A = np.vstack([sqrt_t, np.ones_like(sqrt_t)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    if slope <= 0:
        return {"ok": False,
                "message": "Non-physical (negative) linear-flow slope."}

    xf_sqrtk = (40.92 * abs(qg) * T_R / (slope * h)
                * np.sqrt(1.0 / (phi * mu_ref * ct_ref)))

    return {
        "ok": True,
        "regime": "Linear Flow (Fracture, gas pseudo-pressure)",
        "slope": float(slope),                # psi²/cp per √hr
        "intercept": float(intercept),
        "xf_sqrtk": float(xf_sqrtk),
        "n_points": int(mask.sum()),
        "t_min": t_min, "t_max": t_max,
    }


# =====================================================================
#  Closed-boundary (Pseudo-Steady-State) regression
#
#  After a well has drained out to its no-flow boundary, pressure declines
#  uniformly at the reservoir-volume rate. On a Cartesian (linear-time)
#  plot, ΔP vs t becomes a straight line during PSS:
#
#       Liquid:   dP/dt = m*  = 0.234 · q·B / (φ ct h A)              [psi/hr]
#                 → A = 0.234 · q·B / (φ ct h · m*)                    [ft²]
#                 → V_pore = A · h · φ                                  [ft³]
#                 → V_pore_bbl = V_pore / 5.615
#
#       Gas:      dm(p)/dt = m_g* = 2.356·10⁶ · qg·T / (φ ct h A)      [psi²/cp/hr]
#                 → A = 2.356e6 · qg·T / (φ ct h · m_g*)               [ft²]
#       The 2.356e6 prefactor is the gas analogue of 0.234·(141.2·1000)/24/0.0036...
#       see Lee-Wattenbarger Eq. 6.135. Most practitioners back out drainage
#       AREA, not pore volume, then multiply by net pay.
#
#  Distinguishing feature on log-log: t·dP/dt rises with UNIT slope (=1).
# =====================================================================
def regress_pss(dt: np.ndarray, dp: np.ndarray,
                q: float, B: float, h: float, phi: float, ct: float,
                t_min: float, t_max: float) -> dict:
    """
    Pseudo-steady-state regression — straight-line fit on Cartesian
    Δt vs ΔP plot to extract m* (psi/hr) and drainage area.

    Liquid form. For gas, use ``regress_pss_gas``.
    """
    mask = (dt >= t_min) & (dt <= t_max)
    if mask.sum() < 3:
        return {"ok": False, "message": "Not enough points in the window."}

    t_win = dt[mask]
    p_win = dp[mask]
    A = np.vstack([t_win, np.ones_like(t_win)]).T
    slope, intercept = np.linalg.lstsq(A, p_win, rcond=None)[0]

    if slope <= 0:
        return {"ok": False,
                "message": "Non-physical (negative) PSS slope. "
                           "Window may not be in PSS yet."}

    # Drainage area in ft² (q in STB/D, B in rb/STB, ct in 1/psi, h in ft)
    area_ft2 = 0.234 * abs(q) * B / (phi * ct * h * slope)
    # Pore volume in bbl
    pore_vol_ft3 = area_ft2 * h * phi
    pore_vol_bbl = pore_vol_ft3 / 5.615
    re_ft = np.sqrt(area_ft2 / np.pi)   # equivalent drainage radius

    return {
        "ok": True,
        "regime": "Closed Boundary (PSS)",
        "slope": float(slope),               # psi/hr
        "intercept": float(intercept),
        "area_ft2": float(area_ft2),
        "area_acre": float(area_ft2 / 43560.0),
        "re_ft": float(re_ft),
        "pore_volume_bbl": float(pore_vol_bbl),
        "n_points": int(mask.sum()),
        "t_min": t_min, "t_max": t_max,
    }


def regress_pss_gas(dt: np.ndarray, dm: np.ndarray,
                    qg: float, T_R: float, h: float, phi: float, ct_ref: float,
                    t_min: float, t_max: float) -> dict:
    """
    Pseudo-steady-state regression for a gas well — straight-line fit on
    Cartesian Δt vs Δm(p) plot.

        m_g* = dm/dt  [psi²/cp/hr]
        A    = 2.356·10⁶ · qg · T_R / (φ ct_ref h · m_g*)         [ft²]
    """
    mask = (dt >= t_min) & (dt <= t_max)
    if mask.sum() < 3:
        return {"ok": False, "message": "Not enough points in the window."}

    t_win = dt[mask]
    y_win = dm[mask]
    A = np.vstack([t_win, np.ones_like(t_win)]).T
    slope, intercept = np.linalg.lstsq(A, y_win, rcond=None)[0]

    if slope <= 0:
        return {"ok": False,
                "message": "Non-physical (negative) PSS slope. "
                           "Window may not be in PSS yet."}

    # Gas drainage area  — Lee-Wattenbarger Eq. 6.135 analogue
    # A [ft²] = 2.356e6 · qg [Mscf/D] · T_R [°R] / (φ ct h · m_g* [psi²/cp/hr])
    area_ft2 = 2.356e6 * abs(qg) * T_R / (phi * ct_ref * h * slope)
    pore_vol_ft3 = area_ft2 * h * phi
    re_ft = np.sqrt(area_ft2 / np.pi)
    # Gas-in-place at reservoir conditions (rcf) — not surface; informational
    gip_rcf = pore_vol_ft3

    return {
        "ok": True,
        "regime": "Closed Boundary (PSS, gas)",
        "slope": float(slope),               # psi²/cp/hr
        "intercept": float(intercept),
        "area_ft2": float(area_ft2),
        "area_acre": float(area_ft2 / 43560.0),
        "re_ft": float(re_ft),
        "gip_rcf": float(gip_rcf),
        "n_points": int(mask.sum()),
        "t_min": t_min, "t_max": t_max,
    }
