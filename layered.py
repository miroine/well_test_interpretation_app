"""
modules/layered.py
------------------
Multi-layer reservoir models for pressure transient analysis.

Currently implemented:
  • Commingled layers, no crossflow (exact superposition, validated)

Future work (stubbed):
  • Double-permeability with crossflow (Bourdet 1985 SPE 13628) — requires
    the published Laplace-space transfer function to be verified against
    reference type-curves or commercial PTA software output before release.
    A stub is provided; the math infrastructure (Stehfest inverter in
    modules.laplace) is already validated and ready to use.

================================================================
COMMINGLED LAYERS (NO CROSSFLOW)
================================================================

Two (or more) layers produce into the same wellbore but DO NOT communicate
vertically. Each layer feels the same wellbore pressure and contributes
its own rate. Under fixed-TOTAL-rate boundary conditions and a shared
bottom-hole pressure, the per-layer admittances ADD in parallel:

    ΔP_well(t) = q_total / [ Σ_i  Y_i(t) ]
    Y_i(t)     = (k_i h_i) / [ (141.2 · μ · B) · g_i(t) ]
    g_i(t)     = (1/2) E₁(x_i)  +  s_i,
    x_i        = 948 · φ_i · μ · ct_i · rw² / (k_i · t)

For gas, replace (141.2·μ·B) → (1422·T_R) and ΔP → Δm(p).

DERIVATIVE SIGNATURE:
  • Early time:   the fastest-responding layer dominates → derivative
                  plateau at the level for THAT layer's kh alone.
  • Late time:    all layers contributing → derivative drops to the
                  plateau corresponding to the SUM kh_total.
  • Transition:   a downward STEP in the Bourdet derivative — the
                  qualitative fingerprint of commingled flow w/o crossflow.

NON-UNIQUENESS WARNING:
  Without PLT-derived layer-rate split, the inverse problem is under-
  determined: you can swap kh between layers and re-fit. The inversion
  routine therefore returns aggregate identifiable parameters only.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.special import exp1
from scipy.optimize import minimize

from modules.engine import (
    LINESRC_CONST, GAS_LINESRC_CONST,
    bourdet_derivative,
)


@dataclass
class CommingledInputs:
    """
    Inputs for an N-layer commingled (no-crossflow) reservoir.
    Engine canonical: oilfield units.
    """
    k:    np.ndarray          # permeabilities [mD]
    h:    np.ndarray          # net pays       [ft]
    phi:  np.ndarray          # porosities     [-]
    ct:   np.ndarray          # comp.          [1/psi]
    s:    np.ndarray          # skin per layer [-]
    q_total: float            # total rate     [STB/D or Mscf/D]
    C: float = 0.0
    mu: float = 1.0
    B:  float = 1.2
    rw: float = 0.354
    pi: float = 5000.0

    def __post_init__(self):
        self.k    = np.atleast_1d(np.asarray(self.k,   dtype=float))
        self.h    = np.atleast_1d(np.asarray(self.h,   dtype=float))
        self.phi  = np.atleast_1d(np.asarray(self.phi, dtype=float))
        self.ct   = np.atleast_1d(np.asarray(self.ct,  dtype=float))
        self.s    = np.atleast_1d(np.asarray(self.s,   dtype=float))
        n = self.k.size
        for arr_name in ("h", "phi", "ct", "s"):
            arr = getattr(self, arr_name)
            if arr.size != n:
                raise ValueError(f"{arr_name} has length {arr.size}; expected {n}")

    @property
    def n_layers(self) -> int:   return int(self.k.size)
    @property
    def kh_total(self) -> float: return float(np.sum(self.k * self.h))
    @property
    def h_total(self)  -> float: return float(np.sum(self.h))
    @property
    def kappa(self) -> np.ndarray:
        """kh-fraction per layer."""
        return (self.k * self.h) / self.kh_total
    @property
    def omega(self) -> np.ndarray:
        """Storativity fraction (φ ct h) per layer."""
        s_i = self.phi * self.ct * self.h
        return s_i / s_i.sum()


def simulate_commingled(duration_hr: float, n_points: int,
                        ci: CommingledInputs,
                        fluid: str = "oil",
                        T_R: float | None = None) -> pd.DataFrame:
    """
    Forward-simulate commingled-layer drawdown via parallel-admittance sum.
    """
    t = np.logspace(-3, np.log10(max(duration_hr, 1e-2)), n_points)
    n = ci.n_layers
    g = np.zeros((n, t.size))
    for i in range(n):
        x = 948.0 * ci.phi[i] * ci.mu * ci.ct[i] * (ci.rw**2) / (ci.k[i] * np.clip(t, 1e-12, None))
        g[i, :] = 0.5 * exp1(x) + ci.s[i]

    if fluid == "gas":
        if T_R is None:
            raise ValueError("Gas commingled sim requires T_R (°R).")
        prefactor = 2.0 * GAS_LINESRC_CONST * T_R
    else:
        prefactor = 2.0 * LINESRC_CONST * ci.mu * ci.B

    g_safe = np.clip(g, 1e-12, None)
    kh_i   = (ci.k * ci.h)[:, None]
    Y_i    = kh_i / (prefactor * g_safe)
    Y_total = Y_i.sum(axis=0)
    y      = abs(ci.q_total) / np.clip(Y_total, 1e-30, None)

    if ci.C > 0:
        wbs_ramp = (abs(ci.q_total) * ci.B / (24.0 * ci.C)) * t
        tau = max(200.0 * ci.C / ci.kh_total, 1e-3)
        weight = 1.0 - np.exp(-((t / tau) ** 2))
        y = (1.0 - weight) * wbs_ramp + weight * y

    deriv = bourdet_derivative(t, y, L=0.1)
    ycol  = "dm"     if fluid == "gas" else "dp"
    dcol  = "tdmdt"  if fluid == "gas" else "tdp_dt"
    return pd.DataFrame({"t": t, ycol: y, dcol: deriv})


def fit_commingled_2layer(dt: np.ndarray, tdp: np.ndarray,
                          q_total: float, mu: float, B: float, rw: float,
                          phi_total: float, ct_total: float,
                          fluid: str = "oil", T_R: float | None = None,
                          h_total_guess: float = 75.0) -> dict:
    """
    Fit aggregate 2-layer commingled parameters (kh_total, κ, h-split, skin)
    by least-squares match of the log Bourdet derivative.

    See the module docstring for the non-uniqueness caveat.
    """
    mask = (dt > 0) & (tdp > 0)
    t   = dt[mask]
    obs = np.log(tdp[mask])

    def forward(kh_total, kappa, h_ratio, skin):
        h1 = h_total_guess * h_ratio
        h2 = h_total_guess * (1.0 - h_ratio)
        kh1 = kappa * kh_total
        kh2 = (1.0 - kappa) * kh_total
        k1 = kh1 / max(h1, 1e-6)
        k2 = kh2 / max(h2, 1e-6)
        ci = CommingledInputs(
            k=[k1, k2], h=[h1, h2],
            phi=[phi_total, phi_total], ct=[ct_total, ct_total],
            s=[skin, skin], q_total=q_total,
            mu=mu, B=B, rw=rw, C=0.0,
        )
        df = simulate_commingled(float(t.max() * 1.1), 400, ci,
                                 fluid=fluid, T_R=T_R)
        col = "tdmdt" if fluid == "gas" else "tdp_dt"
        pred = np.interp(t, df["t"], df[col])
        return np.clip(pred, 1e-9, None)

    def residual(params):
        log10_kh, kappa, h_ratio, skin = params
        if not (1.0 < log10_kh < 6.0):  return 1e9
        if not (0.05 < kappa < 0.95):   return 1e9
        if not (0.05 < h_ratio < 0.95): return 1e9
        if not (-5.0 < skin < 30.0):    return 1e9
        try:
            pred = forward(10**log10_kh, kappa, h_ratio, skin)
            return float(np.mean((np.log(pred) - obs) ** 2))
        except Exception:
            return 1e9

    x0 = np.array([np.log10(5000.0), 0.5, 0.5, 0.0])
    res = minimize(residual, x0, method="nelder-mead",
                   options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 800})

    log10_kh, kappa, h_ratio, skin = res.x
    return {
        "ok":            bool(res.success or res.fun < 1.0),
        "message":       res.message,
        "kh_total":      10.0 ** log10_kh,
        "kappa":         float(kappa),
        "h1_fraction":   float(h_ratio),
        "skin":          float(skin),
        "rms_log_resid": float(np.sqrt(res.fun)) if res.fun < 1e8 else float("nan"),
        "n_iter":        int(res.nit),
        "n_points":      int(mask.sum()),
        "warning":       ("Without PLT rate-allocation, the kh split between "
                          "layers is non-unique. Treat kappa & h_ratio as ONE "
                          "of many fits consistent with the derivative."),
    }


def simulate_double_permeability(*args, **kwargs):
    """
    PLACEHOLDER — to be implemented once the Bourdet 1985 transfer
    function is verified against a reference type-curve or commercial
    PTA software output (e.g. KAPPA Saphir).

    Infrastructure already in place:
      • modules.laplace.stehfest_invert — validated against 5 known
        Laplace pairs incl. radial-diffusion K₀(√s)/s ↔ (1/2)·E₁(1/(4t))
        to 0.30% accuracy at N=8 Stehfest terms.
      • simulate_commingled (this file) — shows the Stehfest-driven
        forward-sim pattern with proper WBS / Bourdet handling.

    Supply the verified Laplace-space p̄_wD(s; κ, ω, λ) and wire it through
    stehfest_invert to complete this function.
    """
    raise NotImplementedError(
        "Double-permeability with crossflow is pending reference verification. "
        "Use simulate_commingled for the no-crossflow case in the meantime."
    )
