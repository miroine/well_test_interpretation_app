"""
modules/pvt.py
--------------
PVT correlations and gas pseudo-function transforms for gas-well PTA.

All math is in OILFIELD UNITS internally, matching the rest of the engine:
    p [psia], T [°R],
    μ [cp], Bg [rcf/scf or rb/Mscf when noted],
    k [mD], h [ft], t [hr], q_g [Mscf/D].

Correlations
------------
Z-factor   : Dranchuk-Abou-Kassem (DAK, 1975) — explicit iteration on ρr.
Viscosity  : Lee-Gonzalez-Eakin (LGE, 1966).
Pseudo-criticals : Sutton (1985) for natural gas, with Wichert-Aziz (1972)
                   correction for acid gases (CO₂ + H₂S).

Pseudo-variables
----------------
Pseudo-pressure  m(p) = 2 ∫₀ᵖ p' / (μ(p') Z(p')) dp'             [psi²/cp]
Pseudo-time      tₐ   = ∫₀ᵗ dt' / (μ(p̄) cₜ(p̄))                  [psi·hr/cp]

For gas-well buildup, ΔP is replaced by Δm(p) and Δt by Δtₐ. The same
Bourdet derivative and IARF / linear-flow regressions then apply with no
algorithmic change — only the variables and slope constants differ.

References
----------
- Dranchuk, P.M., & Abou-Kassem, J.H. (1975). JCPT.
- Lee, A.L., Gonzalez, M.H., & Eakin, B.E. (1966). JPT.
- Sutton, R.P. (1985). SPE 14265.
- Wichert, E., & Aziz, K. (1972). Hydrocarbon Processing.
- Lee, J., Rollins, J.B., & Spivey, J.P. (2003). Pressure Transient Testing,
  SPE Textbook Series, Chapters 3 & 8.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


# =====================================================================
#  Inputs container — gas-specific PVT
# =====================================================================
@dataclass
class GasPVT:
    gamma_g: float = 0.65        # specific gravity (air = 1)
    T_res_F: float = 200.0       # reservoir temperature [°F]
    y_CO2: float = 0.0           # mole fraction CO2
    y_H2S: float = 0.0           # mole fraction H2S
    y_N2:  float = 0.0           # mole fraction N2

    @property
    def T_R(self) -> float:
        """Reservoir temperature in Rankine."""
        return self.T_res_F + 459.67


# =====================================================================
#  Pseudo-critical properties — Sutton + Wichert-Aziz
# =====================================================================
def sutton_pseudo_criticals(gamma_g: float) -> tuple[float, float]:
    """
    Sutton (1985) pseudo-critical pressure and temperature for natural gas,
    correlated against gas specific gravity (air = 1).
        Tpc = 169.2 + 349.5*γg - 74.0*γg²    [°R]
        Ppc = 756.8 - 131.0*γg - 3.6*γg²     [psia]
    Validity: 0.55 ≤ γg ≤ 1.7.
    """
    Tpc = 169.2 + 349.5 * gamma_g - 74.0 * gamma_g ** 2
    Ppc = 756.8 - 131.0 * gamma_g - 3.6 * gamma_g ** 2
    return Ppc, Tpc


def wichert_aziz_correction(Ppc: float, Tpc: float,
                            y_CO2: float, y_H2S: float) -> tuple[float, float]:
    """
    Wichert-Aziz (1972) correction for the influence of CO₂ and H₂S
    on the pseudo-critical properties. Returns corrected (Ppc', Tpc').

        A = y_H2S + y_CO2
        B = y_H2S
        ε = 120·(A^0.9 − A^1.6) + 15·(B^0.5 − B^4)
        Tpc' = Tpc − ε
        Ppc' = Ppc · Tpc' / (Tpc + B·(1 − B)·ε)
    """
    A = y_H2S + y_CO2
    B = y_H2S
    if A <= 0.0:
        return Ppc, Tpc
    eps = 120.0 * (A ** 0.9 - A ** 1.6) + 15.0 * (B ** 0.5 - B ** 4)
    Tpc_corr = Tpc - eps
    Ppc_corr = Ppc * Tpc_corr / (Tpc + B * (1.0 - B) * eps)
    return Ppc_corr, Tpc_corr


def reduced_props(p: np.ndarray, T_R: float,
                  pvt: GasPVT) -> tuple[np.ndarray, float]:
    """Return (Ppr, Tpr) — pseudo-reduced pressure (array) and temperature."""
    Ppc, Tpc = sutton_pseudo_criticals(pvt.gamma_g)
    Ppc, Tpc = wichert_aziz_correction(Ppc, Tpc, pvt.y_CO2, pvt.y_H2S)
    p = np.asarray(p, dtype=float)
    Ppr = p / Ppc
    Tpr = T_R / Tpc
    return Ppr, Tpr


# =====================================================================
#  Z-factor — Dranchuk-Abou-Kassem (1975)
# =====================================================================
# DAK constants
_DAK_A = np.array([0.3265, -1.0700, -0.5339, 0.01569, -0.05165,
                   0.5475, -0.7361,  0.1844, 0.1056,   0.6134, 0.7210])


def _dak_rho_residual(rho_r: float, Ppr: float, Tpr: float) -> float:
    """Residual of DAK equation in reduced density rho_r."""
    A = _DAK_A
    c1 = A[0] + A[1]/Tpr + A[2]/Tpr**3 + A[3]/Tpr**4 + A[4]/Tpr**5
    c2 = A[5] + A[6]/Tpr + A[7]/Tpr**2
    c3 = A[8] * (A[6]/Tpr + A[7]/Tpr**2)
    c4 = A[9] * (1.0 + A[10]*rho_r**2) * (rho_r**2 / Tpr**3) * np.exp(-A[10]*rho_r**2)
    # Z = 1 + c1*ρr + c2*ρr² − c3*ρr⁵ + c4
    Z = 1.0 + c1*rho_r + c2*rho_r**2 - c3*rho_r**5 + c4
    # Definition: ρr = 0.27·Ppr / (Z·Tpr)
    return Z * Tpr * rho_r - 0.27 * Ppr


def z_factor_dak(p: np.ndarray, T_R: float, pvt: GasPVT,
                 tol: float = 1e-8, max_iter: int = 50) -> np.ndarray:
    """
    Z-factor via DAK using a damped Newton/secant iteration on reduced density.
    Vectorized over p. Falls back to bisection if Newton struggles.

    Valid range: 0.2 ≤ Ppr ≤ 30, 1.0 ≤ Tpr ≤ 3.0 (per DAK paper).
    """
    Ppr, Tpr = reduced_props(p, T_R, pvt)
    Ppr = np.atleast_1d(Ppr)
    Z = np.empty_like(Ppr, dtype=float)

    for i, ppr in enumerate(Ppr):
        if ppr <= 0:
            Z[i] = 1.0
            continue
        # Bisection on rho_r ∈ (0, 2.5) — guaranteed bracket for the range we care about
        lo, hi = 1e-8, 2.5
        f_lo = _dak_rho_residual(lo, ppr, Tpr)
        f_hi = _dak_rho_residual(hi, ppr, Tpr)
        # Expand if no sign change (rare, very high pressure)
        expand = 0
        while f_lo * f_hi > 0 and expand < 5:
            hi *= 1.5
            f_hi = _dak_rho_residual(hi, ppr, Tpr)
            expand += 1
        if f_lo * f_hi > 0:
            # Final fallback — ideal gas
            Z[i] = 1.0
            continue
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            f_mid = _dak_rho_residual(mid, ppr, Tpr)
            if abs(f_mid) < tol or (hi - lo) < tol:
                break
            if f_lo * f_mid < 0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
        rho_r = 0.5 * (lo + hi)
        Z[i] = 0.27 * ppr / (rho_r * Tpr) if rho_r > 0 else 1.0

    return Z


# =====================================================================
#  Gas viscosity — Lee-Gonzalez-Eakin (1966)
# =====================================================================
def viscosity_lge(p: np.ndarray, T_R: float, pvt: GasPVT,
                  Z: np.ndarray | None = None) -> np.ndarray:
    """
    Lee-Gonzalez-Eakin gas viscosity correlation [cp].
        μg = 1e-4 · K · exp(X · ρg^Y)
    with
        K = (9.4 + 0.02·Mg)·T^1.5 / (209 + 19·Mg + T)
        X = 3.5 + 986/T + 0.01·Mg
        Y = 2.4 − 0.2·X
        ρg [g/cc] = p·Mg / (Z·R·T),    R = 10.7316 psi·ft³/(lbmol·°R) → need consistent units
    A practical compact form (T in °R, p in psia, Mg = 28.96·γg):
        ρg [lbm/ft³] = p·Mg / (Z·R·T),    then ρg [g/cc] = ρg_field / 62.428
    """
    Mg = 28.96 * pvt.gamma_g
    T = T_R
    p = np.atleast_1d(np.asarray(p, dtype=float))
    if Z is None:
        Z = z_factor_dak(p, T_R, pvt)

    R_field = 10.7316  # psi·ft³/(lbmol·°R)
    rho_lbmft3 = p * Mg / (Z * R_field * T)        # [lbm/ft³]
    rho_gcc = rho_lbmft3 / 62.428                  # → [g/cc]

    K = (9.4 + 0.02 * Mg) * T ** 1.5 / (209.0 + 19.0 * Mg + T)
    X = 3.5 + 986.0 / T + 0.01 * Mg
    Y = 2.4 - 0.2 * X
    mu = 1e-4 * K * np.exp(X * (rho_gcc ** Y))    # [cp]
    return mu


# =====================================================================
#  Gas FVF and isothermal compressibility
# =====================================================================
def gas_fvf(p: np.ndarray, T_R: float, Z: np.ndarray) -> np.ndarray:
    """
    Bg in rb/Mscf  (reservoir barrels per Mscf of surface gas).
        Bg = 5.0287 · Z·T / p           [rb/Mscf]
    (Equivalently 0.02827·Z·T/p in rcf/scf; ×1000/5.615 → rb/Mscf.)
    """
    p = np.asarray(p, dtype=float)
    return 5.0287 * Z * T_R / np.clip(p, 1e-6, None)


def gas_compressibility(p: np.ndarray, T_R: float, pvt: GasPVT,
                        Z: np.ndarray | None = None) -> np.ndarray:
    """
    Isothermal compressibility of a real gas:
        cg = 1/p  −  (1/Z) · (∂Z/∂p)_T          [1/psi]
    The Z-derivative is computed numerically with a small relative step.
    """
    p = np.atleast_1d(np.asarray(p, dtype=float))
    if Z is None:
        Z = z_factor_dak(p, T_R, pvt)
    dp = np.maximum(p * 1e-4, 1.0)                # bigger of 0.01% or 1 psi
    Z_plus = z_factor_dak(p + dp, T_R, pvt)
    Z_minus = z_factor_dak(np.clip(p - dp, 1e-6, None), T_R, pvt)
    dZ_dp = (Z_plus - Z_minus) / (2.0 * dp)
    return 1.0 / np.clip(p, 1e-6, None) - dZ_dp / np.clip(Z, 1e-6, None)


# =====================================================================
#  PVT table builder — one call returns everything on a pressure grid
# =====================================================================
def build_pvt_table(pvt: GasPVT, p_max: float = 10_000.0,
                    n: int = 200) -> dict[str, np.ndarray]:
    """
    Build a PVT lookup table over [1, p_max] psia for fast pseudo-function
    integration. Returns dict with keys: p, Z, mu, Bg, cg.
    """
    p = np.linspace(1.0, p_max, n)
    Z = z_factor_dak(p, pvt.T_R, pvt)
    mu = viscosity_lge(p, pvt.T_R, pvt, Z=Z)
    Bg = gas_fvf(p, pvt.T_R, Z)
    cg = gas_compressibility(p, pvt.T_R, pvt, Z=Z)
    return {"p": p, "Z": Z, "mu": mu, "Bg": Bg, "cg": cg}


# =====================================================================
#  Pseudo-pressure m(p) and Pseudo-time tₐ
# =====================================================================
def pseudo_pressure_table(pvt_table: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Tabulate m(p) = 2 ∫₀ᵖ p' / (μ(p') Z(p')) dp'  using cumulative trapezoid.
    Returns (p_array, m_array) with m(p) in psi²/cp.
    """
    p = pvt_table["p"]
    integrand = 2.0 * p / (pvt_table["mu"] * pvt_table["Z"])
    # Cumulative trapezoidal integration starting at 0 (extrapolated)
    m = np.zeros_like(p)
    m[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(p))
    return p, m


def pseudo_pressure(p_query: np.ndarray, p_table: np.ndarray,
                    m_table: np.ndarray) -> np.ndarray:
    """Interpolate m(p) at arbitrary pressures."""
    return np.interp(np.asarray(p_query, dtype=float), p_table, m_table)


def pseudo_time(t_hr: np.ndarray, p_bar: float, pvt_table: dict) -> np.ndarray:
    """
    Pseudo-time evaluated at a constant average pressure p̄ (the standard
    Agarwal approximation for buildup analysis):
        tₐ(t) = t / (μ(p̄) · cₜ(p̄))
    For multi-rate or strongly transient problems, tₐ should be evaluated
    point-wise — that's a future extension; the constant-p̄ form is the
    industry-standard buildup approximation.
    """
    mu_bar = float(np.interp(p_bar, pvt_table["p"], pvt_table["mu"]))
    cg_bar = float(np.interp(p_bar, pvt_table["p"], pvt_table["cg"]))
    # Note: for a "dry-gas" reservoir cₜ ≈ cg + cw·Sw + cf is dominated by cg
    return np.asarray(t_hr, dtype=float) / (mu_bar * cg_bar)
