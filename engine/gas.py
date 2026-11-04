"""Dry-gas PVT and normalized pseudo-pressure (field units). Math only."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

_A = dict(A1=0.3265, A2=-1.0700, A3=-0.5339, A4=0.01569, A5=-0.05165,
          A6=0.5475, A7=-0.7361, A8=0.1844, A9=0.1056, A10=0.6134, A11=0.7210)


def pseudocritical(gamma_g: float):
    """Sutton (1985) pseudo-criticals for a dry gas: (Tpc [R], Ppc [psia])."""
    return (169.2 + 349.5 * gamma_g - 74.0 * gamma_g ** 2,
            756.8 - 131.0 * gamma_g - 3.6 * gamma_g ** 2)


def z_factor(p_psia: float, T_F: float, gamma_g: float) -> float:
    """Dranchuk-Abou-Kassem Z factor."""
    Tpc, Ppc = pseudocritical(gamma_g)
    Tr = (T_F + 459.67) / Tpc
    Pr = max(p_psia, 1e-6) / Ppc
    a = _A

    def f(Z):
        rho = 0.27 * Pr / (Z * Tr)
        c1 = a["A1"] + a["A2"] / Tr + a["A3"] / Tr ** 3 + a["A4"] / Tr ** 4 + a["A5"] / Tr ** 5
        c2 = a["A6"] + a["A7"] / Tr + a["A8"] / Tr ** 2
        c3 = a["A9"] * (a["A7"] / Tr + a["A8"] / Tr ** 2)
        c4 = a["A10"] * (1 + a["A11"] * rho ** 2) * rho ** 2 / Tr ** 3 * np.exp(-a["A11"] * rho ** 2)
        return 1 + c1 * rho + c2 * rho ** 2 - c3 * rho ** 5 + c4 - Z

    return float(brentq(f, 0.05, 3.0))


def viscosity(p_psia: float, T_F: float, gamma_g: float, Z: float | None = None) -> float:
    """Lee-Gonzalez-Eakin gas viscosity [cp]."""
    T = T_F + 459.67
    M = 28.9647 * gamma_g
    Z = z_factor(p_psia, T_F, gamma_g) if Z is None else Z
    rho = 1.4935e-3 * p_psia * M / (Z * T)          # g/cc
    K = (9.4 + 0.02 * M) * T ** 1.5 / (209.0 + 19.0 * M + T)
    X = 3.5 + 986.0 / T + 0.01 * M
    Y = 2.4 - 0.2 * X
    return 1e-4 * K * np.exp(X * rho ** Y)


def gas_props(p_psia, T_F, gamma_g):
    """Return dict Z, mu [cp], Bg [rb/scf], cg [1/psi]."""
    Z = z_factor(p_psia, T_F, gamma_g)
    mu = viscosity(p_psia, T_F, gamma_g, Z)
    T = T_F + 459.67
    Bg = 0.00504 * Z * T / p_psia
    dp = max(1.0, 1e-3 * p_psia)
    dZdp = (z_factor(p_psia + dp, T_F, gamma_g) - z_factor(p_psia - dp, T_F, gamma_g)) / (2 * dp)
    cg = 1.0 / p_psia - dZdp / Z
    return {"Z": Z, "mu": mu, "Bg": Bg, "cg": cg}


class NormalizedPseudoPressure:
    """p_n(p) = p_i + (mu_i Z_i / p_i) * int_{p_i}^{p} p'/(mu Z) dp' [psi]."""

    def __init__(self, p_i, T_F, gamma_g, p_min=14.7, n=400):
        self.p_i = p_i
        grid = np.linspace(p_min, p_i * 1.05, n)
        integrand = np.array([
            p / (viscosity(p, T_F, gamma_g) * z_factor(p, T_F, gamma_g)) for p in grid])
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(grid))])
        ref = np.interp(p_i, grid, cum)
        Zi = z_factor(p_i, T_F, gamma_g)
        mui = viscosity(p_i, T_F, gamma_g, Zi)
        self.grid = grid
        self.pn = p_i + (mui * Zi / p_i) * (cum - ref)

    def forward(self, p):
        return np.interp(p, self.grid, self.pn)

    def inverse(self, pn):
        return np.interp(pn, self.pn, self.grid)
