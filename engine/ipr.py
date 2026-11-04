"""Inflow performance relationships (field units). Math only."""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------- oil
def pi_pss(k, h, mu, B, re, rw, S):
    """Oil productivity index J [stb/d/psi], pseudo-steady state."""
    return k * h / (141.2 * B * mu * (np.log(re / rw) - 0.75 + S))


def vogel_rate(pr, qmax, pwf):
    x = np.asarray(pwf, float) / pr
    return qmax * (1.0 - 0.2 * x - 0.8 * x ** 2)


def vogel_qmax(q_test, pwf_test, pr):
    x = pwf_test / pr
    return q_test / (1.0 - 0.2 * x - 0.8 * x ** 2)


def composite_ipr(pr, pb, J, pwf):
    """Linear above the bubble point, Vogel below (Standing/Kilns composite)."""
    pwf = np.asarray(pwf, float)
    q = np.empty_like(pwf)
    qb = J * max(pr - pb, 0.0)
    qv = J * min(pb, pr) / 1.8
    above = pwf >= min(pb, pr)
    q[above] = J * (pr - pwf[above])
    x = pwf[~above] / min(pb, pr)
    q[~above] = qb + qv * (1.0 - 0.2 * x - 0.8 * x ** 2)
    return q


def composite_J_from_test(pr, pb, q_test, pwf_test):
    """Productivity index from one stabilised test point."""
    if pwf_test >= pb:
        return q_test / (pr - pwf_test)
    x = pwf_test / pb
    return q_test / ((pr - pb) + pb / 1.8 * (1.0 - 0.2 * x - 0.8 * x ** 2))


def composite_qmax(pr, pb, J):
    return float(composite_ipr(pr, pb, J, np.array([0.0]))[0])


# ------------------------------------------------------------------- gas
def fetkovich_fit(pr, pwf, q):
    """q = C (pr^2 - pwf^2)^n by log-log regression. Returns C, n, r2."""
    pwf, q = np.asarray(pwf, float), np.asarray(q, float)
    dp2 = pr ** 2 - pwf ** 2
    m = (dp2 > 0) & (q > 0)
    x, y = np.log(dp2[m]), np.log(q[m])
    n, lc = np.polyfit(x, y, 1)
    pred = n * x + lc
    r2 = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-30)
    return float(np.exp(lc)), float(n), float(r2)


def fetkovich_rate(pr, C, n, pwf):
    return C * np.clip(pr ** 2 - np.asarray(pwf, float) ** 2, 0.0, None) ** n


def lit_fit(pr, pwf, q):
    """pr^2 - pwf^2 = a q + b q^2 (Forchheimer / LIT) by regression of (dp2/q) on q."""
    pwf, q = np.asarray(pwf, float), np.asarray(q, float)
    m = q > 0
    y = (pr ** 2 - pwf[m] ** 2) / q[m]
    b, a = np.polyfit(q[m], y, 1)
    pred = a + b * q[m]
    r2 = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-30)
    return float(a), float(b), float(r2)


def lit_rate(pr, a, b, pwf):
    dp2 = np.clip(pr ** 2 - np.asarray(pwf, float) ** 2, 0.0, None)
    if abs(b) < 1e-30:
        return dp2 / a
    return (-a + np.sqrt(a * a + 4.0 * b * dp2)) / (2.0 * b)


def aof_lit(pr, a, b):
    return float(lit_rate(pr, a, b, np.array([0.0]))[0])


def aof_fetkovich(pr, C, n):
    return float(C * pr ** (2 * n))
