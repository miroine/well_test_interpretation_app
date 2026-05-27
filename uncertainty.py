"""
modules/uncertainty.py
----------------------
Monte Carlo uncertainty propagation for permeability.

Inverts the radial-flow / Horner-slope relation:
    k = 162.6 * q * B * μ / (m * h)

where m (the semilog slope) is provided by the interpretation, and the
uncertain inputs are triangular distributions for net pay (h) and
viscosity (μ).
"""
from __future__ import annotations
import numpy as np
from scipy.stats import gaussian_kde


def triangular_sample(low: float, mode: float, high: float, n: int,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw n samples from a triangular distribution (low ≤ mode ≤ high)."""
    rng = rng or np.random.default_rng()
    lo, mo, hi = sorted([low, mode, high])
    if hi - lo < 1e-12:
        return np.full(n, mo, dtype=float)
    return rng.triangular(lo, mo, hi, size=n)


def run_monte_carlo(
    n_iter: int,
    q: float,
    B: float,
    m_slope: float,
    h_tri: tuple[float, float, float],
    mu_tri: tuple[float, float, float],
    seed: int = 42,
    fluid: str = "oil",
    T_R: float | None = None,
) -> dict:
    """
    Monte-Carlo permeability propagation.

    Liquid (oil/water, fluid='oil'):
        k_i = 162.6 · q · B · μ_i / (m_slope · h_i)
        m_slope in psi/cycle, q in STB/D, μ in cp, h in ft.

    Gas (fluid='gas'):
        k_i = 1637 · q · T_R / (m_slope · h_i)
        m_slope in psi²/cp per cycle, q in Mscf/D, T_R in °R.
        μ-triangular is sampled and reported but does NOT enter the
        permeability formula directly (it's absorbed in m(p) via pseudo-time).
        For gas, the μ distribution is therefore informational only; the
        dominant uncertain inputs are h, m_slope and T_R.

    Returns the sample vector and the P10/P50/P90 statistics.
    By petroleum convention:
        P90 = conservative (low), P50 = best, P10 = optimistic (high).
    """
    rng = np.random.default_rng(seed)
    h_samples = triangular_sample(*h_tri, n=n_iter, rng=rng)
    mu_samples = triangular_sample(*mu_tri, n=n_iter, rng=rng)

    h_samples = np.clip(h_samples, 1e-3, None)
    m_slope = max(abs(m_slope), 1e-6)

    if fluid.lower() == "gas":
        if T_R is None:
            raise ValueError("Gas Monte-Carlo requires T_R (°R).")
        k = 1637.0 * abs(q) * T_R / (m_slope * h_samples)
    else:
        k = 162.6 * abs(q) * B * mu_samples / (m_slope * h_samples)

    # Petroleum convention: P90 = exceedance-90 (low), P10 = exceedance-10 (high)
    p90 = float(np.percentile(k, 10))   # 10th percentile → 90% chance of exceeding
    p50 = float(np.percentile(k, 50))
    p10 = float(np.percentile(k, 90))   # 90th percentile → 10% chance of exceeding

    return {
        "samples": k,
        "h_samples": h_samples,
        "mu_samples": mu_samples,
        "P90": p90,
        "P50": p50,
        "P10": p10,
        "mean": float(np.mean(k)),
        "std": float(np.std(k)),
        "n_iter": n_iter,
        "fluid": fluid,
    }


def kde_curve(samples: np.ndarray, n_grid: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian KDE PDF curve over the support of the samples."""
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    lo, hi = float(np.min(samples)), float(np.max(samples))
    pad = 0.05 * (hi - lo + 1e-9)
    x = np.linspace(lo - pad, hi + pad, n_grid)
    kde = gaussian_kde(samples)
    return x, kde(x)


def cdf_curve(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical CDF — sorted samples vs cumulative probability."""
    s = np.sort(np.asarray(samples, dtype=float))
    p = np.linspace(0.0, 1.0, len(s))
    return s, p


# =====================================================================
#  Sensitivity analysis
# =====================================================================
def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Rank (Spearman) correlation, robust to non-linear monotonic relations
    and to outliers. Returns 0 if either array is constant."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    # ranks
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def tornado_swings(
    mc_result: dict,
    inputs: dict[str, np.ndarray],
) -> list[dict]:
    """
    Build a tornado-chart payload for the Monte Carlo result.

    For each uncertain input we report:
      - low_k, high_k: the median output k restricted to the bottom-decile
                       and top-decile of THAT input (other inputs free to vary).
                       This isolates the contribution of one input at a time
                       without re-running the simulation.
      - rank_corr   : Spearman correlation between the input and the output k.

    The swing magnitude `|high_k − low_k|` is a robust proxy for the input's
    contribution to overall uncertainty. Sorting bars by swing gives the
    classic petroleum tornado.

    Parameters
    ----------
    mc_result : dict from run_monte_carlo (must contain `samples`)
    inputs    : mapping of input name → sample vector aligned with `samples`
                e.g. {"h": h_samples, "μ": mu_samples}
    """
    k = np.asarray(mc_result["samples"], dtype=float)
    if k.ndim != 1 or len(k) < 50:
        return []

    p50 = float(np.percentile(k, 50))
    out = []
    for name, x in inputs.items():
        x = np.asarray(x, dtype=float)
        if len(x) != len(k):
            continue
        low_thr  = np.percentile(x, 10)
        high_thr = np.percentile(x, 90)
        low_mask  = x <= low_thr
        high_mask = x >= high_thr
        low_k  = float(np.median(k[low_mask]))  if low_mask.sum()  > 0 else p50
        high_k = float(np.median(k[high_mask])) if high_mask.sum() > 0 else p50
        out.append({
            "name": name,
            "low_k": low_k,
            "high_k": high_k,
            "swing": abs(high_k - low_k),
            "rank_corr": spearman_correlation(x, k),
            "p50": p50,
        })
    # Sort descending by swing magnitude (largest contributor at top)
    out.sort(key=lambda d: d["swing"], reverse=True)
    return out
