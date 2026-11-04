"""Bounded parameter sampling with a Gaussian copula; no regression."""
from __future__ import annotations

import numpy as np
from scipy import stats

from . import models as M

DISTRIBUTIONS = ("Uniform", "Log-uniform", "Triangular", "Truncated normal")


def sample_parameters(base, specs, correlation, n=200, seed=0):
    """Sample paired parameter rows using normal-score correlation."""
    if not specs or not isinstance(n, int) or not 3 <= n <= 10000:
        raise ValueError("Select parameters and between 3 and 10000 samples.")
    names = list(specs)
    matrix = np.asarray(correlation, float)
    if matrix.shape != (len(names), len(names)) or not np.isfinite(matrix).all():
        raise ValueError("Correlation matrix must be finite and match the selected parameter order.")
    if (np.any(np.abs(matrix) > 1.) or not np.allclose(matrix, matrix.T, atol=1e-10, rtol=0)
            or not np.allclose(np.diag(matrix), 1., atol=1e-10, rtol=0)):
        raise ValueError("Correlation matrix must be symmetric, between -1 and 1, with diagonal 1.")
    eigenvalues, vectors = np.linalg.eigh(matrix)
    if eigenvalues.min() < -1e-10:
        raise ValueError("Correlation matrix must be positive semidefinite; revise inconsistent correlations.")
    transforms = {}
    for name, spec in specs.items():
        if name not in base.to_dict() or name == "n_frac" or isinstance(getattr(base, name), str):
            raise ValueError(f"{name}: select a continuous numeric model parameter.")
        low, high = float(spec["low"]), float(spec["high"])
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError(f"{name}: low must be finite and strictly less than high.")
        if name != "S" and low <= 0:
            raise ValueError(f"{name}: bounds must be positive.")
        if name in ("phi", "omega", "kv_kh", "perf_frac", "perf_pos", "zw_h") and high > 1:
            raise ValueError(f"{name}: upper bound must not exceed 1.")
        if name == "omega" and high >= 1:
            raise ValueError("omega: upper bound must be below 1.")
        if name == "S" and base.model in M.INFINITE_FRACTURE_MODELS and low < 0:
            raise ValueError("Explicit fracture skin is additional skin; its low bound must be non-negative.")
        distribution = spec["distribution"]
        if distribution == "Uniform":
            transforms[name] = stats.uniform(loc=low, scale=high - low)
        elif distribution == "Log-uniform":
            if low <= 0:
                raise ValueError(f"{name}: log-uniform bounds must be positive.")
            transforms[name] = stats.loguniform(low, high)
        elif distribution == "Triangular":
            mode = float(spec["mode"])
            if not np.isfinite(mode) or not low <= mode <= high:
                raise ValueError(f"{name}: most likely value must lie between low and high.")
            transforms[name] = stats.triang((mode - low) / (high - low), loc=low, scale=high - low)
        elif distribution == "Truncated normal":
            mean, sd = float(spec["mean"]), float(spec["sd"])
            if not np.isfinite([mean, sd]).all() or not low <= mean <= high or sd <= 0:
                raise ValueError(f"{name}: mean must lie within bounds and standard deviation must be positive.")
            transforms[name] = stats.truncnorm((low - mean) / sd, (high - mean) / sd, loc=mean, scale=sd)
        else:
            raise ValueError(f"{name}: unknown distribution {distribution}.")
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal((n, len(names))) @ (vectors * np.sqrt(np.maximum(eigenvalues, 0))).T
    uniform = np.clip(stats.norm.cdf(scores), np.finfo(float).eps, 1 - np.finfo(float).eps)
    samples = {name: transforms[name].ppf(uniform[:, i]) for i, name in enumerate(names)}
    for i in range(n):
        member = M.WellTestParams(**{**base.to_dict(), **{k: float(v[i]) for k, v in samples.items()}})
        if member.boundary == "rectangle" and not (0 < member.xw < member.xe and 0 < member.yw < member.ye):
            raise ValueError(f"Sample {i + 1}: well position must lie inside the rectangle. Revise ranges/correlations.")
        if member.model == "composite3" and member.r_mid <= member.r_inner:
            raise ValueError(f"Sample {i + 1}: middle radius must exceed inner radius. Revise ranges/correlations.")
        M.validate_skin(member)
    if any(not np.isfinite(v).all() for v in samples.values()):
        raise ValueError("Distribution parameters produced non-finite samples; revise them.")
    summary = {k: dict(zip(("p10", "p50", "p90"), map(float, np.percentile(v, [10, 50, 90]))),
                       n=n) for k, v in samples.items()}
    return {"mode": "parameter_sampling", "samples": {k: v.tolist() for k, v in samples.items()},
            "summary": summary, "requested": n, "specs": specs, "correlation": matrix.tolist(), "seed": seed}
