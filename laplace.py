"""
modules/laplace.py
------------------
Numerical inverse Laplace transform via the Stehfest algorithm.

Stehfest (1970) approximates the inverse transform
        f(t) = L⁻¹{ F̄(s) }
by
        f(t) ≈ (ln 2 / t) · Σ_{i=1..N}  V_i · F̄( i · ln 2 / t )

where the weights V_i depend only on N:

        V_i = (-1)^(N/2+i) · Σ_{k=⌊(i+1)/2⌋..min(i,N/2)}
              [ k^(N/2) · (2k)! ] / [ (N/2-k)! · k! · (k-1)! · (i-k)! · (2k-i)! ]

N must be even. Industry-standard choices: N=8 (cheap, 8 evaluations),
N=12 (more accurate), N=14 or 16 (more accurate, risk of catastrophic
cancellation in single precision). We use N=8 by default — sufficient for
the double-permeability solution, which is smooth.

The transform must be:
  • single-valued
  • bounded as |s|→∞
  • free of branch cuts in the right half-plane

These conditions hold for the Bourdet double-permeability response.

Reference: Stehfest, H. (1970). "Algorithm 368: Numerical inversion of
Laplace transforms." Commun. ACM 13(1):47-49.
"""
from __future__ import annotations
import math
import numpy as np
from functools import lru_cache


@lru_cache(maxsize=8)
def stehfest_weights(N: int) -> np.ndarray:
    """Compute the N Stehfest weights V_i (i=1..N)."""
    if N % 2 != 0:
        raise ValueError("Stehfest N must be even.")
    nh = N // 2
    V = np.zeros(N)
    for i in range(1, N + 1):
        k_min = (i + 1) // 2
        k_max = min(i, nh)
        total = 0.0
        for k in range(k_min, k_max + 1):
            num = (k ** nh) * math.factorial(2 * k)
            den = (math.factorial(nh - k) * math.factorial(k)
                   * math.factorial(k - 1) * math.factorial(i - k)
                   * math.factorial(2 * k - i))
            total += num / den
        V[i - 1] = ((-1) ** (nh + i)) * total
    return V


def stehfest_invert(F_bar, t_array: np.ndarray, N: int = 8) -> np.ndarray:
    """
    Numerically invert the Laplace transform F̄(s) at the times in t_array.

    Parameters
    ----------
    F_bar : callable
        F̄(s) — must accept a numpy array of real positive s values and
        return a numpy array of the same shape (real values).
    t_array : array-like
        Real positive times at which to evaluate f(t).
    N : int, even
        Number of Stehfest terms. 8 is robust; 12 sharper but more sensitive
        to floating-point noise.

    Returns
    -------
    f : np.ndarray
        Inverted values f(t) at each t in t_array.
    """
    t = np.asarray(t_array, dtype=float)
    if np.any(t <= 0):
        raise ValueError("Stehfest inversion requires strictly positive times.")
    V = stehfest_weights(N)
    ln2 = math.log(2.0)
    # Vectorise: build a (len(t), N) matrix of s = i·ln2/t values and call F̄ once.
    i_arr = np.arange(1, N + 1)
    # s[j, i-1] = i · ln2 / t[j]
    s_matrix = np.outer(ln2 / t, i_arr)
    F_matrix = F_bar(s_matrix.ravel()).reshape(s_matrix.shape)
    # f(t) = ln2/t · Σ V_i · F̄(i ln2 / t)
    f = (ln2 / t) * (F_matrix @ V)
    return f
