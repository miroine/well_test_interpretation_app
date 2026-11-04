import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from engine import models as M, wellbore as W

P = M.WellTestParams(k=50, h=50, phi=0.18, mu=1.0, ct=1.5e-5, B=1.2, rw=0.354, pi=4000.0, C=0.02, S=3.0)


def test_constant_storage_matches_laplace():
    t = np.geomspace(0.005, 200, 30)
    ref = P.pi - M.unit_rate_response(P, t)[0] * 800
    fx = W.WellboreEffects(C_i=0.02, C_f=0.02)
    r = W.march(P, fx, [(0.0, 800.0)], t)
    assert np.max(np.abs((P.pi - r["pwf"]) / (P.pi - ref) - 1)) < 0.03


def test_changing_storage_lowers_early_drawdown_when_decreasing():
    t = np.geomspace(0.005, 50, 25)
    const = W.march(P, W.WellboreEffects(C_i=0.1, C_f=0.1), [(0.0, 800.0)], t)
    dec = W.march(P, W.WellboreEffects(C_i=0.1, C_f=0.005, alpha=0.2), [(0.0, 800.0)], t)
    assert dec["pwf"][12] < const["pwf"][12]          # decreasing storage reaches the reservoir sooner


def test_rate_dependent_skin():
    t = np.array([5.0, 50.0])
    a = W.march(P, W.WellboreEffects(C_i=0.02, C_f=0.02, D=0.0), [(0.0, 800.0)], t)
    b = W.march(P, W.WellboreEffects(C_i=0.02, C_f=0.02, D=0.005), [(0.0, 800.0)], t)
    extra = (a["pwf"] - b["pwf"])
    expected = M.p_scale(P) * 800 * 0.005 * 800      # extra skin D*q applied to q
    assert np.all(np.abs(extra / expected - 1) < 0.1), (extra, expected)


def test_stress_sensitivity_adds_drawdown_and_buildup_recovers():
    t = np.array([10.0, 100.0, 150.0, 400.0])
    base = W.march(P, W.WellboreEffects(C_i=0.02, C_f=0.02), [(0.0, 800.0), (100.0, 0.0)], t)
    ss = W.march(P, W.WellboreEffects(C_i=0.02, C_f=0.02, gamma=2e-4), [(0.0, 800.0), (100.0, 0.0)], t)
    assert ss["pwf"][0] < base["pwf"][0] and ss["pwf"][1] < base["pwf"][1]
    assert ss["pwf"][-1] > 3990


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn(); print("PASS", n)
