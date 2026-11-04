import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from engine import models as M, analysis as A


def test_horner_slope_recovers_k():
    p = M.WellTestParams(k=40.0, S=3.0, C=0.02)
    q, tp = 1500.0, 200.0
    d = M.synthetic_buildup(p, q, tp, 100.0, n=150)
    ht = A.horner_time(tp, d["dt"])
    m, b = A.semilog_radial(ht, d["pws"], (1.2, 6.0))
    k = A.permeability_from_slope(m, q, p.B, p.mu, p.h)
    assert abs(k - 40.0) / 40.0 < 0.06, k


def test_buildup_returns_toward_pi():
    p = M.WellTestParams()
    d = M.synthetic_buildup(p, 1000.0, 100.0, 5000.0, n=60)
    assert d["pws"][-1] > d["pws"][0] and d["pws"][-1] < p.pi + 1e-6


def test_fit_recovers_parameters():
    truth = M.WellTestParams(k=35.0, S=4.0, C=0.03)
    q, tp = 1200.0, 150.0
    d = M.synthetic_buildup(truth, q, tp, 80.0, n=80, noise_psi=0.0)
    guess = M.WellTestParams(k=15.0, S=0.0, C=0.01)
    fit, info = A.fit_buildup(guess, q, tp, d["dt"], d["pws"], ["k", "S", "C"],
                             pwf_shutin=d["pwf_at_shutin"])
    assert abs(fit.k - 35.0) / 35.0 < 0.05, (fit.k, fit.S, fit.C, info)
    assert abs(fit.S - 4.0) < 0.4, (fit.k, fit.S, fit.C, info)


def test_profile_monotone_and_front():
    p = M.WellTestParams(k=50.0, S=0.0, C=1e-3)
    sched = M.simple_buildup_schedule(1000.0, 1e6)       # essentially a drawdown
    r = np.array([1.0, 10.0, 100.0, 1000.0, 5000.0])
    pr = M.radial_profile(p, r, 100.0, sched)
    assert np.all(np.diff(pr) >= -1e-6)                   # pressure rises with r
    assert pr[-1] > p.pi - 0.5                            # beyond the front
    assert pr[0] < p.pi - 20


if __name__ == "__main__":
    import traceback
    f = 0
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            try: fn(); print("PASS", n)
            except Exception: f += 1; print("FAIL", n); traceback.print_exc()
    sys.exit(1 if f else 0)
