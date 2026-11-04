import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.special import expi
from engine import interference as I, models as M

P = M.WellTestParams(k=50, h=50, phi=0.18, mu=1.0, ct=1.5e-5, B=1.2, rw=0.354, pi=4000.0)


def test_matches_line_source_without_storage():
    t = np.array([20.0, 100.0, 500.0, 1500.0])
    w = I.ActiveWell(x=0, y=0, C=0.0, S=0.0, schedule=[(0.0, 1000.0)])
    dp = I.dp_observed(P, (800.0, 600.0), [w], t)
    ei = -70.6 * 1000 * 1.2 / (50 * 50) * expi(-948 * 0.18 * 1.5e-5 * 1000.0 ** 2 / (50 * t))
    assert np.max(np.abs(dp / ei - 1)) < 0.02, (dp, ei)


def test_storage_delays_response():
    t = np.array([2.0, 5.0, 10.0])
    a = I.dp_observed(P, (500.0, 0.0), [I.ActiveWell(C=0.0)], t)
    b = I.dp_observed(P, (500.0, 0.0), [I.ActiveWell(C=5.0)], t)
    assert np.all(b < a)


def test_pulse_fit_recovers_k_phi():
    w = I.ActiveWell(C=0.05, S=1.0, schedule=[(0.0, 0.0), (20.0, 1000.0), (60.0, 0.0), (120.0, 1000.0), (160.0, 0.0)])
    t = np.linspace(1, 300, 120)
    truth = M.WellTestParams(**{**P.to_dict(), "k": 80.0, "phi": 0.12})
    dp = I.dp_observed(truth, (600.0, 0.0), [w], t)
    rng = np.random.default_rng(1)
    dp_n = dp + rng.normal(0, 0.02, dp.shape)
    p0 = M.WellTestParams(**{**P.to_dict(), "k": 40.0, "phi": 0.2})
    pf, info = I.fit_interference(p0, (600.0, 0.0), [w], t, dp_n)
    assert abs(pf.k / 80 - 1) < 0.1 and abs(pf.phi / 0.12 - 1) < 0.15, (pf.k, pf.phi)


def test_pulse_metrics():
    w = I.ActiveWell(C=0.0, schedule=[(0.0, 1000.0), (50.0, 0.0)])
    t = np.linspace(0.5, 300, 300)
    dp = I.dp_observed(P, (500.0, 0.0), [w], t)
    m = I.pulse_metrics(t, dp, 0.0, 50.0)
    assert m["amplitude"] > 0 and m["lag_peak"] > 0


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn(); print("PASS", n)
