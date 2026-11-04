import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from engine import gas as G, qc, units, samples


def test_z_factor_reference():
    # Standing-Katz chart: Tpr=1.5, Ppr=2.0 -> Z ~ 0.84-0.85 (DAK ~0.8442 ... allow band)
    Tpc, Ppc = G.pseudocritical(0.7)
    z = G.z_factor(2.0 * Ppc, 1.5 * Tpc - 459.67, 0.7)
    assert 0.82 < z < 0.87, z
    # low pressure -> ideal gas
    assert abs(G.z_factor(15.0, 100.0, 0.65) - 1.0) < 0.01


def test_viscosity_range():
    mu = G.viscosity(3000.0, 200.0, 0.7)
    assert 0.015 < mu < 0.030, mu       # typical 0.018-0.025 cp


def test_pseudo_pressure_monotone_and_inverse():
    npp = G.NormalizedPseudoPressure(5000.0, 200.0, 0.7)
    p = np.array([1000.0, 2500.0, 4000.0, 5000.0])
    pn = npp.forward(p)
    assert np.all(np.diff(pn) > 0) and abs(pn[-1] - 5000.0) < 1e-6
    assert np.allclose(npp.inverse(pn), p, atol=1.0)


def test_qc_outliers_and_noise():
    rng = np.random.default_rng(0)
    y = np.linspace(0, 10, 200) + rng.normal(0, 0.05, 200)
    y[50] += 3.0
    assert qc.hampel_outliers(y)[50]
    assert 0.03 < qc.noise_estimate(y) < 0.09


def test_clean_gauge():
    df = pd.DataFrame({"t": [3, 1, 2, 2, np.nan], "p": [1, 2, 3, 4, 5]})
    d, issues = qc.clean_gauge(df, "t", "p")
    assert list(d["t"]) == [1.0, 2.0, 3.0] and len(issues) == 3


def test_units_roundtrip():
    for kind in ("length", "pressure", "rate", "ct", "storage"):
        assert abs(units.to_field(units.from_field(123.0, kind, "Metric"), kind, "Metric") - 123.0) < 1e-9
    assert abs(units.from_field(212.0, "temp", "Metric") - 100.0) < 1e-9


def test_samples_run():
    for fn in samples.SAMPLES.values():
        p, q, tp, d = fn()
        assert np.all(np.isfinite(d["pws"]))
    p, q, tp, d, meta = samples.gas_buildup()
    assert np.all(np.isfinite(d["pws"]))


if __name__ == "__main__":
    import traceback
    f = 0
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            try: fn(); print("PASS", n)
            except Exception: f += 1; print("FAIL", n); traceback.print_exc()
    sys.exit(1 if f else 0)
