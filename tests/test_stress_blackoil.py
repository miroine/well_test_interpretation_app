import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from engine.blackoil import BlackOilSim, SimConfig


def _run(gamma):
    cfg = SimConfig(nr=14, nz=3, h=40, k=65, kv_kh=1.0, phi=0.17, p_datum=3850, Rsi=300,
                    C=0.0, skin=0.0, re=3000, gamma=gamma)
    return BlackOilSim(cfg).run([(0, 800.0), (60.0, 0.0)], 120.0)


def test_stress_sensitivity_deepens_drawdown_and_recovers():
    a, b = _run(0.0), _run(3e-4)
    i = np.argmin(abs(a["t"] - 50.0))
    assert (3850 - b["pwf"][i]) > 1.05 * (3850 - a["pwf"][i])
    assert b["pwf"][-1] > 3800            # reversible model: pressure builds back up


def test_zero_gamma_unchanged():
    a = _run(0.0)
    assert np.all(np.isfinite(a["pwf"]))


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn(); print("PASS", n)
