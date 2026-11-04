import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from engine import laplace as L

f = L.f_homogeneous
TD = np.geomspace(1e-5, 1e4, 50)


def _resp(fn):
    return L.unit_response(fn, TD)


def test_high_conductivity_matches_infinite():
    u = np.geomspace(1e-6, 1e2, 8)
    a = L.pws_fracture(u, f, 0, 0, 0.732)
    b = L.pws_fracture_fc(u, f, 0, 0, 1e4)
    assert np.max(np.abs(b / a - 1)) < 0.03


def test_bilinear_quarter_slope():
    pD, dD = _resp(lambda u: L.pws_fracture_fc(u, f, 0, 0, 0.5))
    i = np.where((TD > 1e-5) & (TD < 5e-4))[0]
    slope = np.polyfit(np.log(TD[i]), np.log(dD[i]), 1)[0]
    assert 0.2 < slope < 0.32, slope


def test_late_radial_half():
    for fn in (lambda u: L.pws_fracture_fc(u, f, 0, 0, 5.0),
               lambda u: L.pws_mfhw(u, f, 0, 0, 50.0, 3, 1.0)):
        pD, dD = _resp(fn)
        assert abs(dD[-1] - 0.5) < 0.02


def test_more_fractures_less_drawdown():
    u = np.array([1e-3])
    p1 = L.pws_mfhw(u, f, 0, 0, 50.0, 1, 1.0)[0]
    p4 = L.pws_mfhw(u, f, 0, 0, 50.0, 4, 1.0)[0]
    assert p4 < p1


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn(); print("PASS", n)
