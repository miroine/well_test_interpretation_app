import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from engine import laplace as L


def _resp(LD, hD, zw=0.5, top="sealed", bottom="sealed", tD=None):
    tD = np.geomspace(1e-1, 1e9, 60) if tD is None else tD
    return tD, *L.unit_response(
        lambda u: L.pws_horizontal(u, L.f_homogeneous, 0.0, 0.0, LD, hD, zw, top, bottom), tD)


def test_early_radial_and_late_radial():
    LD, hD = 2000 / 0.354, 100 / 0.354 * np.sqrt(10)
    tD, pD, dD = _resp(LD, hD)
    early = dD[8:20].max()
    assert abs(early / (0.5 * hD / LD) - 1) < 0.1, (early, 0.5 * hD / LD)
    assert abs(dD[-1] - 0.5) < 0.03, dD[-1]


def test_constant_pressure_top_stabilises():
    LD, hD = 300.0, 150.0
    tD, pD, dD = _resp(LD, hD, zw=0.3, top="const_p", tD=np.geomspace(1e0, 1e8, 40))
    assert dD[-1] < 0.05, dD[-1]


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
