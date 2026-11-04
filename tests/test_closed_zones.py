import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from engine import laplace as L

f = L.f_homogeneous
TD = np.geomspace(1e0, 1e9, 55)


def test_circle_pss_unit_slope_and_matches_rectangle_square_roughly():
    reD = 2000.0
    pD, dD = L.unit_response(lambda u: L.pws_circle(u, f, 0, 0, reD), TD)
    # pseudo-steady state: derivative = 2 tD / reD^2
    i = -3
    assert abs(dD[i] / (2 * TD[i] / reD ** 2) - 1) < 0.03, (dD[i], 2 * TD[i] / reD ** 2)
    # before the boundary: radial 0.5
    assert abs(dD[np.searchsorted(TD, 1e3)] - 0.5) < 0.03


def test_circle_const_p_stabilises():
    pD, dD = L.unit_response(lambda u: L.pws_circle(u, f, 0, 0, 500.0, "const_p"), TD)
    assert dD[-1] < 1e-3


def test_rectangle_square_pss_matches_dietz():
    xe = 1500.0
    A = xe * xe
    tdv = np.array([3e5, 1e6])
    pD, dD = L.unit_response(lambda u: L.pws_rectangle(u, f, 0, 0, xe, xe, xe / 2, xe / 2), tdv)
    ca = 30.8828   # Dietz shape factor, well at the centre of a square
    pss = 2 * np.pi * tdv / A + 0.5 * np.log(4 * A / (np.exp(0.5772156649) * ca))
    assert np.max(np.abs(pD - pss)) < 0.02, (pD, pss)
    assert abs(dD[-1] / (2 * np.pi * tdv[-1] / A) - 1) < 0.02


def test_rectangle_elongated_shows_linear_flow():
    pD, dD = L.unit_response(lambda u: L.pws_rectangle(u, f, 0, 0, 4000.0, 400.0, 2000.0, 200.0), TD)
    i = np.where((TD > 2e3) & (TD < 3e4))[0]
    assert i.size > 5


def test_three_zone_reductions():
    u = np.geomspace(1e-8, 1e2, 12)
    a = L.pws_radial(u, f, 0, 1.0)
    b = L.pws_composite3(u, f, 0, 1.0, 50.0, 400.0, 1, 1, 1, 1)
    assert np.max(np.abs(b / a - 1)) < 1e-6
    c = L.pws_composite(u, f, 0, 0.0, 50.0, 3.0, 2.0)
    d = L.pws_composite3(u, f, 0, 0.0, 50.0, 400.0, 3.0, 2.0, 3.0, 2.0)
    assert np.max(np.abs(d / c - 1)) < 1e-6


def test_three_zone_mobility_levels():
    TDz = np.geomspace(1e1, 1e12, 60)
    pD, dD = L.unit_response(lambda u: L.pws_composite3(u, f, 0, 0, 30.0, 3000.0, 1.0, 1.0, 4.0, 4.0), TDz)
    # inner level 0.5, outer level 0.5 * M13 (tight outer zone doubles... mobility 4x lower -> 4x derivative)
    assert abs(dD[np.searchsorted(TDz, 3e1)] - 0.5) < 0.05
    assert abs(dD[-1] - 2.0) < 0.1, dD[-1]


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn(); print("PASS", n)
