import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.special import expi
from engine import areal as A, models as M

BASE = dict(k=50, h=50, phi=0.18, mu=1.0, ct=1.5e-5, B=1.2, rw=0.354, pi=4000.0)


def _p(**kw):
    return M.WellTestParams(**{**BASE, **kw})


def test_matches_analytic_closed_square():
    cfg = A.ArealConfig(wells=[A.ArealWell(skin=2.0, schedule=[(0.0, 1000.0)])])
    res = A.run(cfg, 1500.0)
    t, dp = A.drawdown_response(res)
    p = _p(C=0.0, S=2.0, boundary="rectangle", xe=6000, ye=6000, xw=3000, yw=3000)
    dpa = M.unit_rate_response(p, t)[0] * 1000
    m = t > 0.1
    assert np.max(np.abs(dp[m] / dpa[m] - 1)) < 0.02


def test_sealing_fault_doubles_slope_and_matches_images():
    cfg = A.ArealConfig(wells=[A.ArealWell(schedule=[(0.0, 1000.0)])],
                        faults=[dict(x1=400.0, y1=-3000.0, x2=400.0, y2=3000.0, mult=0.0)])
    res = A.run(cfg, 400.0)
    t, dp = A.drawdown_response(res)
    p = _p(C=0.0, S=0.0, boundary="fault", d1=400.0)
    dpa = M.unit_rate_response(p, t)[0] * 1000
    m = t > 0.05
    assert np.max(np.abs(dp[m] / dpa[m] - 1)) < 0.04, np.max(np.abs(dp[m] / dpa[m] - 1))


def test_leaky_fault_between_extremes():
    last = []
    for mult in (1.0, 0.1, 0.0):
        cfg = A.ArealConfig(wells=[A.ArealWell(schedule=[(0.0, 1000.0)])],
                            faults=[dict(x1=400.0, y1=-3000.0, x2=400.0, y2=3000.0, mult=mult)])
        t, dp = A.drawdown_response(A.run(cfg, 300.0))
        last.append(dp[-1])
    assert last[0] < last[1] < last[2]


def test_interference_matches_line_source():
    obs = A.ArealWell(name="OBS", x=1000.0, y=0.0, schedule=[(0.0, 0.0)])
    cfg = A.ArealConfig(wells=[A.ArealWell(schedule=[(0.0, 1000.0)]), obs])
    res = A.run(cfg, 200.0)
    t = res["t"][1:]
    dp = res["pwf"][0, 1] - res["pwf"][1:, 1]
    ei = -70.6 * 1000 * 1.2 * 1.0 / (50 * 50) * expi(-948 * 0.18 * 1.0 * 1.5e-5 * 1000.0 ** 2 / (50 * t))
    m = ei > 8.0
    assert np.max(np.abs(dp[m] / ei[m] - 1)) < 0.06, np.max(np.abs(dp[m] / ei[m] - 1))


def test_polygon_and_zone_run():
    L_shape = [(-2000, -2000), (2000, -2000), (2000, 0), (0, 0), (0, 2000), (-2000, 2000)]
    cfg = A.ArealConfig(outline=L_shape, wells=[A.ArealWell(x=-1000, y=-1000, schedule=[(0.0, 500.0)])],
                        zones=[dict(poly=[(-500, -2000), (500, -2000), (500, 0), (-500, 0)], kmult=0.1, phimult=1.0)])
    res = A.run(cfg, 500.0)
    assert np.all(np.isfinite(res["pwf"])) and res["pwf"][-1, 0] < 4000


def test_buildup_returns_to_initial_with_big_reservoir():
    cfg = A.ArealConfig(wells=[A.ArealWell(C=0.01, schedule=[(0.0, 1000.0), (100.0, 0.0)])])
    res = A.run(cfg, 600.0)
    assert res["pwf"][-1, 0] > 3990.0 and res["pwf"][:, 0].min() < 3700


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn(); print("PASS", n)
