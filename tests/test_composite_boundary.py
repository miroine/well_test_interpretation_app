import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from engine import models as M

BASE = dict(k=50, h=40, phi=0.2, mu=1, ct=1.5e-5, B=1.2, rw=0.354, pi=4000, C=0.0, S=0)
T = np.geomspace(1e-2, 5e4, 50)


def test_reduces_to_homogeneous_fault():
    a = M.WellTestParams(**BASE, boundary="fault", d1=800)
    b = M.WellTestParams(**BASE, model="composite", r_inner=100, M=1.0, D=1.0, boundary="fault", d1=800)
    da, db = M.unit_rate_response(a, T), M.unit_rate_response(b, T)
    assert np.abs(da[0] - db[0]).max() < 1e-6 and np.abs(da[1] - db[1]).max() < 1e-6


def test_double_slope_in_outer_zone():
    for Mm in (0.5, 2.0):
        c = M.WellTestParams(**BASE, model="composite", r_inner=100, M=Mm, D=Mm, boundary="fault", d1=800)
        c0 = M.WellTestParams(**BASE, model="composite", r_inner=100, M=Mm, D=Mm)
        ratio = M.unit_rate_response(c, T)[1][-1] / M.unit_rate_response(c0, T)[1][-1]
        assert abs(ratio - 2.0) < 0.05, (Mm, ratio)


def test_const_pressure_and_channel_run():
    for bd, kw in (("const_p_fault", dict(d1=800)), ("channel", dict(d1=600, d2=900)), ("corner", dict(d1=700, d2=700))):
        p = M.WellTestParams(**BASE, model="composite", r_inner=100, M=2.0, D=1.5, boundary=bd, **kw)
        dp, der = M.unit_rate_response(p, T)
        assert np.all(np.isfinite(dp)) and np.all(np.isfinite(der))


def test_warning_when_boundary_inside_inner_zone():
    p = M.WellTestParams(**BASE, model="composite", r_inner=300, M=2.0, D=1.5, boundary="fault", d1=250)
    assert M.boundary_warning(p)


if __name__ == "__main__":
    import traceback
    f = 0
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            try: fn(); print("PASS", n)
            except Exception: f += 1; print("FAIL", n); traceback.print_exc()
    sys.exit(1 if f else 0)
