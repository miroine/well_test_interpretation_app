import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.special import exp1

from engine import laplace as L


def test_line_source_vs_exponential_integral():
    """No storage, no skin: pD(tD) at rw ~ 0.5*E1(1/4tD) for large tD."""
    tD = np.array([1e3, 1e4, 1e5])
    pD, _ = L.unit_response(lambda u: L.pws_radial(u, L.f_homogeneous, 0.0, 0.0), tD)
    ref = 0.5 * (np.log(tD) + 0.80907)
    assert np.allclose(pD, ref, rtol=2e-3), (pD, ref)


def test_derivative_stabilises_at_half():
    # radial flow starts after ~ 50 * CD * exp(2 S) -> 1e8 for CD=100, S=5
    tD = np.logspace(8.5, 10, 4)
    _, d = L.unit_response(lambda u: L.pws_radial(u, L.f_homogeneous, 100.0, 5.0), tD)
    assert np.allclose(d, 0.5, atol=5e-3), d


def test_wellbore_storage_unit_slope():
    tD = np.array([1e-1, 1.0])
    CD = 1e4
    pD, d = L.unit_response(lambda u: L.pws_radial(u, L.f_homogeneous, CD, 0.0), tD)
    assert np.allclose(pD, tD / CD, rtol=1e-2)
    assert np.allclose(d, tD / CD, rtol=1e-2)


def test_dual_porosity_dip():
    k = lambda u: L.f_dual_pss(u, 0.05, 1e-6)
    tD = np.logspace(2, 8, 60)
    _, d = L.unit_response(lambda u: L.pws_radial(u, k, 10.0, 0.0), tD)
    assert d.min() < 0.40 and abs(d[-1] - 0.5) < 5e-3


def test_sealing_fault_doubles_derivative():
    b = {"type": "fault", "d": 300.0, "sealing": True}
    tD = np.logspace(5, 8, 6)
    _, d = L.unit_response(lambda u: L.pws_radial(u, L.f_homogeneous, 100.0, 0.0, b), tD)
    assert abs(d[-1] - 1.0) < 0.03, d


def test_constant_pressure_boundary_drops_derivative():
    b = {"type": "fault", "d": 300.0, "sealing": False}
    tD = np.logspace(5, 8, 6)
    _, d = L.unit_response(lambda u: L.pws_radial(u, L.f_homogeneous, 100.0, 0.0, b), tD)
    assert d[-1] < 0.05, d


def test_channel_half_slope():
    b = {"type": "channel", "d1": 100.0, "d2": 100.0}
    tD = np.logspace(6, 8, 5)
    _, d = L.unit_response(lambda u: L.pws_radial(u, L.f_homogeneous, 10.0, 0.0, b), tD)
    # derivative grows ~ sqrt(t) in linear flow: slope 0.5 on log-log
    slope = np.polyfit(np.log(tD[-3:]), np.log(d[-3:]), 1)[0]
    assert 0.35 < slope < 0.65, slope


def test_composite_limits():
    """M=1, D=1 must reproduce the homogeneous response."""
    tD = np.logspace(1, 5, 5)
    ref, _ = L.unit_response(lambda u: L.pws_radial(u, L.f_homogeneous, 100.0, 2.0), tD)
    comp, _ = L.unit_response(
        lambda u: L.pws_composite(u, L.f_homogeneous, 100.0, 2.0, 50.0, 1.0, 1.0), tD)
    assert np.allclose(ref, comp, rtol=2e-3), (ref, comp)


def test_composite_outer_mobility():
    """Lower outer mobility (M=5) doubles... derivative -> 0.5*M at late time."""
    tD = np.logspace(6, 8, 4)
    _, d = L.unit_response(
        lambda u: L.pws_composite(u, L.f_homogeneous, 10.0, 0.0, 30.0, 5.0, 5.0), tD)
    assert abs(d[-1] - 0.5 * 5.0) < 0.1, d


def test_fracture_half_slope_early():
    tD = np.logspace(-3, -1.5, 5)
    pD, d = L.unit_response(lambda u: L.pws_fracture(u, L.f_homogeneous, 0.0, 0.0), tD)
    slope = np.polyfit(np.log(tD), np.log(pD), 1)[0]
    assert 0.45 < slope < 0.55, slope


def test_limited_entry_reduces_to_full_penetration():
    tD = np.logspace(2, 5, 4)
    ref, _ = L.unit_response(lambda u: L.pws_radial(u, L.f_homogeneous, 10.0, 0.0), tD)
    le, _ = L.unit_response(
        lambda u: L.pws_limited_entry(u, L.f_homogeneous, 10.0, 0.0, 50.0, 1.0, 0.5), tD)
    assert np.allclose(ref, le, rtol=5e-3)


def test_limited_entry_adds_pseudoskin():
    tD = np.array([1e6])
    full, _ = L.unit_response(lambda u: L.pws_radial(u, L.f_homogeneous, 10.0, 0.0), tD)
    le, _ = L.unit_response(
        lambda u: L.pws_limited_entry(u, L.f_homogeneous, 10.0, 0.0, 100.0, 0.25, 0.125), tD)
    assert le[0] > full[0] + 0.5   # pseudo-skin of a few units


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("PASS", name)
            except Exception:
                fails += 1; print("FAIL", name); traceback.print_exc()
    sys.exit(1 if fails else 0)
