import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from engine import models as M, multirate as MR, deconv as DC, analysis as A

P = M.WellTestParams(k=50.0, h=50.0, phi=0.18, mu=1.0, ct=1.5e-5, B=1.2, rw=0.354, pi=4000.0, C=0.005, S=2.0)
SCHED = [(0.0, 1000.0), (60.0, 2000.0), (140.0, 600.0), (220.0, 0.0)]


def test_schedule_from_series():
    t = np.arange(0, 10.0)
    q = np.array([5, 5, 5, 8, 8, 2, 2, 2, 0, 0.0])
    s = MR.schedule_from_series(t + 3.0, q)
    assert s == [(0.0, 5.0), (3.0, 8.0), (5.0, 2.0), (8.0, 0.0)]


def test_local_period_pressure_and_derivative_signs_and_alignment():
    schedule = [(0., 100.), (10., 0.)]
    elapsed = np.geomspace(.01, 9., 50)
    t = np.concatenate([[0.], elapsed, [10.], 10. + elapsed])
    pressure = np.concatenate([[4000.], 4000.-20.*elapsed, [3800.], 3800.+10.*elapsed])
    for period, scale in ((0, 20.), (1, 10.)):
        d = MR.period_diagnostics(schedule, t, pressure, 4000., period)
        mask = d["dt"] < 10
        np.testing.assert_allclose(d["dp"][mask], scale*d["dt"][mask], atol=1e-10)
        expected = A.bourdet_derivative(d["dt"], d["dp"], .1)
        np.testing.assert_allclose(d["der"], expected, equal_nan=True)
        assert np.nanmedian(d["der"]) > 0


def test_local_boundary_interpolation_and_negative_response_is_not_hidden():
    schedule = [(0.,100.), (5.,0.)]
    t = np.array([1.,4.,6.,7.,8.,9.])
    pressure = np.array([3990.,3960.,3970.,3980.,3990.,4000.])
    d = MR.period_diagnostics(schedule,t,pressure,4000.,1)
    assert d["start_pressure"] == 3965.
    np.testing.assert_array_equal(d["dp"], [5.,15.,25.,35.])
    falling = MR.period_diagnostics(schedule,t,-pressure,4000.,1)
    assert np.all(falling["dp"] < 0)


def test_superposition_derivative_plateau():
    t, pw = MR.synthetic_multirate(P, SCHED[:3], 300.0, n=300)
    d = MR.superposition_diagnostics(SCHED[:3], t, pw, P.pi)
    plateau = 0.5 * M.p_scale(P)                        # psi per (stb/d) per ln-cycle
    ok = np.isfinite(d["der"]) & (d["dt"] > 20)
    assert abs(np.median(d["der"][ok]) / plateau - 1) < 0.08, (np.median(d["der"][ok]), plateau)


def test_multirate_fit_recovers_k():
    t, pw = MR.synthetic_multirate(P, SCHED, 330.0, n=300, noise_psi=0.05)
    g = M.WellTestParams(**{**P.to_dict(), "k": 20.0, "S": 0.0, "C": 0.01})
    f, info = MR.fit_multirate(g, SCHED, t, pw, ["k", "S", "C"])
    assert abs(f.k - 50.0) / 50.0 < 0.06 and abs(f.S - 2.0) < 0.6, (f.k, f.S, f.C, info)


def test_deconvolution_recovers_derivative():
    t, pw = MR.synthetic_multirate(P, SCHED, 330.0, n=400, noise_psi=0.02)
    r = DC.deconvolve(SCHED, t, pw, pi=None, n_bins=50, lam=0.5)
    assert abs(r["pi_est"] - P.pi) < 3.0, r["pi_est"]
    # compare with the true unit-rate derivative at mid / late times
    tt = np.array([2.0, 10.0, 50.0, 200.0])
    _, der_true = M.unit_rate_response(P, tt)
    der_dec = np.interp(np.log(tt), np.log(r["t"]), r["deriv"])
    err = np.abs(np.log(der_dec / der_true))
    assert np.all(err < 0.25), (der_dec, der_true)


if __name__ == "__main__":
    import traceback
    f = 0
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            try: fn(); print("PASS", n)
            except Exception: f += 1; print("FAIL", n); traceback.print_exc()
    sys.exit(1 if f else 0)
