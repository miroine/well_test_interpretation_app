import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from engine import models as M, multirate as MR, deconv as DC, samples as S

P = M.WellTestParams(k=50.0, h=50.0, phi=0.18, mu=1.0, ct=1.5e-5, B=1.2, rw=0.354, pi=4000.0, C=0.005, S=2.0)
SCHED = [(0.0, 1000.0), (60.0, 2000.0), (140.0, 600.0), (220.0, 0.0)]


def _ratio(P, r, tt):
    _, true = M.unit_rate_response(P, tt)
    return np.interp(np.log(tt), np.log(r["t"]), r["deriv"]) / true


def test_nonlinear_recovers_derivative_and_pi_with_noise():
    for noise in (0.02, 0.3):
        t, pw = MR.synthetic_multirate(P, SCHED, 330.0, n=400, noise_psi=noise)
        r = DC.deconvolve_nonlinear(SCHED, t, pw, pi=None)
        assert abs(r["pi_est"] - P.pi) < 3.0, (noise, r["pi_est"])
        assert np.all(np.abs(np.log(_ratio(P, r, np.array([2.0, 10.0, 50.0, 200.0])))) < 0.12), noise
        assert np.all(r["deriv"] > 0)


def test_nonlinear_on_dst_with_known_pi():
    p, sched, t, pw = S.dst_record()
    r = DC.deconvolve_nonlinear(sched, t, pw, pi=p.pi)
    assert r["pi_est"] == p.pi
    assert np.all(np.abs(np.log(_ratio(p, r, np.array([0.5, 3.0, 10.0])))) < 0.2)


def test_nonlinear_fits_pressure_at_least_as_well_as_linear():
    t, pw = MR.synthetic_multirate(P, SCHED, 330.0, n=400, noise_psi=0.1)
    a = DC.deconvolve_nonlinear(SCHED, t, pw)
    b = DC.deconvolve(SCHED, t, pw, n_bins=50, lam=0.5)
    assert a["rms"] <= b["rms"] * 1.1


if __name__ == "__main__":
    import traceback
    f = 0
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            try: fn(); print("PASS", n)
            except Exception: f += 1; print("FAIL", n); traceback.print_exc()
    sys.exit(1 if f else 0)
