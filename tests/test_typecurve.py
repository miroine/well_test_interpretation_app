import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from engine import models as M, viz, typecurve as T, laplace as L


def test_auto_match_recovers_k_and_c():
    p = M.WellTestParams(k=50.0, h=50.0, phi=0.2, mu=1.0, ct=1.5e-5, B=1.2, rw=0.354, C=0.02, S=3.0)
    q = 1000.0
    dt = np.logspace(-3, 2.7, 90)
    dp, der = M.unit_rate_response(p, dt)
    dp, der = dp * q, der * q
    # CDe2S for truth
    G_true = M.CD_of(p) * np.exp(2 * p.S)
    m = T.auto_match(dt, dp, der)
    res = T.interpret_match(m["sx"], m["sy"], m["G"], q, p.B, p.mu, p.h, p.phi, p.ct, p.rw)
    assert abs(res["k"] - 50.0) / 50.0 < 0.1, (res, m, G_true)
    assert abs(res["C"] - 0.02) / 0.02 < 0.35, (res, m, G_true)


def test_family_shapes():
    fam = T.family([1e3, 1e10])
    for G, (x, pD, dD) in fam.items():
        assert np.isfinite(pD).all() and np.isfinite(dD).all()
        assert abs(dD[-1] - 0.5) < 0.02, (G, dD[-1])


def test_entire_family_is_positive_smooth_and_has_storage_asymptote():
    x = np.logspace(-7, 7, 400)
    for G, (_, pressure, derivative) in T.family(x_grid=x).items():
        assert np.isfinite(pressure).all() and np.isfinite(derivative).all(), G
        assert (pressure > 0).all() and (derivative > 0).all(), G
        np.testing.assert_allclose(pressure[:8] / x[:8], 1., rtol=.01, err_msg=str(G))
        np.testing.assert_allclose(derivative[:8] / x[:8], 1., rtol=.01, err_msg=str(G))
        assert np.max(np.abs(np.diff(np.log10(pressure)))) < .1, G
        assert np.max(np.abs(np.diff(np.log10(derivative)))) < .15, G
        assert abs(derivative[-1] - .5) < .02, G


def test_low_group_uses_stable_effective_radius_and_matching_same_response():
    x = np.geomspace(1e-5, 1e5, 120)
    for G in (.1, 1., 10., 100., 1e4):
        _, pressure, derivative = T.family([G], x)[G]
        CD = min(100., G)
        expected = L.unit_response(lambda u: L.pws_radial(u, L.f_homogeneous, CD,
                                                        .5 * np.log(G / CD)), x * CD)
        np.testing.assert_allclose(pressure, expected[0])
        np.testing.assert_allclose(derivative, expected[1])
        lp, ld = T._curve_logs(G, x)
        np.testing.assert_allclose(np.exp(lp), pressure)
        np.testing.assert_allclose(np.exp(ld), derivative)


def test_nonnegative_skin_family_is_unchanged():
    x = np.geomspace(.001, 1e7, 100)
    for G in (100., 1e3, 1e12, 1e60):
        expected = L.unit_response(lambda u: L.pws_radial(u, L.f_homogeneous, 100.,
                                                        .5 * np.log(G / 100.)), x * 100.)
        _, pressure, derivative = T.family([G], x)[G]
        np.testing.assert_array_equal(pressure, expected[0])
        np.testing.assert_array_equal(derivative, expected[1])


if __name__ == "__main__":
    import traceback
    f = 0
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            try: fn(); print("PASS", n)
            except Exception: f += 1; print("FAIL", n); traceback.print_exc()
    sys.exit(1 if f else 0)
