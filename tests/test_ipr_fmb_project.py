import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from engine import ipr, fmb, project


def test_vogel_endpoints():
    assert abs(ipr.vogel_rate(3000.0, 1000.0, 0.0) - 1000.0) < 1e-9
    assert abs(ipr.vogel_rate(3000.0, 1000.0, 3000.0)) < 1e-9
    qm = ipr.vogel_qmax(600.0, 2000.0, 3000.0)
    assert abs(ipr.vogel_rate(3000.0, qm, 2000.0) - 600.0) < 1e-9


def test_composite_continuity_and_test_point():
    pr, pb, J = 4000.0, 2500.0, 2.0
    q = ipr.composite_ipr(pr, pb, J, np.array([pb + 1e-6, pb - 1e-6]))
    assert abs(q[0] - q[1]) < 1e-3
    J2 = ipr.composite_J_from_test(pr, pb, ipr.composite_ipr(pr, pb, J, np.array([1000.0]))[0], 1000.0)
    assert abs(J2 - J) < 1e-9


def test_pi_pss_scale():
    j = ipr.pi_pss(k=50, h=50, mu=1.0, B=1.2, re=1000, rw=0.354, S=0)
    ref = 50 * 50 / (141.2 * 1.2 * 1.0 * (np.log(1000 / 0.354) - 0.75))
    assert abs(j - ref) < 1e-9 and 2.0 < j < 2.1
    assert ipr.pi_pss(50, 50, 1.0, 1.2, 1000, 0.354, 5) < j   # positive skin lowers J


def test_fetkovich_and_lit_recover():
    pr = 3000.0
    pwf = np.array([2800.0, 2400.0, 2000.0, 1500.0])
    C, n = 0.0008, 0.85
    q = ipr.fetkovich_rate(pr, C, n, pwf)
    C2, n2, r2 = ipr.fetkovich_fit(pr, pwf, q)
    assert abs(n2 - n) < 1e-6 and abs(C2 / C - 1) < 1e-6 and r2 > 0.999999
    a, b = 120.0, 0.02
    q = ipr.lit_rate(pr, a, b, pwf)
    a2, b2, _ = ipr.lit_fit(pr, pwf, q)
    assert abs(a2 / a - 1) < 1e-6 and abs(b2 / b - 1) < 1e-6
    assert abs(ipr.aof_lit(pr, a, b) - ipr.lit_rate(pr, a, b, np.array([0.0]))[0]) < 1e-9


def test_gas_fmb_recovers_G():
    tank = fmb.GasTank(5000.0, 200.0, 0.7, n=200)
    G_true, J_true = 4.0e6, 0.9           # Mscf, Mscf/d/psi
    t, q, pw, Gp = fmb.simulate_gas(tank, G_true, J_true, lambda t: 1200.0, 1500.0, dt=10.0)
    rng = np.random.default_rng(1)
    qn = q * (1 + rng.normal(0, 0.01, len(q)))
    r = fmb.fit_gas_fmb(tank, Gp, qn, pw, G0=2.0e6)
    assert abs(r["G"] / G_true - 1) < 0.05, r
    assert abs(r["J"] / J_true - 1) < 0.05, r
    line = fmb.gas_fmb_line(tank, Gp, q, pw)
    assert 0.5 * G_true < line["G_line"] < 1.5 * G_true     # quick straight-line estimate, rough


def test_oil_fmb_recovers_N():
    N_true, J_true = 2.0e7, 4.0
    t, q, pw, Np = fmb.simulate_oil(N_true, J_true, 1.8e-5, 1.25, 1.25, 4500.0, lambda t: 1500.0, 1500.0, dt=10.0)
    r = fmb.fit_oil_fmb(4500.0, 1.8e-5, 1.25, 1.25, Np, q, pw, N0=1e7)
    assert abs(r["N"] / N_true - 1) < 0.02 and abs(r["J"] / J_true - 1) < 0.02, r


def test_project_roundtrip():
    st = {"dataset": {"dt": np.array([1.0, 2.0]), "tp": 10.0}, "fits": [{"name": "a", "rms": np.float64(0.1)}],
          "mr": {"sched": [(0.0, 5.0), (3.0, 0.0)]}}
    txt = project.dumps({"well": "W"}, st)
    meta, back = project.loads(txt)
    assert meta["well"] == "W"
    assert np.allclose(back["dataset"]["dt"], [1.0, 2.0]) and back["mr"]["sched"][1] == (3.0, 0.0)
    assert back["fits"][0]["rms"] == 0.1


def test_report_builds_valid_html():
    from html.parser import HTMLParser
    from engine import report
    x = np.logspace(-2, 2, 30)
    svg = report.svg_chart([{"x": x, "y": np.sqrt(x), "kind": "points", "name": "dp"},
                            {"x": x, "y": 0.5 * np.ones_like(x), "kind": "line", "dash": True, "name": "model"}],
                           "time [hr]", "dp [psi]")
    assert svg.startswith("<svg") and svg.count("<circle") == 30 and "<path" in svg
    h = report.build_html("T & <test>", {"Well": "A"}, [{"heading": "R", "rows": [{"k": 1.5, "S": 2}], "svg": svg, "paras": ["x<y"]}])
    HTMLParser().feed(h)
    assert "T &amp; &lt;test&gt;" in h and "x&lt;y" in h


if __name__ == "__main__":
    import traceback
    f = 0
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            try: fn(); print("PASS", n)
            except Exception: f += 1; print("FAIL", n); traceback.print_exc()
    sys.exit(1 if f else 0)
