import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from engine.blackoil import BlackOilSim, SimConfig, RelPerm
from engine import models as M


def _single_phase(nr=20, **kw):
    cfg = SimConfig(nr=nr, nz=4, h=40, k=65, kv_kh=1.0, phi=0.17, p_datum=3850, Rsi=300,
                    C=0.012, skin=3.5, re=5000, **kw)
    return BlackOilSim(cfg)


def test_matches_analytical_drawdown_buildup():
    s = _single_phase()
    sched = [(0, 500.0), (200.0, 0.0)]
    r = s.run(sched, 400.0)
    rp, v = s.rp, s.pvt
    P = s.props(s.u0[0::3][:1], s.u0[1::3][:1], s.u0[2::3][:1], s.sat0[:1])
    p = M.WellTestParams(k=65 * rp.krocw, h=40, phi=0.17, mu=float(P.muo[0]),
                         ct=s.cfg.cr + (1 - rp.swc) * v.co + rp.swc * v.cw,
                         B=float(P.bo[0]), rw=0.354, pi=3850, C=0.012, S=3.5)
    ana = M.pwf_history(p, np.maximum(r["t"], 1e-6), sched)
    t = r["t"]
    for tt in (2, 10, 50, 150):
        i = np.argmin(abs(t - tt))
        ratio = (3850 - r["pwf"][i]) / (3850 - ana[i])
        assert abs(ratio - 1) < 0.02, (tt, ratio)
    for tt in (210, 300, 400):
        i = np.argmin(abs(t - tt))
        assert abs(r["pwf"][i] - ana[i]) < 3.0, (tt, r["pwf"][i], ana[i])


def test_closed_system_mass_balance_two_phase():
    cfg = SimConfig(nr=12, nz=4, h=50, k=50, kv_kh=0.2, phi=0.2, p_datum=2500, Rsi=600, re=1500)
    s = BlackOilSim(cfg)
    r = s.run([(0, 800.0)], 1500.0)
    assert r["GOR"][-1] > 600 * 0.9          # below bubble point: free gas produced
    o0, g0, w0 = r["inplace0"]; o1, g1, w1 = r["inplace"]
    assert abs((o0 - o1) / r["Np"][-1] - 1) < 1e-6
    assert abs((g0 - g1) / r["Gp"][-1] - 1) < 1e-6


def test_partial_penetration_adds_pseudoskin():
    full = BlackOilSim(SimConfig(nr=14, nz=8, h=80, k=100, kv_kh=0.1, p_datum=3500, Rsi=300, re=3000))
    part = BlackOilSim(SimConfig(nr=14, nz=8, h=80, k=100, kv_kh=0.1, p_datum=3500, Rsi=300, re=3000,
                                 perf_from=0.0, perf_to=0.25))
    a = full.run([(0, 1000.0)], 100.0)["pwf"][-1]
    b = part.run([(0, 1000.0)], 100.0)["pwf"][-1]
    assert (3500 - b) > 1.1 * (3500 - a)


def test_gas_cap_and_water_coning_signature():
    def run(q):
        cfg = SimConfig(nr=12, nz=10, h=120, k=300, kv_kh=0.5, phi=0.22, top_depth=8000,
                        p_datum=3500, datum_depth=8060, goc=8030, woc=8090, Rsi=500, re=2500,
                        perf_from=0.35, perf_to=0.55, outer_pv_mult=50)
        return BlackOilSim(cfg).run([(0, q)], 600.0)
    lo, hi = run(500.0), run(4000.0)
    assert hi["GOR"][-1] > 3 * hi["GOR"][0]                 # gas coning
    assert hi["GOR"][-1] > lo["GOR"][-1]                    # worse at higher rate
    assert hi["WC"][-1] > lo["WC"][-1]                      # water coning too
    assert hi["snapshots"][-1]["sg"].max() > 0.6


if __name__ == "__main__":
    import traceback
    f = 0
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            try: fn(); print("PASS", n)
            except Exception: f += 1; print("FAIL", n); traceback.print_exc()
    sys.exit(1 if f else 0)
