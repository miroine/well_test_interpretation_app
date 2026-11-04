import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from engine import samples as S, models as M, analysis as A


def test_ci_contains_truth_and_widens_with_noise():
    p, q, tp, d = S.oil_buildup(noise=0.0)
    quiet = S.oil_buildup(noise=0.05, seed=1)
    noisy = S.oil_buildup(noise=0.6, seed=1)
    cis = []
    for (pp, qq, tt, dd) in (quiet, noisy):
        p0 = M.WellTestParams(**{**pp.to_dict(), "k": 40.0, "S": 1.0, "C": 0.02})
        pf, info = A.fit_buildup(p0, qq, tt, dd["dt"], dd["pws"], ["k", "S", "C"], pwf_shutin=dd["pwf_at_shutin"])
        ci = info["ci"]["k"]
        assert ci["lo"] <= pp.k * 1.1 and ci["hi"] >= pp.k * 0.9
        cis.append(ci["rel_sd"])
    assert cis[1] > cis[0]


def test_sensitivity_curves_ordering():
    p, q, tp, d = S.oil_buildup()
    sens = A.sensitivity_curves(p, q, tp, d["dt"], ["k", "S"])
    assert len(sens["runs"]) == 4
    lo = [r for r in sens["runs"] if r["name"] == "k" and r["factor"] == 0.5][0]["curves"]["dp"][-1]
    hi = [r for r in sens["runs"] if r["name"] == "k" and r["factor"] == 2.0]  [0]["curves"]["dp"][-1]
    assert lo > sens["base"]["dp"][-1] > hi


def test_monte_carlo_brackets_truth():
    p, q, tp, d = S.oil_buildup(noise=0.3, seed=2)
    p0 = M.WellTestParams(**{**p.to_dict(), "k": 50.0, "S": 2.0, "C": 0.02})
    pf, info = A.fit_buildup(p0, q, tp, d["dt"], d["pws"], ["k", "S", "C"], pwf_shutin=d["pwf_at_shutin"])
    mc = A.monte_carlo(pf, q, tp, d["dt"], d["pws"], ["k", "S", "C"], pwf_shutin=d["pwf_at_shutin"], n=8)
    s = mc["summary"]["k"]
    assert s["p10"] <= s["p90"] and abs(s["p50"] / p.k - 1) < 0.1


if __name__ == "__main__":
    import traceback
    f = 0
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            try: fn(); print("PASS", n)
            except Exception: f += 1; print("FAIL", n); traceback.print_exc()
    sys.exit(1 if f else 0)
