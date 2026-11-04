import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from engine import samples as S, models as M, analysis as A


def test_dual_porosity_screening_recovers_omega_lambda():
    p, q, tp, d = S.dual_porosity_buildup()
    p0 = M.WellTestParams(**{**p.to_dict(), "k": 20.0, "S": 2.0, "C": 0.03, "omega": 0.2, "lam": 1e-5})
    pf, info = A.fit_buildup(p0, q, tp, d["dt"], d["pws"], ["k", "S", "C", "omega", "lam"])
    assert abs(pf.omega / p.omega - 1) < 0.25, pf.omega
    assert 0.5 < pf.lam / p.lam < 2.0, pf.lam
    assert abs(pf.k / p.k - 1) < 0.15, pf.k          # noise-limited (derivative noise)
    assert info["rms"] < 0.6, info


if __name__ == "__main__":
    import traceback
    try: test_dual_porosity_screening_recovers_omega_lambda(); print("PASS test_dual_porosity_screening_recovers_omega_lambda")
    except Exception: traceback.print_exc(); sys.exit(1)
