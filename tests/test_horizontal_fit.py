import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from engine import models as M, analysis as A


def test_fit_recovers_horizontal():
    p = M.WellTestParams(model="horizontal", k=40, h=60, Lw=1500, kv_kh=0.2, C=0.01, S=1.0)
    tp, q = 300.0, 1000.0
    dt = np.geomspace(0.02, 300, 45)
    sched = M.simple_buildup_schedule(q, tp)
    pw = M.pwf_history(p, np.concatenate([[tp * (1 - 1e-9)], tp + dt]), sched)
    assert np.all(np.diff(pw[1:]) > -1e-6)
    pws = pw[1:]
    p0 = M.WellTestParams(**{**p.to_dict(), "k": 70.0, "S": 3.0, "C": 0.02, "Lw": 1100.0})
    pf, info = A.fit_buildup(p0, q, tp, dt, pws, ["k", "S", "C", "Lw"], pwf_shutin=pw[0])
    assert abs(pf.k / p.k - 1) < 0.2 and abs(pf.Lw / p.Lw - 1) < 0.35, (pf.k, pf.Lw, pf.S)


if __name__ == "__main__":
    test_fit_recovers_horizontal(); print("PASS")
