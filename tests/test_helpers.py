import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from engine import models as M, analysis as A
from ui import helpers as H


def test_gas_equivalent_matches_manual_scale():
    p = M.WellTestParams(k=12.0, h=60.0, phi=0.12, pi=5200.0, rw=0.354, C=0.005, S=-1.0)
    pg, q_rb, gp = H.gas_equivalent(p, 220.0, 0.68, 8000.0)
    assert 0.0005 < gp["Bg"] < 0.0015 and 0.015 < gp["mu"] < 0.04
    # engine delta-p scale must equal the textbook 141.2 q Bg mu / (k h) form
    assert abs(M.p_scale(pg) * q_rb - 141.2 * q_rb * gp["mu"] / (12.0 * 60.0)) < 1e-9


def test_free_params():
    assert H.fit_free_params("homogeneous", "none") == ["k", "S", "C"]
    assert "d1" in H.fit_free_params("homogeneous", "fault")
    assert {"omega", "lam"} <= set(H.fit_free_params("dual_pss", "none"))


def test_every_spec_param_exists_on_dataclass():
    p = M.WellTestParams()
    for spec in [H.BASE_SPEC] + list(H.MODEL_SPEC.values()) + list(H.BOUNDARY_SPEC.values()):
        for s in spec:
            assert hasattr(p, s[0]), s[0]


def test_all_models_and_boundaries_evaluate():
    for model in H.MODEL_LABELS:
        for b in H.BOUNDARY_LABELS:
            if model == "composite" and b != "none":
                continue
            p = M.WellTestParams(model=model, boundary=b)
            dp, der = M.unit_rate_response(p, np.logspace(-2, 2, 6))
            assert np.isfinite(dp).all() and np.isfinite(der).all(), (model, b)


if __name__ == "__main__":
    import traceback
    f = 0
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            try: fn(); print("PASS", n)
            except Exception: f += 1; print("FAIL", n); traceback.print_exc()
    sys.exit(1 if f else 0)
