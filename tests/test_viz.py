import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from engine import models as M, viz


def test_animation_payload():
    p = M.WellTestParams(boundary="fault", d1=400.0)
    pl = viz.animation_payload(p, 1500.0, 200.0, 150.0, n_frames=31, n_radii=30)
    json.dumps(pl)                                      # serialisable
    assert len(pl["prof"]) == 31 and len(pl["prof"][0]) == 30
    allp = np.array(pl["prof"])
    assert np.isfinite(allp).all() and allp.max() <= p.pi + 1e-6
    # pressure recovers during the buildup at every radius near the well
    assert allp[-1, 0] > allp[0, 0]
    assert pl["rmax"] > p.d1


def test_design_summary():
    p = M.WellTestParams(boundary="fault", d1=500.0)
    s = viz.design_summary(p, 1000.0, 100.0, 200.0)
    assert s["recommended_shutin_hr"] >= s["t_boundary_hr"]
    assert s["ri_end_ft"] > 100


def test_model_curves_derivative_plateau():
    p = M.WellTestParams(k=50.0, S=0.0, C=0.005)
    c = viz.model_curves(p, 1000.0, 1000.0, 500.0, n=100)
    s = M.p_scale(p) * 1000.0
    mid = c["der"][~np.isnan(c["der"])]
    # derivative w.r.t. Agarwal time at radial flow ~ 0.5 * scale
    assert abs(np.nanmedian(mid[40:70]) / (0.5 * s) - 1) < 0.1


if __name__ == "__main__":
    import traceback
    f = 0
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            try: fn(); print("PASS", n)
            except Exception: f += 1; print("FAIL", n); traceback.print_exc()
    sys.exit(1 if f else 0)
