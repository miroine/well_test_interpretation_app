import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import copy
import numpy as np
from engine import calibrate as CAL, sim_presets as SP
from engine.blackoil import BlackOilSim


def test_presets_run():
    for name, p in SP.presets().items():
        r = BlackOilSim(p["cfg"], p["pvt"], p["rp"]).run(p["schedule"], min(p["t_end"], 40.0))
        assert np.all(np.isfinite(r["pwf"])), name


def test_calibration_recovers_k_and_skin():
    p = SP.presets()["Single-phase check (undersaturated oil)"]
    cfg = copy.deepcopy(p["cfg"]); cfg.nr = 12
    truth = CAL.simulate(cfg, p["pvt"], p["rp"], p["schedule"], 400.0)
    t_obs = np.geomspace(0.05, 400.0, 60)
    obs = CAL.synthetic_observations(truth, t_obs, noise_psi=0.05, seed=1)
    start = copy.deepcopy(cfg); start.k = 120.0; start.skin = 8.0
    c, r, info = CAL.calibrate(start, p["pvt"], p["rp"], p["schedule"], obs, ["k", "skin"], max_nfev=30)
    assert abs(c.k / cfg.k - 1) < 0.08, info
    assert abs(c.skin - cfg.skin) < 1.0, info
    assert info["rms"] < 2.0, info


def test_calibration_to_gor_and_water_cut_with_shutin():
    p = SP.presets()["Gas coning (rate step-up)"]
    cfg = copy.deepcopy(p["cfg"]); cfg.nr = 8; cfg.nz = 6
    sched = [(0, 1000.0), (40.0, 3500.0), (80.0, 0.0)]
    truth = CAL.simulate(cfg, p["pvt"], p["rp"], sched, 100.0)
    t_obs = np.geomspace(1.0, 100.0, 30)
    obs = CAL.synthetic_observations(truth, t_obs, noise_psi=0.1, seed=3)
    assert np.isnan(obs["GOR"]).any() and np.isfinite(obs["GOR"]).any()
    start = copy.deepcopy(cfg); start.kv_kh = cfg.kv_kh * 3.0
    c, r, info = CAL.calibrate(start, p["pvt"], p["rp"], sched, obs, ["kv_kh"], use=("pwf", "GOR", "WC"), max_nfev=6)
    assert abs(np.log(c.kv_kh / cfg.kv_kh)) < abs(np.log(start.kv_kh / cfg.kv_kh)), info
    assert info["rms"] < 3.0, info


if __name__ == "__main__":
    import traceback
    f = 0
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            try: fn(); print("PASS", n)
            except Exception: f += 1; print("FAIL", n); traceback.print_exc()
    sys.exit(1 if f else 0)
