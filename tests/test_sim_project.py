import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from engine import project, sim_presets as SP, calibrate as CAL
from engine.blackoil import BlackOilSim


def test_simulator_results_round_trip():
    p = SP.presets()["DST near a gas cap"]
    cfg = p["cfg"]; cfg.nr = 8; cfg.nz = 6
    res = BlackOilSim(cfg, p["pvt"], p["rp"]).run(p["schedule"], 60.0, snap_times=[10.0, 60.0])
    sim = {"res": res, "cfg": cfg, "rp": p["rp"], "pvt": BlackOilSim(cfg, p["pvt"], p["rp"]).pvt_cfg,
           "sched": p["schedule"], "t_end": 60.0, "name": "x"}
    obs = CAL.synthetic_observations(res, np.geomspace(1.0, 60.0, 15), 0.1)
    fit = {"obs": obs, "curve": res, "info": {"free": ["k"], "start": {"k": 1.0}, "fitted": {"k": 2.0},
                                              "rms": 0.3, "nfev": 3, "trace": [1, 2]}, "cfg": cfg, "rp": p["rp"]}
    text = project.dumps({"well": "W"}, {"sim": sim, "sim_fit": fit})
    _, back = project.loads(text)
    b = back["sim"]
    assert np.allclose(b["res"]["pwf"], res["pwf"]) and len(b["res"]["snapshots"]) == len(res["snapshots"])
    assert b["cfg"].goc == cfg.goc and b["cfg"].nr == 8 and b["rp"].swc == p["rp"].swc
    assert np.allclose(b["res"]["snapshots"][-1]["sg"], res["snapshots"][-1]["sg"])
    assert back["sim_fit"]["info"]["fitted"]["k"] == 2.0
    assert np.allclose(back["sim_fit"]["obs"]["pwf"], obs["pwf"])
    assert np.isnan(back["sim_fit"]["obs"]["GOR"]).any()          # shut-in NaNs survive


if __name__ == "__main__":
    import traceback
    try: test_simulator_results_round_trip(); print("PASS test_simulator_results_round_trip")
    except Exception: traceback.print_exc(); sys.exit(1)
