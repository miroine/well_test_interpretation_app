import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from engine import opm_deck as OD, sim_presets as SP
from engine.blackoil import BlackOilSim


def _sim():
    p = SP.presets()["Gas coning (rate step-up)"]
    cfg = p["cfg"]; cfg.nr = 8; cfg.nz = 6; cfg.C = 0.0
    return BlackOilSim(cfg, p["pvt"], p["rp"]), p


def test_deck_structure_and_geometry():
    s, p = _sim()
    d = OD.build_deck(s, p["schedule"], p["t_end"])
    for kw in ("RUNSPEC", "DIMENS", "1 8 6", "STONE2", "PORV", "TRANY", "TRANZ", "SWOF", "SGOF", "PVCO", "PVDG",
               "EQUIL", "COMPDAT", "WCONPROD", "TSTEP", "END"):
        assert kw in d, kw
    # PORV sums back to the model pore volume
    block = d.split("PORV\n")[1].split("/")[0].split()
    assert abs(sum(float(x) for x in block) - s.pv0.sum() / 5.615) / (s.pv0.sum() / 5.615) < 1e-6
    # TSTEP total equals the scheduled time
    ts = [float(x) for x in d.split("TSTEP\n")[1].split("/")[0].split()]
    assert ts and all(x > 0 for x in ts)
    tot = 0.0
    for chunk in d.split("TSTEP\n")[1:]:
        tot += sum(float(x) for x in chunk.split("/")[0].split())
    assert abs(tot - p["t_end"] / 24.0) < 1e-6, (tot, p["t_end"] / 24.0)
    # well connection factors match the model
    comp = d.split("COMPDAT\n")[1].split("\n/")[0].strip().splitlines()
    assert len(comp) == len(s.well_cells)
    assert abs(float(comp[0].split()[7]) - s.WI[0]) / s.WI[0] < 1e-6


def test_csv_parse_and_compare_self_consistency():
    s, p = _sim()
    res = s.run(p["schedule"], 300.0)
    t_d = np.geomspace(0.1, 300.0, 40) / 24.0
    import io
    from scipy.interpolate import interp1d
    gor = np.interp(t_d * 24, res["t"], np.nan_to_num(res["GOR"], nan=0.0)) / 1000.0
    csv = "TIME,WBHP:PROD,WGOR:PROD,WWCT:PROD,WOPR:PROD\n" + "\n".join(
        f"{a},{np.interp(a * 24, res['t'], res['pwf'])},{g},{np.interp(a * 24, res['t'], np.nan_to_num(res['WC'], nan=0.0))},{np.interp(a * 24, res['t'], res['qo'])}"
        for a, g in zip(t_d, gor))
    fl = OD.parse_flow_csv(csv)
    assert abs(fl["t"][-1] - 300.0) < 1e-6 and np.isfinite(fl["pwf"]).all()
    cmp = OD.compare(res, fl)
    assert cmp["pwf_rms"] < 1e-6 and cmp["GOR_rel"] < 1e-6


if __name__ == "__main__":
    import traceback
    f = 0
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            try: fn(); print("PASS", n)
            except Exception: f += 1; print("FAIL", n); traceback.print_exc()
    sys.exit(1 if f else 0)
