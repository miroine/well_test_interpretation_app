"""Ready-made simulator scenarios (synthetic, anonymised)."""
from __future__ import annotations

from .blackoil import SimConfig, RelPerm, PVTConfig


def _preset(cfg, schedule, t_end, desc, rp=None, pvt=None):
    return {"cfg": cfg, "rp": rp or RelPerm(), "pvt": pvt, "schedule": schedule,
            "t_end": t_end, "description": desc}


def presets():
    return {
        "Single-phase check (undersaturated oil)": _preset(
            SimConfig(nr=18, nz=4, h=40, k=65, kv_kh=1.0, phi=0.17, p_datum=3850, Rsi=300,
                      C=0.012, skin=3.5, re=5000),
            [(0, 500.0), (200.0, 0.0)], 400.0,
            "Drawdown then buildup; should reproduce the analytical homogeneous model."),
        "DST near a gas cap": _preset(
            SimConfig(nr=16, nz=10, h=100, k=120, kv_kh=0.15, phi=0.2, top_depth=8000,
                      p_datum=3600, datum_depth=8070, goc=8035, woc=None, Rsi=450, re=3500,
                      perf_from=0.45, perf_to=0.75, C=0.02, skin=1.0),
            [(0, 1500.0), (6.0, 0.0), (20.0, 2500.0), (60.0, 0.0)], 150.0,
            "Flow / buildup / flow / buildup with the perforations 35-70 ft below a gas cap: "
            "late-time derivative shows gas-cap pressure support, GOR rises in the main flow."),
        "Gas coning (rate step-up)": _preset(
            SimConfig(nr=16, nz=12, h=120, k=300, kv_kh=0.5, phi=0.22, top_depth=8000,
                      p_datum=3500, datum_depth=8060, goc=8030, woc=8090, Rsi=500, re=2500,
                      perf_from=0.30, perf_to=0.50, outer_pv_mult=50),
            [(0, 1000.0), (200.0, 4000.0), (700.0, 0.0)], 900.0,
            "Thin oil column between a gas cap and a water leg; the rate step-up cones gas and "
            "water, the final shut-in shows the cone collapsing."),
        "Water coning (bottom water)": _preset(
            SimConfig(nr=16, nz=10, h=100, k=250, kv_kh=0.4, phi=0.2, top_depth=8000,
                      p_datum=3400, datum_depth=8030, goc=None, woc=8080, Rsi=250, re=2500,
                      perf_from=0.0, perf_to=0.3, outer_pv_mult=100),
            [(0, 3500.0)], 800.0,
            "Perforations at the top of the oil column above an active aquifer; water cut climbs "
            "with time at constant rate."),
    }
