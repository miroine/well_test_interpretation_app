"""Built-in sample tests (synthetic, anonymised). Math only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import models as M
from . import gas as G


def oil_buildup(noise=0.0, seed=3):
    p = M.WellTestParams(k=65.0, h=40.0, phi=0.17, mu=0.9, ct=1.8e-5, B=1.25,
                         rw=0.354, pi=3850.0, C=0.012, S=3.5)
    q, tp = 1800.0, 240.0
    d = M.synthetic_buildup(p, q, tp, 120.0, n=160, noise_psi=noise, seed=seed)
    return p, q, tp, d


def noisy_gauge():
    return oil_buildup(noise=0.35, seed=7)


def faulted_oil_buildup():
    p = M.WellTestParams(k=45.0, h=60.0, phi=0.2, mu=1.1, ct=1.4e-5, B=1.2,
                         rw=0.354, pi=4200.0, C=0.02, S=1.0,
                         boundary="fault", d1=420.0)
    q, tp = 2200.0, 300.0
    d = M.synthetic_buildup(p, q, tp, 300.0, n=170, noise_psi=0.05, seed=11)
    return p, q, tp, d


def dual_porosity_buildup():
    p = M.WellTestParams(k=30.0, h=80.0, phi=0.1, mu=0.8, ct=2.0e-5, B=1.3,
                         rw=0.354, pi=5000.0, C=0.01, S=-1.0,
                         model="dual_pss", omega=0.03, lam=2e-7)
    q, tp = 3000.0, 400.0
    d = M.synthetic_buildup(p, q, tp, 400.0, n=170, noise_psi=0.05, seed=5)
    return p, q, tp, d


def gas_buildup():
    """Gas well in normalized pseudo-pressure (psi-equivalent)."""
    T_F, gg, pi_ = 220.0, 0.68, 5200.0
    gp = G.gas_props(pi_, T_F, gg)
    q_mscfd = 8000.0
    q_rb = q_mscfd * 1000.0 * gp["Bg"]                 # rb/d reservoir equivalent
    p = M.WellTestParams(k=12.0, h=60.0, phi=0.12, mu=gp["mu"], ct=gp["cg"] + 4e-6,
                         B=1.0, rw=0.354, pi=pi_, C=0.005, S=-1.5)
    tp = 168.0
    d = M.synthetic_buildup(p, q_rb, tp, 100.0, n=150, noise_psi=0.1, seed=2)
    return p, q_rb, tp, d, {"T_F": T_F, "gamma_g": gg, "q_mscfd": q_mscfd}


SAMPLES = {
    "Oil buildup": oil_buildup,
    "Noisy gauge": noisy_gauge,
    "Oil buildup near sealing fault": faulted_oil_buildup,
    "Dual-porosity oil buildup": dual_porosity_buildup,
}


def to_frame(d: dict) -> pd.DataFrame:
    return pd.DataFrame({"dt_hr": d["dt"], "pws_psi": d["pws"]})


# ---------------------------------------------------------------- multi-rate
def multirate_record():
    """Four-step multi-rate record ending in a final shut-in (e.g. formation-tester style)."""
    from . import multirate as MR
    p = M.WellTestParams(k=80.0, h=35.0, phi=0.2, mu=0.8, ct=1.6e-5, B=1.3, rw=0.354, pi=4400.0,
                         C=0.004, S=1.5)
    sched = [(0.0, 800.0), (48.0, 1600.0), (110.0, 1200.0), (170.0, 2400.0), (230.0, 0.0)]
    t, pw = MR.synthetic_multirate(p, sched, 330.0, n=420, noise_psi=0.03, seed=9)
    return p, sched, t, pw


def dst_record():
    """DST-style sequence: initial flow, initial buildup, main flow, final buildup."""
    from . import multirate as MR
    p = M.WellTestParams(k=25.0, h=30.0, phi=0.15, mu=0.6, ct=1.8e-5, B=1.2, rw=0.354, pi=5200.0,
                         C=0.003, S=-0.5)
    sched = [(0.0, 900.0), (1.0, 0.0), (4.0, 1200.0), (16.0, 0.0)]
    t, pw = MR.synthetic_multirate(p, sched, 40.0, n=360, noise_psi=0.03, seed=13)
    return p, sched, t, pw


MULTIRATE_SAMPLES = {"Multi-rate record (formation-tester style)": multirate_record,
                     "DST (two flow / two buildup periods)": dst_record}
