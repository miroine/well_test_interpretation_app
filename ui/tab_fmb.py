"""08 FMB - flowing material balance (tank-model regression), oil and gas."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from .theme import takeaway

from engine import fmb
from engine import units as U
from . import figures as F


@st.cache_resource(show_spinner=False)
def _tank(pi, T_F, gamma):
    return fmb.GasTank(pi, T_F, gamma, n=300)


def _cum(t, q):
    return np.concatenate([[0.0], np.cumsum(q[:-1] * np.diff(t))])


def _load_csv(system, kind, ver):
    pu, ru = U.label("pressure", system), U.label(kind, system)
    up = st.file_uploader("CSV: time [days], rate, flowing pressure", type=["csv", "txt"], key=f"fmb_up_{ver}")
    if up is None:
        return None
    try:
        df = pd.read_csv(up, sep=None, engine="python")
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return None
    cols = list(df.columns)
    a, b, c = st.columns(3)
    tc = a.selectbox("Time [days]", cols, index=0)
    qc = b.selectbox(f"Rate [{ru}]", cols, index=min(1, len(cols) - 1))
    pc = c.selectbox(f"Flowing pressure [{pu}]", cols, index=min(2, len(cols) - 1))
    if not st.button("Use this file", type="primary"):
        return None
    d = df[[tc, qc, pc]].apply(pd.to_numeric, errors="coerce").dropna().sort_values(tc)
    t = d[tc].to_numpy()
    q = U.to_field(d[qc].to_numpy(), kind, system)
    return t, q, U.to_field(d[pc].to_numpy(), "pressure", system)


def render(meta):
    S = st.session_state
    system, fluid, ver = meta["units"], meta["fluid"], S["ver"]
    pu = U.label("pressure", system)
    kind = "gasrate" if fluid == "Gas" else "rate"
    ru = U.label(kind, system)
    st.caption("Flowing material balance by regression: for a trial original-in-place the average reservoir "
               "pressure follows from the material balance; pseudo-steady inflow q = J (p_avg - pwf) then fixes "
               "the in-place volume and J together. Valid in boundary-dominated flow with a volumetric tank.")
    src = st.radio("Production data", ["Demo (synthetic)", "Upload CSV"], horizontal=True, key=f"fmb_src_{ver}")
    f = lambda v, k: float(U.from_field(v, k, system))
    g = lambda v, k: float(U.to_field(v, k, system))
    if fluid == "Gas":
        c1, c2, c3 = st.columns(3)
        pi = g(c1.number_input(f"Initial pressure [{pu}]", min_value=100.0, value=f(5000.0, "pressure"),
                               key=f"fmb_pi_{system}_{ver}"), "pressure")
        T = g(c2.number_input(f"Temperature [{U.label('temp', system)}]", value=float(U.from_field(200.0, "temp", system)),
                              key=f"fmb_T_{system}_{ver}"), "temp")
        gg = c3.number_input("Gas gravity", 0.55, 1.2, 0.7, 0.01, key=f"fmb_g_{ver}")
        with st.spinner("Preparing gas properties..."):
            tank = _tank(float(pi), float(T), float(gg))
    else:
        c1, c2, c3, c4 = st.columns(4)
        pi = g(c1.number_input(f"Initial pressure [{pu}]", min_value=100.0, value=f(4500.0, "pressure"),
                               key=f"fmb_pi_{system}_{ver}"), "pressure")
        ce = g(c2.number_input(f"Effective compressibility [{U.label('ct', system)}]", min_value=1e-9, value=f(1.8e-5, "ct"),
                               format="%.3e", key=f"fmb_ce_{system}_{ver}"), "ct")
        Boi = c3.number_input("Boi [rb/stb]", 0.5, 5.0, 1.25, 0.01, key=f"fmb_Boi_{ver}")
        Bo = c4.number_input("Bo [rb/stb]", 0.5, 5.0, 1.25, 0.01, key=f"fmb_Bo_{ver}")
    data = None
    if src == "Demo (synthetic)":
        if st.button("Generate demo production history", type="primary"):
            rng = np.random.default_rng(3)
            if fluid == "Gas":
                t, q, pw, _ = fmb.simulate_gas(tank, 4.0e6, 0.9, lambda t: 0.25 * pi, 1500.0, dt=10.0)
            else:
                t, q, pw, _ = fmb.simulate_oil(2.0e7, 4.0, ce, Boi, Bo, pi, lambda t: 0.33 * pi, 1500.0, dt=10.0)
            S["fmb"] = {"fluid": fluid, "t": t, "q": q * (1 + rng.normal(0, 0.01, len(q))), "pwf": pw, "fit": None}
            st.rerun()
    else:
        data = _load_csv(system, kind, ver)
        if data is not None:
            S["fmb"] = {"fluid": fluid, "t": data[0], "q": data[1], "pwf": data[2], "fit": None}
            st.rerun()
    d = S.get("fmb")
    if not d or d["fluid"] != fluid:
        st.info("Generate the demo history or upload a CSV to begin.")
        return
    t, q, pw = (np.asarray(d[k], float) for k in ("t", "q", "pwf"))
    cum = _cum(t, q)
    if st.button("Fit in-place volume and J", type="primary"):
        with st.spinner("Fitting..."):
            if fluid == "Gas":
                r = fmb.fit_gas_fmb(tank, cum, q, pw, G0=2.0 * max(cum.max(), 1.0))
            else:
                r = fmb.fit_oil_fmb(pi, ce, Boi, Bo, cum, q, pw, N0=2.0 * max(cum.max(), 1.0))
        d["fit"] = r
        S["fmb"] = d
        st.rerun()
    r = d.get("fit")
    qx = lambda v: U.from_field(v, kind, system)
    figs = []
    if fluid == "Gas":
        line = fmb.gas_fmb_line(tank, cum, q, pw)
        series = [("normalised rate q / (pn(pi) - pn(pwf))", cum[q > 0], line["qn"])]
        fig1 = F.line_figure("Flowing material balance", "cumulative gas [Mscf]", "normalised rate [Mscf/d per psi]", series)
        if r:
            pa = tank.p_avg(cum, r["G"])
            qm = r["J"] * (tank.npp.forward(pa) - tank.npp.forward(pw))
            m1, m2, m3 = st.columns(3)
            conv = 28.316847 if system == "Metric" else 1.0
            unit = "MM Sm³" if system == "Metric" else "Bscf"
            takeaway(f"Gas in place is {r['G'] * conv / 1e6:.2f} {unit}.", "Material-balance result")
            m1.metric("Gas in place G", f"{r['G'] * conv / 1e6:.3f} {unit}")
            m2.metric("Productivity J", f"{r['J']:.4g} Mscf/d/psi")
            m3.metric("Straight-line G (rough)", f"{line['G_line'] * conv / 1e6:.3f} {unit}" if np.isfinite(line["G_line"]) else "n/a")
            st.caption(f"Regression RMS (relative to median rate): {r['rms']:.3g}. "
                       f"Recovery to date: {100 * cum.max() / r['G']:.1f} % of G.")
    else:
        fig1 = F.line_figure("Rate vs cumulative oil", "cumulative oil [stb]", f"rate [{ru}]", [("data", cum, qx(q))])
        if r:
            pa = pi - cum * Bo / (r["N"] * ce * Boi)
            qm = r["J"] * (pa - pw)
            m1, m2, m3 = st.columns(3)
            conv = 0.158987295 if system == "Metric" else 1.0
            unit = "MM m³" if system == "Metric" else "MMstb"
            takeaway(f"Oil in place is {r['N'] * conv / 1e6:.2f} {unit}; {100 * cum.max() / r['N']:.1f}% recovered to date.", "Material-balance result")
            m1.metric("Oil in place N", f"{r['N'] * conv / 1e6:.3f} {unit}")
            m2.metric("Productivity J", f"{r['J']:.4g} stb/d/psi")
            m3.metric("Recovery to date", f"{100 * cum.max() / r['N']:.2f} %")
            st.caption(f"Regression RMS (relative to median rate): {r['rms']:.3g}.")
    st.plotly_chart(fig1, use_container_width=True)
    series2 = [("data", t, qx(q))]
    if r:
        series2.append(("tank model", t, qx(np.maximum(qm, 0.0))))
    st.plotly_chart(F.line_figure("Rate history and model", "time [days]", f"rate [{ru}]", series2), use_container_width=True)
