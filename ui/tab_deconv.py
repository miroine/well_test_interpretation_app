"""05 Deconvolution - unit-rate response from a variable-rate record."""
from __future__ import annotations

import numpy as np
import streamlit as st
from .theme import takeaway

from engine import deconv as DC
from engine import models as M
from engine import units as U
from . import figures as F
from . import helpers as H


def render(meta):
    S = st.session_state
    system, ver = meta["units"], S["ver"]
    pu, ru = U.label("pressure", system), U.label("rate", system)
    mr = S.get("mr")
    if not mr:
        st.info("Load a multi-rate or DST record in the 'Multi-rate' tab first. "
                "Deconvolution needs a variable-rate history; a single buildup carries too little information.")
        return
    method = st.radio("Method", ["Nonlinear (von Schroeter-type, positive derivative by construction)",
                                 "Linear (bounded least squares)"], horizontal=True, key=f"dc_method_{ver}")
    st.caption("Regularised deconvolution. The nonlinear form fits ln(derivative) on log-spaced nodes; the linear form "
               "solves a bounded least-squares problem. It returns the "
               "unit-rate response including the initial pressure, so late-time boundaries can appear "
               "beyond the duration of any single flow period.")
    a, b, c = st.columns(3)
    nb = a.slider("Resolution (log bins)", 20, 100, 50, 5, key=f"dc_nb_{ver}")
    lam = b.slider("Smoothing", 0.0, 5.0, 0.5, 0.1, key=f"dc_lam_{ver}")
    est_pi = c.checkbox("Estimate initial pressure", value=True, key=f"dc_pi_{ver}")
    pi_known = float(U.from_field(H.params_from_dict(mr["guess"]).pi, "pressure", system))
    pi_in = st.number_input(f"Initial pressure if known [{pu}]", min_value=1.0, value=float(max(pi_known, 1.0)),
                            step=10.0, key=f"dc_pik_{system}_{ver}", disabled=est_pi)
    if st.button("Run deconvolution", type="primary"):
        sched = [tuple(s) for s in mr["schedule"]]
        with st.spinner("Deconvolving..."):
            pik = None if est_pi else float(U.to_field(pi_in, "pressure", system))
            if method.startswith("Nonlinear"):
                r = DC.deconvolve_nonlinear(sched, mr["t"], mr["pwf"], pi=pik, n_nodes=max(20, nb // 1), lam=lam)
            else:
                r = DC.deconvolve(sched, mr["t"], mr["pwf"], pi=pik, n_bins=nb, lam=lam)
        S["deconv"] = r
        st.rerun()
    r = S.get("deconv")
    if not r:
        return
    norm = U.from_field(1.0, "pressure", system) / U.from_field(1.0, "rate", system)
    takeaway(f"Deconvolution recovers an initial pressure of {U.from_field(r['pi_est'], 'pressure', system):.0f} {U.label('pressure', system)} (fit RMS {U.from_field(r['rms'], 'pressure', system):.2g}).", "Deconvolution result")
    m1, m2 = st.columns(2)
    m1.metric("Initial pressure", f"{U.from_field(r['pi_est'], 'pressure', system):.1f} {pu}")
    m2.metric("Pressure-fit RMS", f"{U.from_field(r['rms'], 'pressure', system):.3g} {pu}")
    models = []
    if mr.get("truth") or mr.get("fit"):
        src = mr["fit"]["params"] if mr.get("fit") else mr["truth"]
        pm = H.params_from_dict(src)
        tt = np.logspace(np.log10(r["t"].min()), np.log10(r["t"].max()), 100)
        dp, der = M.unit_rate_response(pm, tt)
        models.append({"name": "regression fit" if mr.get("fit") else "true model", "x": tt, "dp": dp * norm, "der": der * norm})
    st.plotly_chart(F.diag_figure(
        "Deconvolved unit-rate response", "time [hr]", f"pressure per unit rate [{pu} per {ru}]",
        data={"x": r["t"], "dp": r["g"] * norm, "der": r["deriv"] * norm}, models=models), use_container_width=True)
    st.caption("Points: deconvolved response (blue) and derivative (orange). Lines: model for comparison. "
               "Treat features near the ends of the time range with caution; they are the least constrained.")
