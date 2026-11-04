"""07 IPR - oil (Vogel / composite) and gas (Fetkovich, Forchheimer/LIT)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from .theme import takeaway

from engine import ipr
from engine import units as U
from . import figures as F
from .inputs import radius_input


def _oil(system, ver):
    pu, ru = U.label("pressure", system), U.label("rate", system)
    f = lambda v, k: float(U.from_field(v, k, system))
    g = lambda v, k: float(U.to_field(v, k, system))
    c1, c2, c3 = st.columns(3)
    pr = g(c1.number_input(f"Average reservoir pressure [{pu}]", min_value=1.0, value=f(3500.0, "pressure"),
                           step=f(50.0, "pressure"), key=f"ipr_pr_{system}_{ver}"), "pressure")
    pb = g(c2.number_input(f"Bubble-point pressure [{pu}]", min_value=1.0, value=f(2500.0, "pressure"),
                           step=f(50.0, "pressure"), key=f"ipr_pb_{system}_{ver}"), "pressure")
    c4, c5 = st.columns(2)
    pw_t = g(c4.number_input(f"Test flowing pressure [{pu}]", min_value=0.0, value=f(2000.0, "pressure"),
                             step=f(50.0, "pressure"), key=f"ipr_pwt_{system}_{ver}"), "pressure")
    q_t = g(c5.number_input(f"Test rate [{ru}]", min_value=0.1, value=f(900.0, "rate"),
                            step=f(10.0, "rate"), key=f"ipr_qt_{system}_{ver}"), "rate")
    if pw_t >= pr:
        st.warning("Test flowing pressure must be below the reservoir pressure.")
        return
    J = ipr.composite_J_from_test(pr, pb, q_t, pw_t)
    qmax = ipr.composite_qmax(pr, pb, J)
    pw = np.linspace(0.0, pr, 120)
    q = ipr.composite_ipr(pr, pb, J, pw)
    j_display = J * U.from_field(1.0, "rate", system) / U.from_field(1.0, "pressure", system)
    takeaway(f"Absolute open flow is {U.from_field(qmax, 'rate', system):,.0f} {ru}; productivity index J = {j_display:.2g} {ru}/{pu}.", "IPR result")
    m1, m2, m3 = st.columns(3)
    m1.metric("Productivity index J", f"{j_display:.3g} {ru}/{pu}")
    m2.metric("Absolute open flow (pwf = 0)", f"{U.from_field(qmax, 'rate', system):,.0f} {ru}")
    qv = ipr.vogel_qmax(q_t, pw_t, pr)
    m3.metric("Pure Vogel qmax (pb = pr)", f"{U.from_field(qv, 'rate', system):,.0f} {ru}")
    fig = F.line_figure("Oil IPR (composite linear + Vogel)", f"rate [{ru}]", f"flowing pressure [{pu}]",
                        [("composite", U.from_field(q, "rate", system), U.from_field(pw, "pressure", system)),
                         ("Vogel (saturated)", U.from_field(ipr.vogel_rate(pr, qv, pw), "rate", system),
                          U.from_field(pw, "pressure", system))])
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter(x=[U.from_field(q_t, "rate", system)], y=[U.from_field(pw_t, "pressure", system)],
                             mode="markers", name="test point", marker=dict(color=F.RED, size=11)))
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Productivity index from reservoir properties (pseudo-steady state)"):
        e1, e2, e3, e4 = st.columns(4)
        k = e1.number_input("k [md]", min_value=0.01, value=50.0, key=f"ipr_k_{ver}")
        h = e2.number_input(f"h [{U.label('length', system)}]", min_value=0.1, value=f(50.0, "length"), key=f"ipr_h_{system}_{ver}")
        re_ = e3.number_input(f"re [{U.label('length', system)}]", min_value=1.0, value=f(1000.0, "length"), key=f"ipr_re_{system}_{ver}")
        sk = e4.number_input("Skin", min_value=-6.0, max_value=100.0, value=0.0, key=f"ipr_s_{ver}")
        e5, e6, e7 = st.columns(3)
        mu = e5.number_input("Viscosity [cp]", min_value=0.01, value=1.0, key=f"ipr_mu_{ver}")
        B = e6.number_input("Bo [rb/stb]", min_value=0.5, value=1.2, key=f"ipr_B_{ver}")
        rw = radius_input("ipr", .354, system, ver, e7)
        Jr = ipr.pi_pss(k, g(h, "length"), mu, B, g(re_, "length"), rw, sk)
        jr_display = Jr * U.from_field(1.0, "rate", system) / U.from_field(1.0, "pressure", system)
        st.metric(f"Theoretical J [{ru}/{pu}]", f"{jr_display:.3g}", delta=f"{(J / Jr - 1) * 100:+.0f}% vs test-derived", delta_color="off")


def _gas(system, ver):
    pu, ru = U.label("pressure", system), U.label("gasrate", system)
    pr_d = st.number_input(f"Average reservoir pressure [{pu}]", min_value=1.0,
                           value=float(U.from_field(3000.0, "pressure", system)),
                           key=f"gipr_pr_{system}_{ver}")
    pr = float(U.to_field(pr_d, "pressure", system))
    pw0 = np.array([2800.0, 2400.0, 2000.0, 1600.0])
    q0 = ipr.lit_rate(3000.0, 120.0, 0.02, pw0)
    df0 = pd.DataFrame({f"pwf [{pu}]": U.from_field(pw0, "pressure", system),
                        f"q [{ru}]": U.from_field(q0, "gasrate", system)})
    df = st.data_editor(df0, num_rows="dynamic", hide_index=True, key=f"gipr_tbl_{system}_{ver}")
    pw = U.to_field(pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(), "pressure", system)
    q = U.to_field(pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(), "gasrate", system)
    ok = np.isfinite(pw) & np.isfinite(q) & (q > 0) & (pw < pr)
    if ok.sum() < 3:
        st.info("Enter at least three stabilised points with pwf below the reservoir pressure.")
        return
    pw, q = pw[ok], q[ok]
    C, n, r2f = ipr.fetkovich_fit(pr, pw, q)
    a, b, r2l = ipr.lit_fit(pr, pw, q)
    takeaway(f"Fetkovich exponent n = {n:.2f}; absolute open flow about {U.from_field(ipr.aof_fetkovich(pr, C, n), 'gasrate', system):,.0f} {ru}.", "Deliverability result")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fetkovich exponent n", f"{n:.3f}", help="Between 0.5 (fully turbulent) and 1.0 (laminar).")
    c2.metric("Fetkovich AOF", f"{U.from_field(ipr.aof_fetkovich(pr, C, n), 'gasrate', system):,.0f} {ru}")
    c3.metric("LIT / Forchheimer AOF", f"{U.from_field(ipr.aof_lit(pr, a, b), 'gasrate', system):,.0f} {ru}")
    c4.metric("Fit quality R2 (F / LIT)", f"{r2f:.3f} / {r2l:.3f}")
    if not (0.5 <= n <= 1.05):
        st.warning("Fetkovich n is outside 0.5-1.0: check the test points (stabilisation, units).")
    if b < 0 or a < 0:
        st.warning("Negative Forchheimer coefficient: the data are not consistent with a stabilised deliverability test.")
    pgrid = np.linspace(0.0, pr * 0.999, 120)
    xs = lambda v: U.from_field(v, "gasrate", system)
    fig = F.line_figure("Gas IPR", f"rate [{ru}]", f"flowing pressure [{pu}]",
                        [("Fetkovich", xs(ipr.fetkovich_rate(pr, C, n, pgrid)), U.from_field(pgrid, "pressure", system)),
                         ("LIT / Forchheimer", xs(ipr.lit_rate(pr, a, b, pgrid)), U.from_field(pgrid, "pressure", system))])
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter(x=xs(q), y=U.from_field(pw, "pressure", system), mode="markers", name="test points",
                             marker=dict(color=F.RED, size=10)))
    st.plotly_chart(fig, use_container_width=True)


def render(meta):
    S = st.session_state
    st.caption("Inflow performance from stabilised test points. Both fluids use field-unit engines; "
               "values display in the selected unit system.")
    if meta["fluid"] == "Oil":
        _oil(meta["units"], S["ver"])
    else:
        _gas(meta["units"], S["ver"])
