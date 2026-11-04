"""03 Type curve - Gringarten match for a homogeneous reservoir."""
from __future__ import annotations

import numpy as np
import streamlit as st
from .theme import takeaway

from engine import models as M
from engine import typecurve as T
from engine import units as U
from . import figures as F
from . import helpers as H


@st.cache_data(show_spinner=False)
def _family():
    return T.family()


def render(meta):
    S = st.session_state
    system, ver = meta["units"], S["ver"]
    ie = S.get("interp_eng")
    if not ie:
        st.info("Load a dataset and set the known inputs in tab 02 first.")
        return
    p = H.params_from_dict(ie["params"])
    q = ie["q"]
    d = ie["diag"]
    st.caption("Type-curve matching assumes a homogeneous, infinite-acting reservoir with wellbore storage "
               "and skin. Use the diagnostic plot first to confirm the derivative shows that behaviour.")
    st.caption("Solid lines: dimensionless pressure; dashed lines: derivative. Both approach pD = tD/CD "
               "in early wellbore storage. The family uses CD=100 for G>=100 and a zero-skin effective-radius "
               "reference CD=G below 100 to avoid nonphysical negative lumped-skin transients. "
               "G=CD exp(2S) similarity is approximate at finite-radius early times; low-G matches "
               "should be checked against the actual well model in Interpretation.")
    fam = _family()
    c1, c2 = st.columns([1, 3])
    if c1.button("Auto-match", type="primary"):
        ok = np.isfinite(d["der"]) & (d["der"] > 0) & (d["dp"] > 0)
        with st.spinner("Matching..."):
            m = T.auto_match(d["ta"][ok], d["dp"][ok], d["der"][ok])
        S["tc_match"] = m
    m = S.get("tc_match")
    if not m:
        c2.info("Press Auto-match to find the curve and shifts, then refine with the sliders.")
        st.plotly_chart(F.typecurve_figure(fam), use_container_width=True)
        return
    a, b, g = st.columns(3)
    lsx = a.slider("Time shift (log10 of tD/CD per hr)", -6.0, 6.0, float(np.log10(m["sx"])), 0.02,
                   key=f"tc_sx_{ver}_{round(m['sx'], 6)}")
    lsy = b.slider("Pressure shift (log10 of pD per psi)", -6.0, 3.0, float(np.log10(m["sy"])), 0.02,
                   key=f"tc_sy_{ver}_{round(m['sy'], 6)}")
    lG = g.slider("Curve CD e^(2S) (log10)", -1.0, 60.0, float(np.log10(m["G"])), 0.1,
                  key=f"tc_G_{ver}_{round(m['G'], 3)}")
    sx, sy, G = 10 ** lsx, 10 ** lsy, 10 ** lG
    res = T.interpret_match(sx, sy, G, q, p.B, p.mu, p.h, p.phi, p.ct, p.rw)
    takeaway(f"Type-curve match gives k = {res['k']:.3g} md and skin {res['S']:+.2f} (CD = {res['CD']:.3g}).", "Match result")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Permeability k", f"{res['k']:.4g} md")
    r2.metric("Wellbore storage C", f"{U.from_field(res['C'], 'storage', system):.3g} {U.label('storage', system)}")
    r3.metric("CD", f"{res['CD']:.3g}")
    r4.metric("Skin S", f"{res['S']:.3g}")
    st.caption(f"Automatic match misfit (log-space RMS): {m['rms']:.3g}")
    fig = F.typecurve_figure(fam, sx * d["ta"], sy * d["dp"], sy * d["der"], highlight=G)
    st.plotly_chart(fig, use_container_width=True)
