"""00 Test design - predict a buildup before running it."""
from __future__ import annotations

import numpy as np
import streamlit as st
from .theme import takeaway

from engine import models as M
from engine import units as U
from engine import viz
from engine import design_export
from . import figures as F
from . import helpers as H
from . import inputs
from . import session
from . import design_timeline
from . import model_schematic
from . import design_eclipse


@st.cache_data(show_spinner=False)
def _designed_test(params, q, tp, dt_end):
    return viz.designed_test(H.params_from_dict(params), q, tp, dt_end)


def render(meta):
    S = st.session_state
    system, fluid, ver = meta["units"], meta["fluid"], S["ver"]
    pu = U.label("pressure", system)
    left, right = st.columns([1, 2], gap="large")
    with left:
        saved = S.get("design_eng") or {}
        if saved.get("fluid", fluid) != fluid:
            saved = {}
        st.markdown("#### Reservoir and well")
        base = H.params_from_dict(S["design_params"]) if S.get("design_params") else M.WellTestParams()
        p = inputs.params_form("design", base, system, fluid, ver)
        gas = inputs.gas_form("design", system, ver, defaults=saved.get("gas")) if fluid == "Gas" else None
        st.markdown("#### Test sequence")
        q_user = inputs.rate_input("design", fluid, system, ver, float(saved.get("q_user", 6000.0 if fluid == "Gas" else 1500.0)))
        tp = st.number_input("Flow period before shut-in [hr]", 1.0, 1e5, float(saved.get("tp", 200.0)), 10.0,
                             key=f"design_tp_{ver}")
        dt_end = st.number_input("Planned shut-in duration [hr]", 0.1, 1e5, float(saved.get("dt_end", 100.0)), 10.0,
                                 key=f"design_dtend_{ver}")
    S["design_params"] = p.to_dict()
    p_eng, q_eng, gp = H.engine_view(p, fluid, gas, q_user)
    S["design_eng"] = {"params": p_eng.to_dict(), "q": q_eng, "tp": tp, "dt_end": dt_end,
                       "q_user": q_user, "gas": gas, "fluid": fluid}
    summ = viz.design_summary(p_eng, q_eng, tp, dt_end)
    curves = _designed_test(p_eng.to_dict(), q_eng, tp, dt_end)
    cur = curves["buildup"]
    f = lambda arr: U.from_field(arr, "pressure", system)

    with right:
        takeaway(f"Plan at least {summ['recommended_shutin_hr']:.3g} h of shut-in; wellbore storage fades after about {summ['t_wbs_end_hr']:.3g} h.", "Design conclusion")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Storage ends ~", f"{summ['t_wbs_end_hr']:.3g} hr")
        c2.metric("Radius of investigation", H.metric_str(summ["ri_end_ft"], "length", system, "{:,.0f}"))
        c3.metric("Pressure scale (q B mu / k h)", f"{f(summ['pressure_scale_psi']):.3g} {pu}")
        c4.metric("Recommended shut-in", f"{summ['recommended_shutin_hr']:.3g} hr")
        if dt_end < summ["recommended_shutin_hr"]:
            st.warning(
                f"Planned shut-in ({dt_end:g} hr) is shorter than the recommended "
                f"{summ['recommended_shutin_hr']:.3g} hr: radial flow"
                + (" and the boundary response" if "t_boundary_hr" in summ else "")
                + " may not be reached.")
        else:
            st.success("Planned shut-in is long enough to reach radial flow"
                       + (" and see the boundary." if "t_boundary_hr" in summ else "."))
        if gp is not None:
            st.caption(f"Gas at initial pressure: Z = {gp['Z']:.3f}, viscosity = {gp['mu']:.4f} cp, "
                       f"Bg = {gp['Bg'] * 1000:.3f} rb/Mscf. Pressures shown are normalized pseudo-pressure.")
        st.markdown("##### Reservoir model schematic")
        model_schematic.render(p, curves, tp, dt_end, q_user, system, fluid, ver)
        st.markdown("##### Designed rate and pressure")
        st.caption("Drag Start shut-in to adjust flowing time (keeping shut-in duration); "
                   "drag End test to adjust shut-in duration. Release to recompute the model.")
        design_timeline.render(curves, tp, dt_end, q_user, system, fluid, ver)
        st.markdown("##### Superpose drawdown and buildup")
        selected = []
        dcol, bcol = st.columns(2)
        for col, phase in ((dcol, "Drawdown"), (bcol, "Buildup")):
            for quantity in ("pressure change", "derivative"):
                name = f"{phase} {quantity}"
                if col.checkbox(name, value=True, key=f"design_overlay_{phase}_{quantity}"):
                    selected.append(name)
        alignment = st.radio("Diagnostic time alignment", ["Agarwal equivalent time", "Elapsed time"],
                             horizontal=True, key="design_alignment")
        st.caption("Drawdown: initial minus flowing pressure, derivative d(dp)/d ln(t). "
                   "Buildup: pressure rise since shut-in, derivative against the selected time axis. "
                   "Equivalent time = tp x dt / (tp + dt); it aligns radial flow but does not force "
                   "storage or boundary responses to coincide. Non-positive values are omitted on log axes.")
        if selected and q_eng > 0:
            st.plotly_chart(F.design_overlay(curves, pu, f, selected,
                                            equivalent=alignment == "Agarwal equivalent time"),
                            use_container_width=True)
        else:
            st.info("Select at least one curve and a positive production rate to show the diagnostic overlay.")
        a, b = st.columns(2)
        with a:
            st.plotly_chart(F.horner_figure("Predicted Horner plot", cur["ht"], f(cur["pws"]),
                                            f"pws [{pu}]"), use_container_width=True)
        with b:
            t = np.logspace(-2, np.log10(tp + dt_end), 80)
            ri = U.from_field(M.radius_of_investigation(p_eng, t), "length", system)
            vl = []
            if p.boundary != "none":
                vl.append((float(U.from_field(p.d1, "length", system)), "boundary"))
            st.plotly_chart(F.line_figure(
                "Radius of investigation vs time", "time [hr]", f"radius [{U.label('length', system)}]",
                [("r_inv", t, ri)], xlog=True, ylog=True, hlines=vl), use_container_width=True)
        st.markdown("##### Use this design")
        st.download_button("Download designed test (CSV)",
                           design_export.design_csv(curves, q_user, tp, dt_end, system, fluid),
                           file_name="designed_test.csv", mime="text/csv", key="design_csv_export")
        st.caption("CSV includes both sides of the shut-in event at the same time: flowing rate, then zero rate. "
                   "Pressure follows the current unit selection; gas uses normalized pseudo-pressure.")
        with st.expander("Export Eclipse schedule INCLUDE"):
            st.caption("For an existing, completed producer. Insert inside the host deck's SCHEDULE section "
                       "at test start. Match the host deck units to the sidebar. No reservoir geometry or "
                       "predicted pressure is imposed; TSTEP is in days. Add simulator timestep/report "
                       "controls to resolve the transient. The well remains SHUT at test end.")
            eclipse_well = st.text_input("Eclipse well name (1-8 characters)", "PROD", key="design_eclipse_well")
            bhp = st.number_input(f"Eclipse minimum BHP [{pu}]", min_value=0.001,
                                  value=float(U.from_field(14.7, "pressure", system)),
                                  key=f"design_eclipse_bhp_{system}")
            st.caption("Rate control is ORAT for oil or GRAT for gas, using surface rates. "
                       "The minimum BHP constraint may limit the achieved rate.")
            try:
                schedule = design_export.eclipse_schedule(eclipse_well, q_user, tp, dt_end, system, fluid,
                                                           float(U.to_field(bhp, "pressure", system)))
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.download_button("Download Eclipse schedule (.INC)", schedule,
                                   file_name=f"{eclipse_well}_test_schedule.INC", mime="text/plain",
                                   key="design_eclipse_export")
                try:
                    comparison = design_eclipse.comparison_package(
                        eclipse_well, curves, q_user, tp, dt_end, system, fluid,
                        float(U.to_field(bhp, "pressure", system)), p_eng.pi, gas)
                except ValueError as exc:
                    st.error(f"Cannot export BHP comparison: {exc}")
                else:
                    st.download_button(
                        "Download schedule + predicted BHP comparison (.ZIP)", comparison,
                        file_name=f"{eclipse_well}_schedule_comparison.zip", mime="application/zip",
                        key="design_eclipse_comparison")
                st.caption("Comparison ZIP: sampled rate-controlled schedule, predicted_bhp.csv, "
                           "WBHP summary INCLUDE and alignment instructions. These pressures are synthetic, "
                           "not field measurements or imposed BHP controls. Gas BHP is obtained by inverse "
                           "pseudo-pressure mapping; out-of-range values are blank and flagged.")
        design_eclipse.render(p_eng, curves, q_user, tp, dt_end, fluid, gas, eclipse_well,
                              float(U.to_field(bhp, "pressure", system)), system)
        n1, n2 = st.columns([1, 2])
        noise = n1.number_input(f"Gauge noise (1 sigma) [{pu}]", 0.0, 50.0, 0.1, 0.05, key=f"design_noise_{ver}")
        if n2.button("Generate synthetic test from this design", type="primary"):
            d = M.synthetic_buildup(p_eng, q_eng, tp, dt_end, n=140,
                                    noise_psi=float(U.to_field(noise, "pressure", system)), seed=1)
            session.replace_dataset(S, {"name": "Synthetic from test design", "tp": tp, "q_user": q_user,
                            "dt": d["dt"], "pws": d["pws"], "pwf_shutin": d["pwf_at_shutin"],
                            "fluid": fluid, "truth": p.to_dict(),
                            "guess": H.clone(p, k=p.k * 0.5, S=0.0, C=p.C * 0.5, model="homogeneous",
                                             boundary="none").to_dict(),
                            "gas": gas,
                            "history": {
                                "t": np.concatenate([[0.0], curves["drawdown"]["t"], [tp], tp + d["dt"]]),
                                "p": np.concatenate([[p_eng.pi], curves["drawdown"]["pressure"], [d["pwf_at_shutin"]], d["pws"]]),
                                "rate": np.concatenate([np.full(1 + len(curves["drawdown"]["t"]), q_user),
                                                        np.zeros(1 + len(d["dt"]))])}})
            st.rerun()
    return curves
