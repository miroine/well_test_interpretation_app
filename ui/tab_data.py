"""01 Data and QC - load, clean and choose the buildup to interpret."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from .theme import takeaway

from engine import models as M
from engine import qc
from engine import tabular
from engine import samples as SMP
from engine import units as U
from . import figures as F
from . import helpers as H
from . import session
from . import import_data


def _load_sample(name):
    S = st.session_state
    if name == "Gas well (normalized pseudo-pressure)":
        p, q_rb, tp, d, meta = SMP.gas_buildup()
        q_user, fluid = meta["q_mscfd"], "Gas"
        gas = {"T_F": meta["T_F"], "gamma_g": meta["gamma_g"], "cf": 4e-6}
    else:
        p, q_user, tp, d = SMP.SAMPLES[name]()
        fluid, gas = "Oil", None
    guess = H.clone(p, k=p.k * 0.5, S=0.0, C=p.C * 0.5, model="homogeneous", boundary="none")
    session.replace_dataset(S, {"name": name, "tp": tp, "q_user": q_user, "dt": d["dt"], "pws": d["pws"],
                    "pwf_shutin": d["pwf_at_shutin"], "fluid": fluid, "truth": p.to_dict(),
                    "guess": guess.to_dict(), "gas": gas})
    S["design_params"] = None
    st.rerun()


def render(meta):
    S = st.session_state
    system, fluid = meta["units"], meta["fluid"]
    pu = U.label("pressure", system)
    src = st.radio("Data source", ["Sample test", "Upload files", "Current dataset"], horizontal=True,
                   key=f"data_src_{S['ver']}")
    if src == "Sample test":
        names = list(SMP.SAMPLES) + ["Gas well (normalized pseudo-pressure)"]
        pick = st.selectbox("Sample", names, key="data_sample")
        st.caption("These samples are synthetic. For rate-history and DST examples, use Multi-rate; "
                   "for gas-cap and coning scenarios, use Simulator.")
        if st.button("Load sample", type="primary"):
            _load_sample(pick)
    elif src == "Upload files":
        full = st.checkbox("Upload full test history (pressure and rate)", key="qc_upload_full")
        uploaded = import_data.file_table("Upload pressure data (text or Excel)", "qc_pressure_upload")
        if uploaded is not None:
            filename, df = uploaded
            tcol, pcol, time_opts = import_data.pressure_columns(df, "qc_import", fluid, system, full)
            if not full and time_opts["unit"] == "Calendar date/time":
                st.caption("For buildup-only pressure, enter the shut-in timestamp as the shared origin above.")
            if fluid == "Gas":
                st.caption("Gas buildup interpretation expects normalized pseudo-pressure (psi-equivalent or bar-equivalent), "
                           "not raw measured BHP. Convert gauge BHP before importing.")
            kind = "gasrate" if fluid == "Gas" else "rate"
            rate = None
            if full:
                st.caption("Time is elapsed hours from test start, not time since shut-in. "
                           "Choose the start of the final shut-in below. Interpretation assumes the last "
                           "constant pre-shut-in rate; variable-rate interpretation belongs in Multi-rate.")
                rate = import_data.rate_input(df, "qc_rates", system, kind, time_opts["origin"])
                rate, aligned_tp = import_data.alignment_preview(
                    df, tcol, pcol, time_opts, rate, "qc_import", system, kind, buildup=True)
            else:
                aligned_tp = None
            c3, c4, c5 = st.columns(3)
            if aligned_tp is not None:
                tp = float(aligned_tp)
                c3.metric("Flow period before shut-in [hr]", f"{tp:g}")
                c3.caption("Linked to the selected shut-in marker above. Drag the marker to change it.")
            else:
                tp = c3.number_input("Flow period before shut-in [hr]", 0.1, 1e6, 100.0, 10.0)
            q_disp = 0.0
            if full:
                c4.caption("Pre-shut-in rate is taken from the last non-zero recorded rate.")
            else:
                q_disp = c4.number_input(f"Rate before shut-in [{U.label(kind, system)}]", 0.0, 1e9, 1000.0)
            fallback = "last recorded flowing point" if full else "first point"
            pw0 = c5.number_input(f"Last flowing pressure [{pu}] (0 = use {fallback})", 0.0, 1e6, 0.0)
            if st.button("Use this file", type="primary"):
                try:
                    history = None
                    if full and rate is None:
                        raise ValueError("Provide a valid rate file or rate table before loading the full history.")
                    mapped = import_data.mapped_frame(df, tcol, pcol, time_opts, rate)
                    if full:
                        history, issues = qc.clean_history(mapped, "time", "pressure", "rate")
                        history["p"] = U.to_field(history["p"], "pressure", system)
                        if "column" in rate:
                            history["rate"] = U.to_field(history["rate"], kind, system)
                        flowing = history["rate"][(history["t"] <= tp) & (np.abs(history["rate"]) > 1e-10)]
                        if not len(flowing) or flowing[-1] <= 0:
                            raise ValueError("A positive production rate must be recorded before shut-in.")
                        q_disp = float(U.from_field(flowing[-1], kind, system))
                        if "schedule" in rate:
                            q_disp = float(U.from_field(tabular.final_shutin_rate(rate["schedule"], tp), kind, system))
                        if not np.allclose(flowing, flowing[-1]):
                            issues.append("Variable flowing rates: buildup interpretation assumes the last pre-shut-in rate. "
                                          "Use Multi-rate to account for the complete rate history.")
                        if "schedule" in rate:
                            schedule_flow = [q for t, q in rate["schedule"] if t < tp and q > 0]
                            if schedule_flow and not np.allclose(schedule_flow, schedule_flow[-1]):
                                issues.append("Separate rate schedule has variable rates; use Multi-rate for complete superposition.")
                        mask = history["t"] > tp
                        if np.any(np.abs(history["rate"][mask]) > 1e-10):
                            raise ValueError("Non-zero rates occur after the chosen shut-in time. Choose the final shut-in or use Multi-rate.")
                        d = pd.DataFrame({"t": history["t"][mask] - tp,
                                          "p": U.from_field(history["p"][mask], "pressure", system)})
                    else:
                        d, issues = qc.clean_gauge(mapped, "time", "pressure")
                    n_nonpositive = int((d["t"] <= 0).sum())
                    d = d[d["t"] > 0]
                    qc.validate_buildup(d["t"].to_numpy(), d["p"].to_numpy())
                except ValueError as e:
                    st.error(f"Cannot use this file: {e}")
                    return
                if n_nonpositive:
                    issues.append(f"{n_nonpositive} non-positive shut-in times removed")
                p_field = U.to_field(d["p"].to_numpy(), "pressure", system)
                preceding = history["p"][(history["t"] <= tp) & (np.abs(history["rate"]) > 1e-10)] if history is not None else []
                pwf0 = float(U.to_field(pw0, "pressure", system)) if pw0 > 0 else float(
                    preceding[-1] if len(preceding) else p_field[0])
                base = H.params_from_dict(S["design_params"]) if S.get("design_params") else M.WellTestParams()
                session.replace_dataset(S, {"name": filename, "tp": float(tp), "q_user": float(U.to_field(q_disp, kind, system)),
                                "dt": d["t"].to_numpy(), "pws": p_field, "pwf_shutin": pwf0, "fluid": fluid,
                                "truth": None, "guess": base.to_dict(),
                                "gas": {"T_F": 220.0, "gamma_g": 0.68, "cf": 4e-6} if fluid == "Gas" else None,
                                "issues": issues, "pressure_kind": "normalized_pseudo_pressure" if fluid == "Gas" else "pressure",
                                **({"history": history} if history is not None else {}),
                                **({"rate_schedule": rate["schedule"], "rate_end": rate["end"]}
                                   if rate and "schedule" in rate else {})})
                st.rerun()

    ds = S.get("dataset")
    if ds is None:
        st.info("Load a sample, upload text/Excel pressure data, or generate a synthetic test in tab 00.")
        return
    st.markdown(f"#### Active dataset: {ds['name']}")
    if ds.get("fluid") != fluid:
        st.warning(f"This dataset is {ds['fluid'].lower()} but the sidebar fluid is {fluid.lower()}: "
                   "switch the fluid in the sidebar before interpreting.")
    for msg in ds.get("issues", []):
        st.caption(f"Cleaning: {msg}")

    dt, pws = np.asarray(ds["dt"]), np.asarray(ds["pws"])
    out = qc.hampel_outliers(pws)
    noise = qc.noise_estimate(pws)
    takeaway(f"{len(dt)} points, {int(out.sum())} outliers flagged, gauge noise {U.from_field(noise, 'pressure', system):.2g} {U.label('pressure', system)}.", "Data quality")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Points", f"{len(dt)}")
    c2.metric("Outliers flagged", f"{int(out.sum())}")
    c3.metric("Noise (1 sigma)", f"{U.from_field(noise, 'pressure', system):.3g} {pu}")
    c4.metric("Time span", f"{dt.min():.2g} - {dt.max():.3g} hr")
    for w in qc.resolution_flags(dt, pws):
        st.caption(f"Warning: {w}")

    history = qc.response_source(ds)
    manual_key = f"qc_manual_{S['ver']}"
    if manual_key not in S:
        S[manual_key] = list(ds.get("qc_excluded", []))
    excluded = S[manual_key]
    st.markdown("##### Full response and manual point selection")
    if ds.get("history") is None and ds.get("qc_source", {}).get("history") is None:
        st.caption("This dataset contains buildup pressure only; drawdown pressure was not recorded. "
                   "The rate schedule below is specified, not measured. Upload full history to inspect both periods.")
        rate_t = [0.0, ds["tp"], ds["tp"], float(history["t"][-1])]
        rate_q = [ds["q_user"], ds["q_user"], 0.0, 0.0]
        kind = "gasrate" if ds["fluid"] == "Gas" else "rate"
        st.plotly_chart(F.line_figure("Specified rate schedule", "Elapsed test time [hr]",
                        f"Rate [{U.label(kind, system)}]", [("Specified rate", rate_t, U.from_field(np.array(rate_q), kind, system))]),
                        use_container_width=True)
    st.caption("Box/lasso-select pressure or rate points, then mark them for removal. "
               "Use the toolbar to zoom/pan; double-click to reset. All original readings remain visible. "
               "Point IDs are shared by pressure and rate; exclusions never rewrite the recorded rate history.")
    fig = F.qc_response(history, excluded, ds["tp"], system, ds["fluid"],
                        pseudo=ds["fluid"] == "Gas")
    event = st.plotly_chart(fig, use_container_width=True, key=f"qc_response_{S['ver']}",
                            on_select="rerun", selection_mode=("points", "box", "lasso"))
    points = event.get("selection", {}).get("points", []) if event else []
    selected = sorted({int(p["customdata"]) for p in points if p.get("customdata") is not None})
    st.caption(f"{len(selected)} selected; {len(excluded)} marked/applied exclusions.")
    if st.button("Mark selected points for removal", key=f"qc_mark_{S['ver']}", disabled=not selected):
        S[manual_key] = sorted(set(excluded) | set(selected))
        st.rerun()
    with st.expander("Select or restore individual point IDs"):
        st.multiselect("Point IDs to exclude", list(range(len(history["t"]))), key=manual_key)
        st.dataframe(pd.DataFrame({"Point ID": np.arange(len(history["t"])), "Elapsed time [hr]": history["t"],
                                  f"Pressure [{pu}]": U.from_field(history["p"], "pressure", system),
                                  "Marked for removal": np.isin(np.arange(len(history["t"])), S[manual_key])}),
                     hide_index=True)
    a, b = st.columns(2)
    window = ds.get("qc_window", [float(history["t"][0]), float(history["t"][-1])])
    start = a.number_input("Keep from elapsed time [hr]", value=float(window[0]), key=f"qc_start_{S['ver']}")
    end = b.number_input("Keep through elapsed time [hr]", value=float(window[1]), key=f"qc_end_{S['ver']}")
    pressure_range = None
    if st.checkbox("Filter by pressure range", key=f"qc_pressure_filter_{S['ver']}"):
        a, b = st.columns(2)
        low = a.number_input(f"Minimum kept pressure [{pu}]", value=float(U.from_field(history["p"].min(), "pressure", system)),
                             key=f"qc_low_{S['ver']}_{system}")
        high = b.number_input(f"Maximum kept pressure [{pu}]", value=float(U.from_field(history["p"].max(), "pressure", system)),
                              key=f"qc_high_{S['ver']}_{system}")
        pressure_range = U.to_field(np.array([low, high]), "pressure", system)
    a, b = st.columns(2)
    drop = a.checkbox("Remove flagged outliers", value=bool(out.any()), key=f"qc_drop_{S['ver']}")
    ppc = b.slider("Log-time resampling (points per log cycle, 0 = off)", 0, 40, 0, key=f"qc_ppc_{S['ver']}")
    st.caption("Click Apply QC to use these filters in Interpretation and fitting. "
               "The full-response QC chart retains originals for inspection; Interpretation shows only kept pressure readings.")
    if st.button("Apply QC to the active dataset"):
        try:
            updated = qc.filter_response(ds, S[manual_key], start, end, drop, ppc, pressure_range)
        except ValueError as e:
            st.error(f"QC was not applied: {e}")
            return
        session.replace_dataset(S, updated)
        st.rerun()
    if ds.get("qc_source") is not None and st.button("Restore original response (undo all QC)"):
        session.replace_dataset(S, qc.restore_response(ds))
        st.rerun()

    fig = F.line_figure("Shut-in pressure", "shut-in time [hr]", f"pressure [{pu}]",
                        [("pws", dt, U.from_field(pws, "pressure", system))], xlog=True)
    if out.any():
        import plotly.graph_objects as go
        fig.add_trace(go.Scatter(x=dt[out], y=U.from_field(pws[out], "pressure", system), mode="markers",
                                 name="outliers", marker=dict(color=F.RED, size=9, symbol="x")))
    st.plotly_chart(fig, use_container_width=True)
    if ds.get("truth"):
        with st.expander("Reveal the true parameters used to generate this sample"):
            t = ds["truth"]
            st.write({k: t[k] for k in ("model", "k", "S", "C", "boundary", "d1", "omega", "lam") if k in t})
