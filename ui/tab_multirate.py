"""Multi-rate and DST records: rate history, superposition diagnostics, regression."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from .theme import takeaway

from engine import models as M
from engine import analysis as A
from engine import multirate as MR
from engine import samples as SMP
from engine import units as U
from . import figures as F
from . import helpers as H
from . import inputs
from . import multirate_timeline
from .calculation import FitProgress
from . import import_data
from engine import qc


def _load_sample(name):
    S = st.session_state
    p, sched, t, pw = SMP.MULTIRATE_SAMPLES[name]()
    guess = H.clone(p, k=p.k * 0.5, S=0.0, C=p.C * 2.0, model="homogeneous", boundary="none")
    S["mr"] = {"name": name, "schedule": sched, "t": t, "pwf": pw, "guess": guess.to_dict(),
               "truth": p.to_dict(), "fit": None}
    S["deconv"] = None
    S["ver"] += 1
    st.rerun()


def render(meta):
    S = st.session_state
    system, ver = meta["units"], S["ver"]
    pu, ru = U.label("pressure", system), U.label("rate", system)
    if meta["fluid"] == "Gas":
        st.info("Multi-rate analysis is available for oil in this release (switch Fluid to Oil in the sidebar).")
        return
    src = st.radio("Data source", ["Sample", "Upload pressure and rates", "Active Data & QC record"], horizontal=True,
                   key=f"mr_src_{ver}")
    if src == "Sample":
        pick = st.selectbox("Sample", list(SMP.MULTIRATE_SAMPLES), key="mr_sample")
        if st.button("Load multi-rate sample", type="primary"):
            _load_sample(pick)
    elif src == "Active Data & QC record":
        st.caption("Replace the Multi-rate record with the loaded full pressure/rate history. "
                   "Current Interpretation inputs are used as starting parameters; no Saphir settings are inferred.")
        if st.button("Use active pressure and rate history", key=f"mr_from_dataset_{ver}"):
            try:
                working = S.get("interp_model")
                params = H.params_from_dict(working["params"] if working else (S.get("dataset") or {}).get("guess", {}))
                if not S.get("dataset"):
                    raise ValueError("Load a measured full history in Data & QC first.")
                S["mr"] = MR.from_dataset(S["dataset"], params)
            except ValueError as exc:
                st.error(f"Cannot use active record: {exc}")
            else:
                S["deconv"] = None
                S["ver"] += 1
                st.rerun()
    else:
        uploaded = import_data.file_table("Upload pressure data (text or Excel)", "mr_up")
        if uploaded is not None:
            filename, df = uploaded
            tc, pc, time_opts = import_data.pressure_columns(df, "mr_import", "Oil", system, full=True)
            st.caption("Pressure and rate times share the same test origin. Numeric times are not shifted automatically.")
            rate = import_data.rate_input(df, "mr_rates", system, origin=time_opts["origin"])
            rate, _ = import_data.alignment_preview(df, tc, pc, time_opts, rate, "mr_import", system)
            if st.button("Use this file", type="primary"):
                try:
                    if rate is None:
                        raise ValueError("Provide a valid rate history first.")
                    mapped = import_data.mapped_frame(df, tc, pc, time_opts, rate)
                    history, issues = qc.clean_history(mapped, "time", "pressure", "rate")
                    t = history["t"]
                    if np.any(np.diff(t) <= 0):
                        raise ValueError("Multi-rate pressure times must be unique and strictly increasing.")
                    if "column" in rate:
                        if t[0] != 0:
                            raise ValueError("An in-file rate history must include time zero. Otherwise supply a separate rate schedule starting at zero.")
                        sched = MR.clean_schedule(MR.schedule_from_series(
                            t, U.to_field(history["rate"], "rate", system)))
                    else:
                        sched = rate["schedule"]
                    if not any(q > 0 for _, q in sched):
                        raise ValueError("At least one positive production rate is required.")
                    keep = t > 0
                    qc.validate_buildup(t[keep], history["p"][keep])
                except ValueError as exc:
                    st.error(f"Cannot use this file: {exc}")
                    return
                base = H.params_from_dict(S["design_params"]) if S.get("design_params") else M.WellTestParams()
                S["mr"] = {"name": filename, "schedule": sched, "t": t[keep],
                           "pwf": U.to_field(history["p"][keep], "pressure", system),
                           "guess": base.to_dict(), "truth": None, "fit": None}
                S["deconv"] = None
                S["ver"] += 1
                st.rerun()
    mr = S.get("mr")
    if not mr:
        st.info("Load a sample or upload a record. Rates are detected as a piecewise-constant history.")
        return
    sched = [tuple(s) for s in mr["schedule"]]
    t, pw = np.asarray(mr["t"], float), np.asarray(mr["pwf"], float)
    st.markdown(f"#### {mr['name']}")
    rows = [{"Period": i + 1, "Start [hr]": round(s[0], 4), f"Rate [{ru}]": round(float(U.from_field(s[1], 'rate', system)), 4)}
            for i, s in enumerate(sched)]
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        p = inputs.params_form("mr", H.params_from_dict(mr["guess"]), system, "Oil", ver)
        if st.button("Generate whole-history model", key=f"mr_generate_{ver}"):
            try:
                pressure = M.pwf_history(p, t, sched)
                if not np.isfinite(pressure).all():
                    raise ValueError("The chosen parameters produced non-finite pressure.")
            except (ValueError, np.linalg.LinAlgError) as exc:
                st.error(f"Cannot generate whole-history model: {exc}")
            else:
                mr["working_model"] = p.to_dict()
                S["mr"] = mr
        if p.model == "fracture_dual_pss":
            st.info("Combined fracture/dual-porosity model. Match fluid properties and the lambda/skin "
                    "definitions before comparing software. This is not a certified Saphir replica.")
        st.caption("Initial pressure is used for absolute history matching. A successful optimizer exit "
                   "does not establish model correctness or agreement with another simulator.")
    with right:
        ts, dq = MR.rate_changes(sched)
        periods = [i for i in range(len(ts)) if abs(dq[i]) > 1e-12]
        labels = {i: f"period {i + 1} (starts {ts[i]:g} hr, rate change {U.from_field(dq[i], 'rate', system):.4g} {ru})" for i in periods}
        per = st.selectbox("Diagnose period", periods, index=len(periods) - 1, format_func=labels.get, key=f"mr_per_{ver}")
        multirate_timeline.render(mr, system, per, ver)
        Ls = st.slider("Derivative smoothing L", 0.0, 0.5, 0.1, 0.05, key=f"mr_L_{ver}")
        view = st.radio("Diagnostic view", ["Period pressure change and derivative",
                        "Rate-superposition diagnostic"], key=f"mr_diag_view_{ver}")
        overlay = st.multiselect("Periods to superpose", periods, default=periods, format_func=labels.get,
                                 key=f"mr_overlay_{ver}")
        normalized = st.checkbox("Normalize period response by absolute rate change", key=f"mr_normalize_{ver}")
        local = view == "Period pressure change and derivative"
        norm = U.from_field(1.0, "pressure", system)
        if normalized or not local:
            norm /= U.from_field(1.0, "rate", system)
        ylabel = f"Pressure change and derivative [{pu}" + (f" per {ru}]" if normalized or not local else "]")
        fig = F._base("Multi-rate period diagnostics", "Elapsed time since period start [hr]", ylabel)
        import plotly.graph_objects as go
        fit = None if mr.get("working_model") else mr.get("fit")
        if fit:
            try:
                M.validate_skin(H.params_from_dict(fit["params"]))
            except ValueError as exc:
                st.error(f"Cannot plot the saved fit: {exc} Run regression with corrected inputs.")
                fit = None
        model_pw, pf = None, None
        if fit:
            pf = H.params_from_dict(fit["params"])
            model_pw = M.pwf_history(pf, t, sched)
        elif mr.get("working_model"):
            pf = H.params_from_dict(mr["working_model"])
            model_pw = M.pwf_history(pf, t, sched)
        if model_pw is not None:
            history_fig = F.line_figure("Whole-history pressure: data and model", "Elapsed test time [hr]",
                                       f"Pressure [{pu}]",
                                       [("Recorded pressure", t, U.from_field(pw, "pressure", system)),
                                        ("Last fit" if fit else "Generated model", t, U.from_field(model_pw, "pressure", system))])
            st.plotly_chart(history_fig, use_container_width=True)
            st.caption(f"Whole-history pressure RMS: {U.from_field(np.sqrt(np.mean((model_pw - pw) ** 2)), 'pressure', system):.4g} {pu}.")
        hidden = 0
        for color_index, period in enumerate(overlay):
            color = F.PALETTE[color_index % len(F.PALETTE)]
            for is_model in (False, True):
                if is_model and model_pw is None:
                    continue
                readings, initial = (model_pw, pf.pi) if is_model else (pw, p.pi)
                try:
                    if local:
                        boundary = None
                        if is_model:
                            boundary = initial if period == 0 else float(M.pwf_history(
                                pf, np.array([ts[period]]), sched)[0])
                        d = MR.period_diagnostics(sched, t, readings, initial, period,
                                                  Ls if Ls > 0 else 1e-6, boundary)
                        delta, derivative = d["dp"], d["der"]
                        factor = norm / abs(d["dq"]) if normalized else norm
                    else:
                        d = MR.superposition_diagnostics(sched, t, readings, initial,
                                                        Ls if Ls > 0 else 1e-6, period)
                        delta, derivative, factor = np.abs(d["g"]), d["der"], norm
                except ValueError as exc:
                    st.warning(f"Period {period + 1}: {exc}")
                    continue
                for quantity, values, symbol in (("dp", delta, "circle"), ("derivative", derivative, "diamond")):
                    values = values * factor
                    valid = np.isfinite(values) & (values > 0) & (d["dt"] > 0)
                    hidden += int((~valid).sum())
                    fig.add_trace(go.Scatter(x=d["dt"], y=np.where(valid, values, np.nan),
                        name=f"P{period + 1} {('fit' if fit else 'model') if is_model else 'data'} {quantity}",
                        legendgroup=f"P{period + 1}", mode="lines" if is_model else "markers",
                        marker=dict(color=color, size=5, symbol=symbol),
                        line=dict(color=color, dash="dash" if quantity == "derivative" else "solid"),
                        connectgaps=False))
        st.plotly_chart(fig, use_container_width=True)
        if local:
            st.caption("Local dp: pressure drop after a rate increase, pressure rise after a rate decrease/shut-in. "
                       "Derivative is d(dp)/d ln(elapsed time), as in Interpretation. Periods start at their own zero; "
                       "boundary pressure is interpolated if not recorded (first period uses initial pressure). "
                       "Earlier rate-history effects remain: this overlay does not deconvolve them or force curves to match.")
        else:
            st.caption("Advanced rate-superposition view: |pi-p|/dq and derivative with respect to the "
                       "rate-weighted log-time function. Unlike local dp, its pressure term can retain a large offset "
                       "and need not resemble a single buildup. Earlier rates are included in the time transform.")
        if hidden:
            st.caption(f"{hidden} non-positive or undefined plotted values are left as gaps on the log axes "
                       "(including derivative endpoints); negative slopes are not flipped positive.")
        if not overlay:
            st.info("Select one or more periods to superpose.")
    st.markdown("##### Regression on the whole record")
    specs = H.BASE_SPEC + H.MODEL_SPEC.get(p.model, []) + H.BOUNDARY_SPEC.get(p.boundary, [])
    fit_specs = {spec[0]: spec for spec in specs if spec[0] in A.FITTABLE}
    def parameter_label(name):
        _, label, kind, *_ = fit_specs[name]
        if name == "S" and p.model in M.INFINITE_FRACTURE_MODELS:
            label = "Additional fracture skin"
        unit = U.label(kind, system)
        return f"{label} ({name})" + (f" [{unit}]" if unit else "")
    free = st.multiselect("Parameters to fit", list(fit_specs),
                          default=H.fit_free_params(p.model, p.boundary),
                          format_func=parameter_label, key=f"mr_free_{ver}_{p.model}_{p.boundary}")
    st.caption(f"Boundary condition: {H.BOUNDARY_LABELS[p.boundary]}. "
               "Choose the boundary type under Outer boundary; regression fits its selected numerical "
               "geometry parameters, not the categorical sealed/constant-pressure condition.")
    if st.button("Run multi-rate regression", type="primary", disabled=not free):
        try:
            with FitProgress("Fitting multi-rate model") as progress:
                pf, info = MR.fit_multirate(p, sched, t, pw, free, progress=progress)
        except (ValueError, np.linalg.LinAlgError) as exc:
            st.error(f"Cannot fit the multi-rate model: {exc}")
        else:
            mr["fit"] = {"params": pf.to_dict(), "info": info, "free": free, "start_params": p.to_dict()}
            mr.pop("working_model", None)
            S["mr"] = mr
            st.rerun()
    fit = mr.get("fit")
    if fit:
        pf = H.params_from_dict(fit["params"])
        if fit["info"].get("status", 1) > 0:
            st.success("Model fitted to the multi-rate record. The fitted curves and parameter table show the completed result.")
        else:
            st.warning("Fitting stopped before convergence. The returned model is provisional, not a converged fit.")
        if fit["info"].get("failed_starts"):
            st.warning(f"{fit['info']['failed_starts']} fitting attempts failed; the result uses the best returned attempt.")
        takeaway("Whole-record regression: " + ", ".join(f"{n} = {getattr(pf, n):.3g}" for n in fit["free"]) + ".", "Multi-rate result")
        all_specs = {spec[0]: spec for spec in H.BASE_SPEC +
                     [s for group in H.MODEL_SPEC.values() for s in group] +
                     [s for group in H.BOUNDARY_SPEC.values() for s in group]}
        st.dataframe(pd.DataFrame([{"Parameter": all_specs[n][1],
                                    "Fitted": f"{U.from_field(getattr(pf, n), all_specs[n][2], system):.4g}",
                                    "Unit": U.label(all_specs[n][2], system),
                                    "Started at": f"{U.from_field(fit.get('start_params', p.to_dict())[n], all_specs[n][2], system):.4g}"}
                                   for n in fit["free"]]), hide_index=True, use_container_width=True)
        st.caption(f"RMS misfit {fit['info']['rms']:.3g} (scaled) | {fit['info']['nfev']} evaluations.")
