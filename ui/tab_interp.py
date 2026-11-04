"""02 Interpretation - diagnostics, Horner analysis and model matching."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import streamlit as st
from .theme import takeaway

from engine import analysis as A
from engine import models as M
from engine import units as U
from engine import viz
from . import figures as F
from . import helpers as H
from . import inputs
from . import uncertainty
from .calculation import FitProgress

LABELS = {"k": ("Permeability", "perm"), "S": ("Skin", "none"), "C": ("Wellbore storage", "storage"),
          "omega": ("omega", "none"), "lam": ("lambda", "none"), "xf": ("Fracture half-length", "length"),
          "d1": ("Distance 1", "length"), "d2": ("Distance 2", "length"),
          "r_inner": ("Inner radius", "length"), "M": ("Mobility ratio", "none"), "D": ("Diffusivity ratio", "none"),
          "FcD": ("Fracture conductivity FcD", "none"), "Lw": ("Horizontal length", "length"), "kv_kh": ("kv/kh", "none"),
          "f_space": ("Fracture spacing", "length"), "re": ("Reservoir radius", "length"),
          "xe": ("Rectangle length", "length"), "ye": ("Rectangle width", "length"),
          "r_mid": ("Middle-zone outer radius", "length"), "M3": ("Outer mobility ratio", "none"),
          "D3": ("Outer diffusivity ratio", "none"), "zw_h": ("Well depth / thickness", "none")}


def _fmt(v):
    return f"{v:.4g}"


def _model_response(params, q, tp, dt_min, dt_end):
    p = H.params_from_dict(params)
    cur = viz.model_curves(p, q, tp, dt_end, dt_min=dt_min)
    flow_t = np.geomspace(min(1e-3, tp / 1000), tp, 120)
    flowing = M.pwf_history(p, flow_t, [(0., q)])
    return {"diag": cur, "t": np.concatenate([[0.], flow_t, tp + cur["dt"]]),
            "pressure": np.concatenate([[p.pi], flowing, cur["pws"]])}


def _mc_envelope(params, q, tp, dt_min, dt_end, samples):
    dt = np.geomspace(dt_min, dt_end, 140)
    return viz.monte_carlo_envelope(H.params_from_dict(params), q, tp, dt, samples)


def _sync_controls(state, p, gas, system, ver):
    """Apply fit/unit changes before Streamlit creates the keyed widgets."""
    specs = H.BASE_SPEC + H.MODEL_SPEC.get(p.model, []) + H.BOUNDARY_SPEC.get(p.boundary, [])
    for name, _, kind, *_ in specs:
        if gas is not None and name in ("mu", "B", "ct"):
            continue
        state[f"interp_{name}_{system}_{ver}"] = float(U.from_field(getattr(p, name), kind, system))
    state[f"interp_model_{ver}"] = p.model
    state[f"interp_rw_in_{ver}"] = float(p.rw * 12.)
    state.pop(f"_interp_rw_display_{ver}", None)
    if p.model in M.INFINITE_FRACTURE_MODELS:
        state[f"interp_S_total_{system}_{ver}"] = float(p.S + M.fracture_geometric_skin(p))
        state.pop(f"_interp_skin_display_{ver}", None)
    state[f"interp_boundary_{ver}"] = p.boundary
    if p.model == "horizontal":
        state[f"interp_topbc_{ver}"], state[f"interp_botbc_{ver}"] = p.top_bc, p.bot_bc
    if gas is not None:
        state[f"interp_gasT_{system}_{ver}"] = float(U.from_field(gas["T_F"], "temp", system))
        state[f"interp_gasg_{ver}"] = float(gas["gamma_g"])
        state[f"interp_gascf_{system}_{ver}"] = float(U.from_field(gas["cf"], "ct", system))


def render(meta):
    S = st.session_state
    system, fluid, ver = meta["units"], meta["fluid"], S["ver"]
    pu = U.label("pressure", system)
    ds = S.get("dataset")
    if ds is None:
        st.info("Load or generate a dataset first (tabs 00 or 01).")
        S["interp_eng"] = None
        return
    if ds.get("fluid") != fluid:
        st.warning(f"This dataset contains {ds['fluid'].lower()} data. Select {ds['fluid']} in the sidebar to interpret it.")
        for key in ("interp_eng", "horner"):
            S[key] = None
        return
    working = S.get("interp_model")
    base = H.params_from_dict(working["params"] if working else ds["guess"])
    gas_defaults = working.get("gas") if working else ds.get("gas")
    pending = S.pop("_interp_pending", None)
    if pending:
        base = H.params_from_dict(pending["params"])
        gas_defaults = pending["gas"]
    unit_key = f"_interp_units_{ver}"
    if pending or (working and S.get(unit_key) != system):
        _sync_controls(S, base, gas_defaults, system, ver)
    S[unit_key] = system
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown("#### Known inputs and model")
        p = inputs.params_form("interp", base, system, fluid, ver)
        gas = ds.get("gas") if fluid == "Gas" else None
        if fluid == "Gas":
            gas = inputs.gas_form("interp", system, ver, defaults=gas_defaults)
        tp = ds["tp"]
        generate = st.button("Generate model", key=f"interp_generate_{ver}", type="primary")
        st.caption("Generate the response without fitting. Once generated, parameter edits update "
                   "the working model automatically. Fit to data runs regression separately.")
    p_eng, q_eng, gp = H.engine_view(p, fluid, gas, ds["q_user"])
    dt, pws = np.asarray(ds["dt"], float), np.asarray(ds["pws"], float)
    if ds.get("qc_source") is not None:
        st.caption(f"QC applied: {len(dt)} active buildup readings used for diagnostics and fitting. "
                   "Excluded pressure readings are hidden here; recorded rate events remain unchanged. "
                   "Inspect or restore originals in Data & QC.")
    pw0 = ds["pwf_shutin"]
    f = lambda a: U.from_field(a, "pressure", system)
    if generate or working:
        S["interp_model"] = {"params": p.to_dict(), "gas": gas}

    with right:
        Ls = st.slider("Derivative smoothing L (ln-time window)", 0.0, 0.5, 0.1, 0.05, key=f"interp_L_{ver}")
        diag = viz.diagnostic_from_data(tp, dt, pws, pw0, Ls if Ls > 0 else 1e-6)
        S["interp_eng"] = {"params": p_eng.to_dict(), "q": q_eng, "tp": tp, "L": Ls,
                           "diag": {k: np.asarray(v) for k, v in diag.items()}}
        models, responses = [], []
        if S.get("interp_model"):
            try:
                with st.spinner("Generating working model..."):
                    response = _model_response(p_eng.to_dict(), q_eng, tp, float(dt.min()), float(dt.max()))
            except (ValueError, np.linalg.LinAlgError) as exc:
                st.error(f"Cannot generate model: {exc}")
            else:
                cur = response["diag"]
                models.append({"name": "Working model", "x": cur["ta"], "dp": f(cur["dp"]), "der": f(cur["der"])})
                responses.append({**response, "name": "Working model", "q_user": ds["q_user"], "tp": tp})
        else:
            st.info("Click Generate model to compare the current parameters with the data, or Fit to data to run regression.")
        lf = S.get("last_fit")
        if lf and lf.get("ds") == ds["name"]:
            try:
                M.validate_skin(H.params_from_dict(lf["params"]))
            except ValueError as exc:
                st.error(f"Cannot plot the saved fit: {exc} Generate and fit a corrected model.")
                lf = None
        if lf and lf.get("ds") == ds["name"]:
            pf = H.params_from_dict(lf["params"])
            fit_response = _model_response(pf.to_dict(), lf["q"], lf["tp"], float(dt.min()), float(dt.max()))
            cur = fit_response["diag"]
            models.append({"name": "Last fit", "x": cur["ta"], "dp": f(cur["dp"]), "der": f(cur["der"])})
            responses.append({**fit_response, "name": "Last fit", "q_user": lf.get("q_user", ds["q_user"]), "tp": lf["tp"]})
            st.caption("Last fit is a saved regression result; editing inputs moves the Working model, not that snapshot.")
        envelope = None
        mc = S.get("mc")
        if mc and lf and lf.get("ds") == ds["name"]:
            show_band = st.checkbox("Show Monte Carlo P10-P90 response envelope", value=True,
                                    key=f"interp_mc_band_{ver}")
            if show_band:
                try:
                    with st.spinner("Calculating Monte Carlo response envelope..."):
                        envelope = mc.get("envelope")
                        if envelope is None:
                            envelope = _mc_envelope(lf["params"], lf["q"], lf["tp"],
                                                    float(dt.min()), float(dt.max()), mc["samples"])
                except (ValueError, np.linalg.LinAlgError) as exc:
                    st.error(f"Cannot calculate Monte Carlo response envelope: {exc}")
        diagnostic_fig = F.diag_figure(
            "Diagnostic plot (buildup)", "Agarwal equivalent time [hr]", f"dp and derivative [{pu}]",
            data={"x": diag["ta"], "dp": f(diag["dp"]), "der": f(diag["der"])}, models=models)
        pressure_fig = F.interpretation_response(ds, responses, system)
        if envelope:
            F.add_percentile_band(diagnostic_fig, envelope["ta"], envelope["dp"], "dp", system,
                                  "rgba(34,81,255,0.18)", log=True)
            F.add_percentile_band(diagnostic_fig, envelope["ta"], envelope["der"], "derivative", system,
                                  "rgba(194,65,12,0.20)", log=True)
            F.add_percentile_band(pressure_fig, envelope["t"], envelope["pressure"], "pressure", system,
                                  "rgba(34,81,255,0.18)")
            origin = "assumed parameter distributions around Last fit" if mc.get("mode") == "parameter_sampling" else "noisy-data refits of Last fit"
            st.caption(f"Monte Carlo envelope: {envelope['count']} paired parameter sets from {origin}. "
                       "P10/P90 are the 10th/90th numerical percentiles at each time (central 80%); "
                       "P50 is the pointwise median. Not a simultaneous confidence band or a gauge-noise prediction interval. "
                       "Full drawdown is an extrapolation when only buildup was fitted. "
                       "Rate is fixed, so there is no Monte Carlo rate band.")
        st.plotly_chart(diagnostic_fig, use_container_width=True)
        st.plotly_chart(pressure_fig, use_container_width=True)
        st.caption("Pressure is an absolute model prediction, not shifted to force agreement. "
                   "Rates are prescribed surface controls, not regression outputs. "
                   "This buildup workflow assumes constant production followed by shut-in; "
                   "use Multi-rate for variable-rate superposition. Gas uses normalized pseudo-pressure.")

        st.markdown("##### Horner (semilog) analysis")
        opts = [float(x) for x in np.unique(np.round(dt[dt > 0], 4))]
        lo_i, hi_i = int(len(opts) * 0.45), len(opts) - 1
        win = st.select_slider("Radial-flow window (shut-in time [hr])", options=opts,
                               value=(opts[lo_i], opts[hi_i]), key=f"interp_win_{ver}")
        ht = diag["ht"]
        ht_win = ((tp + win[1]) / win[1], (tp + win[0]) / win[0])
        S["horner"] = None
        try:
            m, b0 = A.semilog_radial(ht, pws, ht_win)
            k_h = A.permeability_from_slope(m, q_eng, p_eng.B, p_eng.mu, p_eng.h)
            p1 = m * np.log10((tp + 1.0) / 1.0) + b0
            s_h = A.skin_from_horner(p1, pw0, m, k_h, p_eng.phi, p_eng.mu, p_eng.ct, p_eng.rw)
            x_line = np.logspace(np.log10(max(min(ht_win) * 0.7, 1.0001)), np.log10(max(ht_win) * 2.0), 20)
            line = (x_line, f(m * np.log10(x_line) + b0))
            S["horner"] = {"k": float(k_h), "S": float(s_h), "pstar": float(b0), "m": float(abs(m)),
                           "window": [float(win[0]), float(win[1])]}
            takeaway(f"Radial-flow analysis gives k = {k_h:.3g} md and skin {s_h:+.2f}; extrapolated p* is {f(b0):.0f} {pu}.", "Semilog result")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Permeability k", f"{k_h:.4g} md")
            c2.metric("Skin S", f"{s_h:.3g}")
            c3.metric("Extrapolated p*", f"{f(b0):.1f} {pu}")
            c4.metric("Slope m", f"{f(abs(m)):.3g} {pu}/cycle")
            st.plotly_chart(F.horner_figure("Horner plot", ht, f(pws), f"pws [{pu}]", line=line,
                                            window=ht_win), use_container_width=True)
            if p_eng.boundary != "none":
                st.caption("A boundary is present in the model: restrict the window to the radial-flow "
                           "part before the derivative departs from its plateau.")
        except ValueError as e:
            st.warning(f"Horner fit: {e}. Widen the window.")

    st.markdown("---")
    st.markdown("#### Fit model to data (automatic regression)")
    free_default = H.fit_free_params(p.model, p.boundary)
    options = [n for n in LABELS if n in A.FITTABLE]
    free = st.multiselect("Parameters to fit", options, default=free_default, key=f"interp_free_{ver}_{p.model}_{p.boundary}")
    c1, c2, c3 = st.columns([1, 1, 2])
    wd = c1.number_input("Derivative weight", 0.0, 5.0, 1.0, 0.25, key=f"interp_wd_{ver}")
    ms = c2.checkbox("Multi-start (slower, more robust)", value=True, key=f"interp_ms_{ver}")
    if c3.button("Fit to data", type="primary", disabled=not free or q_eng <= 0, key=f"interp_fit_{ver}"):
        try:
            with FitProgress("Fitting buildup model") as progress:
                pf, info = A.fit_buildup(p_eng, q_eng, tp, dt, pws, free, pwf_shutin=pw0,
                                         weight_deriv=wd, multistart=ms, progress=progress)
        except (ValueError, np.linalg.LinAlgError) as exc:
            st.error(f"Cannot fit this model to the data: {exc}")
        else:
            S["last_fit"] = {"params": pf.to_dict(), "info": info, "ds": ds["name"], "free": free,
                             "q": q_eng, "tp": tp, "fluid": fluid, "q_user": ds["q_user"],
                             "start_params": p_eng.to_dict(), "gas": deepcopy(gas)}
            S["interp_model"] = {"params": pf.to_dict(), "gas": gas}
            S["_interp_pending"] = S["interp_model"]
            S["mc"], S["sens"] = None, None
            st.rerun()
    lf = S.get("last_fit")
    if lf and lf.get("ds") == ds["name"]:
        pf = H.params_from_dict(lf["params"])
        if lf["info"].get("status", 1) <= 0:
            st.warning("The regression stopped before convergence. Treat this fit as provisional; "
                       "review the starting values, free parameters and iteration limits.")
        else:
            st.success("Model fitted to data. The Last fit curves and fitted parameter table show the completed result.")
        if lf["info"].get("failed_starts"):
            st.warning(f"{lf['info']['failed_starts']} fitting attempts failed; the result uses the best returned attempt.")
        rows = []
        for n in lf["free"]:
            lab, kind = LABELS[n]
            if n == "S" and pf.model in M.INFINITE_FRACTURE_MODELS:
                lab = "Additional fracture skin"
            ci = (lf["info"].get("ci") or {}).get(n)
            row = {"Parameter": lab, "Fitted": _fmt(U.from_field(getattr(pf, n), kind, system)),
                   "Unit": U.label(kind, system)}
            if ci:
                row["95 % low"] = _fmt(U.from_field(ci["lo"], kind, system))
                row["95 % high"] = _fmt(U.from_field(ci["hi"], kind, system))
            row["Started at"] = _fmt(U.from_field(lf.get("start_params", p_eng.to_dict())[n], kind, system))
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(f"RMS misfit (log space) {lf['info']['rms']:.3g} | {lf['info']['nfev']} evaluations. "
                   "A low RMS with several parameters free does not prove the model is unique: "
                   "compare alternative models in tab 04.")
        with st.expander("Uncertainty: sensitivity and Monte Carlo"):
            st.caption("Intervals above are linearised from the regression Jacobian and scaled by the misfit. "
                       "Sensitivity changes one parameter at a time. Choose noisy-data refits or "
                       "propagate bounded parameter distributions without refitting.")
            sens_names = st.multiselect("Parameters for sensitivity", lf["free"], default=lf["free"][:2],
                                        key=f"interp_sens_{ver}_{'_'.join(lf['free'])}")
            s1, s2 = st.columns(2)
            if s1.button("Show sensitivity", key=f"interp_sensbtn_{ver}") and sens_names:
                S["sens"] = A.sensitivity_curves(pf, lf["q"], lf["tp"], dt, sens_names)
            sens = S.get("sens")
            if sens is not None:
                models = [{"name": "fit", "x": sens["base"]["ta"], "dp": f(sens["base"]["dp"]), "der": f(sens["base"]["der"])}]
                for r in sens["runs"]:
                    models.append({"name": f"{r['name']} x{r['factor']:g}", "x": r["curves"]["ta"],
                                   "dp": f(r["curves"]["dp"]), "der": f(r["curves"]["der"])})
                fig = F.diag_figure("Sensitivity of the model response", "Agarwal equivalent time [hr]",
                                              f"dp and derivative [{pu}]", data={"x": diag["ta"], "dp": f(diag["dp"]), "der": f(diag["der"])},
                                              models=models)
                st.plotly_chart(fig, use_container_width=True)
            mode = st.radio("Monte Carlo method", ["Noisy-data refits", "Parameter distributions (no refitting)"],
                            key=f"interp_mc_mode_{ver}")
            if mode == "Parameter distributions (no refitting)":
                uncertainty.parameter_sampling(pf, fluid, system, ver, lf["free"], lf["q"], lf["tp"], dt)
            else:
                nmc = s2.number_input("Monte Carlo runs", 5, 60, 15, 5, key=f"interp_nmc_{ver}")
                if s2.button("Run Monte Carlo", key=f"interp_mcbtn_{ver}"):
                    with st.spinner("Refitting noisy copies..."):
                        mc = A.monte_carlo(pf, lf["q"], lf["tp"], dt, pws, lf["free"], pwf_shutin=pw0, n=int(nmc))
                    mc["requested"] = int(nmc)
                    S["mc"] = mc
                    st.rerun()
            mc = S.get("mc")
            if mc:
                uncertainty.sampling_results(mc, system)
                used = min((len(v) for v in mc["samples"].values()), default=0)
                requested = mc.get("requested", used)
                if used < requested:
                    st.warning(f"Only {used} of {requested} Monte Carlo refits returned parameter sets. "
                               "The envelope uses returned sets only; review stability before interpreting it.")
                if used < 3:
                    st.error("At least three Monte Carlo parameter sets are required to show a response envelope.")
            if mc and mc["summary"]:
                summary_specs = {s[0]: (s[1], s[2]) for s in H.BASE_SPEC +
                                 H.MODEL_SPEC.get(pf.model, []) + H.BOUNDARY_SPEC.get(pf.boundary, [])}
                st.dataframe(pd.DataFrame([{"Parameter": summary_specs[n][0],
                                            "Unit": U.label(summary_specs[n][1], system),
                                            **{key.upper(): _fmt(U.from_field(v[key], summary_specs[n][1], system))
                                               for key in ("p10", "p50", "p90")}, "runs": v["n"]}
                                           for n, v in mc["summary"].items() if n in summary_specs]),
                             hide_index=True, use_container_width=True)
        n1, n2 = st.columns([2, 1])
        name = n1.text_input("Name for this interpretation", value=f"{H.MODEL_LABELS[pf.model]} / {H.BOUNDARY_LABELS[pf.boundary]}",
                             key=f"interp_name_{ver}")
        if n2.button("Save to Compare"):
            S["fits"].append({"name": name, "params": deepcopy(lf["params"]), "rms": lf["info"]["rms"], "q": lf["q"],
                              "tp": lf["tp"], "ds": ds["name"], "fluid": fluid,
                              "gas": deepcopy(lf.get("gas", ds.get("gas")))})
            st.success("Saved.")
