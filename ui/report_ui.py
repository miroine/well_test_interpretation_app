"""Assemble the printable HTML report from session state (display units)."""
from __future__ import annotations

import numpy as np

from engine import models as M
from engine import report as R
from engine import units as U
from engine import viz
from . import helpers as H

SCOPES = ["Everything", "Interpretation (no test design)", "Test design only", "Interpretation + simulation"]


def _params_rows(p: M.WellTestParams, system, fluid):
    rows = []
    for name, label, kind, *_ in H.BASE_SPEC:
        if fluid == "Gas" and name in ("mu", "B", "ct"):
            continue
        rows.append({"Parameter": label, "Value": float(U.from_field(getattr(p, name), kind, system)),
                     "Unit": U.label(kind, system)})
    rows.append({"Parameter": "Reservoir model", "Value": H.MODEL_LABELS[p.model], "Unit": ""})
    for name, label, kind, *_ in H.MODEL_SPEC.get(p.model, []):
        rows.append({"Parameter": label, "Value": float(U.from_field(getattr(p, name), kind, system)),
                     "Unit": U.label(kind, system)})
    rows.append({"Parameter": "Outer boundary", "Value": H.BOUNDARY_LABELS[p.boundary], "Unit": ""})
    for name, label, kind, *_ in H.BOUNDARY_SPEC.get(p.boundary, []):
        rows.append({"Parameter": label, "Value": float(U.from_field(getattr(p, name), kind, system)),
                     "Unit": U.label(kind, system)})
    return rows


def _sim_sections(meta, S, f, pu, system):
    """Simulator scenario, results and calibration; returns (sections, findings)."""
    sim, fit = S.get("sim"), S.get("sim_fit")
    if not sim:
        return [], []
    res, cfg = sim["res"], sim["cfg"]
    L = U.label("length", system)
    lf = lambda v: f"{float(U.from_field(v, 'length', system)):,.0f} {L}"
    rate_u = U.label("rate", system)
    rows = [{"Item": "Scenario", "Value": sim.get("name") or "custom"},
            {"Item": "Horizontal permeability", "Value": f"{cfg.k:.4g} md"}, {"Item": "kv/kh", "Value": f"{cfg.kv_kh:.3g}"},
            {"Item": "Thickness / porosity", "Value": f"{lf(cfg.h)} / {cfg.phi:.3g}"},
            {"Item": "Skin / wellbore storage", "Value": f"{cfg.skin:.3g} / {float(U.from_field(cfg.C, 'storage', system)):.3g} {U.label('storage', system)}"},
            {"Item": "Perforated interval (fraction of thickness)", "Value": f"{cfg.perf_from:.2f} - {cfg.perf_to:.2f}"},
            {"Item": "Gas-oil contact", "Value": lf(cfg.goc) if cfg.goc is not None else "none"},
            {"Item": "Water-oil contact", "Value": lf(cfg.woc) if cfg.woc is not None else "none"},
            {"Item": "Rate schedule", "Value": "; ".join(f"{a:g} h: {float(U.from_field(b, 'rate', system)):,.0f} {rate_u}" for a, b in sim["sched"])},
            {"Item": "Simulated time", "Value": f"{sim['t_end']:g} hr"}]
    t = res["t"]
    series = [{"x": t, "y": f(res["pwf"]), "kind": "line", "color": R.BLUE, "name": "simulated pwf"}]
    if fit:
        series += [{"x": fit["curve"]["t"], "y": f(fit["curve"]["pwf"]), "kind": "line", "color": "#0a7d4f", "dash": True, "name": "calibrated"},
                   {"x": fit["obs"]["t"], "y": f(fit["obs"]["pwf"]), "kind": "points", "color": R.ORANGE, "name": "measured"}]
    secs = [{"heading": "Simulator model", "rows": rows},
            {"heading": "Simulated bottom-hole pressure",
             "svg": R.svg_chart(series, "time [hr]", f"pressure [{pu}]", logx=False, logy=False)}]
    gu = U.label("gor", system)
    gor_series = [{"x": t, "y": U.from_field(res["GOR"], "gor", system), "kind": "line", "color": R.ORANGE, "name": "GOR"}]
    if np.isfinite(res["GOR"]).any():
        secs.append({"heading": "Producing GOR and water cut",
                     "paras": [f"Last flowing producing GOR {float(U.from_field(res['GOR'][np.isfinite(res['GOR'])][-1], 'gor', system)):,.0f} {gu}; "
                               f"peak water cut {100 * float(np.nanmax(res['WC'])):.1f} %."],
                     "svg": R.svg_chart(gor_series, "time [hr]", f"GOR [{gu}]", logx=False, logy=False)})
    findings = [f"Simulator ({sim.get('name') or 'custom'}): GOR reaches "
                f"{float(U.from_field(np.nanmax(res['GOR']), 'gor', system)):,.0f} {gu} and water cut "
                f"{100 * float(np.nanmax(res['WC'])):.1f} % during the simulated test."] if np.isfinite(res["GOR"]).any() else []
    if fit:
        info = fit["info"]
        secs.append({"heading": "Simulator calibration", "rows": [
            {"Parameter": n, "Start": f"{info['start'][n]:.4g}", "Calibrated": f"{info['fitted'][n]:.4g}"} for n in info["free"]],
            "paras": [f"Matched: {', '.join(info.get('use', ['pwf']))}; {info['nfev']} simulator evaluations."]})
        findings.append("Calibrated simulator: " + ", ".join(f"{n} = {info['fitted'][n]:.3g}" for n in info["free"]) + ".")
    return secs, findings


def build(meta, S, scope="Everything", *, design_curves=None):
    system, fluid = meta["units"], meta["fluid"]
    pu = U.label("pressure", system)
    f = lambda a: U.from_field(np.asarray(a, float), "pressure", system)
    secs = [{"heading": "Test summary", "rows": [
        {"Item": "Well", "Value": meta["well"]}, {"Item": "Field", "Value": meta["field"]},
        {"Item": "Test date", "Value": str(meta["date"])}, {"Item": "Analyst", "Value": meta["analyst"] or "-"},
        {"Item": "Fluid / units", "Value": f"{fluid} / {system}"}]}]
    de, ds = S.get("design_eng"), S.get("dataset")
    dataset_matches = ds is None or ds.get("fluid", fluid) == fluid
    if not dataset_matches and scope != "Test design only":
        secs.append({"heading": "Dataset fluid mismatch", "paras": [
            f"The active dataset contains {ds['fluid'].lower()} data, but the selected fluid is {fluid.lower()}. "
            "Dataset interpretation results are omitted. Select the matching fluid before exporting."]})
    if scope in ("Everything", "Test design only") and de:
        p = H.params_from_dict(de["params"])
        summ = viz.design_summary(p, de["q"], de["tp"], de["dt_end"])
        cur = design_curves if design_curves is not None else viz.model_curves(p, de["q"], de["tp"], de["dt_end"])
        svg = R.svg_chart([
            {"x": cur["ta"], "y": f(cur["dp"]), "kind": "line", "color": R.BLUE, "name": "dp"},
            {"x": cur["ta"], "y": f(cur["der"]), "kind": "line", "color": R.ORANGE, "dash": True, "name": "derivative"}],
            "Agarwal equivalent time [hr]", f"dp and derivative [{pu}]")
        secs.append({"heading": "Test design", "rows": _params_rows(H.params_from_dict(S["design_params"] or de["params"]), system, fluid),
                     "paras": [f"Flow period {de['tp']:g} hr, planned shut-in {de['dt_end']:g} hr."]})
        secs.append({"heading": "Design predictions", "rows": [
            {"Quantity": "Storage ends (approx.)", "Value": f"{summ['t_wbs_end_hr']:.3g} hr"},
            {"Quantity": "Radius of investigation at end", "Value": f"{U.from_field(summ['ri_end_ft'], 'length', system):,.0f} {U.label('length', system)}"},
            {"Quantity": "Recommended shut-in", "Value": f"{summ['recommended_shutin_hr']:.3g} hr"}], "svg": svg})
    if scope in ("Everything", "Interpretation (no test design)", "Interpretation + simulation") and ds and dataset_matches:
        secs.append({"heading": "Dataset", "rows": [
            {"Item": "Name", "Value": ds["name"]}, {"Item": "Points", "Value": len(ds["dt"])},
            {"Item": "Flow period before shut-in", "Value": f"{ds['tp']:g} hr"}]})
        ie = S.get("interp_eng")
        series = []
        if ie:
            d = ie["diag"]
            series += [{"x": d["ta"], "y": f(d["dp"]), "kind": "points", "color": R.BLUE, "name": "dp (data)"},
                       {"x": d["ta"], "y": f(d["der"]), "kind": "points", "color": R.ORANGE, "name": "derivative (data)"}]
            lf = S.get("last_fit")
            if lf and lf.get("ds") == ds["name"]:
                pf = H.params_from_dict(lf["params"])
                cur = viz.model_curves(pf, ie["q"], ie["tp"], float(np.max(ds["dt"])), n=140, dt_min=float(np.min(ds["dt"])))
                series += [{"x": cur["ta"], "y": f(cur["dp"]), "kind": "line", "color": "#0a7d4f", "name": "fit dp"},
                           {"x": cur["ta"], "y": f(cur["der"]), "kind": "line", "color": "#0a7d4f", "dash": True, "name": "fit derivative"}]
        h = S.get("horner")
        if h:
            secs.append({"heading": "Semilog (Horner) analysis", "rows": [
                {"Quantity": "Permeability", "Value": f"{h['k']:.4g} md"}, {"Quantity": "Skin", "Value": f"{h['S']:.3g}"},
                {"Quantity": "Extrapolated pressure p*", "Value": f"{float(f(h['pstar'])):.1f} {pu}"},
                {"Quantity": "Slope", "Value": f"{float(f(h['m'])):.3g} {pu}/cycle"},
                {"Quantity": "Window (shut-in time)", "Value": f"{h['window'][0]:g} - {h['window'][1]:g} hr"}]})
        lf = S.get("last_fit")
        if lf and lf.get("ds") == ds["name"]:
            pf = H.params_from_dict(lf["params"])
            secs.append({"heading": "Model regression", "rows": _params_rows(pf, system, fluid),
                         "paras": [f"Fitted parameters: {', '.join(lf['free'])}. RMS misfit (log space) {lf['info']['rms']:.3g}."],
                         "svg": R.svg_chart(series, "Agarwal equivalent time [hr]", f"dp and derivative [{pu}]") if series else ""})
        elif series:
            secs.append({"heading": "Diagnostic plot", "svg": R.svg_chart(series, "Agarwal equivalent time [hr]", f"dp and derivative [{pu}]")})
        tc = S.get("tc_match")
        if tc:
            secs.append({"heading": "Type-curve match", "rows": [
                {"Quantity": "Curve CD e^(2S)", "Value": f"{tc['G']:.3g}"}, {"Quantity": "Match RMS (log)", "Value": f"{tc['rms']:.3g}"}]})
        fits = S.get("fits") or []
        if fits:
            secs.append({"heading": "Saved alternative interpretations", "rows": [
                {"Name": x["name"], "Model": H.MODEL_LABELS[H.params_from_dict(x["params"]).model],
                 "k [md]": round(H.params_from_dict(x["params"]).k, 4), "Skin": round(H.params_from_dict(x["params"]).S, 3),
                 "RMS": round(x["rms"], 4)} for x in fits]})
    mr, dc = S.get("mr"), S.get("deconv")
    if scope in ("Everything", "Interpretation (no test design)", "Interpretation + simulation") and mr:
        ts = [s[0] for s in mr["schedule"]]
        secs.append({"heading": "Multi-rate record", "paras": [f"{mr['name']}: {len(mr['schedule'])} rate periods over {max(mr['t']):g} hr."]})
        if mr.get("fit"):
            secs.append({"heading": "Multi-rate regression", "rows": _params_rows(H.params_from_dict(mr["fit"]["params"]), system, "Oil")})
    if scope in ("Everything", "Interpretation (no test design)", "Interpretation + simulation") and dc:
        norm = U.from_field(1.0, "pressure", system) / U.from_field(1.0, "rate", system)
        secs.append({"heading": "Deconvolution", "paras": [f"Estimated initial pressure {float(f(dc['pi_est'])):.1f} {pu}."],
                     "svg": R.svg_chart([{"x": dc["t"], "y": dc["g"] * norm, "kind": "points", "color": R.BLUE, "name": "deconvolved dp"},
                                         {"x": dc["t"], "y": dc["deriv"] * norm, "kind": "points", "color": R.ORANGE, "name": "deconvolved derivative"}],
                                        "time [hr]", f"{pu} per {U.label('rate', system)}")})
    sim_secs, sim_findings = ([], [])
    if scope in ("Everything", "Interpretation + simulation"):
        sim_secs, sim_findings = _sim_sections(meta, S, f, pu, system)
        secs += sim_secs
    if scope == "Interpretation + simulation":
        # executive summary: analytical result vs simulator, up front
        lf = S.get("last_fit") if dataset_matches else None
        ana = []
        if lf:
            pfit = H.params_from_dict(lf["params"])
            ana.append(f"Analytical regression ({H.MODEL_LABELS[pfit.model]}): k = {pfit.k:.4g} md, skin = {pfit.S:.3g}.")
        elif dataset_matches and S.get("horner"):
            h_ = S["horner"]
            ana.append(f"Semilog analysis: k = {h_['k']:.4g} md, skin = {h_['S']:.3g}.")
        paras = ana + sim_findings
        fit_ = S.get("sim_fit")
        if lf and fit_ and "k" in fit_["info"]["fitted"]:
            kr = fit_["info"]["fitted"]["k"] / H.params_from_dict(lf["params"]).k
            paras.append(f"Calibrated simulator permeability is {kr:.2f} x the analytical value" +
                         (" - the two methods agree within 25 %." if 0.75 <= kr <= 1.25 else
                          " - the methods disagree by more than 25 %; check the model choice, vertical flow and fluid contacts."))
        if not paras:
            paras = ["No interpretation or simulation results are available yet."]
        secs.insert(1, {"heading": "Key findings", "paras": paras})
    secs.append({"heading": "Method", "paras": [
        "Analytical Laplace-space solutions with Stehfest inversion; superposition for variable rates. "
        "Simulation: fully implicit three-phase black-oil r-z model with gravity, gas cap / water leg and calibration to pressure, GOR and water cut. "
        "Results depend on the model chosen; compare alternatives before drawing conclusions."]})
    title = {"Everything": "Well test report", "Interpretation (no test design)": "Well test interpretation report",
             "Test design only": "Well test design report",
             "Interpretation + simulation": "Well test interpretation and simulation report"}[scope]
    return R.build_html(title, {"Well": meta["well"], "Field": meta["field"], "Date": meta["date"], "Analyst": meta["analyst"]}, secs)
