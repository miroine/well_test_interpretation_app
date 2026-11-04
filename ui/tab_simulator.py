"""06 Simulator - r-z black-oil model of the well test, with automatic calibration."""
from __future__ import annotations

import copy
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import units as U
from engine import calibrate as CAL
from engine import opm_deck as OD
from engine import sim_presets
from engine.blackoil import BlackOilSim, NewtonFailure, RelPerm, SimConfig
from . import figures as F
from .theme import polish, takeaway


_SYS = ["Field"]


def _fv(v, kind):
    r = U.from_field(v, kind, _SYS[0])
    return float(r) if np.ndim(r) == 0 else r


def _tf(v, kind):
    r = U.to_field(v, kind, _SYS[0])
    return float(r) if np.ndim(r) == 0 else r


def _num(col, label, value, key, ver, kind=None, **kw):
    """Number input in the display unit system; returns the value in field units."""
    if kind is None:
        return col.number_input(label, value=value, key=f"sim_{key}_{ver}", **kw)
    for n in ("min_value", "max_value", "step"):
        if kw.get(n) is not None:
            kw[n] = float(_fv(kw[n], kind)) if kind != "temp" else kw[n]
    if "format" not in kw:
        kw["format"] = "%.5g"
    shown = col.number_input(f"{label} [{U.label(kind, _SYS[0])}]", value=float(_fv(value, kind)),
                             key=f"sim_{key}_{ver}_{_SYS[0]}", **kw)
    return _tf(shown, kind)


def _form(p, ver):
    c = p["cfg"]
    rp = p["rp"]
    with st.expander("Reservoir, fluid contacts and well", expanded=False):
        st.caption("Inputs follow the unit system chosen in the sidebar; the solver runs in field units. Depths are TVD.")
        a = st.columns(5)
        k = _num(a[0], "k horizontal [md]", float(c.k), "k", ver, min_value=0.1, step=5.0)
        kvkh = _num(a[1], "kv/kh", float(c.kv_kh), "kvkh", ver, min_value=0.001, max_value=2.0, step=0.05)
        phi = _num(a[2], "porosity", float(c.phi), "phi", ver, min_value=0.02, max_value=0.45, step=0.01)
        h = _num(a[3], "gross thickness", float(c.h), "h", ver, "length", min_value=5.0, step=5.0)
        top = _num(a[4], "top depth", float(c.top_depth), "top", ver, "length", step=10.0)
        b = st.columns(5)
        pd_ = _num(b[0], "pressure at datum", float(c.p_datum), "pd", ver, "pressure", min_value=100.0, step=50.0)
        dd = _num(b[1], "datum depth", float(c.datum_depth if c.datum_depth is not None else c.top_depth + 0.5 * c.h * (c.perf_from + c.perf_to)), "dd", ver, "length", step=5.0)
        has_goc = b[2].checkbox("Gas cap", value=c.goc is not None, key=f"sim_hasgoc_{ver}")
        goc = _num(b[3], "GOC depth", float(c.goc if c.goc is not None else c.top_depth + 0.3 * c.h), "goc", ver, "length", step=5.0)
        rsi = _num(b[4], "solution GOR (no gas cap)", float(c.Rsi), "rsi", ver, "gor", min_value=0.0, step=25.0)
        d = st.columns(5)
        has_woc = d[0].checkbox("Water leg", value=c.woc is not None, key=f"sim_haswoc_{ver}")
        woc = _num(d[1], "WOC depth", float(c.woc if c.woc is not None else c.top_depth + 0.8 * c.h), "woc", ver, "length", step=5.0)
        re_ = _num(d[2], "drainage radius re", float(c.re), "re", ver, "length", min_value=100.0, step=100.0)
        pvm = _num(d[3], "outer pore-volume multiplier", float(c.outer_pv_mult), "pvm", ver, min_value=1.0, step=10.0,
                   help="1 = closed boundary; large values mimic aquifer / pressure support at re.")
        cr = _num(d[4], "rock compressibility", float(c.cr), "cr", ver, "ct", format="%.2e")
        e = st.columns(5)
        pf = _num(e[0], "perforated from (fraction of h, top)", float(c.perf_from), "pf", ver, min_value=0.0, max_value=0.99, step=0.05)
        pt = _num(e[1], "perforated to", float(c.perf_to), "pt", ver, min_value=0.01, max_value=1.0, step=0.05)
        sk = _num(e[2], "skin", float(c.skin), "skin", ver, step=0.5)
        C = _num(e[3], "wellbore storage", float(c.C), "C", ver, "storage", min_value=0.0, step=0.005, format="%.4f")
        nr = int(_num(e[4], "radial cells", int(c.nr), "nr", ver, min_value=8, max_value=40, step=2))
        nz = int(st.number_input("vertical layers", value=int(c.nz), min_value=2, max_value=24, step=1, key=f"sim_nz_{ver}"))
    with st.expander("Relative permeability and PVT"):
        r = st.columns(6)
        swc = _num(r[0], "Swc", rp.swc, "swc", ver, step=0.01)
        sorw = _num(r[1], "Sorw", rp.sorw, "sorw", ver, step=0.01)
        sorg = _num(r[2], "Sorg", rp.sorg, "sorg", ver, step=0.01)
        sgc = _num(r[3], "Sgc", rp.sgc, "sgc", ver, step=0.01)
        krw = _num(r[4], "krw max", rp.krw_max, "krw", ver, step=0.05)
        krg = _num(r[5], "krg max", rp.krg_max, "krg", ver, step=0.05)
        r2 = st.columns(6)
        krocw = _num(r2[0], "kro at Swc", rp.krocw, "krocw", ver, step=0.05)
        nw = _num(r2[1], "nw", rp.nw, "nw", ver, step=0.1)
        ng = _num(r2[2], "ng", rp.ng, "ng", ver, step=0.1)
        now = _num(r2[3], "now", rp.now, "now", ver, step=0.1)
        nog = _num(r2[4], "nog", rp.nog, "nog", ver, step=0.1)
        api = _num(r2[5], "oil API", 35.0, "api", ver, step=1.0)
        r3 = st.columns(3)
        gg = _num(r3[0], "gas gravity", 0.75, "gg", ver, step=0.01)
        T = _num(r3[1], "temperature", 200.0, "T", ver, "temp", step=5.0)
        co = _num(r3[2], "undersaturated co", 1.2e-5, "co", ver, "ct", format="%.2e")
    cfg = SimConfig(rw=c.rw, re=re_, nr=nr, nz=nz, h=h, top_depth=top, k=k, kv_kh=kvkh, phi=phi, cr=cr,
                    skin=sk, perf_from=min(pf, pt - 0.01), perf_to=pt, C=C, p_datum=pd_, datum_depth=dd,
                    goc=goc if has_goc else None, woc=woc if has_woc else None, Rsi=rsi, outer_pv_mult=pvm)
    rpn = RelPerm(swc=swc, sorw=sorw, sorg=sorg, sgc=sgc, krw_max=krw, krg_max=krg, krocw=krocw,
                  nw=nw, ng=ng, now=now, nog=nog)
    from engine.blackoil import PVTConfig
    pvt = PVTConfig(api=api, gas_gravity=gg, T_F=T, co=co, p_ref=pd_, p_max=max(2.5 * pd_, 6000.0))
    return cfg, rpn, pvt


def _schedule_editor(p, ver):
    ru = U.label("rate", _SYS[0])
    rows = [{"t_start [hr]": float(t), f"oil rate [{ru}]": _fv(float(q), "rate")} for t, q in
            [(e[0], e[1]) for e in p["schedule"]]]
    df = st.data_editor(pd.DataFrame(rows), num_rows="dynamic", hide_index=True,
                        key=f"sim_sched_{ver}_{p['name']}_{_SYS[0]}")
    df = df.apply(pd.to_numeric, errors="coerce").dropna().sort_values("t_start [hr]")
    sched = [(float(a), _tf(float(b), "rate")) for a, b in df.itertuples(index=False)]
    t_end = st.number_input("End time [hr]", value=float(p["t_end"]), min_value=0.1, key=f"sim_tend_{ver}_{p['name']}")
    return sched, float(t_end)


def _plots(res, cfg, obs=None, fitted=None, flow=None):
    t = res["t"]
    tt = np.maximum(t, 1e-3)
    logt = st.checkbox("Log time axis", value=False, key="sim_logt")
    sysm = _SYS[0]
    pu, ru, gu = U.label("pressure", sysm), U.label("rate", sysm), U.label("gor", sysm)
    fp = lambda a: U.from_field(np.asarray(a, float), "pressure", sysm)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tt if logt else t, y=fp(res["pwf"]), mode="lines", name="simulated pwf",
                             line=dict(color=F.BLUE, width=2.4)))
    if fitted is not None:
        fig.add_trace(go.Scatter(x=fitted["t"], y=fp(fitted["pwf"]), mode="lines", name="calibrated",
                                 line=dict(color=F.RED, width=2.4, dash="dash")))
    if obs is not None:
        fig.add_trace(go.Scatter(x=obs["t"], y=fp(obs["pwf"]), mode="markers", name="measured",
                                 marker=dict(color=F.ORANGE, size=5)))
    if flow is not None and np.isfinite(flow["pwf"]).any():
        fig.add_trace(go.Scatter(x=flow["t"], y=fp(flow["pwf"]), mode="lines+markers", name="OPM Flow",
                                 line=dict(color=F.GREEN, width=1.6), marker=dict(size=4)))
    fig.update_layout(title="Flowing / shut-in bottom-hole pressure", xaxis_title="time [hr]",
                      yaxis_title=f"pressure [{pu}]", height=400, margin=dict(l=50, r=20, t=50, b=40),
                      xaxis_type="log" if logt else "linear", template="plotly_white")
    st.plotly_chart(polish(fig), use_container_width=True)
    c1, c2 = st.columns(2)
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=t, y=U.from_field(res["GOR"], "gor", sysm), mode="lines", name=f"producing GOR [{gu}]",
                            line=dict(color=F.RED, width=2.2)))
    f2.add_trace(go.Scatter(x=t, y=res["WC"] * 100, mode="lines", name="water cut [%]", yaxis="y2",
                            line=dict(color=F.BLUE, width=2.2)))
    if fitted is not None and "GOR" in fitted:
        f2.add_trace(go.Scatter(x=fitted["t"], y=U.from_field(fitted["GOR"], "gor", sysm), mode="lines",
                                name="calibrated GOR", line=dict(color=F.RED, width=1.8, dash="dash")))
        f2.add_trace(go.Scatter(x=fitted["t"], y=np.asarray(fitted["WC"], float) * 100, mode="lines", yaxis="y2",
                                name="calibrated water cut", line=dict(color=F.BLUE, width=1.8, dash="dash")))
    if obs is not None and "GOR" in obs and np.isfinite(obs["GOR"]).any():
        f2.add_trace(go.Scatter(x=obs["t"], y=U.from_field(obs["GOR"], "gor", sysm), mode="markers",
                                name="measured GOR", marker=dict(color=F.ORANGE, size=5)))
        f2.add_trace(go.Scatter(x=obs["t"], y=np.asarray(obs["WC"], float) * 100, mode="markers", yaxis="y2",
                                name="measured water cut", marker=dict(color=F.GREEN, size=5)))
    if flow is not None and np.isfinite(flow["GOR"]).any():
        f2.add_trace(go.Scatter(x=flow["t"], y=U.from_field(flow["GOR"], "gor", sysm), mode="lines",
                                name="OPM Flow GOR", line=dict(color=F.GREEN, width=1.4)))
    f2.update_layout(title="GOR and water cut (flowing periods)", xaxis_title="time [hr]", height=340,
                     yaxis=dict(title=f"GOR [{gu}]"), yaxis2=dict(title="water cut [%]", overlaying="y", side="right"),
                     margin=dict(l=50, r=50, t=50, b=40), template="plotly_white", legend=dict(orientation="h"))
    c1.plotly_chart(polish(f2), use_container_width=True)
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=t, y=U.from_field(res["qo"], "rate", sysm), mode="lines", name=f"oil [{ru}]", line=dict(color=F.BLUE)))
    f3.add_trace(go.Scatter(x=t, y=U.from_field(res["qw"], "rate", sysm), mode="lines", name=f"water [{ru}]", line=dict(color=F.ORANGE)))
    f3.add_trace(go.Scatter(x=t, y=U.from_field(res["qg"] / 1000.0, "gasrate", sysm), mode="lines", name=f"gas [{U.label('gasrate', sysm)}]", line=dict(color=F.RED)))
    f3.update_layout(title="Sandface rates", xaxis_title="time [hr]", height=340, margin=dict(l=50, r=20, t=50, b=40),
                     template="plotly_white", legend=dict(orientation="h"))
    c2.plotly_chart(polish(f3), use_container_width=True)
    snaps = res["snapshots"]
    if len(snaps) > 1:
        i = st.select_slider("Snapshot time [hr]", options=list(range(len(snaps))),
                             value=len(snaps) - 1, format_func=lambda j: f"{snaps[j]['t_hr']:.4g}",
                             key="sim_snap")
        sn = snaps[i]
        d = np.linspace(cfg.top_depth, cfg.top_depth + cfg.h, cfg.nz + 1)
        dc = 0.5 * (d[1:] + d[:-1])
        r_c = np.sqrt(np.geomspace(cfg.rw, cfg.re, cfg.nr + 1)[:-1] * np.geomspace(cfg.rw, cfg.re, cfg.nr + 1)[1:])
        cols = st.columns(3)
        for col, (key, name, scale) in zip(cols, [("sg", "Gas saturation", "YlOrRd"), ("sw", "Water saturation", "Blues"),
                                                  ("p", f"Pressure [{pu}]", "Viridis")]):
            hm = go.Figure(go.Heatmap(x=U.from_field(r_c, "length", sysm), y=U.from_field(dc, "length", sysm), z=(fp(sn[key]) if key == "p" else sn[key]).T, colorscale=scale, colorbar=dict(thickness=10)))
            hm.update_layout(title=f"{name} at {sn['t_hr']:.4g} h", xaxis_type="log", xaxis_title=f"radius [{U.label('length', sysm)}]",
                             yaxis=dict(title=f"depth [{U.label('length', sysm)}]", autorange="reversed"), height=330,
                             margin=dict(l=50, r=10, t=50, b=40), template="plotly_white")
            col.plotly_chart(polish(hm), use_container_width=True)


def _opm_section(sim, ver):
    S = st.session_state
    with st.expander("Cross-check against OPM Flow"):
        st.caption("Export the current model as an OPM Flow deck, run it with `flow DECK.DATA`, export the summary "
                   "(e.g. `ecl2csv summary DECK.UNSMRY -o flow.csv`) and upload it here to overlay and quantify "
                   "the difference. Wellbore storage is not modelled in Flow, so set storage to 0 for a fair comparison.")
        try:
            deck = OD.build_deck(BlackOilSim(sim["cfg"], sim["pvt"], sim["rp"]), sim["sched"], sim["t_end"])
            st.download_button("Download OPM Flow deck (.DATA)", deck, file_name="WTW_RZ.DATA", mime="text/plain",
                               key=f"sim_deck_{ver}")
        except Exception as e:
            st.error(f"Could not build the deck: {e}")
        st.markdown("Known differences between the two models:")
        st.markdown("\n".join(f"- {d}" for d in OD.DIFFERENCES))
        unit = st.radio("Time unit in the Flow CSV", ["days", "hours"], horizontal=True, key=f"sim_flowt_{ver}")
        up = st.file_uploader("Flow summary CSV", type=["csv"], key=f"sim_flowup_{ver}")
        if up is not None:
            try:
                S["sim_flow"] = OD.parse_flow_csv(up.getvalue().decode("utf-8"), unit)
            except Exception as e:
                st.error(f"Could not read the Flow results: {e}")
        flow = S.get("sim_flow")
        if flow is not None:
            cmp = OD.compare(sim["res"], flow)
            c = st.columns(3)
            c[0].metric("BHP RMS difference", f"{_fv(cmp['pwf_rms'], 'pressure'):.2f} {U.label('pressure', _SYS[0])}")
            c[1].metric("GOR median relative difference", f"{100 * cmp['GOR_rel']:.1f} %")
            c[2].metric("Water-cut max difference", f"{100 * cmp['WC_abs']:.1f} points")


def render(meta):
    S = st.session_state
    ver = S["ver"]
    _SYS[0] = meta.get("units", "Field")
    st.markdown("#### 06 Simulator - r-z black-oil model with automatic calibration")
    st.caption("Fully implicit, three-phase, radial-vertical grid with a perforated well, wellbore storage, gravity "
               "and variable bubble-point switching. Verified against the analytical single-phase model and "
               "closed-system mass balance (see tests). Inputs and results follow the sidebar unit system.")
    presets = sim_presets.presets()
    name = st.selectbox("Scenario", list(presets), key=f"sim_preset_{ver}")
    p = dict(presets[name]); p["name"] = name
    st.write(p["description"])
    cfg, rp, pvt = _form(p, ver * 100 + list(presets).index(name))
    sched, t_end = _schedule_editor(p, ver)
    if st.button("Run simulation", type="primary", key=f"sim_run_{ver}"):
        try:
            with st.spinner("Running the simulator..."):
                res = BlackOilSim(cfg, pvt, rp).run(sched, t_end, snap_times=list(np.linspace(0, t_end, 6)[1:]))
            S["sim"] = {"res": res, "cfg": cfg, "rp": rp, "pvt": pvt, "sched": sched, "t_end": t_end, "name": name}
            S["sim_inputs"] = {"scenario": name, "cfg": cfg.to_dict(), "schedule": sched, "t_end": t_end}
            S["sim_fit"] = None
        except (NewtonFailure, ValueError) as e:
            st.error(f"Simulation failed: {e}. Try a lower rate, fewer cells, or a smaller end time.")
    sim = S.get("sim")
    if not sim:
        st.info("Press **Run simulation** to compute the scenario.")
        return
    res = sim["res"]
    o0, g0, w0 = res["inplace0"]
    m = st.columns(4)
    conv_v = float(U.from_field(1.0, "rate", _SYS[0]))
    vu = "MMstb" if _SYS[0] == "Field" else "MM m³"
    m[0].metric("Oil in place (model)", f"{o0 * conv_v / 1e6:.2f} {vu}")
    m[1].metric("Cum oil", f"{res['Np'][-1] * conv_v / 1e3:.1f} {'Mstb' if _SYS[0] == 'Field' else 'k m³'}")
    gfin = res['GOR'][np.isfinite(res['GOR'])]
    gmax = gfin[-1] if len(gfin) else float('nan')
    m[2].metric("Last flowing GOR", f"{float(U.from_field(gmax, 'gor', _SYS[0])):,.0f} {U.label('gor', _SYS[0])}")
    m[3].metric("Newton steps", f"{res['steps']}")
    gor = res["GOR"][np.isfinite(res["GOR"])]
    wc = res["WC"][np.isfinite(res["WC"])]
    if len(gor) > 3:
        takeaway(f"GOR moves from {float(U.from_field(gor[0], 'gor', _SYS[0])):,.0f} to {float(U.from_field(gor[-1], 'gor', _SYS[0])):,.0f} {U.label('gor', _SYS[0])} and water cut reaches {100 * (wc[-1] if len(wc) else 0):.1f}% over {res['t'][-1]:,.0f} h of test.", "Simulation result")
    fit = S.get("sim_fit")
    flow = S.get("sim_flow")
    _plots(res, sim["cfg"], obs=(fit or {}).get("obs"), fitted=(fit or {}).get("curve"), flow=flow)
    _opm_section(sim, ver)

    st.markdown("#### Automatic calibration")
    st.caption("Adjusts the selected parameters so the simulated bottom-hole pressure, producing GOR and water cut "
               "match measured data (least squares, whole simulator in the loop). Use a coarse grid for speed.")
    src = st.radio("Measured data", ["Synthetic (this scenario with hidden changes + noise)", f"Upload CSV (time [hr], pwf [{U.label('pressure', _SYS[0])}], optional GOR [{U.label('gor', _SYS[0])}], water cut [-])"],
                   horizontal=True, key=f"sim_src_{ver}")
    obs = None
    if src.startswith("Synthetic"):
        c1, c2, c3, c4 = st.columns(4)
        kf = c1.number_input("true k / model k", value=1.4, min_value=0.2, max_value=5.0, key=f"sim_kf_{ver}")
        sd = c2.number_input("true skin - model skin", value=3.0, key=f"sim_sd_{ver}")
        kvf = c4.number_input("true kv/kh / model kv/kh", value=1.0, min_value=0.1, max_value=10.0, key=f"sim_kvf_{ver}")
        nz_ = _tf(c3.number_input(f"noise [{U.label('pressure', _SYS[0])}]", value=float(_fv(0.3, "pressure")), min_value=0.0, key=f"sim_noise_{ver}_{_SYS[0]}"), "pressure")
        truth = copy.deepcopy(sim["cfg"]); truth.k *= kf; truth.skin += sd; truth.kv_kh *= kvf
        try:
            tr = CAL.simulate(truth, sim["pvt"], sim["rp"], sim["sched"], sim["t_end"])
            obs = CAL.synthetic_observations(tr, np.geomspace(max(sim["t_end"] / 3000, 0.02), sim["t_end"], 70), nz_)
        except (NewtonFailure, ValueError) as e:
            st.warning(f"Could not generate synthetic data: {e}")
    else:
        up = st.file_uploader("CSV with two columns", type=["csv", "txt"], key=f"sim_up_{ver}")
        if up is not None:
            try:
                df = pd.read_csv(io.BytesIO(up.getvalue()), header=None, comment="#")
                df = df.apply(pd.to_numeric, errors="coerce").dropna()
                obs = {"t": df.iloc[:, 0].to_numpy(float), "pwf": _tf(df.iloc[:, 1].to_numpy(float), "pressure")}
                if df.shape[1] >= 3:
                    obs["GOR"] = _tf(df.iloc[:, 2].to_numpy(float), "gor")
                if df.shape[1] >= 4:
                    obs["WC"] = df.iloc[:, 3].to_numpy(float)
            except Exception as e:
                st.error(f"Could not read the file: {e}")
    avail = [u for u in ("pwf", "GOR", "WC") if obs is not None and u in obs and np.isfinite(np.asarray(obs[u], float)).any()]
    use = st.multiselect("Match", avail, default=["pwf"] if "pwf" in avail else avail, key=f"sim_use_{ver}",
                         help="Pressure, producing GOR and water cut. GOR and water cut are only compared while flowing.")
    free = st.multiselect("Parameters to calibrate", list(CAL.CAL_PARAMS), default=["k", "skin"], key=f"sim_free_{ver}")
    if obs is not None and free and use and st.button("Calibrate", key=f"sim_cal_{ver}"):
        coarse = copy.deepcopy(sim["cfg"]); coarse.nr = min(coarse.nr, 14); coarse.nz = min(coarse.nz, 8)
        try:
            with st.spinner("Calibrating (each iteration runs the simulator)..."):
                c, r, info = CAL.calibrate(coarse, sim["pvt"], sim["rp"], sim["sched"], obs, free, use=tuple(use), max_nfev=25)
                curve = CAL.simulate(c, sim["pvt"], r, sim["sched"], sim["t_end"])
            S["sim_fit"] = {"obs": obs, "curve": curve, "info": info, "cfg": c, "rp": r}
            st.rerun()
        except Exception as e:
            st.error(f"Calibration failed: {e}")
    fit = S.get("sim_fit")
    if fit:
        info = fit["info"]
        if list(info.get("use", ["pwf"])) == ["pwf"]:
            st.success(f"Calibrated in {info['nfev']} evaluations, RMS misfit {_fv(info['rms'], 'pressure'):.2f} {U.label('pressure', _SYS[0])}.")
        else:
            st.success(f"Calibrated to {', '.join(info['use'])} in {info['nfev']} evaluations (weighted RMS {info['rms']:.2f}).")
        st.dataframe(pd.DataFrame({"parameter": info["free"], "start": [info["start"][n] for n in info["free"]],
                                   "calibrated": [info["fitted"][n] for n in info["free"]]}), hide_index=True)
