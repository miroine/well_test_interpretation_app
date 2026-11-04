"""09 Advanced - 2D areal model, interference / pulse tests, wellbore effects, large-gauge tools."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from engine import analysis as A
from engine import areal as AR
from engine import interference as IF
from engine import models as M
from engine import reduce as RD
from engine import units as U
from engine import wellbore as WB
from . import figures as F
from . import helpers as H
from . import inputs
from .theme import takeaway


def render(meta):
    S = st.session_state
    st.caption("Tools for cases the single-well analytical models cannot represent. All calculations run in "
               "field units; inputs and results follow the unit system in the sidebar.")
    if meta["fluid"] == "Gas":
        st.info("The advanced tools use a liquid (slightly compressible) formulation. Enter gas properties as "
                "effective liquid values or use the analytical tabs for real-gas pseudo-pressure analysis.")
    t1, t2, t3, t4 = st.tabs(["2D areal model", "Interference and pulse test", "Wellbore effects", "Large gauge records"])
    with t1:
        _areal(meta, S)
    with t2:
        _interference(meta, S)
    with t3:
        _wellbore(meta, S)
    with t4:
        _gauge(meta, S)


# ------------------------------------------------------------------ helpers
def _len(system, label, value_ft, key, lo=1.0, hi=1e6, step=50.0, container=st):
    u = U.label("length", system)
    v = container.number_input(f"{label} [{u}]", min_value=float(U.from_field(lo, "length", system)),
                               max_value=float(U.from_field(hi, "length", system)),
                               value=float(U.from_field(value_ft, "length", system)),
                               step=float(U.from_field(step, "length", system)), key=key)
    return float(U.to_field(v, "length", system))


def _reservoir(prefix, meta, ver, container=st):
    base = H.params_from_dict(st.session_state["design_params"]) if st.session_state.get("design_params") \
        else M.WellTestParams()
    p = inputs.params_form(prefix, base, meta["units"], "Oil", ver, container)
    if p.model not in ("homogeneous", "dual_pss", "dual_slab"):
        p = H.clone(p, model="homogeneous")
        container.caption("This tool uses a homogeneous or dual-porosity reservoir; the model was reset to homogeneous.")
    return p


# ----------------------------------------------------------------- areal 2D
_PRESETS = {
    "Sealing fault, 600 ft from the well": dict(fault=(600.0, 0.0), lens=None, obs=False),
    "Leaky fault (10 % transmissibility)": dict(fault=(600.0, 0.1), lens=None, obs=True),
    "Low-permeability lens east of the well": dict(fault=None, lens=0.05, obs=False),
    "Interference: observation well across a leaky fault": dict(fault=(500.0, 0.3), lens=None, obs=True),
    "Closed rectangle (no faults)": dict(fault=None, lens=None, obs=False, closed=True),
}


def _areal_cfg(p: M.WellTestParams, preset, size, q, tp, tsi, system, ver):
    pr = _PRESETS[preset]
    half = size / 2.0
    wells = [AR.ArealWell(name="Test well", x=0.0, y=0.0, rw=p.rw, skin=p.S, C=p.C,
                          schedule=[(0.0, q), (tp, 0.0)] if tsi else [(0.0, q)])]
    faults, zones = [], []
    if pr["fault"] is not None:
        d, mult = pr["fault"]
        faults.append(dict(x1=d, y1=-half, x2=d, y2=half, mult=mult))
    if pr["lens"] is not None:
        zones.append(dict(poly=[(300.0, -half), (half, -half), (half, half), (300.0, half)], kmult=pr["lens"], phimult=1.0))
    if pr["obs"]:
        wells.append(AR.ArealWell(name="Observation", x=1000.0, y=0.0, rw=p.rw, schedule=[(0.0, 0.0)]))
    outline = [(-half, -half), (half, -half), (half, half), (-half, half)]
    return AR.ArealConfig(k=p.k, h=p.h, phi=p.phi, mu=p.mu, ct=p.ct, B=p.B, pi=p.pi, outline=outline,
                          faults=faults, zones=zones,
                          wells=wells, d0=20.0, growth=0.22)


def _areal(meta, S):
    system, ver = meta["units"], S["ver"]
    pu, ru = U.label("pressure", system), U.label("rate", system)
    st.markdown("#### 2D areal numerical model")
    st.caption("Plan-view finite-volume model: faults (sealing or leaky), permeability lenses, several wells. "
               "Use it to test a geometry the analytical boundary models cannot describe, then compare the "
               "pressure derivative with your data.")
    left, right = st.columns([1, 2])
    with left:
        p = _reservoir("adv_ar", meta, ver)
        preset = st.selectbox("Geometry", list(_PRESETS), key=f"ar_pre_{ver}")
        size = _len(system, "Model size (square)", 8000.0, f"ar_size_{system}_{ver}", 2000.0, 1e5, 500.0)
        q = float(U.to_field(st.number_input(f"Rate [{ru}]", 1.0, 1e9, float(U.from_field(1000.0, "rate", system)),
                                             key=f"ar_q_{system}_{ver}"), "rate", system))
        tsi = st.checkbox("Shut in after the flow period", value=True, key=f"ar_si_{ver}")
        tp = st.number_input("Flow period [hr]", 1.0, 1e5, 100.0, 10.0, key=f"ar_tp_{ver}")
        tend = st.number_input("Total simulated time [hr]", 10.0, 1e6, 400.0, 50.0, key=f"ar_te_{ver}")
        go = st.button("Run areal model", type="primary")
    if go:
        cfg = _areal_cfg(p, preset, size, q, tp, tsi, system, ver)
        with st.spinner("Solving..."):
            res = AR.run(cfg, float(tend), snap_times=[float(tp)])
        S["areal"] = {"res": res, "cfg": cfg.to_dict(), "preset": preset, "tp": float(tp), "tsi": bool(tsi), "q": q}
        st.rerun()
    ar = S.get("areal")
    with right:
        if not ar:
            st.info("Choose a geometry and run the model.")
            return
        res = ar["res"]
        t = res["t"][1:]
        pw = res["pwf"][1:, 0]
        p_i = res["pwf"][0, 0]
        fig = F.line_figure("Test-well flowing pressure", "time [hr]", f"pressure [{pu}]",
                            [("test well", t, U.from_field(pw, "pressure", system))] +
                            ([("observation well", t, U.from_field(res["pwf"][1:, 1], "pressure", system))]
                             if res["pwf"].shape[1] > 1 else []), xlog=False)
        st.plotly_chart(fig, use_container_width=True)
        tp_ = ar["tp"]
        if ar["tsi"]:
            m = t > tp_
            dt = t[m] - tp_
            dp = pw[m] - pw[np.argmin(abs(t - tp_))]
        else:
            m = t > 0
            dt, dp = t[m], p_i - pw[m]
        ok = dp > 0
        if ok.sum() > 8:
            der = A.bourdet_derivative(dt[ok], dp[ok])
            fig = F.diag_figure("Pressure change and derivative (test well)", "elapsed time [hr]",
                                f"dp and derivative [{pu}]",
                                data=dict(x=dt[ok], dp=U.from_field(dp[ok], "pressure", system),
                                          der=U.from_field(der, "pressure", system)))
            st.plotly_chart(fig, use_container_width=True)
        import plotly.graph_objects as go_
        fld = res["snaps"].get(float(ar["tp"]), res["final"])
        hm = go_.Figure(go_.Heatmap(x=res["xc"], y=res["yc"], z=U.from_field(fld.T, "pressure", system),
                                    colorscale="Blues"))
        hm.update_layout(title="Pressure map at the end of the flow period", height=420)
        st.plotly_chart(hm, use_container_width=True)
        takeaway(f"{res['n']} active cells; lowest flowing pressure {U.from_field(float(pw.min()), 'pressure', system):.0f} {pu}.",
                 "Areal model")


# ------------------------------------------------------------ interference
def _interference(meta, S):
    system, ver = meta["units"], S["ver"]
    pu, ru = U.label("pressure", system), U.label("rate", system)
    st.markdown("#### Interference and pulse test")
    st.caption("Pressure at an observation well caused by one or more active wells, with wellbore storage and "
               "skin at the active well. Fit permeability and porosity (or any property) to the observed response.")
    left, right = st.columns([1, 2])
    with left:
        p = _reservoir("adv_if", meta, ver)
        r = _len(system, "Distance to observation well", 800.0, f"if_r_{system}_{ver}", 10.0, 1e5, 50.0)
        q = float(U.to_field(st.number_input(f"Active-well rate [{ru}]", 1.0, 1e9, float(U.from_field(1000.0, "rate", system)),
                                             key=f"if_q_{system}_{ver}"), "rate", system))
        mode = st.radio("Test type", ["Interference (continuous)", "Pulse test"], horizontal=True, key=f"if_mode_{ver}")
        c1, c2 = st.columns(2)
        on = c1.number_input("Pulse length [hr]", 1.0, 1e4, 24.0, 1.0, key=f"if_on_{ver}")
        off = c2.number_input("Pause between pulses [hr]", 1.0, 1e4, 48.0, 1.0, key=f"if_off_{ver}")
        npulse = st.slider("Number of pulses", 1, 6, 3, key=f"if_n_{ver}")
        noise = st.number_input(f"Synthetic gauge noise [{pu}]", 0.0, 10.0, float(U.from_field(0.02, "pressure", system)),
                                0.01, key=f"if_noise_{system}_{ver}")
        run = st.button("Simulate and fit", type="primary", key="if_run")
    if run:
        if mode.startswith("Interference"):
            sched = [(0.0, q)]
            tend = on * 10
        else:
            sched = [(0.0, 0.0)]
            for i in range(int(npulse)):
                sched += [(off + i * (on + off), q), (off + i * (on + off) + on, 0.0)]
            tend = off + npulse * (on + off) + off
        w = IF.ActiveWell(C=p.C, S=p.S, schedule=sched)
        t = np.linspace(tend / 150, tend, 150)
        truth = IF.dp_observed(p, (r, 0.0), [w], t)
        rng = np.random.default_rng(5)
        dp_obs = truth + rng.normal(0, float(U.to_field(noise, "pressure", system)) if noise else 0.0, t.shape)
        start = H.clone(p, k=p.k * 0.6, phi=min(p.phi * 1.5, 0.5))
        with st.spinner("Fitting..."):
            pf, info = IF.fit_interference(start, (r, 0.0), [w], t, dp_obs, free=("k", "phi"))
        S["interf"] = {"t": t, "dp": dp_obs, "fit": IF.dp_observed(pf, (r, 0.0), [w], t), "k": pf.k, "phi": pf.phi,
                       "true_k": p.k, "true_phi": p.phi, "info": info, "sched": sched, "mode": mode}
        st.rerun()
    d = S.get("interf")
    with right:
        if not d:
            st.info("Set up the test and run it.")
            return
        takeaway(f"Fitted k = {U.from_field(d['k'], 'perm', system):.3g} {U.label('perm', system)} and porosity = {d['phi']:.3f} "
                 f"(synthetic truth {U.from_field(d['true_k'], 'perm', system):.3g} and {d['true_phi']:.3f}).", "Interference result")
        m1, m2, m3 = st.columns(3)
        m1.metric("Permeability", f"{U.from_field(d['k'], 'perm', system):.3g} {U.label('perm', system)}")
        m2.metric("Porosity", f"{d['phi']:.3f}")
        m3.metric("RMS misfit", f"{U.from_field(d['info']['rms_psi'], 'pressure', system):.3g} {pu}")
        fig = F.line_figure("Observation-well pressure change", "time [hr]", f"pressure change [{pu}]",
                            [("fit", d["t"], U.from_field(d["fit"], "pressure", system))])
        import plotly.graph_objects as go_
        fig.add_trace(go_.Scatter(x=d["t"], y=U.from_field(d["dp"], "pressure", system), mode="markers",
                                  name="observed", marker=dict(color=F.ORANGE, size=5)))
        st.plotly_chart(fig, use_container_width=True)
        if d["mode"].startswith("Pulse"):
            on_t = [s for s, q_ in d["sched"] if q_ > 0]
            off_t = [s for s, q_ in d["sched"] if q_ == 0 and s > 0]
            rows = []
            for a, b in zip(on_t, off_t):
                mt = IF.pulse_metrics(d["t"], d["dp"], a, b)
                rows.append({"Pulse start [hr]": round(a, 1), "Lag to peak after pulse end [hr]": round(mt["lag_peak"], 1),
                             "Lag to onset [hr]": round(mt["lag_onset"], 1),
                             f"Amplitude [{pu}]": round(float(U.from_field(mt["amplitude"], "pressure", system)), 4)})
            st.dataframe(pd.DataFrame(rows))


# --------------------------------------------------------- wellbore effects
def _wellbore(meta, S):
    system, ver = meta["units"], S["ver"]
    pu, ru = U.label("pressure", system), U.label("rate", system)
    st.markdown("#### Changing storage, rate-dependent skin and stress-sensitive permeability")
    st.caption("Time-domain wellbore model coupled to the analytical reservoir response. Compare the base case "
               "(constant storage, constant skin, constant permeability) with the effects switched on.")
    left, right = st.columns([1, 2])
    with left:
        p = _reservoir("adv_wb", meta, ver)
        q = float(U.to_field(st.number_input(f"Rate [{ru}]", 1.0, 1e9, float(U.from_field(800.0, "rate", system)),
                                             key=f"wb_q_{system}_{ver}"), "rate", system))
        tp = st.number_input("Flow period [hr]", 1.0, 1e5, 100.0, 10.0, key=f"wb_tp_{ver}")
        tend = st.number_input("Total time [hr]", 2.0, 1e6, 300.0, 10.0, key=f"wb_te_{ver}")
        stor = U.label("storage", system)
        ci = st.number_input(f"Storage at the start of each period [{stor}]", 1e-5, 100.0,
                             float(U.from_field(max(p.C, 0.005) * 4, "storage", system)), 0.001, format="%.4g", key=f"wb_ci_{system}_{ver}")
        cf = st.number_input(f"Storage at long time [{stor}]", 1e-5, 100.0,
                             float(U.from_field(max(p.C, 0.005), "storage", system)), 0.001, format="%.4g", key=f"wb_cf_{system}_{ver}")
        al = st.number_input("Storage change time constant [hr]", 0.01, 1e4, 0.5, 0.1, key=f"wb_al_{ver}")
        D = st.number_input("Rate-dependent skin coefficient D [1/(stb/d)]", 0.0, 1.0, 0.0, 0.0005, format="%.5f", key=f"wb_D_{ver}")
        gam = st.number_input("Permeability modulus gamma [1/psi]", 0.0, 0.01, 0.0, 1e-5, format="%.2e", key=f"wb_g_{ver}")
        go = st.button("Compute", type="primary", key="wb_go")
    if go:
        sched = [(0.0, q), (float(tp), 0.0)]
        t = np.unique(np.concatenate([np.geomspace(0.01, float(tp), 60), float(tp) + np.geomspace(0.01, float(tend) - float(tp), 60)]))
        fx0 = WB.WellboreEffects(C_i=p.C, C_f=p.C, alpha=1.0)
        fx = WB.WellboreEffects(C_i=float(U.to_field(ci, "storage", system)), C_f=float(U.to_field(cf, "storage", system)),
                                alpha=float(al), D=float(D), gamma=float(gam))
        with st.spinner("Marching in time..."):
            base = WB.march(p, fx0, sched, t)
            eff = WB.march(p, fx, sched, t)
        S["wbfx"] = {"t": t, "base": base["pwf"], "eff": eff["pwf"], "qsf": eff["qsf"], "tp": float(tp), "pi": p.pi}
        st.rerun()
    d = S.get("wbfx")
    with right:
        if not d:
            st.info("Set the effects and compute.")
            return
        t = d["t"]
        fig = F.line_figure("Flowing and build-up pressure", "time [hr]", f"pressure [{pu}]",
                            [("base case", t, U.from_field(d["base"], "pressure", system)),
                             ("with effects", t, U.from_field(d["eff"], "pressure", system))])
        st.plotly_chart(fig, use_container_width=True)
        m = t > d["tp"]
        dt = t[m] - d["tp"]
        series = []
        for name, y in (("base case", d["base"]), ("with effects", d["eff"])):
            dp = y[m] - y[np.argmin(abs(t - d["tp"]))]
            ok = dp > 0
            if ok.sum() > 8:
                series.append((name, dt[ok], dp[ok], A.bourdet_derivative(dt[ok], dp[ok])))
        if series:
            fig = F.diag_figure("Build-up diagnostic plot", "shut-in time [hr]", f"dp and derivative [{pu}]",
                                models=[dict(name=n, x=x, dp=U.from_field(dp, "pressure", system),
                                             der=U.from_field(der, "pressure", system)) for n, x, dp, der in series])
            st.plotly_chart(fig, use_container_width=True)
        q_end = float(np.interp(d["tp"] * 0.99, t, d["qsf"]))
        takeaway(f"At the end of the flow period the sandface rate is {U.from_field(q_end, 'rate', system):,.0f} {ru}; "
                 f"the pressure difference between the two cases is {U.from_field(float(abs(d['base'][np.argmin(abs(t - d['tp']))] - d['eff'][np.argmin(abs(t - d['tp']))])), 'pressure', system):.2g} {pu}.",
                 "Wellbore effects")


# -------------------------------------------------------------- large gauges
def _synthetic_gauge(n=40000):
    p = M.WellTestParams(k=40, h=50, C=0.02, S=2.0, pi=4000.0)
    sched = [(0.0, 900.0), (60.0, 0.0), (120.0, 700.0), (180.0, 0.0)]
    t = np.linspace(0.02, 260.0, n)
    clean = M.pwf_history(p, t, sched)
    rng = np.random.default_rng(11)
    temp = 95.0 + (80.0 - 95.0) * np.exp(-(t % 60) / 6.0) * (np.sin(t / 30) ** 2 * 0 + 1) + rng.normal(0, 0.03, n)
    return t, clean + rng.normal(0, 0.15, n), temp, sched, p


def _gauge(meta, S):
    system, ver = meta["units"], S["ver"]
    pu = U.label("pressure", system)
    st.markdown("#### Large gauge records")
    st.caption("Denoise with a wavelet filter, reduce hundreds of thousands of points to a few hundred while "
               "keeping the events, detect rate changes from the pressure alone and look at the temperature channel.")
    up = st.file_uploader("Gauge CSV (columns: time [hr], pressure, optional temperature)", type=["csv"], key=f"ga_up_{ver}")
    c1, c2, c3 = st.columns(3)
    strength = c1.slider("Wavelet threshold strength", 0.0, 2.0, 1.0, 0.1, key=f"ga_str_{ver}")
    dp_tol = c2.number_input(f"Reduction: keep a point when pressure moves by [{pu}]", 0.001, 100.0,
                             float(U.from_field(0.3, "pressure", system)), 0.05, key=f"ga_tol_{system}_{ver}")
    logstep = c3.slider("Reduction: minimum log-time density (%)", 1, 20, 3, key=f"ga_log_{ver}")
    if st.button("Load or generate and process", type="primary", key="ga_go"):
        if up is not None:
            df = pd.read_csv(io.BytesIO(up.getvalue()))
            t = df.iloc[:, 0].to_numpy(float)
            pr = U.to_field(df.iloc[:, 1].to_numpy(float), "pressure", system)
            temp = df.iloc[:, 2].to_numpy(float) if df.shape[1] > 2 else None
            sched = None
        else:
            t, pr, temp, sched, _ = _synthetic_gauge()
        den, sig = RD.wavelet_denoise(pr, strength=strength)
        tol = float(U.to_field(dp_tol, "pressure", system))
        tr, pr_r, idx = RD.smart_reduce(t, den, dp_tol=tol, log_step=logstep / 100.0)
        events = RD.detect_events(t[:: max(1, len(t) // 4000)], den[:: max(1, len(t) // 4000)])
        out = {"t": t, "raw": pr, "den": den, "sigma": sig, "tr": tr, "pr": pr_r, "events": events,
               "sched": sched, "temp": temp}
        if temp is not None:
            try:
                out["trelax"], out["tres"] = RD.temperature_relaxation(t[t < (t[0] + 60)], temp[t < (t[0] + 60)])
            except Exception:
                out["trelax"] = None
        S["gauge"] = out
        st.rerun()
    g = S.get("gauge")
    if not g:
        st.info("Upload a record or press the button to generate a synthetic 40 000-point gauge.")
        return
    takeaway(f"{len(g['t']):,} points reduced to {len(g['tr']):,} ({100 * len(g['tr']) / len(g['t']):.1f} %); "
             f"estimated gauge noise {U.from_field(g['sigma'], 'pressure', system):.3g} {pu}; "
             f"{len(g['events'])} candidate rate changes.", "Gauge processing")
    sub = max(1, len(g["t"]) // 3000)
    fig = F.line_figure("Raw, denoised and reduced pressure", "time [hr]", f"pressure [{pu}]",
                        [("raw", g["t"][::sub], U.from_field(g["raw"][::sub], "pressure", system)),
                         ("denoised", g["t"][::sub], U.from_field(g["den"][::sub], "pressure", system))],
                        vlines=[(e, "") for e in g["events"][:12]])
    import plotly.graph_objects as go_
    fig.add_trace(go_.Scatter(x=g["tr"], y=U.from_field(g["pr"], "pressure", system), mode="markers",
                              name="reduced set", marker=dict(color=F.RED, size=4)))
    st.plotly_chart(fig, use_container_width=True)
    if g.get("events"):
        st.write("Detected event times [hr]: " + ", ".join(f"{e:.1f}" for e in g["events"][:15]))
    if g.get("sched"):
        st.caption("Actual rate changes in the synthetic record: " + ", ".join(f"{s:g}" for s, _ in g["sched"][1:]))
    if g.get("temp") is not None:
        sub2 = max(1, len(g["t"]) // 2000)
        fig = F.line_figure("Temperature channel", "time [hr]", "temperature [degF]",
                            [("temperature", g["t"][::sub2], g["temp"][::sub2])])
        st.plotly_chart(fig, use_container_width=True)
        if g.get("trelax"):
            r = g["trelax"]
            st.caption(f"Temperature relaxation in the first period: T0 {r['T0']:.1f}, final {r['Tf']:.1f}, "
                       f"time constant {r['tau']:.1f} hr.")
