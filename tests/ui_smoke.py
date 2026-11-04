"""Smoke test of app.py and every tab using a *stand-in* for Streamlit and Plotly.

It cannot prove the UI renders (run the real app for that) but it executes all
UI code paths - number_input type rules, session-state flow, button handlers,
engine calls, JSON/HTML generation - and fails on any Python error.
"""
import sys, types, runpy, pathlib, datetime, json, re
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class StreamlitAPIException(Exception):
    pass


class Rerun(Exception):
    pass


PRESSED = set()
CONSUMED = set()
LOG = {"charts": 0, "html": [], "downloads": []}
UPLOAD = {}


def _same_type(*vals):
    vals = [v for v in vals if v is not None]
    kinds = {type(v) if type(v) in (int, float) else float for v in vals}
    return len(kinds) <= 1


class Ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, name):
        if name in ("markdown", "caption", "title", "info", "warning", "success", "error", "write",
                    "metric", "dataframe", "header", "subheader", "text", "code"):
            return lambda *a, **k: None
        raise AttributeError(name)

    def plotly_chart(self, fig, **k):
        LOG["charts"] += 1
        assert isinstance(fig, Fig)

    def columns(self, spec, **k):
        return [Ctx() for _ in range(spec if isinstance(spec, int) else len(spec))]
    def tabs(self, labels, key=None, **k):
        selected = st.session_state.get(key, labels[0])
        containers = [Ctx() for _ in labels]
        for label, container in zip(labels, containers):
            container.open = label == selected
        return containers
    def expander(self, *a, **k): return Ctx()
    def spinner(self, *a, **k): return Ctx()
    def progress(self, *a, **k): return Ctx()
    def empty(self): return None
    def container(self, *a, **k): return Ctx()

    def button(self, label, **k):
        # like Streamlit: a click is True for exactly one script run
        for p in PRESSED:
            if p in label and (label not in CONSUMED):
                CONSUMED.add(label)
                return True
        return False
    def number_input(self, label, min_value=None, max_value=None, value=None, step=None, format=None, key=None, **k):
        if not _same_type(min_value, max_value, value, step):
            raise StreamlitAPIException(f"mixed numeric types in {label!r}: {min_value, max_value, value, step}")
        v = min_value if value is None else value
        if v is None: v = 0.0
        if min_value is not None and v < min_value: raise StreamlitAPIException(f"{label!r}: value {v} < min {min_value}")
        if max_value is not None and v > max_value: raise StreamlitAPIException(f"{label!r}: value {v} > max {max_value}")
        return v
    def slider(self, label, min_value=None, max_value=None, value=None, step=None, key=None, **k):
        if not _same_type(min_value, max_value, value, step):
            raise StreamlitAPIException(f"mixed numeric types in slider {label!r}")
        if value < min_value or value > max_value: raise StreamlitAPIException(f"slider {label!r} out of range {value}")
        return value
    def select_slider(self, label, options=(), value=None, key=None, **k):
        for v in (value if isinstance(value, tuple) else (value,)):
            if v not in options: raise StreamlitAPIException(f"select_slider {label!r} value not in options")
        return value
    def selectbox(self, label, options, index=0, format_func=str, key=None, **k):
        o = list(options); format_func(o[index]); return o[index]
    def radio(self, label, options, index=0, horizontal=False, key=None, **k):
        return list(options)[index]
    def multiselect(self, label, options, default=None, key=None, **k):
        d = list(default or []); assert all(x in options for x in d), (label, d, options); return d
    def checkbox(self, label, value=False, key=None, **k): return value
    def text_input(self, label, value="", key=None, **k): return value
    def date_input(self, label, value=None, **k): return value
    def file_uploader(self, *a, key=None, **k):
        return UPLOAD.get(key)
    def data_editor(self, df, **k): return df
    def download_button(self, label, data, file_name=None, mime=None, **k):
        LOG["downloads"].append((label, data, file_name)); return False
    def rerun(self): raise Rerun()


st = types.ModuleType("streamlit")
ctx = Ctx()
for n in dir(Ctx):
    if not n.startswith("_"):
        setattr(st, n, getattr(ctx, n))
for n in ("markdown", "caption", "title", "info", "warning", "success", "error", "write", "metric",
          "dataframe", "header", "subheader", "text", "code"):
    setattr(st, n, getattr(ctx, n))


class SS(dict):
    __getattr__ = dict.get
    def __setattr__(self, k, v): self[k] = v


st.session_state = SS()
st.sidebar = Ctx()
st.set_page_config = lambda **k: None
st.cache_data = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
st.cache_resource = st.cache_data
comp = types.ModuleType("streamlit.components.v1")
def _html(h, **k): LOG["html"].append(h)
comp.html = _html
st.components = types.ModuleType("streamlit.components"); st.components.v1 = comp
comp2 = types.ModuleType("streamlit.components.v2")
comp2.component = lambda *a, **k: (lambda **kwargs: {})
st.components.v2 = comp2
sys.modules.update({"streamlit": st, "streamlit.components": st.components,
                    "streamlit.components.v1": comp, "streamlit.components.v2": comp2})


class Fig:
    def __init__(self, *a, **k): self.traces = []
    def update_layout(self, *a, **k): return self
    def update_xaxes(self, *a, **k): return self
    def update_yaxes(self, *a, **k): return self
    def add_trace(self, t, *a, **k): self.traces.append(t); return self
    def add_vline(self, *a, **k): return self
    def add_hline(self, *a, **k): return self


go = types.ModuleType("plotly.graph_objects")
go.Figure = Fig
go.Scatter = lambda **k: k
go.Heatmap = lambda **k: k
plotly = types.ModuleType("plotly"); plotly.graph_objects = go
sys.modules.update({"plotly": plotly, "plotly.graph_objects": go})


def run_app(max_reruns=8):
    for _ in range(max_reruns):
        try:
            runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
            return
        except Rerun:
            continue
    raise RuntimeError("too many reruns")


def scenario(name, fluid, units, pressed, ds_check=None, active_tab="00 Test design"):
    st.session_state.clear()
    st.session_state["active_tab"] = active_tab
    PRESSED.clear(); PRESSED.update(pressed); CONSUMED.clear()
    # monkeypatch sidebar radios for fluid / units
    orig = Ctx.radio
    def radio(self, label, options, index=0, horizontal=False, key=None, **k):
        if label == "Fluid": return fluid
        if label == "Units": return units
        return orig(self, label, options, index, horizontal, key)
    Ctx.radio = radio
    st.radio = types.MethodType(radio, ctx)
    LOG["charts"] = 0; LOG["html"].clear(); LOG["downloads"].clear()
    run_app()
    Ctx.radio = orig
    ds = st.session_state.get("dataset")
    print(f"PASS {name}: charts={LOG['charts']} dataset={'yes' if ds else 'no'} "
          f"fit={'yes' if st.session_state.get('last_fit') else 'no'} fits={len(st.session_state['fits'])} "
          f"anim_html={len(LOG['html'])}")


def check_downloads():
    from html.parser import HTMLParser
    rep = [d for d in LOG["downloads"] if d[0].startswith("Export report")]
    proj = [d for d in LOG["downloads"] if d[0] == "Save project"]
    assert rep and proj, "report / project download buttons missing"
    HTMLParser().feed(rep[-1][1])
    assert "<h1>" in rep[-1][1] and "Test summary" in rep[-1][1]
    json.loads(proj[-1][1])
    return rep[-1][1], proj[-1][1]


if __name__ == "__main__":
    scenario("idle, oil, field", "Oil", "Field", set())
    scenario("idle, oil, metric", "Oil", "Metric", set())
    scenario("idle, gas, field", "Gas", "Field", set())
    scenario("generate synthetic from design", "Oil", "Field", {"Generate synthetic"})
    scenario("sample + fit + save + type curve", "Oil", "Field", {"Load sample", "Fit to data", "Save to Compare", "Auto-match"})
    scenario("fit + uncertainty", "Oil", "Field", {"Load sample", "Fit to data", "Show sensitivity", "Run Monte Carlo"})
    lf_ = st.session_state["last_fit"]
    assert lf_ and lf_["info"]["ci"]["k"]["lo"] < lf_["info"]["ci"]["k"]["hi"] and st.session_state.get("mc")
    print("PASS fit uncertainty: CI + sensitivity + Monte Carlo ran")
    scenario("gas sample (metric)", "Gas", "Metric", {"Load sample"})
    assert not LOG["html"], "hidden buildup animation should not be computed"
    scenario("selected buildup animation", "Gas", "Metric", set(), active_tab="Buildup animation")
    # animation HTML must contain valid payload JSON
    assert LOG["html"], "selected animation tab should render"
    m = re.search(r"const D = (\{.*?\});\nconst nF", LOG["html"][-1], re.S)
    assert m and json.loads(m.group(1))["prof"], "animation payload not embedded"
    print("PASS animation payload embedded and parseable")

    # ---------------- phase 2 scenarios
    scenario("multi-rate sample + fit + deconvolution", "Oil", "Field",
             {"Load multi-rate sample", "Run multi-rate regression", "Run deconvolution"})
    assert st.session_state["mr"]["fit"] and st.session_state["deconv"], "multirate/deconv did not run"
    rep, proj = check_downloads()
    assert "Multi-rate record" in rep and "Deconvolution" in rep
    scenario("DST sample metric", "Oil", "Metric", {"Load multi-rate sample"})
    scenario("IPR oil metric", "Oil", "Metric", set())
    scenario("IPR gas field", "Gas", "Field", set())
    scenario("FMB gas demo + fit", "Gas", "Field", {"Generate demo", "Fit in-place"})
    fm = st.session_state["fmb"]["fit"]
    assert fm and abs(fm["G"] / 4.0e6 - 1) < 0.08, fm
    print("PASS fmb gas recovered G =", round(fm["G"] / 1e6, 3), "Bscf (true 4.0)")
    scenario("FMB oil demo + fit (metric)", "Oil", "Metric", {"Generate demo", "Fit in-place"})
    fo = st.session_state["fmb"]["fit"]
    assert fo and abs(fo["N"] / 2.0e7 - 1) < 0.05, fo
    # full interpretation flow -> report -> project round trip
    scenario("full flow for report", "Oil", "Field", {"Load sample", "Fit to data", "Auto-match", "Save to Compare"})
    rep, proj = check_downloads()
    assert "Model regression" in rep and "Semilog" in rep and "<svg" in rep
    from engine import project as P
    _, back = P.loads(proj)
    assert back["dataset"] and len(back["fits"]) == 1
    class Up:  # emulate an uploaded project file
        name, size = "p.wtw.json", len(proj)
        def getvalue(self): return proj.encode()
    UPLOAD["proj_up"] = Up()
    scenario("open project", "Oil", "Field", set())
    assert st.session_state["dataset"] is not None and len(st.session_state["fits"]) == 1
    print("PASS project open restores dataset + fits")

    # ---------------- phase 3: simulator
    scenario("simulator run + calibrate", "Oil", "Field", {"Run simulation", "Calibrate"})
    sim = st.session_state["sim"]
    assert sim and len(sim["res"]["t"]) > 10 and sim["res"]["snapshots"], "simulator did not run"
    fit = st.session_state["sim_fit"]
    assert fit and fit["info"]["rms"] < 3.0, fit and fit["info"]
    print("PASS simulator: steps =", sim["res"]["steps"], "calibrated", {k: round(v, 3) for k, v in fit["info"]["fitted"].items()})
    rep, proj = check_downloads()
    pj = json.loads(proj)
    assert pj["state"]["sim"] and pj["state"]["sim"]["res"]["t"], "simulator results missing from the project file"
    deck = [d for d in LOG["downloads"] if d[0].startswith("Download OPM")]
    assert deck and "DIMENS" in deck[-1][1] and "COMPDAT" in deck[-1][1], "OPM deck not offered"
    print("PASS simulator saved in project + OPM deck offered")
    from ui import report_ui
    rep_all = report_ui.build({"well": "W", "field": "F", "date": "d", "analyst": "", "fluid": "Oil", "units": "Field"},
                              st.session_state, "Interpretation + simulation")
    assert "Key findings" in rep_all and "Simulator calibration" in rep_all and "<svg" in rep_all
    rep_m = report_ui.build({"well": "W", "field": "F", "date": "d", "analyst": "", "fluid": "Oil", "units": "Metric"},
                            st.session_state, "Everything")
    assert "Simulator model" in rep_m
    print("PASS combined interpretation + simulation report")
    class Up2:
        name, size = "s.wtw.json", len(proj)
        def getvalue(self): return proj.encode()
    UPLOAD["proj_up"] = Up2()
    scenario("open project with simulator", "Oil", "Field", set())
    s2 = st.session_state["sim"]
    assert s2 and abs(s2["cfg"].k - sim["cfg"].k) < 1e-9 and len(s2["res"]["t"]) == len(sim["res"]["t"])
    print("PASS project open restores simulator run")
    UPLOAD.clear()
    scenario("simulator metric run + calibrate", "Oil", "Metric", {"Run simulation", "Calibrate"})
    simm = st.session_state["sim"]
    assert abs(simm["cfg"].k - sim["cfg"].k) < 1e-6 and abs(simm["cfg"].h - sim["cfg"].h) < 1e-6, "metric round trip changed the model"
    assert abs(simm["cfg"].p_datum - sim["cfg"].p_datum) < 1e-3, (simm["cfg"].p_datum, sim["cfg"].p_datum)
    assert abs(st.session_state["sim_fit"]["info"]["fitted"]["k"] - fit["info"]["fitted"]["k"]) / fit["info"]["fitted"]["k"] < 0.05
    print("PASS simulator metric round trip")

    # ---------------- v4: advanced tab
    scenario("advanced: areal + interference + wellbore + gauge", "Oil", "Field",
             {"Run areal model", "Simulate and fit", "Compute", "Load or generate"})
    S_ = st.session_state
    assert S_["areal"] and S_["areal"]["res"]["pwf"].shape[0] > 10, "areal model did not run"
    assert S_["interf"] and S_["interf"]["info"]["rms_psi"] < 0.5, S_["interf"] and S_["interf"]["info"]
    assert S_["wbfx"] and np.all(np.isfinite(S_["wbfx"]["eff"])), "wellbore effects did not run"
    assert S_["gauge"] and len(S_["gauge"]["tr"]) < 0.1 * len(S_["gauge"]["t"]), "gauge reduction failed"
    print("PASS advanced tab: areal, interference fit, wellbore effects, gauge reduction")
    scenario("advanced metric", "Oil", "Metric", {"Run areal model", "Simulate and fit", "Compute", "Load or generate"})
    assert abs(st.session_state["interf"]["k"] / st.session_state["interf"]["true_k"] - 1) < 0.15
    print("PASS advanced tab metric: interference fit recovers k within 15 %")
    # new analytical models flow through the interpretation tab
    from engine import models as M_
    for mdl, bnd in (("horizontal", "none"), ("mfhw", "none"), ("fracture_fc", "none"),
                     ("composite3", "none"), ("homogeneous", "rectangle"), ("homogeneous", "circle")):
        p_ = M_.WellTestParams(model=mdl, boundary=bnd)
        dp_, d_ = M_.unit_rate_response(p_, np.geomspace(1e-2, 1e3, 12))
        assert np.all(np.isfinite(dp_)) and np.all(np.isfinite(d_)), (mdl, bnd)
    print("PASS all new models produce finite responses")
