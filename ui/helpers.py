"""UI-side helpers that do not import Streamlit (so they can be unit-tested)."""
from __future__ import annotations

from dataclasses import fields

import numpy as np

from engine import units as U
from engine import gas as G
from engine import models as M

# name, label, unit kind, min, max, step(display), format
BASE_SPEC = [
    ("k",   "Permeability k",          "perm",     1e-3, 1e5,  1.0,   "%.4g"),
    ("h",   "Net thickness h",         "length",   0.1,  5e3,  1.0,   "%.4g"),
    ("phi", "Porosity",                "none",     0.005, 0.6, 0.01,  "%.3f"),
    ("mu",  "Viscosity",               "visc",     0.005, 1e4, 0.05,  "%.4g"),
    ("ct",  "Total compressibility",   "ct",       1e-9, 1.0,  1e-6,  "%.3e"),
    ("B",   "Formation volume factor", "none",     0.5,  5.0,  0.01,  "%.3f"),
    ("rw",  "Wellbore radius",         "length",   0.01, 5.0,  0.01,  "%.4g"),
    ("pi",  "Initial pressure",        "pressure", 1.0,  2e4,  10.0,  "%.1f"),
    ("C",   "Wellbore storage C",      "storage",  1e-6, 100.0, 0.001, "%.4g"),
    ("S",   "Skin",                    "none",     -7.0, 100.0, 0.5,  "%.2f"),
]
MODEL_SPEC = {
    "dual_pss":  [("omega", "Storativity ratio omega", "none", 1e-4, 0.99, 0.01, "%.4f"),
                  ("lam", "Interporosity flow lambda", "none", 1e-10, 1.0, 1e-7, "%.3e")],
    "dual_slab": [("omega", "Storativity ratio omega", "none", 1e-4, 0.99, 0.01, "%.4f"),
                  ("lam", "Interporosity flow lambda", "none", 1e-10, 1.0, 1e-7, "%.3e")],
    "fracture":  [("xf", "Fracture half-length xf", "length", 1.0, 5e3, 10.0, "%.4g")],
    "composite": [("r_inner", "Inner-zone radius", "length", 2.0, 1e4, 10.0, "%.4g"),
                  ("M", "Mobility ratio M (inner/outer)", "none", 0.01, 100.0, 0.1, "%.3f"),
                  ("D", "Diffusivity ratio (inner/outer)", "none", 0.01, 100.0, 0.1, "%.3f")],
    "limited_entry": [("kv_kh", "kv/kh", "none", 1e-4, 1.0, 0.01, "%.4f"),
                      ("perf_frac", "Perforated fraction b/h", "none", 0.02, 1.0, 0.05, "%.3f"),
                      ("perf_pos", "Perforation centre depth / h", "none", 0.01, 0.99, 0.05, "%.3f")],
    "fracture_fc": [("xf", "Fracture half-length xf", "length", 1.0, 5e3, 10.0, "%.4g"),
                    ("FcD", "Dimensionless conductivity FcD", "none", 0.05, 1e4, 1.0, "%.4g")],
    "mfhw": [("xf", "Fracture half-length xf", "length", 1.0, 5e3, 10.0, "%.4g"),
             ("FcD", "Dimensionless conductivity FcD", "none", 0.05, 1e4, 1.0, "%.4g"),
             ("n_frac", "Number of transverse fractures", "none", 1.0, 12.0, 1.0, "%.0f"),
             ("f_space", "Fracture spacing", "length", 10.0, 5e3, 10.0, "%.4g")],
    "composite3": [("r_inner", "Inner-zone radius", "length", 2.0, 1e4, 10.0, "%.4g"),
                   ("r_mid", "Middle-zone outer radius", "length", 5.0, 1e5, 10.0, "%.4g"),
                   ("M", "Mobility ratio inner/middle", "none", 0.01, 100.0, 0.1, "%.3f"),
                   ("D", "Diffusivity ratio inner/middle", "none", 0.01, 100.0, 0.1, "%.3f"),
                   ("M3", "Mobility ratio inner/outer", "none", 0.01, 100.0, 0.1, "%.3f"),
                   ("D3", "Diffusivity ratio inner/outer", "none", 0.01, 100.0, 0.1, "%.3f")],
    "horizontal": [("Lw", "Horizontal well length", "length", 20.0, 2e4, 50.0, "%.4g"),
                   ("kv_kh", "kv/kh", "none", 1e-4, 1.0, 0.01, "%.4f"),
                   ("zw_h", "Well depth below top / h", "none", 0.02, 0.98, 0.05, "%.3f")],
}
BOUNDARY_SPEC = {
    "fault":         [("d1", "Distance to fault", "length", 10.0, 5e4, 10.0, "%.4g")],
    "const_p_fault": [("d1", "Distance to constant-pressure boundary", "length", 10.0, 5e4, 10.0, "%.4g")],
    "channel":       [("d1", "Distance to boundary 1", "length", 10.0, 5e4, 10.0, "%.4g"),
                      ("d2", "Distance to boundary 2", "length", 10.0, 5e4, 10.0, "%.4g")],
    "circle":        [("re", "Reservoir radius", "length", 20.0, 1e6, 50.0, "%.4g")],
    "circle_cp":     [("re", "Reservoir radius", "length", 20.0, 1e6, 50.0, "%.4g")],
    "rectangle":     [("xe", "Rectangle length (x)", "length", 20.0, 1e6, 50.0, "%.4g"),
                      ("ye", "Rectangle width (y)", "length", 20.0, 1e6, 50.0, "%.4g"),
                      ("xw", "Well position x", "length", 1.0, 1e6, 50.0, "%.4g"),
                      ("yw", "Well position y", "length", 1.0, 1e6, 50.0, "%.4g")],
    "corner":        [("d1", "Distance to fault 1", "length", 10.0, 5e4, 10.0, "%.4g"),
                      ("d2", "Distance to fault 2", "length", 10.0, 5e4, 10.0, "%.4g")],
}
MODEL_LABELS = {
    "homogeneous": "Homogeneous", "dual_pss": "Dual porosity (pseudo-steady)",
    "dual_slab": "Dual porosity (transient slab)", "composite": "Radial composite",
    "fracture": "Infinite-conductivity fracture", "limited_entry": "Limited entry (partial penetration)",
    "horizontal": "Horizontal well",
    "composite3": "Three-zone radial composite",
    "fracture_fc": "Finite-conductivity fracture",
    "mfhw": "Multi-fractured horizontal well",
}
BOUNDARY_LABELS = {
    "none": "Infinite acting", "fault": "Sealing fault",
    "const_p_fault": "Constant-pressure boundary", "channel": "Parallel sealing faults (channel)",
    "corner": "Intersecting sealing faults (90 deg)",
    "circle": "Closed circle (sealed)", "circle_cp": "Circle, constant-pressure edge",
    "rectangle": "Closed rectangle (sealed)",
}


def spec_value_to_display(params: M.WellTestParams, name, kind, system):
    return float(U.from_field(getattr(params, name), kind, system))


def display_to_field(value, kind, system):
    return float(U.to_field(value, kind, system))


def gas_equivalent(p: M.WellTestParams, T_F: float, gamma_g: float, q_mscfd: float,
                   cf: float = 4e-6):
    """Return (params, q_rb_per_day) that make the oil-type engine reproduce a gas
    well in normalized pseudo-pressure (psi-equivalent). Fluid props at initial p."""
    gp = G.gas_props(p.pi, T_F, gamma_g)
    out = M.WellTestParams(**p.to_dict())
    out.mu, out.B, out.ct = gp["mu"], 1.0, gp["cg"] + cf
    return out, q_mscfd * 1000.0 * gp["Bg"], gp


def clone(p: M.WellTestParams, **kw) -> M.WellTestParams:
    d = p.to_dict()
    d.update(kw)
    return M.WellTestParams(**d)


def params_from_dict(d: dict) -> M.WellTestParams:
    names = {f.name for f in fields(M.WellTestParams)}
    return M.WellTestParams(**{k: v for k, v in d.items() if k in names})


def metric_str(value, kind, system, fmt="{:,.3g}"):
    return f"{fmt.format(U.from_field(value, kind, system))} {U.label(kind, system)}"


def fit_free_params(model: str, boundary: str):
    free = ["k", "S", "C"]
    free += {"dual_pss": ["omega", "lam"], "dual_slab": ["omega", "lam"],
             "fracture": ["xf"], "composite": ["r_inner", "M", "D"],
             "fracture_fc": ["xf", "FcD"], "composite3": ["r_inner", "r_mid", "M", "M3"], "mfhw": ["xf", "FcD"], "horizontal": ["Lw", "kv_kh"]}.get(model, [])
    if boundary in ("circle", "circle_cp"):
        free.append("re")
    elif boundary == "rectangle":
        free += ["xe", "ye"]
    if boundary in ("fault", "const_p_fault"):
        free.append("d1")
    elif boundary in ("channel", "corner"):
        free += ["d1", "d2"]
    return free


def engine_view(p: M.WellTestParams, fluid: str, gas: dict | None, q_user_field: float):
    """(engine params, engine rate, gas props) for either fluid.

    Oil: params unchanged, rate in stb/d.   Gas: normalized pseudo-pressure
    equivalent with rate in reservoir bbl/d.
    """
    if fluid == "Gas" and gas:
        pe, q_rb, gp = gas_equivalent(p, gas["T_F"], gas["gamma_g"], q_user_field, gas.get("cf", 4e-6))
        return pe, q_rb, gp
    return p, q_user_field, None
