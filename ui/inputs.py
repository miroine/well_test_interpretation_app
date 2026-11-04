"""Streamlit input widgets. Keys carry the unit system and a version counter so
that switching units / loading a dataset re-initialises values (a Streamlit
keyed widget ignores `value=` once its state exists)."""
from __future__ import annotations

import numpy as np
import streamlit as st

from engine import models as M
from engine import units as U
from . import helpers as H


def radius_input(prefix, radius_ft, system, ver, container=st):
    """Radius, never diameter; return feet regardless of display preference."""
    unit = container.selectbox("Well radius input unit", [U.label("length", system), "in"],
                               key=f"{prefix}_rw_unit_{system}_{ver}",
                               help="Radius, not diameter. 12 inches = 1 foot; 1 inch = 0.0254 m.")
    factor = 12. if unit == "in" else float(U.from_field(1., "length", system))
    key = f"{prefix}_rw_{'in' if unit == 'in' else system}_{ver}"
    tracking = f"_{prefix}_rw_display_{ver}"
    previous = st.session_state.get(tracking)
    if previous and previous["key"] != key:
        old = st.session_state.get(previous["key"], previous["value"])
        st.session_state[key] = float(old) / previous["factor"] * factor
    value = container.number_input(f"Wellbore radius [{unit}]", min_value=.01 * factor,
                                    max_value=5. * factor, value=float(radius_ft * factor),
                                    step=.01 * factor, format="%.4g", key=key)
    st.session_state[tracking] = {"key": key, "factor": factor, "value": float(value)}
    return float(value) / factor


def _number(prefix, spec, params, system, ver, container):
    name, label, kind, lo, hi, step, fmt = spec
    if name == "rw":
        return radius_input(prefix, params.rw, system, ver, container)
    if name == "S":
        model = st.session_state.get(f"{prefix}_model_{ver}", params.model)
        if model in M.INFINITE_FRACTURE_MODELS:
            if params.S < 0:
                container.warning("This saved model contains unsupported negative additional fracture skin. "
                                  "Re-enter the intended total equivalent radial skin below; the input starts "
                                  "from zero additional skin, not from the invalid saved value.")
            # Geometry widgets are rendered later; use their current session values.
            radius = st.session_state.get(f"_{prefix}_rw_display_{ver}")
            rw = radius["value"] / radius["factor"] if radius else params.rw
            xf = float(U.to_field(st.session_state.get(f"{prefix}_xf_{system}_{ver}",
                                  U.from_field(params.xf, "length", system)), "length", system))
            geometric = M.fracture_geometric_skin(H.clone(params, model=model, rw=rw, xf=xf))
            definition = container.selectbox("Fracture skin definition",
                ["Additional skin", "Total equivalent radial skin"], index=0 if model == "fracture_dual_pss" else 1,
                key=f"{prefix}_skin_definition_{ver}",
                help="The explicit fracture already represents stimulation. Total skin includes its late-time geometric contribution.")
            tracking = f"_{prefix}_skin_display_{ver}"
            previous = st.session_state.get(tracking)
            if definition == "Total equivalent radial skin":
                total_key = f"{prefix}_S_total_{system}_{ver}"
                if previous and previous["definition"] != definition:
                    st.session_state[total_key] = float(previous["additional"] + geometric)
                value = container.number_input("Total equivalent radial skin", min_value=float(geometric),
                    max_value=float(geometric + hi), value=float(geometric + max(params.S, 0.)),
                    step=float(step), format=fmt, key=total_key)
                additional = float(value - geometric)
                container.caption(f"Late-time approximation: S_total = S_additional + ln(2 rw/xf). "
                                  f"Geometric skin = {geometric:.3f}; additional skin = {additional:.3f}. "
                                  "Not an exact early-time equivalence. Regression fits additional skin.")
                st.session_state[tracking] = {"definition": definition, "additional": additional}
                return additional
            container.caption(f"Fracture geometry already gives approximate equivalent radial skin {geometric:.3f}. "
                              "Additional skin must be non-negative; choose total equivalent radial skin to enter negative total values.")
            additional_key = f"{prefix}_S_{system}_{ver}"
            if previous and previous["definition"] != definition:
                st.session_state[additional_key] = float(previous["additional"])
            value = container.number_input("Additional fracture skin", min_value=0., max_value=float(hi),
                value=float(np.clip(params.S, 0., hi)), step=float(step), format=fmt, key=additional_key)
            st.session_state[tracking] = {"definition": definition, "additional": float(value)}
            return float(value)
    unit = U.label(kind, system)
    lo_d, hi_d = U.from_field(lo, kind, system), U.from_field(hi, kind, system)
    step_d = float(step) * (U.from_field(1.0, kind, system) if kind != "temp" else 1.0)
    disp = float(np.clip(U.from_field(getattr(params, name), kind, system), lo_d, hi_d))
    val = container.number_input(
        f"{label} [{unit}]" if unit else label, min_value=float(lo_d), max_value=float(hi_d),
        value=disp, step=step_d, format=fmt, key=f"{prefix}_{name}_{system}_{ver}")
    return float(U.to_field(val, kind, system))


def params_form(prefix: str, base: M.WellTestParams, system: str, fluid: str, ver: int,
                container=st) -> M.WellTestParams:
    """Reservoir + well + model + boundary form. Returns params in field units."""
    vals = {}
    skip = {"mu", "B", "ct"} if fluid == "Gas" else set()
    with container.expander("Reservoir, fluid and wellbore", expanded=True):
        for spec in H.BASE_SPEC:
            if spec[0] in skip:
                continue
            vals[spec[0]] = _number(prefix, spec, base, system, ver, st)
    models = list(H.MODEL_LABELS)
    model = container.selectbox(
        "Reservoir model", models, index=models.index(base.model) if base.model in models else 0,
        format_func=H.MODEL_LABELS.get, key=f"{prefix}_model_{ver}")
    vals["model"] = model
    if model in H.MODEL_SPEC:
        with container.expander("Model parameters", expanded=True):
            for spec in H.MODEL_SPEC[model]:
                vals[spec[0]] = _number(prefix, spec, base, system, ver, st)
            if model == "fracture_dual_pss":
                st.caption("Warren-Root PSS reservoir with an infinite-conductivity fracture. "
                           "Lambda is dimensionless on well radius rw; internally lambda_xf = lambda_rw (xf/rw)^2. "
                           "Confirm reference-length and transfer conventions before comparing other software.")
            if model == "horizontal":
                bcs = {"sealed": "Sealed", "const_p": "Constant pressure"}
                vals["top_bc"] = st.selectbox("Top boundary (gas cap)", list(bcs), index=list(bcs).index(base.top_bc),
                                              format_func=bcs.get, key=f"{prefix}_topbc_{ver}")
                vals["bot_bc"] = st.selectbox("Bottom boundary (aquifer)", list(bcs), index=list(bcs).index(base.bot_bc),
                                              format_func=bcs.get, key=f"{prefix}_botbc_{ver}")
    bks = list(H.BOUNDARY_LABELS)
    boundary = container.selectbox(
        "Outer boundary", bks, index=bks.index(base.boundary) if base.boundary in bks else 0,
        format_func=H.BOUNDARY_LABELS.get, key=f"{prefix}_boundary_{ver}")
    vals["boundary"] = boundary
    if boundary in H.BOUNDARY_SPEC:
        with container.expander("Boundary geometry", expanded=True):
            for spec in H.BOUNDARY_SPEC[boundary]:
                vals[spec[0]] = _number(prefix, spec, base, system, ver, st)
    out = H.clone(base, **vals)
    from engine import models as _M
    warn = _M.boundary_warning(out)
    if warn:
        container.warning(warn)
    return out


def gas_form(prefix: str, system: str, ver: int, container=st, defaults: dict | None = None) -> dict:
    defaults = defaults or {"T_F": 220.0, "gamma_g": 0.68, "cf": 4e-6}
    with container.expander("Gas properties", expanded=True):
        T = st.number_input(f"Reservoir temperature [{U.label('temp', system)}]",
                            value=float(U.from_field(defaults["T_F"], "temp", system)), step=5.0,
                            key=f"{prefix}_gasT_{system}_{ver}")
        g = st.number_input("Gas gravity (air = 1)", 0.55, 1.2, float(defaults["gamma_g"]), 0.01, key=f"{prefix}_gasg_{ver}")
        cf = st.number_input(f"Formation + water compressibility [{U.label('ct', system)}]",
                             value=float(U.from_field(defaults["cf"], "ct", system)), format="%.3e",
                             key=f"{prefix}_gascf_{system}_{ver}")
    return {"T_F": float(U.to_field(T, "temp", system)), "gamma_g": float(g),
            "cf": float(U.to_field(cf, "ct", system))}


def rate_input(prefix: str, fluid: str, system: str, ver: int, default_field: float, container=st):
    kind = "gasrate" if fluid == "Gas" else "rate"
    disp = float(U.from_field(default_field, kind, system))
    v = container.number_input(f"Production rate before shut-in [{U.label(kind, system)}]",
                               min_value=0.0, value=disp, step=max(disp * 0.05, 1.0),
                               key=f"{prefix}_q_{fluid}_{system}_{ver}")
    return float(U.to_field(v, kind, system))
