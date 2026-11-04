"""Buildup animation - pressure diffusion, derivative and flow regimes in motion."""
from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

from engine import viz
from . import helpers as H
from .animation import build_html


def _payload(params_json: str, q: float, tp: float, dt_end: float):
    p = H.params_from_dict(json.loads(params_json))
    return viz.animation_payload(p, q, tp, dt_end)


def render(meta):
    S = st.session_state
    options = []
    if S.get("design_eng"):
        options.append("Test design (tab 00)")
    lf = S.get("last_fit")
    if lf and S.get("dataset"):
        options.append("Latest regression (tab 02)")
    if not options:
        st.info("Define a test design in tab 00 first.")
        return
    src = st.radio("Animate", options, horizontal=True)
    if src.startswith("Test design"):
        de = S["design_eng"]
        params, q, tp, dt_end = de["params"], de["q"], de["tp"], de["dt_end"]
    else:
        ds = S["dataset"]
        params, q, tp = lf["params"], lf["q"], lf["tp"]
        dt_end = float(max(ds["dt"]))
    with st.spinner("Computing frames..."):
        payload = _payload(json.dumps(params, sort_keys=True), float(q), float(tp), float(dt_end))
    components.html(build_html(payload), height=720, scrolling=False)
