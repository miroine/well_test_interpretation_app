"""Offline, bidirectional rate/pressure timeline with accessible timing handles."""
from __future__ import annotations

import math
from pathlib import Path

import streamlit as st
from streamlit.components.v2 import component

from engine import units as U

_CSS = """
    .timeline { font: 13px "Segoe UI", Arial, sans-serif; color: #051C2C; }
    svg { display: block; width: 100%; background: white; }
    .handle { cursor: ew-resize; touch-action: none; outline: none; }
    .handle:focus .grip { stroke: #2251FF; stroke-width: 4; }
    .status { min-height: 2.8em; margin: 4px 0; line-height: 1.4; }
    .legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 4px 0; }
    """
_JS = Path(__file__).with_suffix(".js").read_text(encoding="utf-8")


def _timeline(**kwargs):
    return component(
        "welltest_design_timeline", html='<div class="timeline"></div>',
        css=_CSS, js=_JS,
    )(**kwargs)


def validate_edit(edit, tp, dt_end):
    """Reject malformed, out-of-range or stale browser events."""
    if not isinstance(edit, dict) or set(edit) != {"tp", "dt_end", "base"}:
        raise ValueError("The timeline returned an invalid timing edit.")
    base = edit["base"]
    values = (edit["tp"], edit["dt_end"])
    if (not isinstance(base, list) or len(base) != 2 or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in (*base, *values))):
        raise ValueError("Timeline durations must be finite numbers.")
    if not (math.isclose(base[0], tp, abs_tol=1e-6) and math.isclose(base[1], dt_end, abs_tol=1e-6)):
        raise ValueError("The design changed while dragging. Please drag the updated timeline again.")
    if not (1.0 <= values[0] <= 1e5 and 0.1 <= values[1] <= 1e5):
        raise ValueError("Flow duration must be 1-100,000 hr; shut-in must be 0.1-100,000 hr.")
    return tuple(float(v) for v in values)


def render(curves, tp, dt_end, q_user, system, fluid, ver):
    key = f"design_timeline_{ver}"

    def apply_edit():
        value = st.session_state[key]["timing"]
        try:
            new_tp, new_dt = validate_edit(value, st.session_state[f"design_tp_{ver}"],
                                           st.session_state[f"design_dtend_{ver}"])
        except ValueError as exc:
            st.session_state["_design_timing_error"] = str(exc)
        else:
            st.session_state[f"design_tp_{ver}"] = new_tp
            st.session_state[f"design_dtend_{ver}"] = new_dt
            st.session_state.pop("_design_timing_error", None)

    if st.session_state.get("_design_timing_error"):
        st.error(st.session_state["_design_timing_error"])
    timeline = curves["timeline"]
    kind = "gasrate" if fluid == "Gas" else "rate"
    _timeline(
        key=key, height="content",
        data={"tp": tp, "dt_end": dt_end,
              "time": timeline["t"].tolist(),
              "pressure": U.from_field(timeline["pressure"], "pressure", system).tolist(),
              "rate": float(U.from_field(q_user, kind, system)),
              "pressure_unit": U.label("pressure", system), "rate_unit": U.label(kind, system)},
        on_timing_change=apply_edit,
    )
