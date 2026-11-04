"""Editable period boundaries shared by recorded pressure and rate history."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from streamlit.components.v2 import component

from engine import units as U
from .design_timeline import _CSS

_JS = Path(__file__).with_suffix(".js").read_text(encoding="utf-8")


def validate_edit(edit, schedule, end):
    if not isinstance(edit, dict) or set(edit) != {"base", "index", "start"}:
        raise ValueError("Invalid period timing edit.")
    try:
        base = np.asarray(edit["base"], float)
        current = np.asarray(schedule, float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid period schedule.") from exc
    if base.shape != current.shape or not np.isfinite(base).all() or not np.array_equal(base, current):
        raise ValueError("The schedule changed while editing. Drag the updated boundary again.")
    i, start = edit["index"], edit["start"]
    if isinstance(i, bool) or not isinstance(i, int) or not 1 <= i < len(schedule):
        raise ValueError("The first period starts at zero and cannot be moved.")
    if isinstance(start, bool) or not isinstance(start, (int, float)) or not np.isfinite(start):
        raise ValueError("Period start must be a finite number.")
    stop = schedule[i + 1][0] if i + 1 < len(schedule) else end
    if not schedule[i - 1][0] < start < stop:
        raise ValueError("Period starts must stay strictly ordered and inside the recorded test.")
    updated = [tuple(row) for row in schedule]
    updated[i] = (float(start), updated[i][1])
    return updated


def apply_edit(state, edit):
    mr = state["mr"]
    updated = validate_edit(edit, mr["schedule"], float(np.max(mr["t"])))
    state["mr"] = {**mr, "schedule": updated, "fit": None}
    state["deconv"] = None


def render(mr, system, selected, ver):
    key = f"mr_timeline_{ver}"

    def changed():
        try:
            apply_edit(st.session_state, st.session_state[key]["timing"])
        except ValueError as exc:
            st.session_state["_mr_timing_error"] = str(exc)
        else:
            st.session_state.pop("_mr_timing_error", None)

    if st.session_state.get("_mr_timing_error"):
        st.error(st.session_state["_mr_timing_error"])
    st.caption("Shaded bands identify the same periods on both plots; the diagnosed period is emphasized. "
               "Drag a vertical start marker on either plot, then release to apply. "
               "Recorded pressure stays unchanged; old regression/deconvolution results are cleared. "
               "The first start remains at zero. Arrow keys: 0.1 hr; Shift: 1 hr.")
    component("welltest_multirate_timeline", html='<div class="timeline"></div>', css=_CSS, js=_JS)(
        key=key, height="content", on_timing_change=changed,
        data={"schedule": [list(row) for row in mr["schedule"]], "time": np.asarray(mr["t"]).tolist(),
              "pressure": U.from_field(np.asarray(mr["pwf"]), "pressure", system).tolist(),
              "rates": [float(U.from_field(row[1], "rate", system)) for row in mr["schedule"]],
              "pressure_unit": U.label("pressure", system), "rate_unit": U.label("rate", system),
              "selected": int(selected)})
