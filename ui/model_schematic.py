"""Model-aware SVG schematic; playback reuses the designed test's solver samples."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from streamlit.components.v2 import component

from engine import units as U
from . import helpers as H

_JS = Path(__file__).with_suffix(".js").read_text(encoding="utf-8")
_CSS = """
.schematic { font: 14px "Segoe UI", Arial, sans-serif; color: #051C2C; }
.controls { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin: 8px 0; }
button, select { font: inherit; background: white; color: #051C2C; border: 1px solid #D5DEE9;
  padding: 6px 12px; border-radius: 2px; cursor: pointer; }
button:focus-visible, input:focus-visible, select:focus-visible { outline: 2px solid #2251FF; }
input[type=range] { flex: 1; min-width: 120px; accent-color: #2251FF; }
svg { width: 100%; display: block; background: #F7F9FC; border: 1px solid #D5DEE9; }
.readout { font-variant-numeric: tabular-nums; min-height: 3em; line-height: 1.5; }
.parameters { display: flex; flex-wrap: wrap; gap: 5px 16px; font-size: 12px; margin: 8px 0; }
.note { font-size: 12px; color: #4a5f75; }
"""


def payload(p, curves, tp, dt_end, q_user, system, fluid):
    """Keep schematic geometry dimensionless; only labels/readouts change units."""
    specs = H.MODEL_SPEC.get(p.model, []) + H.BOUNDARY_SPEC.get(p.boundary, [])
    details = []
    for name, label, kind, *_ in specs:
        value = float(U.from_field(getattr(p, name), kind, system))
        details.append(f"{label}: {value:.4g} {U.label(kind, system)}".strip())
    timeline = curves["timeline"]
    return {
        "model": p.model, "model_label": H.MODEL_LABELS[p.model],
        "boundary": p.boundary, "boundary_label": H.BOUNDARY_LABELS[p.boundary],
        "params": p.to_dict(), "details": details, "tp": tp, "dt_end": dt_end,
        "time": np.asarray(timeline["t"], float).tolist(),
        "pressure": U.from_field(np.asarray(timeline["pressure"], float), "pressure", system).tolist(),
        "pressure_unit": U.label("pressure", system),
        "pressure_label": "Normalized pseudo-pressure" if fluid == "Gas" else "Bottom-hole pressure",
        "rate": float(U.from_field(q_user, "gasrate" if fluid == "Gas" else "rate", system)),
        "rate_unit": U.label("gasrate" if fluid == "Gas" else "rate", system),
    }


def render(p, curves, tp, dt_end, q_user, system, fluid, ver):
    component("welltest_model_schematic", html='<div class="schematic"></div>', css=_CSS, js=_JS)(
        key=f"design_schematic_{ver}", data=payload(p, curves, tp, dt_end, q_user, system, fluid),
        height="content",
    )
