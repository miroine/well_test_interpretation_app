"""04 Compare - saved interpretations side by side."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from .theme import takeaway

from engine import models as M
from engine import units as U
from engine import viz
from . import figures as F
from . import helpers as H


def render(meta):
    S = st.session_state
    system = meta["units"]
    pu = U.label("pressure", system)
    fits, ds, ie = S["fits"], S.get("dataset"), S.get("interp_eng")
    if not fits:
        st.info("Run a regression in tab 02 and press 'Save to Compare' to build a list of alternatives.")
        return
    rows = []
    for f in fits:
        p = H.params_from_dict(f["params"])
        rows.append({"Name": f["name"], "Model": H.MODEL_LABELS[p.model], "Boundary": H.BOUNDARY_LABELS[p.boundary],
                     "k [md]": round(p.k, 4), "Skin": round(p.S, 3),
                     f"C [{U.label('storage', system)}]": float(f"{U.from_field(p.C, 'storage', system):.4g}"),
                     "RMS (log)": round(f["rms"], 4)})
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)
    best = int(np.argmin([f["rms"] for f in fits]))
    takeaway(f"Lowest-misfit model: {fits[best]['name']} (RMS {fits[best]['rms']:.3g}); misfit alone cannot discriminate between models.", "Model comparison")
    st.caption(f"Lowest misfit: {fits[best]['name']}. Misfit alone cannot distinguish models that fit "
               "equally well - check that each parameter set is physically plausible and consistent with geology.")
    models = []
    for f in fits:
        p = H.params_from_dict(f["params"])
        dtmax = float(np.max(ds["dt"])) if ds else 100.0
        dtmin = float(np.min(ds["dt"])) if ds else 1e-3
        cur = viz.model_curves(p, f["q"], f["tp"], dtmax, n=120, dt_min=dtmin)
        models.append({"name": f["name"], "x": cur["ta"], "dp": U.from_field(cur["dp"], "pressure", system),
                       "der": U.from_field(cur["der"], "pressure", system)})
    data = None
    if ds and ie:
        d = ie["diag"]
        data = {"x": d["ta"], "dp": U.from_field(d["dp"], "pressure", system),
                "der": U.from_field(d["der"], "pressure", system)}
    st.plotly_chart(F.diag_figure("Saved interpretations vs data", "Agarwal equivalent time [hr]",
                                  f"dp and derivative [{pu}]", data=data, models=models), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.download_button("Download table (CSV)", df.to_csv(index=False).encode(), "interpretations.csv", "text/csv")
    if c2.button("Clear saved interpretations"):
        S["fits"] = []
        st.rerun()
