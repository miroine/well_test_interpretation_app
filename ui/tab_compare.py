"""04 Compare - saved interpretations side by side."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import streamlit as st
from .theme import takeaway

from engine import models as M
from engine import units as U
from engine import viz
from . import figures as F
from . import helpers as H


def _manage_saved(state, meta):
    fits = state["fits"]
    revision = state.get("_compare_version", 0)
    version = f"{state.get('ver', 0)}_{revision}"
    with st.expander("Manage saved interpretations", expanded=True):
        st.caption("Rename, load for parameter editing, or delete saved alternatives. "
                   "Deleting here keeps uploaded data, the working model and Last fit. "
                   "Changes affect this session; download the project again to keep them.")
        selected = st.selectbox("Saved interpretation", range(len(fits)),
                                format_func=lambda i: f"{i + 1}. {fits[i]['name']}",
                                key=f"compare_selected_{version}")
        saved = fits[selected]
        name = st.text_input("Interpretation name", value=saved["name"],
                             key=f"compare_name_{version}_{selected}")
        rename, edit = st.columns(2)
        if rename.button("Rename saved interpretation", key=f"compare_rename_{version}"):
            if not name.strip():
                st.error("Enter a non-empty interpretation name.")
            else:
                state["fits"] = [{**f, "name": name.strip()} if i == selected else f
                                 for i, f in enumerate(fits)]
                state["_compare_version"] = revision + 1
                st.rerun()
        if edit.button("Load for editing in Interpretation", key=f"compare_edit_{version}"):
            ds = state.get("dataset")
            if not ds or saved.get("ds") != ds.get("name"):
                st.error("Load the dataset used by this interpretation before editing its parameters.")
            elif ds.get("fluid") != meta["fluid"] or saved.get("fluid", ds["fluid"]) != meta["fluid"]:
                st.error("Select the saved interpretation's fluid in the sidebar before editing.")
            else:
                working = {"params": deepcopy(saved["params"]),
                           "gas": deepcopy(saved.get("gas", ds.get("gas")))}
                state["interp_model"] = working
                state["_interp_pending"] = deepcopy(working)
                state["mc"], state["sens"] = None, None
                st.success("Loaded as the working model. Open Interpretation to edit parameters, "
                           "then refit and Save to Compare as a new alternative. "
                           "The saved snapshot and Last fit have not changed.")
        deleting = st.multiselect("Interpretations to delete", range(len(fits)),
                                  format_func=lambda i: f"{i + 1}. {fits[i]['name']}",
                                  key=f"compare_delete_selection_{version}")
        confirmed = st.checkbox("Confirm deletion of saved interpretations",
                                key=f"compare_confirm_{version}")
        delete, clear = st.columns(2)
        if delete.button("Delete selected interpretations", disabled=not confirmed or not deleting,
                         key=f"compare_delete_{version}"):
            state["fits"] = [f for i, f in enumerate(fits) if i not in deleting]
            state["_compare_version"] = revision + 1
            st.rerun()
        if clear.button("Clear saved interpretations", disabled=not confirmed,
                        key=f"compare_clear_{version}"):
            state["fits"] = []
            state["_compare_version"] = revision + 1
            st.rerun()


def render(meta):
    S = st.session_state
    system = meta["units"]
    pu = U.label("pressure", system)
    fits, ds, ie = S["fits"], S.get("dataset"), S.get("interp_eng")
    if not fits:
        st.info("Run a regression in tab 02 and press 'Save to Compare' to build a list of alternatives.")
        return
    _manage_saved(S, meta)
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
        try:
            cur = viz.model_curves(p, f["q"], f["tp"], dtmax, n=120, dt_min=dtmin)
        except (ValueError, np.linalg.LinAlgError) as exc:
            st.warning(f"Cannot plot saved interpretation '{f['name']}': {exc} "
                       "Load it for editing or delete it using the controls above.")
            continue
        models.append({"name": f["name"], "x": cur["ta"], "dp": U.from_field(cur["dp"], "pressure", system),
                       "der": U.from_field(cur["der"], "pressure", system)})
    data = None
    if ds and ie:
        d = ie["diag"]
        data = {"x": d["ta"], "dp": U.from_field(d["dp"], "pressure", system),
                "der": U.from_field(d["der"], "pressure", system)}
    st.plotly_chart(F.diag_figure("Saved interpretations vs data", "Agarwal equivalent time [hr]",
                                  f"dp and derivative [{pu}]", data=data, models=models), use_container_width=True)
    st.download_button("Download table (CSV)", df.to_csv(index=False).encode(), "interpretations.csv", "text/csv")
