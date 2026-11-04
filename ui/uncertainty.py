"""Controls for bounded forward-model uncertainty, distinct from noise refits."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from engine import uncertainty as UC, units as U, viz
from . import helpers as H


def parameter_sampling(p, fluid, system, ver, free, q, tp, dt):
    specs = H.BASE_SPEC + H.MODEL_SPEC.get(p.model, []) + H.BOUNDARY_SPEC.get(p.boundary, [])
    available = {s[0]: s for s in specs if s[0] != "n_frac" and
                 not (fluid == "Gas" and s[0] in ("mu", "B", "ct"))}
    tag = f"uncertainty_{ver}_{system}_{p.model}_{p.boundary}"
    names = st.multiselect("Uncertain parameters", list(available),
                           default=[n for n in free if n in available],
                           format_func=lambda n: available[n][1], key=f"{tag}_names")
    st.caption("Low/high are hard bounds. Other parameters stay at Last fit. "
               "Skin is the engine's additional skin for explicit fractures. "
               "These are assumed input distributions, not fitted confidence intervals.")
    choices = {}
    for name in names:
        _, label, kind, lower, upper, *_ = available[name]
        if name == "S" and p.model in ("fracture", "fracture_dual_pss"):
            lower = 0.
        center = float(np.clip(getattr(p, name), lower, upper))
        width = abs(center) * .2 if center != 0 else .5
        lo, hi = max(lower, center - width), min(upper, center + width)
        with st.expander(f"{label} [{U.label(kind, system)}]"):
            distribution = st.selectbox("Distribution", UC.DISTRIBUTIONS, key=f"{tag}_{name}_dist")
            a, b = st.columns(2)
            low = a.number_input("Low", value=float(U.from_field(lo, kind, system)),
                                 format="%.8g", key=f"{tag}_{name}_low")
            high = b.number_input("High", value=float(U.from_field(hi, kind, system)),
                                  format="%.8g", key=f"{tag}_{name}_high")
            row = {"distribution": distribution, "low": float(U.to_field(low, kind, system)),
                   "high": float(U.to_field(high, kind, system))}
            if distribution == "Triangular":
                mode = st.number_input("Most likely", value=float(U.from_field(center, kind, system)),
                                       format="%.8g", key=f"{tag}_{name}_mode")
                row["mode"] = float(U.to_field(mode, kind, system))
            if distribution == "Truncated normal":
                mean = st.number_input("Mean before truncation", value=float(U.from_field(center, kind, system)),
                                       format="%.8g", key=f"{tag}_{name}_mean")
                sd = st.number_input("Standard deviation before truncation",
                                     value=float(U.from_field((hi - lo) / 4, kind, system)),
                                     format="%.8g", key=f"{tag}_{name}_sd")
                row.update(mean=float(U.to_field(mean, kind, system)), sd=float(U.to_field(sd, kind, system)))
            choices[name] = row
    correlation = np.eye(len(names))
    if names and st.checkbox("Correlate parameters (Gaussian copula)", key=f"{tag}_correlated"):
        st.caption("Edit both halves symmetrically; diagonal = 1. This is normal-score correlation, "
                   "not parameter Pearson correlation. The matrix must be positive semidefinite.")
        edited = st.data_editor(pd.DataFrame(correlation, index=names, columns=names),
                                key=f"{tag}_matrix_{'_'.join(names)}", num_rows="fixed")
        correlation = edited.to_numpy(dtype=float)
    runs = st.number_input("Parameter samples", 5, 1000, 100, 5, key=f"{tag}_runs")
    seed = st.number_input("Sampling seed", 0, 2147483647, 0, key=f"{tag}_seed")
    st.caption("Large ensembles and complex fracture models can take time. "
               "No noisy-data refitting is performed. Invalid combinations stop the run; no samples are silently dropped. "
               "Changing these controls does not change the last completed result until you run again.")
    if st.button("Run parameter uncertainty", key=f"{tag}_run", disabled=not names):
        try:
            for name, row in choices.items():
                lower, upper = available[name][3:5]
                if row["low"] < lower or row["high"] > upper:
                    raise ValueError(f"{name}: range must stay within supported bounds "
                                     f"{U.from_field(lower, available[name][2], system):g} to "
                                     f"{U.from_field(upper, available[name][2], system):g}.")
            with st.spinner("Sampling correlated parameters and calculating response envelopes..."):
                result = UC.sample_parameters(p, choices, correlation, int(runs), int(seed))
                grid = np.geomspace(float(np.min(dt)), float(np.max(dt)), 140)
                result["envelope"] = viz.monte_carlo_envelope(p, q, tp, grid, result["samples"])
        except (ValueError, np.linalg.LinAlgError) as exc:
            st.error(f"Cannot calculate parameter uncertainty: {exc}")
        else:
            st.session_state["mc"] = result
            st.rerun()


def sampling_results(mc, system):
    if mc.get("mode") != "parameter_sampling":
        return
    st.info("Parameter-range propagation, without refitting. P10/P50/P90 describe your assumed "
            "input uncertainty, not posterior probabilities or data-derived confidence limits.")
    st.caption(f"Last completed ensemble: {mc['requested']} samples; seed {mc['seed']}. "
               "Samples and results are held in this session only.")
    frame = pd.DataFrame(mc["samples"])
    if len(frame.columns) > 1:
        st.markdown("**Achieved parameter Pearson correlations (finite ensemble)**")
        st.dataframe(frame.corr(), use_container_width=True)
    all_specs = {s[0]: s for s in H.BASE_SPEC + [s for group in H.MODEL_SPEC.values() for s in group] +
                 [s for group in H.BOUNDARY_SPEC.values() for s in group]}
    displayed = pd.DataFrame({name: U.from_field(frame[name].to_numpy(), all_specs[name][2], system)
                              for name in frame})
    displayed.columns = [f"{name} [{U.label(all_specs[name][2], system)}]" for name in frame]
    st.download_button("Download paired parameter samples (CSV)", displayed.to_csv(index=False).encode(),
                       "parameter_samples.csv", "text/csv")
