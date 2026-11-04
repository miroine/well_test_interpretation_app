"""Explicit setup and downloads for numerical Eclipse design approximations."""
from __future__ import annotations

from dataclasses import asdict
import json

import streamlit as st

from engine import design_export as E
from engine import eclipse_design as D
from engine import units as U


@st.cache_data(show_spinner=False, max_entries=8)
def comparison_package(well, curves, q_user, tp, dt_end, system, fluid, minimum_bhp, pi, gas):
    return E.zip_files(E.comparison_files(well, curves, q_user, tp, dt_end, system, fluid, minimum_bhp, pi, gas))


def render(p, curves, q_user, tp, dt_end, fluid, gas, well, minimum_bhp, system):
    saved = st.session_state.get("_design_deck_setup", {})
    switching = st.session_state.get("_design_deck_setup_units") != system
    def number(field, label, kind, low, high, default, step=None, fmt=None):
        factor = (1.0 if system == "Field" else 16.01846337396) if kind == "density" else float(U.from_field(1.0, kind, system))
        unit = ("lb/ft3" if system == "Field" else "kg/m3") if kind == "density" else U.label(kind, system)
        key = f"design_deck_{field}_{system}"
        value = float(saved.get(field, default)) * factor
        if switching:
            st.session_state[key] = value
        kwargs = {"min_value": low * factor, "value": value, "key": key}
        if high is not None:
            kwargs["max_value"] = high * factor
        if step is not None:
            kwargs["step"] = step * factor
        if fmt:
            kwargs["format"] = fmt
        return st.number_input(f"{label} [{unit}]", **kwargs) / factor
    with st.expander("Export full Eclipse .DATA model"):
        st.warning("Numerical approximation, not an exact analytical-model conversion. "
                   f"The deck follows project units ({system.upper()}). Dual-porosity models export only a late-time "
                   "single-continuum surrogate: their omega/lambda transition is NOT represented. "
                   "Analytical wellbore storage C is NOT represented for any model.")
        notes = D.assumptions(p, fluid, system)
        for note in notes:
            st.caption(note)
        st.markdown(f"**Export grid and datum ({system} units)**")
        a, b, c = st.columns(3)
        nx = a.number_input("Base grid X refinement", 3, 151, 31, 2, key="design_deck_nx")
        ny = b.number_input("Base grid Y refinement", 3, 151, 31, 2, key="design_deck_ny")
        nz = c.number_input("Base vertical layers", 1, 50, 5, 1, key="design_deck_nz")
        extent = number("half_extent_ft", "Far-field half-extent", "length", 10.0, 1e6, 5000.0, 100.0)
        st.caption("Model feature edges are inserted automatically, so final cell counts differ from base "
                   "refinement. Rectangle/circle dimensions and fault/channel distances come from the design. "
                   "The half-extent controls remaining far-field directions.")
        depth = number("top_depth_ft", "Reservoir top depth", "length", 0.0, 1e5, 8000.0, 100.0)
        st.markdown("**Additional fluid and fracture inputs**")
        rock = number("rock_compressibility", "Deck rock compressibility", "ct", 0.0, None,
                      float(gas["cf"] if fluid == "Gas" and gas else min(3e-6, p.ct / 2)), fmt="%.4e")
        density = number("oil_density_lbft3", "Stock-tank oil density", "density", 1.0, 100.0, 53.0, 1.0)
        if fluid == "Gas":
            st.caption("Dry-gas PVT uses the design temperature and gas gravity; deck rock compressibility "
                       "should match the design formation compressibility for comparison. Oil density is inactive.")
        width, fracture_phi, infinite_fcd = 0.02, 0.2, 1e4
        if p.model in ("fracture", "fracture_fc", "mfhw"):
            width = number("fracture_width_ft", "Physical fracture width", "length", 1e-5, 10.0, 0.02, fmt="%.5g")
            fracture_phi = st.number_input("Fracture porosity", 0.001, 0.999, 0.2, key="design_deck_frac_phi")
            if p.model == "fracture":
                infinite_fcd = st.number_input("Finite proxy for infinite fracture FcD", 1e2, 1e8, 1e4,
                                               key="design_deck_infinite_fcd")
        buffer = st.number_input("Constant-pressure buffer pore-volume multiplier", 1.0, 1e12, 1e6,
                                 format="%.4g", key="design_deck_buffer")
        count = st.number_input("Report points per flowing / shut-in period", 3, 500, 60, 1,
                                key="design_deck_reports")
        setup = D.DeckSetup(int(nx), int(ny), int(nz), extent, depth, rock, density,
                            width, fracture_phi, infinite_fcd, buffer, int(count))
        st.session_state["_design_deck_setup"] = {**saved, **asdict(setup)}
        st.session_state["_design_deck_setup_units"] = system
        fingerprint = json.dumps({"params": p.to_dict(), "setup": asdict(setup), "well": well,
                                  "q": q_user, "tp": tp, "dt": dt_end, "fluid": fluid,
                                  "gas": gas, "bhp": minimum_bhp, "units": system}, sort_keys=True, allow_nan=False)
        accepted = st.checkbox("I accept the listed numerical approximations and will validate the deck in my simulator",
                               key=f"design_deck_accept_{p.model}_{p.boundary}")
        if st.button("Build Eclipse .DATA model", key="design_deck_build"):
            if not accepted:
                st.error("Review and accept the numerical approximations before building the deck.")
            else:
                try:
                    with st.spinner("Building grid, fluid tables and schedule..."):
                        deck, package, metadata = D.deck_package(
                            p, setup, well, curves, q_user, tp, dt_end, fluid, minimum_bhp, gas, system=system)
                except ValueError as exc:
                    st.session_state.pop("_design_eclipse_deck", None)
                    st.error(f"Cannot export this deck: {exc}")
                else:
                    st.session_state["_design_eclipse_deck"] = {
                        "fingerprint": fingerprint, "deck": deck, "package": package, "metadata": metadata}
        result = st.session_state.get("_design_eclipse_deck")
        if result and result["fingerprint"] == fingerprint and accepted:
            meta = result["metadata"]
            st.success(f"Built {meta['grid_shape']} grid; {meta['active_cells']:,} active cells and "
                       f"{meta['completion_count']} well completions. Simulator execution is unverified.")
            st.download_button("Download full Eclipse model (.DATA)", result["deck"],
                               file_name=f"{well}.DATA", mime="text/plain", key="design_deck_data")
            st.download_button("Download full model + BHP comparison (.ZIP)", result["package"],
                               file_name=f"{well}_model_comparison.zip", mime="application/zip",
                               key="design_deck_zip")
        elif result:
            st.info("Export inputs changed or acknowledgement is missing. Build again to download a current deck.")
