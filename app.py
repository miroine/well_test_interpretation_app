"""
app.py
------
Main Streamlit entry point for Aquila — Pressure Transient Analysis.

Author : Merouane Hamdani
License: MIT (see LICENSE)
Version: 1.0

Tabs:
    1) Well Test Design & Economics
    2) Interpretation & Regression
    3) Uncertainty Modeling & Report

Unit handling
-------------
The user picks a unit system in the sidebar (Field or SI). The engine remains
in field units internally; this UI layer is the ONLY place that converts:
    UI input  →  to_field()    →  engine
    engine    →  from_field()  →  UI display
Every label, axis title and metric is built from `units.label(...)`.

Disclaimer
----------
Aquila is provided "as is", without warranty. Outputs are intended to
assist, not replace, the work of a qualified petroleum engineer. The
visual identity is inspired by publicly-visible corporate-design
conventions; Aquila is NOT affiliated with Equinor ASA.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from styles import (
    inject_css, banner_html, get_plotly_layout,
    COLOR_RED, COLOR_DARK, COLOR_TEAL, COLOR_WHITE,
)
from modules.engine import (
    WellInputs, simulate_drawdown, wbs_clearance_time, fair_schulz_transition,
    decimate, detect_flow_periods, build_buildup_derivatives,
    regress_radial, regress_linear,
    # Gas additions
    FluidType, GasWellInputs, simulate_gas_drawdown, regress_radial_gas,
    wbs_clearance_time_gas, bourdet_derivative,
    # Phase A additions
    regress_linear_gas, regress_pss, regress_pss_gas,
    # Model-vs-data helpers
    horner_time, line_source_dp,
)
from modules.pvt import GasPVT, build_pvt_table, pseudo_pressure_table, pseudo_pressure
from modules.layered import (
    CommingledInputs, simulate_commingled, fit_commingled_2layer,
)
from modules.economics import EconomicsInputs, compute_costs
from modules.uncertainty import run_monte_carlo, kde_curve, cdf_curve
from modules.reporter import build_report
from modules.units import (
    UnitSystem, to_field, from_field, label, fmt, axis_title,
    temperature_to_F, temperature_from_F,
)


# =====================================================================
#  Cached compute kernels
#
#  Streamlit re-runs the entire script on every interaction. Wrap the
#  pure-math kernels in @st.cache_data so they only re-execute when their
#  *inputs* change. Hash all primitives; mutable args (DataFrames) are
#  hashed by Streamlit automatically.
# =====================================================================
@st.cache_data(show_spinner=False)
def cached_simulate(k, h, q, s, C, mu, B, phi, ct, rw, pi, duration_hr, n_points,
                    re_ft=None):
    w = WellInputs(k=k, h=h, q=q, s=s, C=C, mu=mu, B=B, phi=phi, ct=ct, rw=rw, pi=pi)
    df = simulate_drawdown(duration_hr=duration_hr, n_points=n_points, w=w, re_ft=re_ft)
    return df, wbs_clearance_time(w)


@st.cache_data(show_spinner=False)
def cached_synthetic_dataset(mu, B, phi, ct, rw, pi):
    """Build the demo 2-flow / 2-buildup dataset (cached so toggling
    decimation doesn't regenerate the underlying signal)."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 120.0, 6000)
    q1, q2 = 2500.0, 2200.0
    rate = np.where(t < 30.0, q1,
            np.where(t < 50.0, 0.0,
            np.where(t < 90.0, q2, 0.0)))
    w_demo = WellInputs(k=45.0, h=70.0, q=1.0, s=2.0, C=0.012,
                        mu=mu, B=B, phi=phi, ct=ct, rw=rw, pi=pi)
    dp = np.zeros_like(t)
    for q_step, t_step in [(q1, 0.0), (-q1, 30.0), (q2, 50.0), (-q2, 90.0)]:
        m = t >= t_step
        dp[m] += q_step * fair_schulz_transition(np.maximum(t[m] - t_step, 1e-6), w_demo)
    p = pi - dp + rng.normal(0.0, 1.5, len(t))
    return pd.DataFrame({"time": t, "pressure": p, "rate": rate})


@st.cache_data(show_spinner=False)
def cached_read_csv(file_bytes: bytes, system: str) -> pd.DataFrame:
    """Read uploaded CSV, validate, and convert ALL rows to field units (vectorized)."""
    from io import BytesIO
    df_raw = pd.read_csv(BytesIO(file_bytes))
    df_raw.columns = [c.strip().lower() for c in df_raw.columns]

    missing = [c for c in ("time", "pressure", "rate") if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}.")

    # Coerce to numeric — non-numeric entries become NaN and we drop them
    for col in ("time", "pressure", "rate"):
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
    before = len(df_raw)
    df_raw = df_raw.dropna(subset=["time", "pressure", "rate"]).reset_index(drop=True)
    dropped = before - len(df_raw)

    # Sort by time, drop duplicate timestamps (gauge buffer artefacts)
    df_raw = (df_raw
              .sort_values("time", kind="mergesort")
              .drop_duplicates(subset="time", keep="first")
              .reset_index(drop=True))

    # Vectorized field-unit conversion using factor lookup
    from modules.units import _TABLE  # internal but stable
    if system == UnitSystem.FIELD.value:
        tf_t = tf_p = tf_r = 1.0
    else:
        tf_t = _TABLE["time"].to_field
        tf_p = _TABLE["pressure"].to_field
        tf_r = _TABLE["rate"].to_field

    out = pd.DataFrame({
        "time":     df_raw["time"].to_numpy()     * tf_t,
        "pressure": df_raw["pressure"].to_numpy() * tf_p,
        "rate":     df_raw["rate"].to_numpy()     * tf_r,
    })
    out.attrs["dropped_rows"] = dropped
    out.attrs["raw_row_count"] = before
    return out


@st.cache_data(show_spinner=False)
def cached_buildup_pack(df_use: pd.DataFrame, rate_thr_field: float, L: float):
    """Cache the event detection + derivative pipeline (re-runs only on
    decimation / smoothing-L changes, not on slider movement)."""
    periods = detect_flow_periods(
        df_use["time"].to_numpy(),
        df_use["rate"].to_numpy(),
        rate_threshold=rate_thr_field,
    )
    buildups = build_buildup_derivatives(df_use, periods, L=L)
    return periods, buildups


# =====================================================================
#  Page config + global styling
# =====================================================================
st.set_page_config(
    page_title="Aquila — PTA",
    page_icon="assets/aquila_icon.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(inject_css(), unsafe_allow_html=True)

# --- Aquila branded header --------------------------------------------
_LOGO_PATH = Path(__file__).parent / "assets" / "aquila_logo.svg"
if _LOGO_PATH.exists():
    _logo_svg = _LOGO_PATH.read_text(encoding="utf-8")
    # Embed inline so styling matches the page perfectly
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; justify-content:space-between;
                    border-bottom:2px solid #001E2B; padding-bottom:8px; margin-bottom:18px;">
            <div style="display:flex; align-items:center;">{_logo_svg}</div>
            <div style="color:#5a5a5a; font-size:12px; text-align:right; line-height:1.45;">
                Pressure-transient design · interpretation · uncertainty<br>
                <span style="color:#001E2B;">by</span>
                <span style="color:#001E2B; font-weight:600;">Merouane Hamdani</span>
                &nbsp;·&nbsp;
                <span style="color:#00707F; font-weight:600;">v1.0</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # Fallback if assets/ is missing
    st.markdown(
        banner_html("Aquila",
                    "Pressure transient design · interpretation · uncertainty · "
                    "by Merouane Hamdani"),
        unsafe_allow_html=True,
    )

# Session state defaults
ss = st.session_state
ss.setdefault("interpretation_result", None)
ss.setdefault("mc_result", None)
ss.setdefault("project_name", "North Sea — Field Alpha — Well A-1H")
ss.setdefault("unit_system", UnitSystem.FIELD.value)
ss.setdefault("fluid_type", FluidType.OIL.value)


# =====================================================================
#  Cached PVT-table builder for gas wells
# =====================================================================
@st.cache_data(show_spinner=False)
def cached_pvt_table(gamma_g: float, T_F: float,
                     y_CO2: float, y_H2S: float, y_N2: float,
                     p_max: float = 10_000.0) -> dict:
    """Build & cache a PVT table. Re-runs only when gas composition or T changes."""
    pvt = GasPVT(gamma_g=gamma_g, T_res_F=T_F,
                 y_CO2=y_CO2, y_H2S=y_H2S, y_N2=y_N2)
    table = build_pvt_table(pvt, p_max=p_max, n=200)
    p_pp, m_pp = pseudo_pressure_table(table)
    table["p_pp"] = p_pp
    table["m_pp"] = m_pp
    table["T_R"] = pvt.T_R
    return table


def is_gas() -> bool:
    """True if the user has selected a gas well."""
    return ss["fluid_type"] == FluidType.GAS.value


# =====================================================================
#  Sidebar — global project metadata + UNIT SYSTEM SWITCH
# =====================================================================

# ----- State preservation across unit-system changes -----
# Every numeric input below is keyed on session_state, and the registry below
# tells us the physical quantity each one represents. When the user toggles
# Field ↔ SI, we convert ALL tracked values in-place so no data is lost.
_TRACKED_INPUTS: dict[str, str] = {
    # sidebar
    "ui_mu":  "viscosity",
    "ui_B":   "fvf",
    "ui_phi": "porosity",
    "ui_ct":  "compressibility",
    "ui_rw":  "wellbore_radius",
    "ui_pi":  "pressure",
    # tab 1 — design
    "ui_k":      "permeability",
    "ui_h":      "height",
    "ui_q":      "rate",
    "ui_s":      "skin",
    "ui_C":      "wbs",
    "ui_dur":    "time",
    "ui_oil":    "oil_price",
    "ui_dq":     "rate",
    "ui_dt":     "time",
    # tab 3 — Monte Carlo
    "mc_q":      "rate",
    "h_lo":      "height",
    "h_mo":      "height",
    "h_hi":      "height",
    "mu_lo":     "viscosity",
    "mu_mo":     "viscosity",
    "mu_hi":     "viscosity",
}


def _convert_state_on_system_change(old_sys: str, new_sys: str) -> None:
    """Walk the tracked-input registry and re-express every cached value in
    the new system. Operates directly on st.session_state so widgets pick up
    the new numbers on the next render."""
    if old_sys == new_sys:
        return
    for key, quantity in _TRACKED_INPUTS.items():
        if key not in ss:
            continue
        try:
            field_val = to_field(float(ss[key]), quantity, old_sys)
            ss[key] = float(from_field(field_val, quantity, new_sys))
        except Exception:
            # If anything goes wrong, leave the stored value alone rather
            # than corrupt session state.
            pass


with st.sidebar:
    st.header("Project")
    ss["project_name"] = st.text_input("Project / Well", ss["project_name"])

    st.markdown("---")
    st.header("Units")
    new_system = st.radio(
        "Unit system",
        options=[UnitSystem.FIELD.value, UnitSystem.SI.value],
        index=0 if ss["unit_system"] == UnitSystem.FIELD.value else 1,
        horizontal=True,
        help=("Engine math is always run in oil-field units; this switch only "
              "affects how values are entered and displayed. **Existing input "
              "values are converted automatically when you flip the switch.**"),
    )
    if new_system != ss["unit_system"]:
        _convert_state_on_system_change(ss["unit_system"], new_system)
        ss["unit_system"] = new_system
        st.rerun()    # force a clean re-render with converted defaults
    U = ss["unit_system"]   # short alias used throughout this file

    # ----- Fluid type -----
    st.markdown("---")
    st.header("Fluid")
    new_fluid = st.radio(
        "Fluid type",
        options=[FluidType.OIL.value, FluidType.GAS.value, FluidType.WATER.value],
        index=[FluidType.OIL.value, FluidType.GAS.value, FluidType.WATER.value]
              .index(ss["fluid_type"]),
        horizontal=True,
        help=("Gas wells use pseudo-pressure m(p) and pseudo-time tₐ instead of "
              "ΔP and Δt. Water behaves like a slightly-compressible liquid "
              "and uses the same math as oil."),
    )
    if new_fluid != ss["fluid_type"]:
        ss["fluid_type"] = new_fluid
        # Reset interpretation/MC results since they depend on the fluid model
        ss["interpretation_result"] = None
        ss["mc_result"] = None
        ss["interp_slope_m_field"] = None
        ss["interp_k_field"] = None
        ss["interp_skin_field"] = None
        st.rerun()

    # ----- Gas PVT panel (only for gas wells) -----
    pvt_table = None
    if is_gas():
        with st.expander("Gas PVT — composition & temperature", expanded=False):
            T_disp = st.number_input(
                f"Reservoir temperature ({label('temperature', U)})",
                -20.0, 500.0,
                200.0 if U == UnitSystem.FIELD.value else 93.3,
                1.0, key="pvt_T",
            )
            gamma_g = st.number_input(
                "Gas specific gravity γg (air = 1)",
                0.55, 1.7, 0.65, 0.01, key="pvt_gamma",
            )
            y_CO2 = st.slider("CO₂ mole fraction", 0.0, 0.30, 0.0, 0.01, key="pvt_co2")
            y_H2S = st.slider("H₂S mole fraction", 0.0, 0.30, 0.0, 0.01, key="pvt_h2s")
            y_N2  = st.slider("N₂ mole fraction",  0.0, 0.20, 0.0, 0.01, key="pvt_n2")

        # Build / cache PVT table for downstream use
        T_F = temperature_to_F(T_disp, U)
        pvt_table = cached_pvt_table(
            gamma_g=gamma_g, T_F=T_F,
            y_CO2=y_CO2, y_H2S=y_H2S, y_N2=y_N2,
        )
        # Stash so other tabs/cells can grab it without re-passing
        ss["pvt_table"] = pvt_table
    else:
        ss["pvt_table"] = None

    # --- Reservoir defaults (entered in chosen system, converted to field) ---
    st.markdown("---")
    st.header("Reservoir Defaults")

    if is_gas():
        st.caption("Viscosity μ, FVF Bg, and compressibility cₜ come from PVT.")
        # Use neutral placeholder values; engine never uses them for gas
        mu_disp = 1.0
        B_disp = 1.0
        ct_disp = 1e-5
        phi_disp = st.number_input(
            f"Porosity φ ({label('porosity', U)})",
            0.01, 0.40, 0.15, 0.01, key="ui_phi",
        )
        rw_default_disp = 0.354 if U == UnitSystem.FIELD.value else 0.108
        rw_disp = st.number_input(
            f"Wellbore radius r𝓌 ({label('wellbore_radius', U)})",
            0.01, 3.0, rw_default_disp, 0.001, format="%.3f", key="ui_rw",
        )
        pi_default_disp = 5000.0 if U == UnitSystem.FIELD.value else 345.0
        pi_disp = st.number_input(
            f"Initial pressure Pᵢ ({label('pressure', U)})",
            1.0, 2000.0 if U == UnitSystem.SI.value else 20000.0,
            pi_default_disp, 1.0, key="ui_pi",
        )
    else:
        mu_disp = st.number_input(
            f"Viscosity μ ({label('viscosity', U)})",
            0.1, 50.0, 1.0, 0.1, key="ui_mu",
        )
        B_disp = st.number_input(
            f"FVF B ({label('fvf', U)})",
            1.0, 3.0, 1.20, 0.01, key="ui_B",
        )
        phi_disp = st.number_input(
            f"Porosity φ ({label('porosity', U)})",
            0.01, 0.40, 0.15, 0.01, key="ui_phi",
        )
        # Compressibility default differs between systems (avoid awkward scientific-notation entry)
        ct_default_disp = 1.0e-5 if U == UnitSystem.FIELD.value else 1.45e-4
        ct_disp = st.number_input(
            f"Total compressibility cₜ ({label('compressibility', U)})",
            1e-7, 1e-2, ct_default_disp, 1e-6, format="%.2e", key="ui_ct",
        )
        rw_default_disp = 0.354 if U == UnitSystem.FIELD.value else 0.108
        rw_disp = st.number_input(
            f"Wellbore radius r𝓌 ({label('wellbore_radius', U)})",
            0.01, 3.0, rw_default_disp, 0.001, format="%.3f", key="ui_rw",
        )
        pi_default_disp = 5000.0 if U == UnitSystem.FIELD.value else 345.0
        pi_disp = st.number_input(
            f"Initial pressure Pᵢ ({label('pressure', U)})",
            1.0, 2000.0 if U == UnitSystem.SI.value else 20000.0,
            pi_default_disp, 1.0, key="ui_pi",
        )

    # Convert sidebar defaults to field units once (engine always in field)
    mu_default   = to_field(mu_disp,  "viscosity",       U)
    B_default    = to_field(B_disp,   "fvf",             U)
    phi_default  = to_field(phi_disp, "porosity",        U)
    ct_default   = to_field(ct_disp,  "compressibility", U)
    rw_default   = to_field(rw_disp,  "wellbore_radius", U)
    pi_default   = to_field(pi_disp,  "pressure",        U)

    # For gas, μ / B / cₜ are PVT-derived at pi (overrides the placeholders above).
    if is_gas() and pvt_table is not None:
        mu_default = float(np.interp(pi_default, pvt_table["p"], pvt_table["mu"]))
        # Bg here is in rb/Mscf — different unit but only used by economics gas-FVF
        B_default = float(np.interp(pi_default, pvt_table["p"], pvt_table["Bg"]))
        ct_default = float(np.interp(pi_default, pvt_table["p"], pvt_table["cg"]))

    st.markdown("---")
    fluid_msg = f"**{ss['fluid_type']}** well"
    if is_gas() and pvt_table is not None:
        fluid_msg += f"  ·  μg(pi)={mu_default:.4f} cp  ·  cg(pi)={ct_default:.2e} 1/psi"
    st.caption(f"Active unit system: **{U}**  ·  {fluid_msg}")

    # =====================================================================
    #  About / License / Disclaimer
    # =====================================================================
    with st.expander("ℹ  About · License · Disclaimer", expanded=False):
        st.markdown(
            """
            <div style="font-size:12px; line-height:1.55; color:#001E2B;">
            <p style="margin:4px 0;"><b>Aquila</b> — Pressure Transient Analysis<br>
            Version <span style="color:#00707F;">1.0</span> · 2026</p>

            <p style="margin:4px 0;">Author: <b>Merouane Hamdani</b></p>

            <p style="margin:8px 0 4px 0;"><b style="color:#FF1243;">Disclaimer.</b>
            This software is provided <i>as is</i>, without warranty of any kind.
            Outputs are intended to assist — not replace — the work of a
            qualified petroleum engineer. PVT correlations and inversion
            routines are valid only within their published applicability
            ranges. Decisions on the basis of this output are the user's sole
            responsibility.</p>

            <p style="margin:8px 0 4px 0;"><b style="color:#FF1243;">No Equinor affiliation.</b>
            Aquila's visual identity is <i>inspired by</i> publicly-available
            corporate-design conventions; the project is not produced by, nor
            affiliated with, Equinor ASA, and the Equinor name, logo and
            wordmark are not used.</p>

            <p style="margin:8px 0 4px 0;"><b>Licensed under MIT.</b>
            See the <code>LICENSE</code> file for full terms and the additional
            engineering-use disclaimer.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =====================================================================
#  Onboarding — collapsible, dismissible
# =====================================================================
ss.setdefault("show_onboarding", True)
if ss["show_onboarding"]:
    with st.expander("👋  Welcome to Aquila — Quick-start guide  (click to expand)",
                     expanded=True):
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.markdown(
                """
**Aquila** is a pressure-transient analysis (PTA) workbench. Three tabs,
left to right, mirror the actual PTA workflow:

1. **Design & Economics** — Pick a fluid, set candidate reservoir parameters,
   see the predicted log-log diagnostic plot, and check whether your test
   duration is long enough to clear wellbore storage. Get an exposure
   estimate for the operation.

2. **Interpretation** — Upload your gauge CSV (columns: `time`, `pressure`,
   `rate`) or use the built-in demo dataset. Aquila auto-detects flow
   periods and buildups, computes Bourdet derivatives, and lets you fit
   **radial / linear / closed-boundary** regimes with a draggable window.

3. **Uncertainty & Report** — Triangular Monte Carlo on k, with P10/P50/P90,
   a tornado sensitivity chart, and a one-click PDF report.

**Tips for first use:**
- Pick **Field** or **SI** units in the sidebar — everything converts.
- Pick **Oil / Gas / Water** in the sidebar — gas wells unlock the PVT panel.
- After running a regression, scroll down to **📊 Model vs Data** to see
  the analytical model overlaid on your buildup in a four-panel view.
- Most inputs have ℹ️ tooltips — hover over the label.
"""
            )
        with col_right:
            st.markdown("**Typical workflow:**")
            st.markdown(
                "```\n"
                "Sidebar   → set units & fluid\n"
                "          → for gas: set PVT\n"
                "Tab 1     → forward-sim a candidate well\n"
                "          → check WBS clearance\n"
                "Tab 2     → upload CSV (or use demo)\n"
                "          → pick a buildup\n"
                "          → drag the regression window\n"
                "          → review Model-vs-Data panel\n"
                "Tab 3     → run Monte Carlo\n"
                "          → download PDF report\n"
                "```"
            )
            if st.button("Don't show this again", key="dismiss_onboarding"):
                ss["show_onboarding"] = False
                st.rerun()

# =====================================================================
#  Tabs
# =====================================================================
tab_design, tab_interp, tab_uncert = st.tabs([
    "1 · Design & Economics",
    "2 · Interpretation",
    "3 · Uncertainty & Report",
])


# =====================================================================
#  TAB 1 — Design & Economics
# =====================================================================
with tab_design:
    st.subheader("Forward Simulation — Drawdown / Buildup")

    # System-aware default values for the design inputs.
    # Note: max bounds are kept generous so a unit-system switch never clips a
    # legitimate converted value (e.g. 5000 ft → 1524 m must still fit).
    if U == UnitSystem.FIELD.value:
        k_default, h_default, q_default = 50.0, 75.0, 2500.0
        s_default, C_default, dur_default = 1.5, 0.01, 72.0
        k_max, h_max, q_max, C_max = 5000.0, 5000.0, 100_000.0, 2.0
        C_fmt = "%.4f"
        C_step = 0.001
    else:
        k_default, h_default, q_default = 50.0, 22.86, 397.47
        s_default, C_default, dur_default = 1.5, 0.0231, 72.0
        k_max, h_max, q_max, C_max = 5000.0, 1525.0, 16_000.0, 4.7
        C_fmt = "%.4f"
        C_step = 0.001

    # --- Rate label and quantity differ between liquid and gas ---
    if is_gas():
        rate_quantity = "gas_rate"
        rate_default = q_default * 2     # gas wells are typically higher rate
        rate_max = 1_000_000.0
        rate_step = 100.0
    else:
        rate_quantity = "rate"
        rate_default = q_default
        rate_max = q_max
        rate_step = 10.0

    c1, c2, c3 = st.columns(3)
    with c1:
        k_in = st.number_input(
            f"Permeability k ({label('permeability', U)})",
            0.01, k_max, k_default, 1.0, key="ui_k",
        )
        h_in = st.number_input(
            f"Net pay h ({label('height', U)})",
            0.1, h_max, h_default, 1.0, key="ui_h",
        )
    with c2:
        q_in = st.number_input(
            f"Rate q ({label(rate_quantity, U)})",
            1.0, rate_max, rate_default, rate_step,
            key="ui_q_gas" if is_gas() else "ui_q",
        )
        s_in = st.number_input("Skin s (–)", -5.0, 50.0, s_default, 0.5, key="ui_s")
    with c3:
        C_in = st.number_input(
            f"Wellbore storage C ({label('wbs', U)})",
            1e-5, C_max, C_default, C_step, format=C_fmt, key="ui_C",
        )
        duration_in = st.number_input(
            f"Planned duration ({label('time', U)})",
            0.5, 720.0, dur_default, 1.0, key="ui_dur",
        )

    # Optional closed-boundary input — folded into an expander to avoid clutter
    with st.expander("Closed boundary (optional)", expanded=False):
        use_boundary = st.checkbox(
            "Add a closed circular drainage boundary at radius re",
            value=False, key="ui_use_re",
            help="When ON, the forward simulation includes a pseudo-steady-state "
                 "contribution that takes over after the boundary is felt. "
                 "The Bourdet derivative then rises with unit slope at late time."
        )
        if use_boundary:
            re_default_disp = 500.0 if U == UnitSystem.FIELD.value else 150.0
            re_in = st.number_input(
                f"Drainage radius re ({label('length', U)})",
                10.0, 1e4, re_default_disp, 10.0, key="ui_re",
            )
            re_field = to_field(re_in, "length", U)
        else:
            re_field = None

    # --- convert all design inputs to field for the engine ---
    duration_field = to_field(duration_in, "time", U)
    k_field  = to_field(k_in, "permeability", U)
    h_field  = to_field(h_in, "height",       U)
    q_field  = to_field(q_in, rate_quantity,  U)   # field = STB/D or Mscf/D
    C_field  = to_field(C_in, "wbs",          U)

    if is_gas():
        # ---------------------- GAS PATH ----------------------
        gw = GasWellInputs(
            k=k_field, h=h_field, qg=q_field, s=s_in, C=C_field,
            T_R=pvt_table["T_R"], pi=pi_default, phi=phi_default, rw=rw_default,
        )
        sim_df = simulate_gas_drawdown(
            duration_hr=duration_field, n_points=300,
            gw=gw, pvt_table=pvt_table, re_ft=re_field,
        )
        t_clear_field = wbs_clearance_time_gas(gw, pvt_table)

        # Store a 'well' shim for downstream code that expects a WellInputs-like
        # object (used by the regression panel and report snapshot).
        class _GasShim:
            def __init__(self, gw, mu, ct, B):
                self.k = gw.k; self.h = gw.h; self.q = gw.qg; self.s = gw.s
                self.C = gw.C; self.mu = mu; self.B = B; self.phi = gw.phi
                self.ct = ct; self.rw = gw.rw; self.pi = gw.pi
        well = _GasShim(gw, mu_default, ct_default, B_default)
    else:
        # ---------------------- OIL / WATER PATH ----------------------
        well = WellInputs(
            k=k_field, h=h_field, q=q_field, s=s_in, C=C_field,
            mu=mu_default, B=B_default, phi=phi_default,
            ct=ct_default, rw=rw_default, pi=pi_default,
        )
        sim_df, t_clear_field = cached_simulate(
            k=well.k, h=well.h, q=well.q, s=well.s, C=well.C,
            mu=well.mu, B=well.B, phi=well.phi, ct=well.ct,
            rw=well.rw, pi=well.pi,
            duration_hr=duration_field, n_points=300,
            re_ft=re_field,
        )

    t_clear_disp = from_field(t_clear_field, "time", U)

    # Rule-of-thumb warning (talks in user's units)
    if duration_in < 1.5 * t_clear_disp:
        st.warning(
            f"⚠️  Planned duration **{duration_in:.1f} {label('time', U)}** may be "
            f"too short to clear wellbore storage. Estimated clearance time ≈ "
            f"**{t_clear_disp:.1f} {label('time', U)}**. Consider extending the test."
        )
    else:
        st.success(
            f"✓ Planned duration ({duration_in:.1f} {label('time', U)}) comfortably "
            f"exceeds the estimated wellbore-storage clearance time "
            f"(~{t_clear_disp:.1f} {label('time', U)})."
        )

    # --- Plot: ΔP/t·dP/dt for liquid; Δm(p)/t·dm/dt for gas ---
    fig = go.Figure()
    sim_t_disp = sim_df["t"].to_numpy()
    if is_gas():
        # Convert Δm(p) and t·dm/dt to the user's pressure-squared/cp unit
        mp_factor = from_field(1.0, "pseudo_pressure", U)
        sim_dp_disp    = sim_df["dm"].to_numpy()    * mp_factor
        sim_tdpdt_disp = sim_df["tdmdt"].to_numpy() * mp_factor
        y_qty   = "pseudo_pressure"
        y_label = "Δm(p), t·dm/dt"
        dp_legend = "Δm(p)"
        deriv_legend = "Bourdet derivative  t·dm/dt"
    else:
        dp_factor = from_field(1.0, "delta_pressure", U)
        sim_dp_disp    = sim_df["dp"].to_numpy()     * dp_factor
        sim_tdpdt_disp = sim_df["tdp_dt"].to_numpy() * dp_factor
        y_qty   = "delta_pressure"
        y_label = "ΔP, t·dP/dt"
        dp_legend = "ΔP"
        deriv_legend = "Bourdet derivative  t·dP/dt"

    fig.add_trace(go.Scatter(
        x=sim_t_disp, y=sim_dp_disp, mode="lines",
        name=dp_legend, line=dict(color=COLOR_DARK, width=3),
    ))
    fig.add_trace(go.Scatter(
        x=sim_t_disp, y=sim_tdpdt_disp, mode="lines",
        name=deriv_legend, line=dict(color=COLOR_RED, width=3),
    ))
    layout = get_plotly_layout("Log-log diagnostic plot")
    layout["xaxis"].update(type="log", title=axis_title("Elapsed time Δt", "time", U))
    layout["yaxis"].update(type="log", title=axis_title(y_label, y_qty, U))
    fig.update_layout(layout)
    st.plotly_chart(fig, use_container_width=True)

    # Download the simulated trace for QC
    y_unit_label = label(y_qty, U)
    sim_export = pd.DataFrame({
        f"t_{label('time', U)}":            sim_t_disp,
        f"dy_{y_unit_label}":               sim_dp_disp,
        f"tdy_dt_{y_unit_label}":           sim_tdpdt_disp,
    })
    st.download_button(
        "⬇  Download simulation (CSV)",
        data=sim_export.to_csv(index=False).encode("utf-8"),
        file_name=f"simulation_{ss['fluid_type'].lower()}_{U.lower()}.csv",
        mime="text/csv",
        key="dl_sim",
    )

    # ============================================================
    #  Multi-Layer Forward Simulation (commingled, no crossflow)
    # ============================================================
    st.markdown("---")
    with st.expander("⛏ Multi-Layer Forward Simulation (commingled, no crossflow)",
                     expanded=False):
        st.caption(
            "Two-layer commingled response — both layers share the wellbore "
            "pressure but DO NOT communicate vertically. The total rate q is "
            "split between layers in proportion to each layer's instantaneous "
            "admittance (parallel-resistor analogue). The qualitative signature "
            "is a STEP DOWN in the Bourdet derivative as the slower layer "
            "starts contributing. Double-permeability with crossflow is a "
            "future addition."
        )

        # Sensible system-aware defaults
        if U == UnitSystem.FIELD.value:
            l_k1_d, l_k2_d = 200.0, 5.0
            l_h1_d, l_h2_d = 15.0,  60.0
        else:
            l_k1_d, l_k2_d = 200.0, 5.0
            l_h1_d, l_h2_d = 4.6,   18.3

        cL1, cL2, cL3 = st.columns(3)
        with cL1:
            st.markdown("**Layer 1**")
            l_k1 = st.number_input(f"k₁ ({label('permeability', U)})",
                                    0.01, k_max, l_k1_d, 1.0, key="ui_l_k1")
            l_h1 = st.number_input(f"h₁ ({label('height', U)})",
                                    0.1, h_max, l_h1_d, 1.0, key="ui_l_h1")
            l_s1 = st.number_input("s₁ (–)", -5.0, 50.0, 0.0, 0.5, key="ui_l_s1")
        with cL2:
            st.markdown("**Layer 2**")
            l_k2 = st.number_input(f"k₂ ({label('permeability', U)})",
                                    0.01, k_max, l_k2_d, 1.0, key="ui_l_k2")
            l_h2 = st.number_input(f"h₂ ({label('height', U)})",
                                    0.1, h_max, l_h2_d, 1.0, key="ui_l_h2")
            l_s2 = st.number_input("s₂ (–)", -5.0, 50.0, 0.0, 0.5, key="ui_l_s2")
        with cL3:
            st.markdown("**Common**")
            l_phi1 = st.number_input("φ₁", 0.01, 0.40, phi_default, 0.01, key="ui_l_phi1")
            l_phi2 = st.number_input("φ₂", 0.01, 0.40, phi_default, 0.01, key="ui_l_phi2")
            l_dur  = st.number_input(f"Duration ({label('time', U)})",
                                      0.5, 720.0, dur_default, 1.0, key="ui_l_dur")

        # Build and run forward sim
        if is_gas():
            l_ci = CommingledInputs(
                k=[to_field(l_k1, "permeability", U), to_field(l_k2, "permeability", U)],
                h=[to_field(l_h1, "height", U),       to_field(l_h2, "height", U)],
                phi=[l_phi1, l_phi2],
                ct=[ct_default, ct_default],
                s=[l_s1, l_s2],
                q_total=q_field,
                mu=mu_default, B=B_default,
                rw=rw_default, pi=pi_default, C=0.0,
            )
            df_l = simulate_commingled(to_field(l_dur, "time", U), 500, l_ci,
                                       fluid="gas", T_R=pvt_table["T_R"])
            ycol, dcol = "dm", "tdmdt"
        else:
            l_ci = CommingledInputs(
                k=[to_field(l_k1, "permeability", U), to_field(l_k2, "permeability", U)],
                h=[to_field(l_h1, "height", U),       to_field(l_h2, "height", U)],
                phi=[l_phi1, l_phi2],
                ct=[ct_default, ct_default],
                s=[l_s1, l_s2],
                q_total=q_field,
                mu=mu_default, B=B_default,
                rw=rw_default, pi=pi_default, C=0.0,
            )
            df_l = simulate_commingled(to_field(l_dur, "time", U), 500, l_ci,
                                       fluid="oil")
            ycol, dcol = "dp", "tdp_dt"

        # Summary metrics
        kh1_disp = l_k1 * l_h1
        kh2_disp = l_k2 * l_h2
        kh_tot_disp = kh1_disp + kh2_disp
        mA, mB, mC = st.columns(3)
        mA.metric(f"k₁h₁ ({label('permeability', U)}·{label('height', U)})",
                  f"{kh1_disp:,.0f}")
        mB.metric(f"k₂h₂ ({label('permeability', U)}·{label('height', U)})",
                  f"{kh2_disp:,.0f}")
        mC.metric(f"kh_total ({label('permeability', U)}·{label('height', U)})",
                  f"{kh_tot_disp:,.0f}   (κ₁={kh1_disp/kh_tot_disp:.2f})")

        # Plot — comparison with single-equivalent kh
        fig_l = go.Figure()
        # Layered response
        fig_l.add_trace(go.Scatter(
            x=df_l["t"], y=df_l[ycol] * analysis_factor,
            mode="lines", name=("Δm(p) (layered)" if is_gas() else "ΔP (layered)"),
            line=dict(color=COLOR_DARK, width=3),
        ))
        fig_l.add_trace(go.Scatter(
            x=df_l["t"], y=df_l[dcol] * analysis_factor,
            mode="lines",
            name=("t·dm/dt (layered)" if is_gas() else "t·dP/dt (layered)"),
            line=dict(color=COLOR_RED, width=3),
        ))

        # Reference: equivalent single-layer with kh_total/h_total
        try:
            if is_gas():
                gw_eq = GasWellInputs(
                    k=l_ci.kh_total / l_ci.h_total, h=l_ci.h_total,
                    qg=l_ci.q_total, s=0, C=0,
                    T_R=pvt_table["T_R"], pi=pi_default,
                    phi=(l_phi1 + l_phi2) / 2.0, rw=rw_default,
                )
                df_eq = simulate_gas_drawdown(to_field(l_dur, "time", U), 500,
                                               gw_eq, pvt_table)
                eq_y = df_eq["dm"] * analysis_factor
                eq_d = df_eq["tdmdt"] * analysis_factor
            else:
                w_eq = WellInputs(
                    k=l_ci.kh_total / l_ci.h_total, h=l_ci.h_total,
                    q=l_ci.q_total, s=0, C=0,
                    mu=mu_default, B=B_default,
                    phi=(l_phi1 + l_phi2) / 2.0,
                    ct=ct_default, rw=rw_default, pi=pi_default,
                )
                df_eq = simulate_drawdown(to_field(l_dur, "time", U), 500, w_eq)
                eq_y = df_eq["dp"] * analysis_factor
                eq_d = df_eq["tdp_dt"] * analysis_factor

            fig_l.add_trace(go.Scatter(
                x=df_eq["t"], y=eq_d, mode="lines",
                name="t·dP/dt (single-layer, equiv kh)",
                line=dict(color=COLOR_TEAL, width=2, dash="dash"),
            ))
        except Exception:
            pass

        layout = get_plotly_layout("Layered response vs equivalent single-layer")
        layout["xaxis"].update(type="log", title=axis_title("Δt", "time", U))
        layout["yaxis"].update(type="log", title=axis_title(analysis_label,
                                                             analysis_qty, U))
        fig_l.update_layout(layout)
        st.plotly_chart(fig_l, use_container_width=True)

        st.download_button(
            "⬇  Download layered simulation (CSV)",
            data=df_l.to_csv(index=False).encode("utf-8"),
            file_name=f"layered_{ss['fluid_type'].lower()}_{U.lower()}.csv",
            mime="text/csv",
            key="dl_layered",
        )

    # ============================================================
    #  Economic Exposure
    # ============================================================
    st.markdown("---")
    st.subheader("Economic Exposure")

    # Price + deferred-rate quantities depend on fluid
    if is_gas():
        price_quantity = "gas_price"
        # ~$3-5/Mscf typical; SI default ≈ $176/10³m³
        price_default = 4.0 if U == UnitSystem.FIELD.value else 141.0
        defer_rate_quantity = "gas_rate"
        defer_rate_default = q_default * 2
        defer_rate_max = 1_000_000.0
        defer_rate_step = 100.0
    else:
        price_quantity = "oil_price"
        price_default = 78.0 if U == UnitSystem.FIELD.value else 490.0
        defer_rate_quantity = "rate"
        defer_rate_default = q_default
        defer_rate_max = q_max
        defer_rate_step = 50.0

    e1, e2, e3 = st.columns(3)
    with e1:
        rig_rate = st.number_input("Rig day rate (USD/day)", 0.0, 5_000_000.0,
                                   180_000.0, 5_000.0, key="ui_rig")
        total_days = st.number_input("Total operational days",
                                     0.1, 60.0, 5.0, 0.5, key="ui_days")
    with e2:
        service_fees = st.number_input("Fixed service fees (USD)",
                                       0.0, 50_000_000.0, 350_000.0, 10_000.0,
                                       key="ui_fees")
        oil_price_in = st.number_input(
            f"Price ({label(price_quantity, U)})",
            0.1, 5000.0, price_default, 0.1,
            key="ui_oil_gas" if is_gas() else "ui_oil",
        )
    with e3:
        deferred_rate_in = st.number_input(
            f"Deferred production rate ({label(defer_rate_quantity, U)})",
            0.0, defer_rate_max, defer_rate_default, defer_rate_step,
            key="ui_dq_gas" if is_gas() else "ui_dq",
        )
        defer_dur_in = st.number_input(
            f"Deferred period ({label('time', U)})",
            0.0, 1500.0, float(duration_in), 1.0, key="ui_dt",
        )

    econ = compute_costs(EconomicsInputs(
        rig_day_rate=rig_rate,
        total_days=total_days,
        fixed_service_fees=service_fees,
        deferred_rate=to_field(deferred_rate_in, defer_rate_quantity, U),
        test_duration_hr=to_field(defer_dur_in, "time", U),
        oil_price=to_field(oil_price_in, price_quantity, U),
    ))

    # All monetary values are USD in both systems — no conversion needed
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rig cost",          f"${econ['rig_cost']:,.0f}")
    m2.metric("Service fees",      f"${econ['service_fees']:,.0f}")
    m3.metric("Deferred revenue",  f"${econ['deferred_loss']:,.0f}")
    m4.metric("Total exposure",    f"${econ['total_exposure']:,.0f}")

    donut = go.Figure(go.Pie(
        labels=list(econ["breakdown"].keys()),
        values=list(econ["breakdown"].values()),
        hole=0.62,
        marker=dict(
            colors=[COLOR_DARK, COLOR_TEAL, COLOR_RED],
            line=dict(color=COLOR_WHITE, width=2),
        ),
        textinfo="label+percent",
        textfont=dict(color=COLOR_DARK, size=12),
    ))
    donut.update_layout(get_plotly_layout("Cost breakdown"))
    donut.update_layout(showlegend=True, height=380)
    st.plotly_chart(donut, use_container_width=True)

    # Persist snapshot for the report (display the user's units)
    ss["design_snapshot"] = {
        "unit_system": U,
        "well_inputs_display": (
            {
                f"Fluid":                                        ss["fluid_type"],
                f"Permeability k ({label('permeability', U)})": fmt("permeability", k_in),
                f"Net pay h ({label('height', U)})":             fmt("height", h_in),
                f"Rate q ({label(rate_quantity, U)})":           fmt(rate_quantity, q_in),
                "Skin s":                                        f"{s_in:.2f}",
                f"Wellbore storage C ({label('wbs', U)})":       fmt("wbs", C_in),
                f"Duration ({label('time', U)})":                f"{duration_in:.1f}",
                f"Reservoir T ({label('temperature', U)})":      f"{ss.get('pvt_T', '—')}",
                f"Gas gravity γg":                               f"{ss.get('pvt_gamma', '—')}",
                f"μg(pi) (cp)":                                  f"{mu_default:.4f}",
                f"cg(pi) (1/psi)":                               f"{ct_default:.3e}",
            } if is_gas() else
            {
                f"Fluid":                                        ss["fluid_type"],
                f"Permeability k ({label('permeability', U)})": fmt("permeability", k_in),
                f"Net pay h ({label('height', U)})":             fmt("height", h_in),
                f"Rate q ({label('rate', U)})":                  fmt("rate", q_in),
                "Skin s":                                        f"{s_in:.2f}",
                f"Wellbore storage C ({label('wbs', U)})":       fmt("wbs", C_in),
                f"Duration ({label('time', U)})":                f"{duration_in:.1f}",
                f"Viscosity μ ({label('viscosity', U)})":        fmt("viscosity", mu_disp),
                f"FVF B ({label('fvf', U)})":                    fmt("fvf", B_disp),
            }
        ),
        "well_field": {  # field values for later MC reuse
            "q_field": well.q,
        },
        "econ": {
            "Rig cost (USD)":            f"${econ['rig_cost']:,.0f}",
            "Service fees (USD)":        f"${econ['service_fees']:,.0f}",
            ("Deferred production (Mscf)" if is_gas() else "Deferred production (bbl)"):
                f"{econ['deferred_bbl']:,.0f}",
            "Deferred revenue (USD)":    f"${econ['deferred_loss']:,.0f}",
            "Total exposure (USD)":      f"${econ['total_exposure']:,.0f}",
        },
    }


# =====================================================================
#  TAB 2 — Interpretation
# =====================================================================
with tab_interp:
    st.subheader("Field-data Interpretation")

    st.markdown(
        f"Upload a CSV with columns **time** ({label('time', U)}), "
        f"**pressure** ({label('pressure', U)}), and **rate** ({label('rate', U)}). "
        f"Values will be internally converted to field units for the engine. "
        f"If no file is uploaded, a synthetic dataset is generated."
    )

    up = st.file_uploader("Upload pressure / rate CSV", type=["csv"])

    if up is not None:
        # Cached parsing + vectorized unit conversion. Re-reads only if a NEW
        # file is uploaded or the active unit system changes.
        try:
            df = cached_read_csv(up.getvalue(), U)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            st.stop()

        dropped = df.attrs.get("dropped_rows", 0)
        if dropped:
            st.warning(f"Dropped {dropped:,} non-numeric / blank row(s) from the upload.")
        # Rate threshold for event detection: 1 % of max rate (in field units),
        # floored at a sensible minimum.
        rate_thr_field = max(10.0, 0.01 * float(np.max(np.abs(df["rate"]))))
    else:
        # Cached synthetic 2-flow / 2-buildup demo dataset (analytic superposition)
        df = cached_synthetic_dataset(
            mu=mu_default if not is_gas() else 1.0,
            B=B_default if not is_gas() else 1.2,
            phi=phi_default,
            ct=ct_default if not is_gas() else 1e-5,
            rw=rw_default, pi=pi_default,
        )
        rate_thr_field = 50.0

    # ---------------------- GAS pseudo-pressure transform ----------------------
    # For gas wells, the math must operate on m(p) and tₐ, not ΔP and Δt.
    # We transform df["pressure"] → m(p) in place; "pressure_psi" is preserved.
    # The Bourdet-derivative / event-detection pipeline below is agnostic to
    # what the "pressure" column represents (any monotone scalar field works).
    if is_gas() and pvt_table is not None:
        df = df.copy()
        df["pressure_psi"] = df["pressure"]    # keep the raw gauge pressure
        # Map every measured psi → psi²/cp using the cached m(p) table
        df["pressure"] = pseudo_pressure(
            df["pressure"].to_numpy(),
            pvt_table["p_pp"], pvt_table["m_pp"],
        )
        st.info(
            f"🔬  Gas mode: pressures transformed to pseudo-pressure **m(p)** "
            f"(psi²/cp) for analysis. Bourdet, IARF and skin all use Δm(p) and "
            f"the gas-specific slope constant 1637 in place of 162.6."
        )

    c1, c2 = st.columns(2)
    n_decim = c1.slider("Decimation — keep every Nth point", 1, 50, 5)
    L_smooth = c2.slider("Bourdet log-time smoothing L", 0.0, 0.5, 0.10, 0.01)

    df_use = decimate(df, n_decim)
    st.caption(f"Rows after decimation: **{len(df_use):,}** (raw: {len(df):,})")

    # --- Raw signal (vectorized conversion to user units) ---
    q_factor = from_field(1.0, "rate" if not is_gas() else "gas_rate", U)
    q_unit_label = label("rate" if not is_gas() else "gas_rate", U)

    # For raw display, show actual gauge pressure even in gas mode
    if is_gas() and "pressure_psi" in df_use.columns:
        raw_p = df_use["pressure_psi"].to_numpy()
    else:
        raw_p = df_use["pressure"].to_numpy()
    p_factor = from_field(1.0, "pressure", U)
    dp_factor = from_field(1.0, "delta_pressure", U)
    p_disp = raw_p * p_factor
    q_disp = df_use["rate"].to_numpy() * q_factor

    raw_fig = go.Figure()
    raw_fig.add_trace(go.Scatter(
        x=df_use["time"], y=p_disp, mode="lines",
        name=f"Pressure ({label('pressure', U)})",
        line=dict(color=COLOR_DARK, width=2), yaxis="y1",
    ))
    raw_fig.add_trace(go.Scatter(
        x=df_use["time"], y=q_disp, mode="lines",
        name=f"Rate ({q_unit_label})",
        line=dict(color=COLOR_RED, width=2, dash="dot"), yaxis="y2",
    ))
    layout = get_plotly_layout("Raw pressure & rate")
    layout["xaxis"].update(title=axis_title("Time", "time", U))
    layout["yaxis"].update(title=axis_title("Pressure", "pressure", U))
    layout["yaxis2"] = dict(
        title=f"Rate ({q_unit_label})", overlaying="y", side="right",
        gridcolor="#E5E5E5", linecolor=COLOR_DARK, showgrid=False,
    )
    raw_fig.update_layout(layout)
    st.plotly_chart(raw_fig, use_container_width=True)

    # ---- For gas, the analysis-variable display quantity is pseudo_pressure ----
    if is_gas():
        analysis_qty = "pseudo_pressure"
        analysis_factor = from_field(1.0, "pseudo_pressure", U)
        analysis_label = "Δm(p), t·dm/dt"
        rate_q_for_threshold = q_factor   # use gas rate
    else:
        analysis_qty = "delta_pressure"
        analysis_factor = dp_factor
        analysis_label = "ΔP, t·dP/dt"

    # --- Event detection + Bourdet derivatives (cached) ---
    periods, buildups = cached_buildup_pack(df_use, rate_thr_field, L_smooth)
    n_flow = sum(1 for p in periods if p["type"] == "flow")
    n_bu = sum(1 for p in periods if p["type"] == "buildup")
    st.info(f"Detected **{len(periods)}** flow periods → {n_flow} drawdowns · {n_bu} buildups.")

    # Diagnostic plot — vectorized conversion to user units
    st.markdown("#### Superposed buildup diagnostic")
    palette = [COLOR_RED, COLOR_TEAL, COLOR_DARK, "#E07A00", "#7A1FA2"]
    dfig = go.Figure()
    for i, bu in enumerate(buildups):
        col = palette[i % len(palette)]
        dp_d = bu["dp"]     * analysis_factor
        td_d = bu["tdp_dt"] * analysis_factor
        q_ref_disp = bu["q_ref"] * q_factor
        bu_label = f"Buildup #{i + 1}  (q_ref={q_ref_disp:.1f} {q_unit_label})"
        legend_dp     = ("Δm(p)"   if is_gas() else "ΔP")
        legend_deriv  = ("t·dm/dt" if is_gas() else "t·dP/dt")
        dfig.add_trace(go.Scatter(
            x=bu["dt"], y=dp_d, mode="markers",
            name=f"{legend_dp}  {bu_label}",
            marker=dict(color=col, size=5, symbol="circle-open"),
            legendgroup=str(i),
        ))
        dfig.add_trace(go.Scatter(
            x=bu["dt"], y=td_d, mode="markers",
            name=f"{legend_deriv}  {bu_label}",
            marker=dict(color=col, size=6, symbol="diamond"),
            legendgroup=str(i),
        ))
    layout = get_plotly_layout("Log-log buildup derivatives")
    layout["xaxis"].update(type="log", title=axis_title("Δt", "time", U))
    layout["yaxis"].update(type="log", title=axis_title(analysis_label, analysis_qty, U))
    dfig.update_layout(layout)
    st.plotly_chart(dfig, use_container_width=True)

    # --- Regression panel ---
    st.markdown("#### Visual time-window regression")
    if not buildups:
        st.warning("No buildup detected — cannot run regression.")
    else:
        sel_idx = st.selectbox(
            "Buildup to analyze", range(len(buildups)),
            format_func=lambda i: f"Buildup #{i + 1}  "
                                  f"(q_ref={buildups[i]['q_ref'] * q_factor:.1f} "
                                  f"{q_unit_label})",
        )
        bu = buildups[sel_idx]

        dt_min, dt_max = float(np.min(bu["dt"])), float(np.max(bu["dt"]))
        window = st.slider(
            f"Regression window (Δt, {label('time', U)})",
            min_value=dt_min, max_value=dt_max,
            value=(dt_min, dt_max),
            step=(dt_max - dt_min) / 200.0,
        )
        # All three regimes are now supported for both fluids.
        regime_options = [
            "Radial Flow (IARF)",
            "Linear Flow (Fracture)",
            "Closed Boundary (PSS)",
        ]
        regime = st.radio("Flow regime", regime_options, horizontal=True)

        # Dispatch on (regime, fluid). Each branch sets `res` and populates
        # st metrics and the session-state `interpretation_result` summary.
        # ─── (Radial × Gas) ───
        if regime.startswith("Radial") and is_gas():
            mu_ref = float(np.interp(well.pi, pvt_table["p"], pvt_table["mu"]))
            cg_ref = float(np.interp(well.pi, pvt_table["p"], pvt_table["cg"]))
            res = regress_radial_gas(
                dt=bu["dt"], dm_deriv_tdmdt=bu["tdp_dt"],
                dm=bu["dp"],
                qg=abs(bu["q_ref"]) if abs(bu["q_ref"]) > 1 else well.q,
                T_R=pvt_table["T_R"], h=well.h,
                phi=phi_default, mu_ref=mu_ref, ct_ref=cg_ref, rw=rw_default,
                t_min=window[0], t_max=window[1],
            )
            if res["ok"]:
                k_disp        = res["k_mD"]
                plateau_disp  = res["plateau_tdmdt"] * analysis_factor
                m_slope_field = res["semilog_slope_m_g"]
                m_slope_disp  = m_slope_field * analysis_factor
                skin_calc     = res.get("skin")
                flatness      = res.get("flatness_geom_std", 1.0)

                a, b, c, d = st.columns(4)
                a.metric(f"k  ({label('permeability', U)})", f"{k_disp:.2f}")
                b.metric(f"Plateau t·dm/dt ({label('pseudo_pressure', U)})",
                         f"{plateau_disp:.2e}")
                c.metric("Skin s (–)", f"{skin_calc:+.2f}" if skin_calc is not None else "—")
                d.metric("Points in window", f"{res['n_points']}")

                if flatness < 1.15:
                    st.success(f"✓ Window is plateau-like (flatness = {flatness:.2f}).")
                elif flatness < 1.40:
                    st.warning(f"⚠ Window is only weakly flat (flatness = {flatness:.2f}).")
                else:
                    st.error(f"✗ Window is NOT a clean plateau (flatness = {flatness:.2f}).")

                interp_summary = {
                    "Regime": res["regime"],
                    f"Permeability k ({label('permeability', U)})": f"{k_disp:.2f}",
                    f"Semilog slope m_g ({label('slope_psp', U)})": f"{m_slope_disp:.3e}",
                    f"Window Δt ({label('time', U)})": f"{window[0]:.2f} – {window[1]:.2f}",
                    "Window flatness (geom-std)": f"{flatness:.3f}",
                    "Status": "Converged",
                }
                if skin_calc is not None:
                    interp_summary["Skin s (–)"] = f"{skin_calc:+.2f}"
                ss["interpretation_result"] = interp_summary
                ss["interp_slope_m_field"] = m_slope_field
                ss["interp_is_gas"] = True
                ss["interp_k_field"] = res["k_mD"]
                ss["interp_skin_field"] = skin_calc if skin_calc is not None else 0.0
            else:
                st.error(res["message"])

        # ─── (Radial × Liquid) ───
        elif regime.startswith("Radial"):
            res = regress_radial(
                dt=bu["dt"], dp_deriv_tdpdt=bu["tdp_dt"],
                q=abs(bu["q_ref"]) if abs(bu["q_ref"]) > 1 else well.q,
                B=B_default, mu=mu_default, h=well.h,
                t_min=window[0], t_max=window[1],
                dp=bu["dp"], phi=phi_default, ct=ct_default, rw=rw_default,
            )
            if res["ok"]:
                k_disp        = from_field(res["k_mD"],            "permeability",   U)
                plateau_disp  = from_field(res["plateau_tdpdt"],   "delta_pressure", U)
                m_slope_disp  = from_field(res["semilog_slope_m"], "slope_semilog",  U)
                skin_calc     = res.get("skin")
                flatness      = res.get("flatness_geom_std", 1.0)

                a, b, c, d = st.columns(4)
                a.metric(f"k  ({label('permeability', U)})", f"{k_disp:.2f}")
                b.metric(f"Plateau t·dP/dt ({label('delta_pressure', U)})", f"{plateau_disp:.2f}")
                c.metric("Skin s (–)", f"{skin_calc:+.2f}" if skin_calc is not None else "—")
                d.metric("Points in window", f"{res['n_points']}")

                if flatness < 1.15:
                    st.success(f"✓ Window is plateau-like (flatness = {flatness:.2f}).")
                elif flatness < 1.40:
                    st.warning(f"⚠ Window is only weakly flat (flatness = {flatness:.2f}).")
                else:
                    st.error(f"✗ Window is NOT a clean plateau (flatness = {flatness:.2f}).")

                interp_summary = {
                    "Regime": res["regime"],
                    f"Permeability k ({label('permeability', U)})": f"{k_disp:.2f}",
                    f"Semilog slope m ({label('slope_semilog', U)})": f"{m_slope_disp:.2f}",
                    f"Window Δt ({label('time', U)})": f"{window[0]:.2f} – {window[1]:.2f}",
                    "Window flatness (geom-std)": f"{flatness:.3f}",
                    "Status": "Converged",
                }
                if skin_calc is not None:
                    interp_summary["Skin s (–)"] = f"{skin_calc:+.2f}"
                ss["interpretation_result"] = interp_summary
                ss["interp_slope_m_field"] = res["semilog_slope_m"]
                ss["interp_is_gas"] = False
                ss["interp_k_field"] = res["k_mD"]
                ss["interp_skin_field"] = skin_calc if skin_calc is not None else 0.0
            else:
                st.error(res["message"])

        # ─── (Linear × Gas) ───
        elif regime.startswith("Linear") and is_gas():
            mu_ref = float(np.interp(well.pi, pvt_table["p"], pvt_table["mu"]))
            cg_ref = float(np.interp(well.pi, pvt_table["p"], pvt_table["cg"]))
            res = regress_linear_gas(
                dt=bu["dt"], dm=bu["dp"],
                qg=abs(bu["q_ref"]) if abs(bu["q_ref"]) > 1 else well.q,
                T_R=pvt_table["T_R"], h=well.h,
                phi=phi_default, mu_ref=mu_ref, ct_ref=cg_ref,
                t_min=window[0], t_max=window[1],
            )
            if res["ok"]:
                xf_disp    = from_field(res["xf_sqrtk"], "frac_capacity", U)
                slope_disp = res["slope"] * analysis_factor
                a, b, c = st.columns(3)
                a.metric(f"xf · √k  ({label('frac_capacity', U)})", f"{xf_disp:.1f}")
                b.metric("Slope (psi²/cp/√hr)", f"{slope_disp:.2e}")
                c.metric("Points in window", f"{res['n_points']}")
                ss["interpretation_result"] = {
                    "Regime": res["regime"],
                    f"Fracture capacity xf·√k ({label('frac_capacity', U)})": f"{xf_disp:.1f}",
                    "Sqrt-time slope (psi²/cp/√hr)": f"{slope_disp:.3e}",
                    f"Window Δt ({label('time', U)})": f"{window[0]:.2f} – {window[1]:.2f}",
                    "Status": "Converged",
                }
            else:
                st.error(res["message"])

        # ─── (Linear × Liquid) ───
        elif regime.startswith("Linear"):
            res = regress_linear(
                dt=bu["dt"], dp=bu["dp"],
                q=abs(bu["q_ref"]) if abs(bu["q_ref"]) > 1 else well.q,
                B=B_default, mu=mu_default, h=well.h,
                ct=ct_default, phi=phi_default,
                t_min=window[0], t_max=window[1],
            )
            if res["ok"]:
                xf_disp    = from_field(res["xf_sqrtk"], "frac_capacity", U)
                slope_disp = from_field(res["slope"],    "slope_sqrt_t",  U)
                a, b, c = st.columns(3)
                a.metric(f"xf · √k  ({label('frac_capacity', U)})", f"{xf_disp:.1f}")
                b.metric(f"Slope ({label('slope_sqrt_t', U)})",     f"{slope_disp:.2f}")
                c.metric("Points in window", f"{res['n_points']}")
                ss["interpretation_result"] = {
                    "Regime": res["regime"],
                    f"Fracture capacity xf·√k ({label('frac_capacity', U)})": f"{xf_disp:.1f}",
                    f"Sqrt-time slope ({label('slope_sqrt_t', U)})": f"{slope_disp:.2f}",
                    f"Window Δt ({label('time', U)})": f"{window[0]:.2f} – {window[1]:.2f}",
                    "Status": "Converged",
                }
            else:
                st.error(res["message"])

        # ─── (Closed Boundary PSS × Gas) ───
        elif regime.startswith("Closed") and is_gas():
            cg_ref = float(np.interp(well.pi, pvt_table["p"], pvt_table["cg"]))
            res = regress_pss_gas(
                dt=bu["dt"], dm=bu["dp"],
                qg=abs(bu["q_ref"]) if abs(bu["q_ref"]) > 1 else well.q,
                T_R=pvt_table["T_R"], h=well.h, phi=phi_default, ct_ref=cg_ref,
                t_min=window[0], t_max=window[1],
            )
            if res["ok"]:
                a, b, c, d = st.columns(4)
                a.metric("Area (acre)", f"{res['area_acre']:.1f}")
                b.metric(f"Drainage re ({label('length', U)})",
                         f"{from_field(res['re_ft'], 'length', U):.0f}")
                c.metric("Slope m_g* (psi²/cp/hr)", f"{res['slope']:.3e}")
                d.metric("Points", f"{res['n_points']}")
                st.info("ℹ Closed-boundary regression assumes the window is in fully-developed PSS. "
                        "Check the log-log derivative — late time should rise with unit slope.")
                ss["interpretation_result"] = {
                    "Regime": res["regime"],
                    "Drainage area (acre)": f"{res['area_acre']:.1f}",
                    f"Drainage radius re ({label('length', U)})":
                        f"{from_field(res['re_ft'], 'length', U):.0f}",
                    "Slope m_g* (psi²/cp/hr)": f"{res['slope']:.3e}",
                    f"Window Δt ({label('time', U)})": f"{window[0]:.2f} – {window[1]:.2f}",
                    "Status": "Converged",
                }
            else:
                st.error(res["message"])

        # ─── (Closed Boundary PSS × Liquid) ───
        else:  # Closed × oil/water
            res = regress_pss(
                dt=bu["dt"], dp=bu["dp"],
                q=abs(bu["q_ref"]) if abs(bu["q_ref"]) > 1 else well.q,
                B=B_default, h=well.h, phi=phi_default, ct=ct_default,
                t_min=window[0], t_max=window[1],
            )
            if res["ok"]:
                a, b, c, d = st.columns(4)
                a.metric("Area (acre)", f"{res['area_acre']:.1f}")
                b.metric(f"Drainage re ({label('length', U)})",
                         f"{from_field(res['re_ft'], 'length', U):.0f}")
                c.metric(f"Pore vol ({label('rate', U)}·hr)",
                         f"{from_field(res['pore_volume_bbl'], 'rate', U):,.0f}")
                d.metric("Points", f"{res['n_points']}")
                slope_psi_hr = from_field(res['slope'], 'delta_pressure', U)
                st.info(f"ℹ PSS slope m* = {slope_psi_hr:.3f} {label('delta_pressure', U)}/hr. "
                        "Check that the late-time derivative rises with unit slope.")
                ss["interpretation_result"] = {
                    "Regime": res["regime"],
                    "Drainage area (acre)": f"{res['area_acre']:.1f}",
                    f"Drainage radius re ({label('length', U)})":
                        f"{from_field(res['re_ft'], 'length', U):.0f}",
                    f"Pore volume ({label('rate', U)}·hr)":
                        f"{from_field(res['pore_volume_bbl'], 'rate', U):,.0f}",
                    f"PSS slope m* ({label('delta_pressure', U)}/hr)":
                        f"{slope_psi_hr:.3f}",
                    f"Window Δt ({label('time', U)})": f"{window[0]:.2f} – {window[1]:.2f}",
                    "Status": "Converged",
                }
            else:
                st.error(res["message"])

        # Window overlay (vectorized conversion) — fluid-aware
        ovl = go.Figure()
        dp_d = bu["dp"]     * analysis_factor
        td_d = bu["tdp_dt"] * analysis_factor
        legend_dp     = ("Δm(p)"   if is_gas() else "ΔP")
        legend_deriv  = ("t·dm/dt" if is_gas() else "t·dP/dt")
        ovl.add_trace(go.Scatter(
            x=bu["dt"], y=td_d, mode="markers",
            name=legend_deriv, marker=dict(color=COLOR_RED, size=6, symbol="diamond"),
        ))
        ovl.add_trace(go.Scatter(
            x=bu["dt"], y=dp_d, mode="markers",
            name=legend_dp, marker=dict(color=COLOR_DARK, size=5, symbol="circle-open"),
        ))
        ovl.add_vrect(x0=window[0], x1=window[1],
                      fillcolor=COLOR_TEAL, opacity=0.15, line_width=0,
                      annotation_text="Regression window",
                      annotation_position="top left",
                      annotation_font_color=COLOR_DARK)

        # === Regime-specific reference overlay ===
        if regime.startswith("Radial") and res.get("ok"):
            plateau_key = "plateau_tdmdt" if is_gas() else "plateau_tdpdt"
            plateau_disp = res[plateau_key] * analysis_factor
            x_ref = np.array([bu["dt"].min(), bu["dt"].max()])
            ovl.add_trace(go.Scatter(
                x=x_ref, y=[plateau_disp, plateau_disp],
                mode="lines",
                name=f"IARF plateau ({plateau_disp:.3g})",
                line=dict(color=COLOR_TEAL, width=2, dash="dash"),
            ))
        elif regime.startswith("Linear") and res.get("ok"):
            slope_field_disp = res["slope"] * analysis_factor
            x_min = max(bu["dt"].min(), 1e-3)
            x_max = bu["dt"].max()
            x_ref = np.linspace(x_min, x_max, 50)
            y_ref = slope_field_disp * np.sqrt(x_ref)
            ovl.add_trace(go.Scatter(
                x=x_ref, y=y_ref, mode="lines",
                name="Linear-flow √t fit",
                line=dict(color=COLOR_TEAL, width=2, dash="dash"),
            ))
        elif regime.startswith("Closed") and res.get("ok"):
            # PSS reference: ΔP grows linearly in t  → t·dP/dt grows linearly in t
            # (unit slope on log-log). Draw both:
            #   (a) The straight ΔP = m*·t + b fit on Cartesian-like view
            #   (b) A unit-slope reference line for the derivative
            slope_field = res["slope"] * analysis_factor
            intercept_field = res["intercept"] * analysis_factor
            x_min = max(bu["dt"].min(), 1e-3)
            x_max = bu["dt"].max()
            x_ref = np.geomspace(x_min, x_max, 50)
            y_ref = slope_field * x_ref + intercept_field
            ovl.add_trace(go.Scatter(
                x=x_ref, y=np.clip(y_ref, 1e-9, None), mode="lines",
                name="PSS fit  ΔP = m*·t + b",
                line=dict(color=COLOR_TEAL, width=2, dash="dash"),
            ))
            # Derivative unit-slope reference, anchored at window mid-point
            t_mid = np.sqrt(x_min * x_max)
            tdpdt_mid_field = float(np.interp(t_mid, bu["dt"], bu["tdp_dt"]))
            tdpdt_mid_disp  = tdpdt_mid_field * analysis_factor
            y_unit = tdpdt_mid_disp * (x_ref / t_mid)   # slope = 1 on log-log
            ovl.add_trace(go.Scatter(
                x=x_ref, y=y_unit, mode="lines",
                name="Unit-slope reference (PSS)",
                line=dict(color=COLOR_DARK, width=1, dash="dot"),
            ))

        layout = get_plotly_layout("Selected window — log-log")
        layout["xaxis"].update(type="log", title=axis_title("Δt", "time", U))
        layout["yaxis"].update(type="log",
                               title=axis_title(analysis_label, analysis_qty, U))
        ovl.update_layout(layout)
        st.plotly_chart(ovl, use_container_width=True)

        # === Downloads — buildup table and regression summary ===
        st.markdown("##### Downloads")
        y_unit = label(analysis_qty, U)
        bu_df = pd.DataFrame({
            f"dt_{label('time', U)}":  bu["dt"],
            f"dy_{y_unit}":            bu["dp"]     * analysis_factor,
            f"tdy_dt_{y_unit}":        bu["tdp_dt"] * analysis_factor,
        })
        st.download_button(
            "⬇  Download selected buildup (CSV)",
            data=bu_df.to_csv(index=False).encode("utf-8"),
            file_name=f"buildup_{sel_idx + 1}_{ss['fluid_type'].lower()}_{U.lower()}.csv",
            mime="text/csv",
            key="dl_buildup",
        )

        # ============================================================
        #  Model vs. Data — four-panel workspace
        # ============================================================
        st.markdown("---")
        with st.expander("📊 Model vs Data — full PTA workspace",
                         expanded=False):
            st.caption(
                "Four-panel view showing the analytical model alongside the "
                "measured data: log-log diagnostic (top-left), superposition / "
                "Horner plot (top-right), raw pressure history (bottom-left), "
                "and rate schedule (bottom-right). Use the controls below to "
                "tweak the model parameters and see the match update."
            )

            # Use current regression result (if any) as starting point.
            # Otherwise seed from sidebar defaults.
            seed_k = float(ss.get("interp_k_field", well.k))
            seed_s = float(ss.get("interp_skin_field", well.s if not is_gas() else 0.0))

            mvc1, mvc2, mvc3 = st.columns(3)
            with mvc1:
                mv_k = st.number_input(
                    f"Model k ({label('permeability', U)})",
                    0.01, 50000.0, seed_k, 1.0, key="mv_k",
                )
            with mvc2:
                mv_s = st.number_input(
                    "Model skin s (–)", -5.0, 50.0, seed_s, 0.1, key="mv_s",
                )
            with mvc3:
                mv_C_default = float(well.C) if well.C > 0 else 0.001
                mv_C = st.number_input(
                    f"Model WBS C ({label('wbs', U)})",
                    1e-6, 5.0, mv_C_default, 1e-4, format="%.4f", key="mv_C",
                )

            # Build a model trace at the SAME Δt sample points as the
            # selected buildup. Honour fluid type.
            dt_obs = bu["dt"]
            dp_obs = bu["dp"]
            tdp_obs = bu["tdp_dt"]
            t_p = bu.get("t_p", float(dt_obs.max() * 2))
            p0_bu = bu.get("p0", float(well.pi))

            if is_gas():
                gw_model = GasWellInputs(
                    k=mv_k, h=well.h, qg=abs(bu["q_ref"]) if abs(bu["q_ref"]) > 1 else well.q,
                    s=mv_s, C=mv_C, T_R=pvt_table["T_R"], pi=well.pi,
                    phi=phi_default, rw=rw_default,
                )
                t_grid = np.logspace(np.log10(max(dt_obs.min(), 1e-3)),
                                     np.log10(dt_obs.max()), 300)
                df_model = simulate_gas_drawdown(float(t_grid.max()), 300,
                                                   gw_model, pvt_table)
                model_y = np.interp(dt_obs, df_model["t"], df_model["dm"])
                model_d = np.interp(dt_obs, df_model["t"], df_model["tdmdt"])
            else:
                w_model = WellInputs(
                    k=mv_k, h=well.h,
                    q=abs(bu["q_ref"]) if abs(bu["q_ref"]) > 1 else well.q,
                    s=mv_s, C=mv_C, mu=mu_default, B=B_default,
                    phi=phi_default, ct=ct_default,
                    rw=rw_default, pi=well.pi,
                )
                df_model = simulate_drawdown(float(dt_obs.max() * 1.1), 300, w_model)
                model_y = np.interp(dt_obs, df_model["t"], df_model["dp"])
                model_d = np.interp(dt_obs, df_model["t"], df_model["tdp_dt"])

            # --- Panel construction with subplots ---
            from plotly.subplots import make_subplots
            fig4 = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Log-log diagnostic — ΔP & Bourdet derivative",
                    "Horner / superposition plot",
                    "Raw pressure history",
                    "Rate schedule",
                ),
                horizontal_spacing=0.12, vertical_spacing=0.16,
            )

            # ---- Panel 1: log-log diagnostic with model overlay ----
            fig4.add_trace(go.Scatter(
                x=dt_obs, y=dp_obs * analysis_factor, mode="markers",
                name=("Δm(p) data" if is_gas() else "ΔP data"),
                marker=dict(color=COLOR_TEAL, size=5, symbol="circle"),
                showlegend=True,
            ), row=1, col=1)
            fig4.add_trace(go.Scatter(
                x=dt_obs, y=tdp_obs * analysis_factor, mode="markers",
                name=("t·dm/dt data" if is_gas() else "t·dP/dt data"),
                marker=dict(color=COLOR_RED, size=5, symbol="diamond"),
                showlegend=True,
            ), row=1, col=1)
            # Model overlay — dashed lines
            fig4.add_trace(go.Scatter(
                x=dt_obs, y=model_y * analysis_factor, mode="lines",
                name=("Δm(p) model" if is_gas() else "ΔP model"),
                line=dict(color=COLOR_DARK, width=2, dash="dash"),
                showlegend=True,
            ), row=1, col=1)
            fig4.add_trace(go.Scatter(
                x=dt_obs, y=model_d * analysis_factor, mode="lines",
                name=("t·dm/dt model" if is_gas() else "t·dP/dt model"),
                line=dict(color=COLOR_RED, width=2, dash="dash"),
                showlegend=True,
            ), row=1, col=1)
            fig4.update_xaxes(type="log", title_text=f"Δt ({label('time', U)})", row=1, col=1)
            fig4.update_yaxes(type="log", title_text=label(analysis_qty, U), row=1, col=1)

            # ---- Panel 2: Horner plot ----
            t_H_obs = horner_time(dt_obs, t_p)
            # On a Horner plot, BHP increases as t_H decreases (late time = small t_H).
            # The traditional plot uses log-x. We plot ΔP vs t_H so it's directly
            # comparable to the screenshot's superposition plot.
            fig4.add_trace(go.Scatter(
                x=t_H_obs, y=dp_obs * analysis_factor, mode="markers",
                name="Data", marker=dict(color=COLOR_TEAL, size=5),
                showlegend=False,
            ), row=1, col=2)
            fig4.add_trace(go.Scatter(
                x=t_H_obs, y=model_y * analysis_factor, mode="lines",
                name="Model", line=dict(color=COLOR_DARK, width=2, dash="dash"),
                showlegend=False,
            ), row=1, col=2)
            fig4.update_xaxes(type="log", title_text="Horner time (t_p + Δt)/Δt", row=1, col=2)
            fig4.update_yaxes(title_text=label(analysis_qty, U), row=1, col=2)

            # ---- Panel 3: Raw pressure history (full test) ----
            if is_gas() and "pressure_psi" in df_use.columns:
                hist_p = df_use["pressure_psi"].to_numpy() * p_factor
            else:
                hist_p = df_use["pressure"].to_numpy() * p_factor
            fig4.add_trace(go.Scatter(
                x=df_use["time"], y=hist_p, mode="lines",
                name="BHP", line=dict(color=COLOR_DARK, width=1.5),
                showlegend=False,
            ), row=2, col=1)
            fig4.update_xaxes(title_text=f"Time ({label('time', U)})", row=2, col=1)
            fig4.update_yaxes(title_text=f"Pressure ({label('pressure', U)})", row=2, col=1)

            # ---- Panel 4: Rate schedule (step function) ----
            rate_disp = df_use["rate"].to_numpy() * q_factor
            fig4.add_trace(go.Scatter(
                x=df_use["time"], y=rate_disp, mode="lines",
                name="Rate", line=dict(color=COLOR_RED, width=1.5, shape="hv"),
                showlegend=False,
            ), row=2, col=2)
            fig4.update_xaxes(title_text=f"Time ({label('time', U)})", row=2, col=2)
            fig4.update_yaxes(title_text=f"Rate ({q_unit_label})", row=2, col=2)

            # Layout from styles.py, then enlarge
            layout = get_plotly_layout("")
            layout.pop("title", None)        # subplots provide their own titles
            fig4.update_layout(
                **{k: v for k, v in layout.items() if k not in ("xaxis", "yaxis")},
                height=720,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0),
            )
            st.plotly_chart(fig4, use_container_width=True)

            # --- Match quality metrics ---
            # Mean absolute log-error on the derivative — primary match indicator
            with np.errstate(divide="ignore", invalid="ignore"):
                log_err = np.log(np.clip(model_d, 1e-12, None)) - np.log(np.clip(tdp_obs, 1e-12, None))
            mae_log = float(np.mean(np.abs(log_err[np.isfinite(log_err)])))
            rms_log = float(np.sqrt(np.mean(log_err[np.isfinite(log_err)] ** 2)))
            q1, q2, q3 = st.columns(3)
            q1.metric("MAE log(t·dP/dt)", f"{mae_log:.3f}")
            q2.metric("RMS log(t·dP/dt)", f"{rms_log:.3f}")
            quality = "Excellent" if rms_log < 0.05 else ("Good" if rms_log < 0.15 else
                       ("Fair" if rms_log < 0.30 else "Poor — tune parameters"))
            q3.metric("Match quality", quality)

        # ============================================================
        #  Multi-layer commingled inverse fit (advanced)
        # ============================================================
        st.markdown("---")
        with st.expander("⛏ Multi-Layer Fit — Commingled (no crossflow)",
                         expanded=False):
            st.caption(
                "Fit an aggregated two-layer commingled model to the selected "
                "buildup. Because a single buildup cannot resolve the k×h "
                "split between layers without PLT-derived rate allocations, "
                "the fit reports only the IDENTIFIABLE aggregate parameters "
                "(kh_total and one κ-h_ratio combination consistent with the "
                "derivative). Treat the κ / h₁-fraction split as one of many "
                "equivalent solutions."
            )
            if st.button("Run layered fit", key="btn_layered_fit"):
                with st.spinner("Fitting two-layer commingled model…"):
                    h_guess_field = well.h    # use single-layer h estimate as anchor
                    if is_gas():
                        cg_ref = float(np.interp(well.pi, pvt_table["p"], pvt_table["cg"]))
                        fit = fit_commingled_2layer(
                            dt=bu["dt"], tdp=bu["tdp_dt"],
                            q_total=abs(bu["q_ref"]) if abs(bu["q_ref"]) > 1 else well.q,
                            mu=mu_default, B=B_default, rw=rw_default,
                            phi_total=phi_default, ct_total=cg_ref,
                            fluid="gas", T_R=pvt_table["T_R"],
                            h_total_guess=h_guess_field,
                        )
                    else:
                        fit = fit_commingled_2layer(
                            dt=bu["dt"], tdp=bu["tdp_dt"],
                            q_total=abs(bu["q_ref"]) if abs(bu["q_ref"]) > 1 else well.q,
                            mu=mu_default, B=B_default, rw=rw_default,
                            phi_total=phi_default, ct_total=ct_default,
                            fluid="oil",
                            h_total_guess=h_guess_field,
                        )

                # Display
                if fit["ok"]:
                    st.success(f"Fit converged in {fit['n_iter']} iterations  ·  "
                               f"RMS log-residual = {fit['rms_log_resid']:.4f}")
                else:
                    st.warning(f"Fit did not fully converge: {fit['message']}  ·  "
                               f"RMS log-residual = {fit['rms_log_resid']:.4f}")

                kh_disp = fit["kh_total"]
                # Display in user units (kh has units of permeability×height)
                f1, f2, f3 = st.columns(3)
                f1.metric(f"kh_total ({label('permeability', U)}·{label('height', U)})",
                          f"{kh_disp:,.1f}")
                f2.metric("κ  (= k₁h₁ / kh_total)", f"{fit['kappa']:.3f}")
                f3.metric("h₁ / h_total", f"{fit['h1_fraction']:.3f}")

                st.info(f"⚠  {fit['warning']}")
            else:
                st.caption("Press the button above to fit the two-layer model.")


# =====================================================================
#  TAB 3 — Uncertainty & Reporting
# =====================================================================
with tab_uncert:
    st.subheader("Monte Carlo — Permeability uncertainty")

    if not ss.get("interp_slope_m_field"):
        st.info(
            "Run an IARF regression in **Tab 2** first to capture a semilog "
            "slope. A default slope is used until then."
        )
    m_slope_field = float(ss.get("interp_slope_m_field", 50.0))
    mc_is_gas = ss.get("interp_is_gas", False) or is_gas()

    # Slope display: psi/cycle for liquid, psi²/cp/cycle for gas
    if mc_is_gas:
        m_slope_disp = m_slope_field * from_field(1.0, "slope_psp", U)
        m_slope_disp_str = f"{m_slope_disp:.3e} {label('slope_psp', U)}"
    else:
        m_slope_disp = from_field(m_slope_field, "slope_semilog", U)
        m_slope_disp_str = f"{m_slope_disp:.2f} {label('slope_semilog', U)}"

    # System-aware + fluid-aware defaults for triangular inputs
    if U == UnitSystem.FIELD.value:
        h_lo_d, h_mo_d, h_hi_d   = 50.0, 75.0, 100.0
        h_min_d, h_max_d         = 1.0,   1000.0
        h_step                   = 1.0
        mu_lo_d, mu_mo_d, mu_hi_d = 0.8, 1.0, 1.3
        q_default_disp           = 2500.0 if not mc_is_gas else 5000.0
        q_max_d                  = 50000.0 if not mc_is_gas else 1_000_000.0
    else:
        h_lo_d, h_mo_d, h_hi_d   = 15.2, 22.86, 30.5
        h_min_d, h_max_d         = 0.3,   305.0
        h_step                   = 0.5
        mu_lo_d, mu_mo_d, mu_hi_d = 0.8, 1.0, 1.3
        q_default_disp           = 397.5 if not mc_is_gas else 141.0
        q_max_d                  = 8000.0 if not mc_is_gas else 28_000.0

    rate_quantity_mc = "gas_rate" if mc_is_gas else "rate"

    # Try to seed default q from Tab 1's snapshot (always in field internally)
    snap_q_field = ss.get("design_snapshot", {}).get("well_field", {}).get("q_field")
    if snap_q_field is not None:
        q_default_disp = from_field(snap_q_field, rate_quantity_mc, U)

    c1, c2, c3 = st.columns(3)
    with c1:
        n_iter = st.number_input("Iterations", 500, 100_000, 5_000, 500)
        q_ref_in = st.number_input(
            f"Rate q ({label(rate_quantity_mc, U)})",
            1.0, q_max_d, float(q_default_disp), 10.0,
            key="mc_q_gas" if mc_is_gas else "mc_q",
        )
    with c2:
        st.caption(f"Net pay h ({label('height', U)}) — Triangular")
        h_lo = st.number_input("Min",  h_min_d, h_max_d, h_lo_d, h_step, key="h_lo")
        h_mo = st.number_input("Mode", h_min_d, h_max_d, h_mo_d, h_step, key="h_mo")
        h_hi = st.number_input("Max",  h_min_d, h_max_d, h_hi_d, h_step, key="h_hi")
    with c3:
        if mc_is_gas:
            st.caption("Viscosity μg — Informational (does not enter gas k formula)")
        else:
            st.caption(f"Viscosity μ ({label('viscosity', U)}) — Triangular")
        mu_lo = st.number_input("Min",  0.001, 50.0, mu_lo_d, 0.01, key="mu_lo")
        mu_mo = st.number_input("Mode", 0.001, 50.0, mu_mo_d, 0.01, key="mu_mo")
        mu_hi = st.number_input("Max",  0.001, 50.0, mu_hi_d, 0.01, key="mu_hi")

    st.caption(
        f"Using semilog slope m = **{m_slope_disp_str}** (from interpretation). "
        + ("Gas formula: k = 1637 · q · T / (m_g · h)." if mc_is_gas
           else "Liquid formula: k = 162.6 · q · B · μ / (m · h).")
    )

    # Validate triangular bounds: require low ≤ mode ≤ high (strict ≤, allow degenerate)
    h_bounds_ok  = (h_lo <= h_mo <= h_hi)
    mu_bounds_ok = (mu_lo <= mu_mo <= mu_hi)
    bounds_ok    = h_bounds_ok and mu_bounds_ok and (n_iter >= 500)

    if not h_bounds_ok:
        st.error(f"Net pay h: require **Min ≤ Mode ≤ Max** (got {h_lo} / {h_mo} / {h_hi}).")
    if not mu_bounds_ok:
        st.error(f"Viscosity μ: require **Min ≤ Mode ≤ Max** (got {mu_lo} / {mu_mo} / {mu_hi}).")

    if st.button("Run Monte Carlo", disabled=not bounds_ok):
        with st.spinner(f"Running {int(n_iter):,} iterations…"):
            T_R_mc = (pvt_table["T_R"] if (mc_is_gas and pvt_table is not None) else None)
            ss["mc_result"] = run_monte_carlo(
                n_iter=int(n_iter),
                q=to_field(q_ref_in, rate_quantity_mc, U),
                B=B_default,
                m_slope=m_slope_field,
                h_tri=(
                    to_field(h_lo, "height", U),
                    to_field(h_mo, "height", U),
                    to_field(h_hi, "height", U),
                ),
                mu_tri=(
                    to_field(mu_lo, "viscosity", U),
                    to_field(mu_mo, "viscosity", U),
                    to_field(mu_hi, "viscosity", U),
                ),
                fluid="gas" if mc_is_gas else "oil",
                T_R=T_R_mc,
            )
            ss["mc_result"]["display_system"] = U

    mc = ss["mc_result"]
    if mc:
        # P-values are permeabilities — same unit (mD) in both systems, no conversion
        P90_d = from_field(mc["P90"], "permeability", U)
        P50_d = from_field(mc["P50"], "permeability", U)
        P10_d = from_field(mc["P10"], "permeability", U)
        unit  = label("permeability", U)

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "P90  Conservative", f"{P90_d:.2f} {unit}",
            help="Petroleum convention: 90% chance the TRUE value exceeds this. "
                 "Mathematically the 10th percentile of the sample distribution.",
        )
        m2.metric(
            "P50  Best estimate", f"{P50_d:.2f} {unit}",
            help="The median — 50% chance the true value is above, 50% below.",
        )
        m3.metric(
            "P10  Optimistic", f"{P10_d:.2f} {unit}",
            help="Petroleum convention: 10% chance the TRUE value exceeds this. "
                 "Mathematically the 90th percentile.",
        )

        # Convert samples once for display (vectorized — permeability factor
        # happens to be 1.0 since mD is the same in both systems, but kept
        # general for clarity / future-proofing).
        k_factor = from_field(1.0, "permeability", U)
        samples_disp = mc["samples"] * k_factor

        x_kde, y_kde = kde_curve(samples_disp)
        x_cdf, y_cdf = cdf_curve(samples_disp)

        # PDF
        pdf_fig = go.Figure()
        pdf_fig.add_trace(go.Scatter(
            x=x_kde, y=y_kde, mode="lines",
            name="PDF (Gaussian KDE)",
            line=dict(color=COLOR_RED, width=3),
            fill="tozeroy", fillcolor="rgba(255,18,67,0.18)",
        ))
        for lab, val, col in [("P90", P90_d, COLOR_TEAL),
                              ("P50", P50_d, COLOR_DARK),
                              ("P10", P10_d, COLOR_RED)]:
            pdf_fig.add_vline(
                x=val, line=dict(color=col, width=2, dash="dash"),
                annotation_text=f"{lab}={val:.2f}",
                annotation_position="top",
                annotation_font_color=col,
            )
        layout = get_plotly_layout("Permeability — Probability Density")
        layout["xaxis"].update(title=axis_title("k", "permeability", U))
        layout["yaxis"].update(title="Density")
        pdf_fig.update_layout(layout)
        st.plotly_chart(pdf_fig, use_container_width=True)

        # CDF
        cdf_fig = go.Figure()
        cdf_fig.add_trace(go.Scatter(
            x=x_cdf, y=y_cdf, mode="lines",
            name="Empirical CDF",
            line=dict(color=COLOR_DARK, width=3),
        ))
        for lab, val, col, p in [("P90", P90_d, COLOR_TEAL, 0.10),
                                 ("P50", P50_d, COLOR_DARK, 0.50),
                                 ("P10", P10_d, COLOR_RED,  0.90)]:
            cdf_fig.add_trace(go.Scatter(
                x=[val, val], y=[0, p], mode="lines",
                line=dict(color=col, width=1.5, dash="dash"),
                showlegend=False,
            ))
            cdf_fig.add_trace(go.Scatter(
                x=[val], y=[p], mode="markers+text",
                marker=dict(color=col, size=10, symbol="circle"),
                text=[f"  {lab}"], textposition="middle right",
                textfont=dict(color=col, size=11),
                showlegend=False,
            ))
        layout = get_plotly_layout("Permeability — Expectation Curve (CDF)")
        layout["xaxis"].update(title=axis_title("k", "permeability", U))
        layout["yaxis"].update(title="Cumulative probability", range=[0, 1])
        cdf_fig.update_layout(layout)
        st.plotly_chart(cdf_fig, use_container_width=True)

        # ====================================================================
        #  Sensitivity ranking — tornado chart
        #
        #  Reveals which uncertain input dominates the spread of permeability.
        #  Bars: median k restricted to the bottom-decile (low_k) vs top-decile
        #  (high_k) of each input. Annotations show the Spearman rank correlation.
        # ====================================================================
        from modules.uncertainty import tornado_swings
        # Inputs are in field units; convert to user system for axis display
        h_factor   = from_field(1.0, "height",    U)
        mu_factor  = from_field(1.0, "viscosity", U)
        torn = tornado_swings(mc, inputs={
            "Net pay h":   mc["h_samples"]  * h_factor,
            "Viscosity μ": mc["mu_samples"] * mu_factor,
        })

        if torn:
            st.markdown("##### Sensitivity ranking — which input drives the spread?")
            top_swing = max(t["swing"] for t in torn) or 1.0
            torn_fig = go.Figure()
            for t in torn:
                low_k_d  = from_field(t["low_k"],  "permeability", U)
                high_k_d = from_field(t["high_k"], "permeability", U)
                p50_d    = from_field(t["p50"],    "permeability", U)
                lo, hi = min(low_k_d, high_k_d), max(low_k_d, high_k_d)
                torn_fig.add_trace(go.Bar(
                    y=[t["name"]],
                    x=[hi - lo],
                    base=[lo],
                    orientation="h",
                    marker=dict(
                        color=COLOR_RED if t["rank_corr"] >= 0 else COLOR_TEAL,
                        line=dict(color=COLOR_DARK, width=1),
                    ),
                    name=t["name"],
                    showlegend=False,
                    text=[f"ρ={t['rank_corr']:+.2f}"],
                    textposition="outside",
                    textfont=dict(color=COLOR_DARK, size=11),
                ))
            # P50 reference line
            torn_fig.add_vline(
                x=from_field(torn[0]["p50"], "permeability", U),
                line=dict(color=COLOR_DARK, width=2, dash="dot"),
                annotation_text="P50",
                annotation_position="top",
                annotation_font_color=COLOR_DARK,
            )
            layout = get_plotly_layout("Tornado — k vs input decile")
            layout["xaxis"].update(title=axis_title("k", "permeability", U))
            layout["yaxis"].update(title="", autorange="reversed")
            torn_fig.update_layout(layout, height=240, margin=dict(l=120))
            st.plotly_chart(torn_fig, use_container_width=True)
            st.caption(
                "Bars span the median k for the bottom vs top decile of each "
                "input. Red = positive Spearman correlation with k, teal = negative. "
                "The longest bar is the dominant uncertainty driver."
            )

        # === Downloads — full sample vectors for external QC ===
        st.markdown("##### Downloads")
        mc_export = pd.DataFrame({
            f"k_{label('permeability', U)}": mc["samples"]   * k_factor,
            f"h_{label('height',       U)}": mc["h_samples"] * h_factor,
            f"mu_{label('viscosity',   U)}": mc["mu_samples"] * mu_factor,
        })
        st.download_button(
            "⬇  Download Monte Carlo samples (CSV)",
            data=mc_export.to_csv(index=False).encode("utf-8"),
            file_name=f"monte_carlo_samples_{U.lower()}.csv",
            mime="text/csv",
            key="dl_mc",
        )
    else:
        st.info("Configure the inputs and press **Run Monte Carlo**.")

    st.markdown("---")
    st.subheader("Compile PDF report")

    if st.button("Generate PDF report"):
        well_inputs = ss.get("design_snapshot", {}).get("well_inputs_display", {})
        econ = ss.get("design_snapshot", {}).get("econ", {})
        interp = ss.get("interpretation_result")

        # Build a system-aware MC summary for the report
        report_mc = None
        if mc:
            unit_str = label("permeability", U)
            report_mc = {
                "n_iter": mc["n_iter"],
                "P90":   from_field(mc["P90"],  "permeability", U),
                "P50":   from_field(mc["P50"],  "permeability", U),
                "P10":   from_field(mc["P10"],  "permeability", U),
                "mean":  from_field(mc["mean"], "permeability", U),
                "std":   from_field(mc["std"],  "permeability", U),
                "unit":  unit_str,
            }

        pdf_bytes = build_report(
            project_name=ss["project_name"],
            unit_system=U,
            well_inputs=well_inputs,
            economics=econ,
            interpretation=interp,
            monte_carlo=report_mc,
        )
        ss["pdf_bytes"] = pdf_bytes
        st.success("PDF generated. Use the button below to download.")

    if ss.get("pdf_bytes"):
        st.download_button(
            label="⬇  Download PTA Report (PDF)",
            data=ss["pdf_bytes"],
            file_name=f"PTA_report_{ss['project_name'].replace(' ', '_')}.pdf",
            mime="application/pdf",
        )


# =====================================================================
#  Footer
# =====================================================================
st.markdown(
    f"""
    <div style="margin-top:48px; padding-top:14px; border-top:1px solid #E5E5E5;
                color:#5A5A5A; font-size:11px; line-height:1.5;
                display:flex; justify-content:space-between; flex-wrap:wrap;">
        <div>
            <b style="color:#001E2B;">Aquila</b> v1.0
            &nbsp;·&nbsp; © 2026 Merouane Hamdani
            &nbsp;·&nbsp; <span style="color:#00707F;">MIT License</span>
        </div>
        <div style="text-align:right;">
            Active units: <b>{U}</b>
            &nbsp;·&nbsp; Not affiliated with Equinor ASA
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
