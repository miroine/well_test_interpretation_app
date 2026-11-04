"""Well Test Workbench - Streamlit front end.

Run:  streamlit run app.py
Numerics live in engine/ (no Streamlit imports); the UI lives in ui/.
"""
import datetime
import hashlib

import streamlit as st

from engine import project
from ui import session
from ui import (report_ui, tab_animation, tab_compare, tab_data, tab_deconv, tab_design, tab_fmb,
                tab_info, tab_interp, tab_ipr, tab_multirate, tab_simulator, tab_typecurve, theme,
                tab_advanced)

st.set_page_config(page_title="Well Test Workbench", page_icon=None, layout="wide")

theme.apply()

S = st.session_state
session.initialize(S)

with st.sidebar:
    st.markdown("### Well Test Workbench")
    st.caption("Pressure transient analysis")
    up = st.file_uploader("Open project", type=["json"], key="proj_up")
    if up is not None:
        content = up.getvalue()
        pid = hashlib.sha256(content).hexdigest()
        if S.get("_proj_id") != pid:
            try:
                saved_meta, restored = project.loads(content.decode("utf-8"))
                session.restore(S, saved_meta, restored)
            except (ValueError, KeyError, TypeError, UnicodeDecodeError) as e:
                st.error(f"Could not open the project: {e}")
            else:
                S["_proj_id"] = pid
                st.rerun()
    well = st.text_input("Well", "Well-A", key="meta_well")
    field = st.text_input("Field", "Field-X", key="meta_field")
    date = st.date_input("Test date", datetime.date.today(), key="meta_date")
    analyst = st.text_input("Analyst", "", key="meta_analyst")
    fluid = st.radio("Fluid", ["Oil", "Gas"], horizontal=True, key="meta_fluid")
    units = st.radio("Units", ["Field", "Metric"], horizontal=True,
                     key="meta_units",
                     help="Calculations always run in field units; inputs and results are converted.")
    st.caption("Field: psi, ft, stb/d, Mscf/d.  Metric: bar, m, m3/d, Sm3/d.")

meta = {"well": well, "field": field, "date": date, "analyst": analyst, "fluid": fluid, "units": units}

st.title("Well Test Workbench")
st.markdown("Plan with confidence. Interpret with evidence.")
st.caption(f"{well} | {field} | {fluid} | {units} units")

tabs = st.tabs(["00 Test design", "01 Data & QC", "02 Interpretation", "03 Type curve", "04 Compare",
                "Multi-rate", "05 Deconvolution", "06 Simulator", "07 IPR", "08 FMB", "09 Advanced",
                "Buildup animation", "Documentation & formulas", "How it works"], key="active_tab", on_change="rerun")
with tabs[0]:
    design_curves = tab_design.render(meta)
with tabs[1]:
    tab_data.render(meta)
with tabs[2]:
    tab_interp.render(meta)
with tabs[3]:
    tab_typecurve.render(meta)
with tabs[4]:
    tab_compare.render(meta)
with tabs[5]:
    tab_multirate.render(meta)
with tabs[6]:
    tab_deconv.render(meta)
with tabs[7]:
    tab_simulator.render(meta)
with tabs[8]:
    tab_ipr.render(meta)
with tabs[9]:
    tab_fmb.render(meta)
with tabs[10]:
    tab_advanced.render(meta)
with tabs[11]:
    if tabs[11].open:
        tab_animation.render(meta)
with tabs[12]:
    tab_info.documentation()
with tabs[13]:
    tab_info.how_it_works()

# Save and report controls sit last so they see this run's results.
with st.sidebar:
    st.markdown("---")
    st.markdown("**Project**")
    st.download_button("Save project", project.dumps(meta, S), file_name=f"{well or 'project'}.wtw.json",
                       mime="application/json")
    st.markdown("**Report**")
    scope = st.selectbox("Contents", report_ui.SCOPES)
    st.download_button("Export report (HTML, print to PDF)",
                       report_ui.build(meta, S, scope, design_curves=design_curves["buildup"]),
                       file_name=f"{well or 'report'}_report.html", mime="text/html")
