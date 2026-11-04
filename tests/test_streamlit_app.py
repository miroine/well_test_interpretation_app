"""Real Streamlit integration tests; no stand-in UI objects."""
import logging
import base64
import csv
import io
import json
import pathlib
import sys
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

import streamlit as st
import numpy as np
from streamlit.testing.v1 import AppTest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine import models, project, viz, units
from ui import tab_design, tab_animation


def qc_test_app():
    import numpy as np
    import streamlit as st
    from ui import session, tab_data
    from engine import models
    session.initialize(st.session_state)
    if st.session_state["dataset"] is None:
        st.session_state["dataset"] = {
            "name": "Full response", "fluid": "Oil", "tp": 2., "q_user": 100.,
            "dt": np.arange(1., 7.), "pws": np.arange(101., 107.), "pwf_shutin": 100.,
            "guess": models.WellTestParams().to_dict(), "truth": None, "gas": None,
            "history": {"t": np.array([0., 1., 2., 2., 3., 4., 5., 6., 7., 8.]),
                        "p": np.array([110., 105., 100., 100., 101., 102., 103., 104., 105., 106.]),
                        "rate": np.array([100., 100., 100., 0., 0., 0., 0., 0., 0., 0.])}}
    system = st.radio("Units", ["Field", "Metric"], key="qc_test_units")
    tab_data.render({"units": system, "fluid": "Oil"})


def interpretation_test_app():
    import streamlit as st
    import numpy as np
    from ui import session, tab_interp
    from engine import models, viz
    session.initialize(st.session_state)
    if st.session_state["dataset"] is None:
        p = models.WellTestParams(k=80., S=2., C=0.01)
        d = models.synthetic_buildup(p, 1000., 48., 24., n=60)
        curves = viz.designed_test(p, 1000., 48., 24., n=60)
        st.session_state["dataset"] = {
            "name": "Interpretation test", "fluid": "Oil", "tp": 48., "q_user": 1000.,
            "dt": d["dt"], "pws": d["pws"], "pwf_shutin": d["pwf_at_shutin"],
            "guess": models.WellTestParams(k=40., S=0., C=0.01).to_dict(), "gas": None,
            "history": {"t": curves["timeline"]["t"], "p": curves["timeline"]["pressure"],
                        "rate": np.where(curves["timeline"]["t"] <= 48., 1000., 0.)}}
    system = st.radio("Units", ["Field", "Metric"], key="interp_test_units")
    tab_interp.render({"units": system, "fluid": "Oil"})


def interpretation_gas_test_app():
    import streamlit as st
    from ui import session, tab_interp
    from engine import samples
    session.initialize(st.session_state)
    if st.session_state["dataset"] is None:
        p, q, tp, d, meta = samples.gas_buildup()
        st.session_state["dataset"] = {
            "name": "Gas interpretation", "fluid": "Gas", "tp": tp, "q_user": meta["q_mscfd"],
            "dt": d["dt"], "pws": d["pws"], "pwf_shutin": d["pwf_at_shutin"],
            "guess": p.to_dict(), "gas": {"T_F": meta["T_F"], "gamma_g": meta["gamma_g"], "cf": 4e-6}}
    system = st.radio("Units", ["Field", "Metric"], key="interp_test_units")
    tab_interp.render({"units": system, "fluid": "Gas"})


class StreamlitJourneys(unittest.TestCase):
    def app(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        return app

    def check_run(self, app):
        app.run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertFalse(app.error, [e.value for e in app.error])

    def chart(self, app, title):
        return next(json.loads(c.proto.spec) for c in app.get("plotly_chart")
                    if json.loads(c.proto.spec)["layout"]["title"]["text"] == title)

    def test_generate_model_tracks_parameter_edits_and_pressure_rate(self):
        app = AppTest.from_function(interpretation_test_app, default_timeout=120).run()
        self.assertFalse(app.exception)
        diag = self.chart(app, "Diagnostic plot (buildup)")
        self.assertEqual(len(diag["data"]), 2)
        app.button(key="interp_generate_0").click()
        self.check_run(app)
        diag = self.chart(app, "Diagnostic plot (buildup)")
        model_dp = next(t for t in diag["data"] if t["name"] == "Working model dp")["y"]
        model_der = next(t for t in diag["data"] if t["name"] == "Working model derivative")["y"]
        data_dp = diag["data"][0]["y"]
        pressure = self.chart(app, "Pressure and rate: data and model")
        model_p = next(t for t in pressure["data"] if t["name"] == "Working model pressure")["y"]
        self.assertTrue(any(t["name"] == "Recorded surface rate" for t in pressure["data"]))
        app.number_input(key="interp_k_Field_0").set_value(100.)
        self.check_run(app)
        changed = self.chart(app, "Diagnostic plot (buildup)")
        self.assertNotEqual(model_dp, next(t for t in changed["data"] if t["name"] == "Working model dp")["y"])
        self.assertNotEqual(model_der, next(t for t in changed["data"] if t["name"] == "Working model derivative")["y"])
        self.assertEqual(data_dp, changed["data"][0]["y"])
        pressure = self.chart(app, "Pressure and rate: data and model")
        self.assertNotEqual(model_p, next(t for t in pressure["data"] if t["name"] == "Working model pressure")["y"])
        app.radio(key="interp_test_units").set_value("Metric")
        self.check_run(app)
        self.assertAlmostEqual(app.number_input(key="interp_k_Metric_0").value, 100.)
        app.number_input(key="interp_S_Metric_0").set_value(4.)
        self.check_run(app)
        app.radio(key="interp_test_units").set_value("Field")
        self.check_run(app)
        self.assertAlmostEqual(app.number_input(key="interp_S_Field_0").value, 4.)
        _, state = project.loads(project.dumps({}, app.session_state.filtered_state))
        self.assertEqual(state["interp_model"]["params"]["S"], 4.)

    def test_fit_adopts_results_and_keeps_snapshot_when_working_model_changes(self):
        app = AppTest.from_function(interpretation_test_app, default_timeout=120).run()
        app.checkbox(key="interp_ms_0").uncheck()
        app.multiselect(key="interp_free_0_homogeneous_none").set_value(["k", "S"])
        app.button(key="interp_fit_0").click()
        self.check_run(app)
        fit = app.session_state["last_fit"]
        if fit["info"]["status"] > 0:
            self.assertTrue(any("Model fitted to data" in success.value for success in app.success))
        else:
            self.assertTrue(any("before convergence" in warning.value for warning in app.warning))
            self.assertFalse(any("Model fitted to data" in success.value for success in app.success))
        self.assertAlmostEqual(app.number_input(key="interp_k_Field_0").value, fit["params"]["k"])
        self.assertAlmostEqual(app.number_input(key="interp_S_Field_0").value, fit["params"]["S"])
        diag = self.chart(app, "Diagnostic plot (buildup)")
        fitted_dp = next(t for t in diag["data"] if t["name"] == "Last fit dp")["y"]
        self.assertEqual(fitted_dp, next(t for t in diag["data"] if t["name"] == "Working model dp")["y"])
        app.number_input(key="interp_k_Field_0").set_value(fit["params"]["k"] * 1.5)
        self.check_run(app)
        diag = self.chart(app, "Diagnostic plot (buildup)")
        self.assertEqual(fitted_dp, next(t for t in diag["data"] if t["name"] == "Last fit dp")["y"])
        self.assertNotEqual(fitted_dp, next(t for t in diag["data"] if t["name"] == "Working model dp")["y"])
        self.assertEqual(app.session_state["last_fit"]["start_params"]["k"], 40.)

    def test_fit_calculation_panel_and_converged_success_label(self):
        from engine import analysis
        from ui import calculation
        app = AppTest.from_function(interpretation_test_app, default_timeout=120).run()
        def converged(p, *args, progress=None, **kwargs):
            progress(0, 2, "Fitting attempt 1 of 2")
            progress(1, 2, "Completed attempt 1")
            progress(2, 2, "Completed attempt 2")
            return p, {"status": 1, "rms": .1, "nfev": 4}
        original_progress = calculation.st.progress
        bars = []
        def capture(*args, **kwargs):
            bars.append((args, kwargs))
            return original_progress(*args, **kwargs)
        with patch.object(analysis, "fit_buildup", side_effect=converged), \
                patch.object(calculation.st, "progress", side_effect=capture):
            app.button(key="interp_fit_0").click()
            self.check_run(app)
        self.assertTrue(bars)
        self.assertEqual(bars[0][0][0], 0.)
        self.assertTrue(any("Model fitted to data" in success.value for success in app.success))

    def test_generated_model_uses_selected_model_and_boundary(self):
        app = AppTest.from_function(interpretation_test_app, default_timeout=120).run()
        app.button(key="interp_generate_0").click()
        self.check_run(app)
        initial = self.chart(app, "Diagnostic plot (buildup)")
        y = next(t for t in initial["data"] if t["name"] == "Working model derivative")["y"]
        app.selectbox(key="interp_model_0").select("fracture_fc")
        self.check_run(app)
        fracture = self.chart(app, "Diagnostic plot (buildup)")
        yf = next(t for t in fracture["data"] if t["name"] == "Working model derivative")["y"]
        self.assertNotEqual(y, yf)
        app.selectbox(key="interp_boundary_0").select("fault")
        self.check_run(app)
        app.number_input(key="interp_d1_Field_0").set_value(50.)
        self.check_run(app)
        fault = self.chart(app, "Diagnostic plot (buildup)")
        self.assertNotEqual(yf, next(t for t in fault["data"] if t["name"] == "Working model derivative")["y"])

    def test_monte_carlo_bands_overlay_both_plots_and_survive_working_edits(self):
        from engine import analysis
        samples = {"k": [40., 60., 80., 100., 120.], "S": [4., 3., 2., 1., 0.]}
        mc = {"samples": samples, "summary": {
            n: {"p10": float(np.percentile(v, 10)), "p50": float(np.percentile(v, 50)),
                "p90": float(np.percentile(v, 90)), "n": 5} for n, v in samples.items()}, "sigma": .1}
        app = AppTest.from_function(interpretation_test_app, default_timeout=120).run()
        app.checkbox(key="interp_ms_0").uncheck()
        app.multiselect(key="interp_free_0_homogeneous_none").set_value(["k", "S"])
        app.button(key="interp_fit_0").click()
        self.check_run(app)
        with patch.object(analysis, "monte_carlo", return_value=mc):
            app.button(key="interp_mcbtn_0").click()
            self.check_run(app)
        diag = self.chart(app, "Diagnostic plot (buildup)")
        names = [t["name"] for t in diag["data"]]
        self.assertIn("MC derivative P10-P90", names)
        self.assertIn("MC dp P10-P90", names)
        pressure = self.chart(app, "Pressure and rate: data and model")
        band = next(t for t in pressure["data"] if t["name"] == "MC pressure P10-P90")
        self.assertEqual(band["fill"], "tonexty")
        original = band["y"]
        app.number_input(key="interp_k_Field_0").set_value(150.)
        self.check_run(app)
        pressure = self.chart(app, "Pressure and rate: data and model")
        self.assertEqual(original, next(t for t in pressure["data"] if t["name"] == "MC pressure P10-P90")["y"])
        app.radio(key="interp_test_units").set_value("Metric")
        self.check_run(app)
        pressure = self.chart(app, "Pressure and rate: data and model")
        metric_band = next(t for t in pressure["data"] if t["name"] == "MC pressure P10-P90")
        values = np.frombuffer(base64.b64decode(original["bdata"]), dtype=original["dtype"])
        converted = np.frombuffer(base64.b64decode(metric_band["y"]["bdata"]), dtype=metric_band["y"]["dtype"])
        np.testing.assert_allclose(converted, values * units.PSI_BAR)
        app.checkbox(key="interp_mc_band_0").uncheck()
        self.check_run(app)
        self.assertFalse(any(t["name"].startswith("MC ") for t in self.chart(app, "Diagnostic plot (buildup)")["data"]))
        app.button(key="interp_fit_0").click()
        self.check_run(app)
        self.assertIsNone(app.session_state["mc"])

    def test_generated_gas_model_uses_surface_rates_and_metric_pressure(self):
        app = AppTest.from_function(interpretation_gas_test_app, default_timeout=120).run()
        app.button(key="interp_generate_0").click()
        self.check_run(app)
        eng = app.session_state["interp_eng"]
        app.radio(key="interp_test_units").set_value("Metric")
        self.check_run(app)
        chart = self.chart(app, "Pressure and rate: data and model")
        self.assertIn("pseudo-pressure", chart["layout"]["yaxis"]["title"]["text"])
        self.assertIn("bar", chart["layout"]["yaxis"]["title"]["text"])
        rate = next(t for t in chart["data"] if t["name"] == "Working model rate control (not fitted)")
        values = np.frombuffer(base64.b64decode(rate["y"]["bdata"]), dtype=rate["y"]["dtype"])
        np.testing.assert_allclose(values, [8000 * 28.316847, 8000 * 28.316847, 0, 0])
        self.assertAlmostEqual(app.session_state["interp_eng"]["q"], eng["q"])
        app.number_input(key="interp_gasg_0").set_value(0.75)
        self.check_run(app)
        self.assertNotEqual(app.session_state["interp_eng"]["q"], eng["q"])

    def test_full_response_manual_qc_restore_and_units(self):
        app = AppTest.from_function(qc_test_app).run()
        self.assertFalse(app.exception)
        spec = next(json.loads(c.proto.spec) for c in app.get("plotly_chart")
                    if json.loads(c.proto.spec)["layout"]["title"]["text"] == "Full pressure and rate response")
        self.assertIn("customdata", spec["data"][0])
        self.assertEqual(spec["layout"]["yaxis2"]["title"]["text"], f"Rate [{units.label('rate', 'Field')}]")
        app.multiselect(key="qc_manual_0").set_value([1, 5])
        self.check_run(app)
        app.session_state["last_fit"] = {"old": True}
        next(b for b in app.button if b.label == "Apply QC to the active dataset").click()
        self.check_run(app)
        self.assertEqual(list(app.session_state["dataset"]["dt"]), [1, 3, 4, 5, 6])
        self.assertIsNone(app.session_state["last_fit"])
        app.radio(key="qc_test_units").set_value("Metric")
        self.check_run(app)
        spec = next(json.loads(c.proto.spec) for c in app.get("plotly_chart")
                    if json.loads(c.proto.spec)["layout"]["title"]["text"] == "Full pressure and rate response")
        self.assertIn("bar", spec["layout"]["yaxis"]["title"]["text"])
        self.assertIn(units.label("rate", "Metric"), spec["layout"]["yaxis2"]["title"]["text"])
        next(b for b in app.button if b.label == "Restore original response (undo all QC)").click()
        self.check_run(app)
        self.assertEqual(list(app.session_state["dataset"]["dt"]), [1, 2, 3, 4, 5, 6])
        self.assertNotIn("qc_source", app.session_state["dataset"])

    def test_plot_selection_maps_pressure_and_rate_to_original_point_ids(self):
        original_chart = st.plotly_chart
        def chart_event(fig, *args, **kwargs):
            result = original_chart(fig, *args, **kwargs)
            if str(kwargs.get("key", "")).startswith("qc_response_"):
                return {"selection": {"points": [{"curve_number": 0, "customdata": 5},
                                                  {"curve_number": 3, "customdata": 5},
                                                  {"curve_number": 3, "customdata": 1}]}}
            return result
        with patch.object(st, "plotly_chart", side_effect=chart_event):
            app = AppTest.from_function(qc_test_app).run()
            app.button(key="qc_mark_0").click()
            self.check_run(app)
            self.assertEqual(app.multiselect(key="qc_manual_0").value, [1, 5])
            next(b for b in app.button if b.label == "Apply QC to the active dataset").click()
            self.check_run(app)
            self.assertEqual(list(app.session_state["dataset"]["dt"]), [1, 3, 4, 5, 6])

    def test_qc_rejects_too_few_points_without_losing_data(self):
        app = AppTest.from_function(qc_test_app).run()
        app.multiselect(key="qc_manual_0").set_value(list(range(10)))
        self.check_run(app)
        next(b for b in app.button if b.label == "Apply QC to the active dataset").click()
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any("At least three" in e.value for e in app.error))
        self.assertEqual(len(app.session_state["dataset"]["dt"]), 6)
        self.assertEqual(app.session_state["ver"], 0)

    def test_metric_full_history_csv_upload_preserves_flow_and_shut_in(self):
        def upload(*args, **kwargs):
            file = io.BytesIO(b"time,pressure,rate\n0,210,100\n1,200,100\n2,190,100\n2,190,0\n3,191,0\n4,192,0\n5,193,0\n6,194,0\n")
            file.name = "full_history.csv"
            return file
        with patch.object(st, "file_uploader", side_effect=upload):
            app = AppTest.from_function(qc_test_app).run()
            app.radio(key="qc_test_units").set_value("Metric")
            app.radio(key="data_src_0").set_value("Upload files")
            self.check_run(app)
            app.checkbox(key="qc_upload_full").check()
            self.check_run(app)
            next(n for n in app.number_input if n.label == "Flow period before shut-in [hr]").set_value(2.)
            next(b for b in app.button if b.label == "Use this file").click()
            self.check_run(app)
            ds = app.session_state["dataset"]
            self.assertEqual(list(ds["history"]["t"]), [0, 1, 2, 2, 3, 4, 5, 6])
            self.assertEqual(list(ds["dt"]), [1, 2, 3, 4])
            self.assertAlmostEqual(ds["history"]["p"][0], 210 * 14.5037738)
            self.assertAlmostEqual(ds["history"]["rate"][0], 100 / 0.158987295)
            self.assertAlmostEqual(ds["q_user"], 100 / 0.158987295)
            self.assertAlmostEqual(ds["pwf_shutin"], 190 * 14.5037738)

    def test_design_timing_overlay_generation_and_save(self):
        app = self.app()
        app.number_input(key="design_tp_0").set_value(72.0)
        app.number_input(key="design_dtend_0").set_value(36.0)
        self.check_run(app)
        app.checkbox(key="design_overlay_Drawdown_pressure change").uncheck()
        app.checkbox(key="design_overlay_Buildup_pressure change").uncheck()
        app.radio(key="design_alignment").set_value("Elapsed time")
        self.check_run(app)
        charts = [json.loads(c.proto.spec) for c in app.get("plotly_chart")]
        overlay = next(c for c in charts if c["layout"]["title"]["text"] == "Drawdown and buildup diagnostics")
        self.assertEqual([t["name"] for t in overlay["data"]], ["Drawdown derivative", "Buildup derivative"])
        self.assertIn("Elapsed time within", overlay["layout"]["xaxis"]["title"]["text"])
        app.radio(key="meta_units").set_value("Metric")
        self.check_run(app)
        design = app.session_state["design_eng"]
        self.assertEqual((design["tp"], design["dt_end"]), (72.0, 36.0))
        meta = {"well": "W", "fluid": "Oil", "units": "Metric"}
        _, saved = project.loads(project.dumps(meta, app.session_state.filtered_state))
        self.assertEqual(saved["design_eng"]["tp"], 72.0)
        self.assertEqual(saved["design_eng"]["dt_end"], 36.0)
        next(b for b in app.button if b.label == "Generate synthetic test from this design").click()
        self.check_run(app)
        dataset = app.session_state["dataset"]
        self.assertEqual(dataset["tp"], 72.0)
        self.assertAlmostEqual(max(dataset["dt"]), 36.0)
        self.assertEqual(app.session_state["design_eng"]["tp"], 72.0)

    def test_design_zero_rate_and_all_curves_hidden(self):
        app = self.app()
        for phase in ("Drawdown", "Buildup"):
            for quantity in ("pressure change", "derivative"):
                app.checkbox(key=f"design_overlay_{phase}_{quantity}").uncheck()
        self.check_run(app)
        self.assertTrue(any("Select at least one curve" in i.value for i in app.info))
        app.number_input(key="design_q_Oil_Field_0").set_value(0.0)
        self.check_run(app)
        self.assertEqual(app.session_state["design_eng"]["q"], 0.0)

    def test_design_exports_follow_timing_units_and_surface_gas_rate(self):
        downloads = {}
        original_download = st.download_button
        def capture(label, data, *args, **kwargs):
            downloads[kwargs.get("key")] = data
            return original_download(label, data, *args, **kwargs)
        with patch.object(st, "download_button", side_effect=capture):
            app = self.app()
            app.number_input(key="design_tp_0").set_value(72.0)
            app.number_input(key="design_dtend_0").set_value(36.0)
            app.text_input(key="design_eclipse_well").set_value("TEST-1")
            self.check_run(app)
            self.assertIn("'TEST-1' 'OPEN' 'ORAT' 1500", downloads["design_eclipse_export"])
            self.assertIn("TSTEP\n  3 /", downloads["design_eclipse_export"])
            self.assertIn("TSTEP\n  1.5 /", downloads["design_eclipse_export"])
            app.radio(key="meta_fluid").set_value("Gas")
            app.radio(key="meta_units").set_value("Metric")
            self.check_run(app)
            design = app.session_state["design_eng"]
            rows = list(csv.DictReader(io.StringIO(downloads["design_csv_export"])))
            self.assertIn("normalized_pseudo_pressure_bar", rows[0])
            self.assertAlmostEqual(float(rows[0]["surface_rate_Sm3_per_day"]),
                                   design["q_user"] * 28.316847)
            self.assertEqual(float(rows[-1]["time_hr"]), 108.0)
            self.assertIn("'GRAT' 2*", downloads["design_eclipse_export"])
            app.number_input(key="design_eclipse_bhp_Metric").set_value(10.0)
            self.check_run(app)
            self.assertIn("2* 10 /", downloads["design_eclipse_export"])
            app.text_input(key="design_eclipse_well").set_value("INVALID'WELL")
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Eclipse well name" in e.value for e in app.error))
            self.assertFalse(any(d.key == "design_eclipse_export" for d in app.get("download_button")))

    def test_full_deck_build_acknowledgement_and_stale_export_guard(self):
        downloads = {}
        original_download = st.download_button
        def capture(label, data, *args, **kwargs):
            downloads[kwargs.get("key")] = data
            return original_download(label, data, *args, **kwargs)
        with patch.object(st, "download_button", side_effect=capture):
            app = self.app()
            app.button(key="design_deck_build").click()
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(any("accept the numerical approximations" in e.value for e in app.error))
            self.assertFalse(any(d.key == "design_deck_data" for d in app.get("download_button")))
            app.selectbox(key="design_model_0").select("mfhw")
            self.check_run(app)
            app.checkbox(key="design_deck_accept_mfhw_none").check()
            app.number_input(key="design_deck_nx").set_value(9)
            app.number_input(key="design_deck_ny").set_value(9)
            app.button(key="design_deck_build").click()
            self.check_run(app)
            result = app.session_state["_design_eclipse_deck"]
            self.assertEqual(result["metadata"]["source_design"]["model"], "mfhw")
            self.assertEqual(result["metadata"]["completion_count"], 4)
            self.assertIn("\nFIELD\n", downloads["design_deck_data"])
            with zipfile.ZipFile(io.BytesIO(downloads["design_deck_zip"])) as z:
                self.assertIn("PROD.DATA", z.namelist())
                reference = z.read("predicted_bhp.csv").decode()
                self.assertIn("synthetic_bhp_psi", reference)
            with zipfile.ZipFile(io.BytesIO(downloads["design_eclipse_comparison"])) as z:
                self.assertIn("WBHP", z.read("summary.INC").decode())
            app.number_input(key="design_tp_0").set_value(72.0)
            self.check_run(app)
            self.assertFalse(any(d.key == "design_deck_data" for d in app.get("download_button")))
            self.assertTrue(any("Build again" in i.value for i in app.info))
            app.button(key="design_deck_build").click()
            self.check_run(app)
            self.assertEqual(app.session_state["_design_eclipse_deck"]["metadata"]["flow_hr"], 72)
            app.number_input(key="design_deck_half_extent_ft_Field").set_value(6000.0)
            app.number_input(key="design_deck_fracture_width_ft_Field").set_value(0.03)
            self.check_run(app)
            app.radio(key="meta_units").set_value("Metric")
            self.check_run(app)
            self.assertAlmostEqual(app.number_input(key="design_deck_half_extent_ft_Metric").value, 6000 * 0.3048)
            self.assertAlmostEqual(app.number_input(key="design_deck_fracture_width_ft_Metric").value, 0.03 * 0.3048)
            self.assertIn("[kg/m3]", app.number_input(key="design_deck_oil_density_lbft3_Metric").label)
            self.assertFalse(any(d.key == "design_deck_data" for d in app.get("download_button")))
            app.number_input(key="design_deck_top_depth_ft_Metric").set_value(2500.0)
            app.button(key="design_deck_build").click()
            self.check_run(app)
            self.assertIn("\nMETRIC\n", downloads["design_deck_data"])
            self.assertNotIn("\nFIELD\n", downloads["design_deck_data"])
            with zipfile.ZipFile(io.BytesIO(downloads["design_deck_zip"])) as z:
                self.assertIn("synthetic_bhp_bar", z.read("predicted_bhp.csv").decode())
            app.radio(key="meta_units").set_value("Field")
            self.check_run(app)
            self.assertAlmostEqual(app.number_input(key="design_deck_top_depth_ft_Field").value, 2500 / 0.3048)
            self.assertAlmostEqual(app.number_input(key="design_deck_half_extent_ft_Field").value, 6000)
            self.assertFalse(any(d.key == "design_deck_data" for d in app.get("download_button")))

    def test_multifracture_design_avoids_shared_cache_and_defers_animation(self):
        app = self.app()
        app.selectbox(key="design_model_0").select("mfhw")
        with patch.object(viz, "model_curves", wraps=viz.model_curves) as curves, \
             patch.object(viz, "animation_payload", wraps=viz.animation_payload) as animation:
            self.check_run(app)
            self.assertEqual(curves.call_count, 1)
            animation.assert_not_called()
            app.checkbox(key="design_overlay_Drawdown_pressure change").uncheck()
            self.check_run(app)
            self.assertEqual(curves.call_count, 2)
            animation.assert_not_called()
            app.number_input(key="design_tp_0").set_value(72.0)
            self.check_run(app)
            self.assertEqual(curves.call_count, 3)
            app.session_state["active_tab"] = "Buildup animation"
            self.check_run(app)
            self.assertEqual(animation.call_count, 1)
            self.assertEqual(animation.call_args.args[2], 72.0)
            self.assertEqual(curves.call_count, 4)
            app.session_state["active_tab"] = "00 Test design"
            self.check_run(app)
            self.assertEqual(animation.call_count, 1)

    def test_oil_sample_metric_and_fluid_guard(self):
        app = self.app()
        next(b for b in app.button if b.label == "Load sample").click()
        self.check_run(app)
        self.assertEqual(app.session_state["dataset"]["fluid"], "Oil")
        next(b for b in app.button if b.label == "Auto-match").click()
        self.check_run(app)
        self.assertIsNotNone(app.session_state["tc_match"])
        app.radio(key="meta_units").set_value("Metric")
        self.check_run(app)
        self.assertIsNotNone(app.session_state["interp_eng"])
        app.radio(key="meta_fluid").set_value("Gas")
        self.check_run(app)
        self.assertIsNone(app.session_state["interp_eng"])
        self.assertTrue(any("Select Oil" in w.value for w in app.warning))

    def test_new_model_and_boundary_defaults_are_valid(self):
        app = self.app()
        next(b for b in app.button if b.label == "Load sample").click()
        self.check_run(app)
        ver = app.session_state["ver"]
        app.selectbox(key=f"interp_model_{ver}").select("composite3")
        self.check_run(app)
        self.assertIn("M3", app.multiselect(key=f"interp_free_{ver}_composite3_none").value)
        app.selectbox(key=f"interp_model_{ver}").select("homogeneous")
        app.selectbox(key=f"interp_boundary_{ver}").select("circle")
        self.check_run(app)
        self.assertIn("re", app.multiselect(key=f"interp_free_{ver}_homogeneous_circle").value)

    def test_sensitivity_converts_units_and_new_fit_clears_uncertainty(self):
        app = self.app()
        next(b for b in app.button if b.label == "Load sample").click()
        self.check_run(app)
        next(b for b in app.button if b.label == "Fit to data").click()
        self.check_run(app)
        next(b for b in app.button if b.label == "Show sensitivity").click()
        self.check_run(app)
        self.assertIsNotNone(app.session_state["sens"])
        app.radio(key="meta_units").set_value("Metric")
        self.check_run(app)
        charts = [json.loads(c.proto.spec) for c in app.get("plotly_chart")]
        sensitivity = next(c for c in charts if c["layout"]["title"]["text"] == "Sensitivity of the model response")
        self.assertIn("bar", sensitivity["layout"]["yaxis"]["title"]["text"])
        next(b for b in app.button if b.label == "Fit to data").click()
        self.check_run(app)
        self.assertIsNone(app.session_state["sens"])
        self.assertIsNone(app.session_state["mc"])

    def test_project_restoration_survives_real_widget_rerun(self):
        app = self.app()
        p = models.WellTestParams().to_dict()
        payload = project.dumps(
            {"well": "Restored well", "field": "North", "date": "2026-11-04",
             "analyst": "Analyst", "fluid": "Gas", "units": "Metric"},
            {"fits": [], "design_params": p,
             "design_eng": {"params": p, "q_user": 4321.0, "q": 4321.0, "tp": 72.0,
                            "dt_end": 36.0, "gas": {"T_F": 180.0, "gamma_g": 0.72, "cf": 5e-6}}},
        )
        original_uploader = st.file_uploader
        upload = SimpleNamespace(getvalue=lambda: payload.encode("utf-8"))
        def uploader(*args, **kwargs):
            return upload if kwargs.get("key") == "proj_up" else original_uploader(*args, **kwargs)
        with patch.object(st, "file_uploader", side_effect=uploader):
            self.check_run(app)
        self.assertEqual(app.text_input(key="meta_well").value, "Restored well")
        self.assertEqual(app.radio(key="meta_units").value, "Metric")
        design = app.session_state["design_eng"]
        self.assertAlmostEqual(design["q_user"], 4321.0)
        self.assertEqual(design["tp"], 72.0)
        self.assertEqual(design["dt_end"], 36.0)
        self.assertAlmostEqual(design["gas"]["T_F"], 180.0)
        self.assertAlmostEqual(design["gas"]["gamma_g"], 0.72)
        meta, _ = project.loads(payload)
        _, back = project.loads(project.dumps(meta, app.session_state.filtered_state))
        self.assertEqual(back["design_eng"]["tp"], 72.0)


if __name__ == "__main__":
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    unittest.main()
