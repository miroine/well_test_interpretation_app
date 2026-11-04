import pathlib
import sys
import unittest
from unittest.mock import patch

import numpy as np
import streamlit as st
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from engine import project, models, multirate
from ui import helpers
from ui import multirate_timeline as T


def period_app():
    import numpy as np
    import streamlit as st
    from ui import session, tab_multirate
    from engine import models
    session.initialize(st.session_state)
    if not st.session_state["mr"]:
        st.session_state["mr"] = {"name": "Period test", "schedule": [(0.,100.),(10.,200.),(20.,0.)],
                                  "t": np.arange(1.,31.), "pwf": np.arange(4000.,4030.),
                                  "guess": models.WellTestParams().to_dict(), "fit": None}
    system = st.radio("Units", ["Field", "Metric"], key="test_mr_units")
    tab_multirate.render({"fluid":"Oil", "units":system})


class PeriodTimingTests(unittest.TestCase):
    def setUp(self):
        self.schedule = [(0., 100.), (10., 200.), (20., 0.)]

    def test_move_retains_rates_and_pressure_clears_derived_results(self):
        t, pressure = np.arange(1., 31.), np.arange(4000., 4030.)
        state = {"mr": {"schedule": self.schedule, "t": t, "pwf": pressure, "fit": {"old": True}},
                 "deconv": {"old": True}}
        T.apply_edit(state, {"base": self.schedule, "index": 1, "start": 12.5})
        self.assertEqual(state["mr"]["schedule"], [(0., 100.), (12.5, 200.), (20., 0.)])
        np.testing.assert_array_equal(state["mr"]["pwf"], pressure)
        self.assertIsNone(state["mr"]["fit"])
        self.assertIsNone(state["deconv"])
        _, saved = project.loads(project.dumps({}, state))
        self.assertEqual(saved["mr"]["schedule"][1][0], 12.5)

    def test_invalid_or_stale_edits_do_not_mutate_schedule(self):
        for index, start in ((0, 1.), (1, 0.), (1, 20.), (2, 30.), (2, np.nan), (True, 12.)):
            with self.subTest(index=index, start=start), self.assertRaises(ValueError):
                T.validate_edit({"base": self.schedule, "index": index, "start": start}, self.schedule, 30.)
        with self.assertRaisesRegex(ValueError, "changed"):
            T.validate_edit({"base": [(0., 100.)], "index": 1, "start": 12.}, self.schedule, 30.)
        self.assertEqual(self.schedule[1], (10., 200.))

    def test_final_start_stays_inside_record_and_repeated_stale_event_is_rejected(self):
        edit = {"base": self.schedule, "index": 2, "start": 25.}
        updated = T.validate_edit(edit, self.schedule, 30.)
        self.assertEqual(updated[-1], (25., 0.))
        with self.assertRaisesRegex(ValueError, "changed"):
            T.validate_edit(edit, updated, 30.)

    def test_real_render_after_edit_updates_schedule_table_and_metric_payload(self):
        payloads = []
        original = T.component
        def capture(*args, **kwargs):
            widget = original(*args, **kwargs)
            def render(**options):
                payloads.append(options["data"])
                return widget(**options)
            return render
        with patch.object(T, "component", side_effect=capture):
            app = AppTest.from_function(period_app, default_timeout=120).run()
            self.assertFalse(app.exception)
            T.apply_edit(app.session_state, {"base": self.schedule, "index": 1, "start": 12.5})
            app.radio(key="test_mr_units").set_value("Metric")
            app.run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(payloads[-1]["schedule"][1][0], 12.5)
            self.assertAlmostEqual(payloads[-1]["rates"][1], 200 * .158987295)
            self.assertEqual(payloads[-1]["pressure_unit"], "bar")
            self.assertIn(12.5, app.dataframe[0].value["Start [hr]"].tolist())

    def test_selecting_periods_overlays_pressure_and_derivative(self):
        import json
        app = AppTest.from_function(period_app, default_timeout=120).run()
        self.assertFalse(app.exception)
        def chart():
            return next(json.loads(c.proto.spec) for c in app.get("plotly_chart")
                        if json.loads(c.proto.spec)["layout"]["title"]["text"] == "Multi-rate period diagnostics")
        self.assertEqual(len(chart()["data"]),6)
        app.multiselect(key="mr_overlay_0").set_value([1,2])
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual([t["name"] for t in chart()["data"]],
                         ["P2 data dp","P2 data derivative","P3 data dp","P3 data derivative"])
        app.checkbox(key="mr_normalize_0").check()
        app.radio(key="test_mr_units").set_value("Metric")
        app.run()
        self.assertFalse(app.exception)
        self.assertIn("bar per",chart()["layout"]["yaxis"]["title"]["text"])
        app.radio(key="mr_diag_view_0").set_value("Rate-superposition diagnostic")
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(chart()["data"]),4)

    def test_boundary_and_model_fit_parameters_follow_selection(self):
        app = AppTest.from_function(period_app, default_timeout=120).run()
        for boundary, names in (("circle", ["re"]), ("circle_cp", ["re"]),
                                ("channel", ["d1", "d2"]), ("rectangle", ["xe", "ye"])):
            app.selectbox(key="mr_boundary_0").select(boundary)
            app.run()
            self.assertFalse(app.exception)
            selector = app.multiselect(key=f"mr_free_0_homogeneous_{boundary}")
            self.assertTrue(set(names) <= set(selector.value))
        app.selectbox(key="mr_model_0").select("fracture_fc")
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue({"xf", "FcD", "xe", "ye"} <=
                        set(app.multiselect(key="mr_free_0_fracture_fc_rectangle").value))
        for model in helpers.MODEL_LABELS:
            for boundary in helpers.BOUNDARY_LABELS:
                from engine import analysis
                specs = helpers.BASE_SPEC + helpers.MODEL_SPEC.get(model, []) + helpers.BOUNDARY_SPEC.get(boundary, [])
                options = {s[0] for s in specs if s[0] in analysis.FITTABLE}
                self.assertLessEqual(set(helpers.fit_free_params(model, boundary)), options)

    def test_closed_circle_radius_reaches_regression_and_metric_results(self):
        app = AppTest.from_function(period_app, default_timeout=120).run()
        app.selectbox(key="mr_boundary_0").select("circle")
        app.run()
        app.multiselect(key="mr_free_0_homogeneous_circle").set_value(["re"])
        def fit(p, schedule, t, pressure, free, *, progress=None):
            self.assertEqual(p.boundary, "circle")
            self.assertEqual(free, ["re"])
            return models.WellTestParams(**{**p.to_dict(), "re": 500.}), {"rms": .1, "nfev": 3}
        with patch.object(multirate, "fit_multirate", side_effect=fit):
            next(b for b in app.button if b.label == "Run multi-rate regression").click()
            app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["mr"]["fit"]["params"]["re"], 500.)
        self.assertTrue(any("Model fitted to the multi-rate record" in s.value for s in app.success))
        app.radio(key="test_mr_units").set_value("Metric")
        app.run()
        self.assertFalse(app.exception)
        results = app.dataframe[-1].value
        self.assertEqual(results.iloc[0]["Parameter"], "Reservoir radius")
        self.assertEqual(results.iloc[0]["Unit"], "m")
        self.assertAlmostEqual(float(results.iloc[0]["Fitted"]), 152.4)


if __name__ == "__main__":
    unittest.main()
