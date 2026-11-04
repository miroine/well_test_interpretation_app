import pathlib
import sys
import unittest

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def radius_app():
    import streamlit as st
    from ui.inputs import radius_input
    system = st.radio("Units", ["Field", "Metric"], key="units")
    st.session_state["radius_ft"] = radius_input("test", .354, system, 0)


class RadiusInputTests(unittest.TestCase):
    def test_inches_and_project_unit_switches_preserve_radius(self):
        app = AppTest.from_function(radius_app).run()
        self.assertFalse(app.exception)
        app.number_input(key="test_rw_Field_0").set_value(.5).run()
        app.selectbox(key="test_rw_unit_Field_0").select("in").run()
        self.assertAlmostEqual(app.number_input(key="test_rw_in_0").value, 6.)
        app.number_input(key="test_rw_in_0").set_value(4.25).run()
        self.assertAlmostEqual(app.session_state["radius_ft"], 4.25 / 12.)
        app.selectbox(key="test_rw_unit_Field_0").select("ft").run()
        self.assertAlmostEqual(app.number_input(key="test_rw_Field_0").value, 4.25 / 12.)
        app.radio(key="units").set_value("Metric").run()
        self.assertAlmostEqual(app.number_input(key="test_rw_Metric_0").value, 4.25 * .0254)
        app.selectbox(key="test_rw_unit_Metric_0").select("in").run()
        self.assertAlmostEqual(app.number_input(key="test_rw_in_0").value, 4.25)
        self.assertFalse(app.exception, [e.message for e in app.exception])

    def test_interpretation_sync_updates_inches(self):
        from engine.models import WellTestParams
        from ui.tab_interp import _sync_controls
        state = {"_interp_rw_display_0": {"key": "stale"}}
        _sync_controls(state, WellTestParams(rw=.5), None, "Metric", 0)
        self.assertEqual(state["interp_rw_in_0"], 6.)
        self.assertNotIn("_interp_rw_display_0", state)


if __name__ == "__main__":
    unittest.main()
