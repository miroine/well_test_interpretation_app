import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest

from engine import project


def saved_app():
    import numpy as np
    import streamlit as st
    from engine import models as M
    from ui import session, tab_compare
    session.initialize(st.session_state)
    if "seeded" not in st.session_state:
        p = M.WellTestParams().to_dict()
        st.session_state["fits"] = [
            {"name": name, "params": dict(p), "rms": .1 + i * .1,
             "q": 1000., "tp": 24., "ds": "Measured", "fluid": "Oil"}
            for i, name in enumerate(("First", "Second", "Second"))]
        st.session_state["dataset"] = {"name": "Measured", "fluid": "Oil", "dt": np.geomspace(.01, 10., 20)}
        st.session_state["last_fit"] = {"marker": "unchanged"}
        st.session_state["seeded"] = True
    tab_compare.render({"units": "Field", "fluid": "Oil"})


def editing_app():
    import streamlit as st
    from engine import models as M
    from ui import session, tab_compare, tab_interp
    session.initialize(st.session_state)
    if st.session_state["dataset"] is None:
        p = M.WellTestParams(k=80.)
        data = M.synthetic_buildup(p, 1000., 24., 10., n=30)
        st.session_state["dataset"] = {
            "name": "Measured", "fluid": "Oil", "tp": 24., "q_user": 1000.,
            "dt": data["dt"], "pws": data["pws"], "pwf_shutin": data["pwf_at_shutin"],
            "guess": M.WellTestParams(k=40.).to_dict(), "gas": None}
        st.session_state["fits"] = [{"name": "Alternative", "params": p.to_dict(), "rms": .1,
                                     "q": 1000., "tp": 24., "ds": "Measured"}]
    meta = {"units": "Field", "fluid": "Oil"}
    tab_interp.render(meta)
    tab_compare.render(meta)


class SavedInterpretationTests(unittest.TestCase):
    def app(self):
        app = AppTest.from_function(saved_app, default_timeout=120).run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        return app

    def test_rename_rejects_empty_and_keeps_parameters(self):
        app = self.app()
        original = app.session_state["fits"][0]["params"].copy()
        app.text_input(key="compare_name_0_0_0").set_value("  ").run()
        app.button(key="compare_rename_0_0").click().run()
        self.assertTrue(app.error)
        self.assertEqual(app.session_state["fits"][0]["name"], "First")
        app.text_input(key="compare_name_0_0_0").set_value(" Revised ").run()
        app.button(key="compare_rename_0_0").click().run()
        self.assertEqual(app.session_state["fits"][0]["name"], "Revised")
        self.assertEqual(app.session_state["fits"][0]["params"], original)
        self.assertFalse(app.exception)
        _, state = project.loads(project.dumps({}, {"fits": app.session_state["fits"]}))
        self.assertEqual(state["fits"][0]["name"], "Revised")

    def test_delete_is_confirmed_and_handles_duplicate_names(self):
        app = self.app()
        self.assertTrue(app.button(key="compare_clear_0_0").disabled)
        app.multiselect(key="compare_delete_selection_0_0").set_value([1]).run()
        self.assertTrue(app.button(key="compare_delete_0_0").disabled)
        app.checkbox(key="compare_confirm_0_0").check().run()
        app.button(key="compare_delete_0_0").click().run()
        self.assertFalse(app.exception)
        self.assertEqual([f["name"] for f in app.session_state["fits"]], ["First", "Second"])
        self.assertAlmostEqual(app.session_state["fits"][1]["rms"], .3)
        self.assertFalse(app.checkbox(key="compare_confirm_0_1").value)
        self.assertEqual(app.session_state["dataset"]["name"], "Measured")
        self.assertEqual(app.session_state["last_fit"], {"marker": "unchanged"})

    def test_clear_all_does_not_remove_active_results_or_dataset(self):
        app = self.app()
        app.checkbox(key="compare_confirm_0_0").check().run()
        app.button(key="compare_clear_0_0").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["fits"], [])
        self.assertEqual(app.session_state["dataset"]["name"], "Measured")
        self.assertEqual(app.session_state["last_fit"], {"marker": "unchanged"})
        self.assertTrue(app.info)

    def test_load_for_editing_is_independent_and_requires_matching_data(self):
        app = self.app()
        app.button(key="compare_edit_0_0").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["_interp_pending"]["params"],
                         app.session_state["fits"][0]["params"])
        app.session_state["interp_model"]["params"]["k"] = 999.
        self.assertNotEqual(app.session_state["fits"][0]["params"]["k"], 999.)
        self.assertEqual(app.session_state["last_fit"], {"marker": "unchanged"})
        app.session_state["dataset"] = {"name": "Other", "fluid": "Oil", "dt": [.01, 1.]}
        app.run()
        app.button(key="compare_edit_0_0").click().run()
        self.assertTrue(app.error)
        self.assertEqual(app.session_state["interp_model"]["params"]["k"], 999.)

    def test_editing_updates_working_controls_not_saved_snapshot(self):
        app = AppTest.from_function(editing_app, default_timeout=120).run()
        app.button(key="compare_edit_0_0").click().run()
        app.run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(app.number_input(key="interp_k_Field_0").value, 80.)
        app.number_input(key="interp_k_Field_0").set_value(90.).run()
        self.assertEqual(app.session_state["interp_model"]["params"]["k"], 90.)
        self.assertEqual(app.session_state["fits"][0]["params"]["k"], 80.)
        self.assertIsNone(app.session_state["last_fit"])

    def test_invalid_legacy_skin_does_not_block_deleting_snapshot(self):
        app = self.app()
        fits = app.session_state["fits"]
        fits[0]["params"].update(model="fracture", S=-4.)
        app.session_state["fits"] = fits
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Cannot plot saved interpretation" in w.value for w in app.warning))
        app.multiselect(key="compare_delete_selection_0_0").set_value([0, 2]).run()
        app.checkbox(key="compare_confirm_0_0").check().run()
        app.button(key="compare_delete_0_0").click().run()
        self.assertEqual(len(app.session_state["fits"]), 1)
        self.assertAlmostEqual(app.session_state["fits"][0]["rms"], .2)


if __name__ == "__main__":
    unittest.main()
