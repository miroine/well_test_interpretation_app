import pathlib
import sys
import unittest

import numpy as np
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from engine import models as M, analysis as A


def fracture_app():
    import streamlit as st
    from engine import models as M
    from ui import inputs
    p = M.WellTestParams(model="fracture", k=15, h=100, phi=.23, ct=3e-6,
                         xf=280, rw=.354, C=.01, S=0)
    result = inputs.params_form("skin", p, "Field", "Oil", 0)
    st.session_state["parameters"] = result.to_dict()


class FractureSkinTests(unittest.TestCase):
    def test_screenshot_total_skin_converts_to_stable_additional_skin(self):
        app = AppTest.from_function(fracture_app, default_timeout=120).run()
        app.number_input(key="skin_S_total_Field_0").set_value(-4.).run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        p = M.WellTestParams(**app.session_state["parameters"])
        self.assertAlmostEqual(p.S, -4. - np.log(2 * .354 / 280))
        times = np.geomspace(.0001, 156, 300)
        pressure = M.pwf_history(p, times, [(0., 1500.), (84., 0.)])
        self.assertTrue(np.isfinite(pressure).all())
        self.assertGreater(pressure.min(), 3000)
        self.assertLessEqual(pressure.max(), p.pi)
        prepared = M.prepare_pwf_history(times, [(0., 1500.), (84., 0.)])
        np.testing.assert_array_equal(prepared(p), pressure)
        app.selectbox(key="skin_skin_definition_0").select("Additional skin").run()
        self.assertAlmostEqual(app.session_state["parameters"]["S"], p.S)
        app.selectbox(key="skin_skin_definition_0").select("Total equivalent radial skin").run()
        self.assertAlmostEqual(app.number_input(key="skin_S_total_Field_0").value, -4)

    def test_negative_additional_skin_rejected_in_both_response_paths(self):
        p = M.WellTestParams(model="fracture", S=-4., xf=280.)
        for evaluate in (lambda p: M.pwf_history(p, [1., 10.], [(0., 1000.)]),
                         M.prepare_pwf_history([1., 10.], [(0., 1000.)])):
            with self.assertRaisesRegex(ValueError, "non-negative additional skin"):
                evaluate(p)

    def test_regression_fits_nonnegative_additional_skin(self):
        p = M.WellTestParams(model="fracture", S=2., xf=280.)
        data = M.synthetic_buildup(p, 1500., 84., 72., n=40)
        p.S = 0.
        fitted, info = A.fit_buildup(p, 1500., 84., data["dt"], data["pws"], ["S"],
                                     pwf_shutin=data["pwf_at_shutin"], multistart=False)
        self.assertGreaterEqual(fitted.S, 0)
        self.assertAlmostEqual(fitted.S, 2., delta=.1)


if __name__ == "__main__":
    unittest.main()
