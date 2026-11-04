import pathlib
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy import stats
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from engine import models as M, uncertainty as UC, viz


def sampling_app():
    import numpy as np
    import streamlit as st
    from engine import models as M
    from ui import uncertainty
    system = st.radio("Units", ["Field", "Metric"], key="test_units")
    p = M.WellTestParams(k=50., S=1.)
    uncertainty.parameter_sampling(p, "Oil", system, 0, ["k", "S"], 1000., 24., np.geomspace(.01, 10., 20))
    if st.session_state.get("mc"):
        uncertainty.sampling_results(st.session_state["mc"], system)


def interpretation_sampling_app():
    import numpy as np
    import streamlit as st
    from engine import models as M
    from ui import session, tab_interp
    session.initialize(st.session_state)
    if st.session_state["dataset"] is None:
        p = M.WellTestParams(k=50., S=1.)
        data = M.synthetic_buildup(p, 1000., 24., 10., n=30)
        st.session_state["dataset"] = {
            "name": "Sampling test", "fluid": "Oil", "tp": 24., "q_user": 1000.,
            "dt": data["dt"], "pws": data["pws"], "pwf_shutin": data["pwf_at_shutin"],
            "guess": p.to_dict(), "gas": None}
        st.session_state["last_fit"] = {"params": p.to_dict(), "ds": "Sampling test",
            "free": ["k", "S"], "q": 1000., "tp": 24., "q_user": 1000.,
            "info": {"rms": .1, "nfev": 1, "status": 1}}
    tab_interp.render({"units": "Field", "fluid": "Oil"})


class ParameterUncertaintyTests(unittest.TestCase):
    def test_marginal_distributions_bounds_and_percentiles(self):
        definitions = {
            "k": {"distribution": "Uniform", "low": 10., "high": 100.},
            "C": {"distribution": "Log-uniform", "low": .001, "high": 1.},
            "S": {"distribution": "Triangular", "low": 0., "high": 10., "mode": 2.},
            "xf": {"distribution": "Truncated normal", "low": 10., "high": 100., "mean": 40., "sd": 20.}}
        result = UC.sample_parameters(M.WellTestParams(), definitions, np.eye(4), 10000, 123)
        expected = [stats.uniform(10., 90.), stats.loguniform(.001, 1.),
                    stats.triang(.2, loc=0., scale=10.), stats.truncnorm(-1.5, 3., loc=40., scale=20.)]
        for (name, spec), marginal in zip(definitions.items(), expected):
            values = np.asarray(result["samples"][name])
            self.assertTrue(np.all((values >= spec["low"]) & (values <= spec["high"])))
            self.assertLess(stats.kstest(values, marginal.cdf).statistic, .025)
            self.assertAlmostEqual(result["summary"][name]["p50"], np.percentile(values, 50))

    def test_correlation_and_reproducibility(self):
        specs = {n: {"distribution": "Uniform", "low": 1., "high": 10.} for n in ("k", "h")}
        result = UC.sample_parameters(M.WellTestParams(), specs, [[1., .8], [.8, 1.]], 10000, 45)
        self.assertEqual(result, UC.sample_parameters(M.WellTestParams(), specs, [[1., .8], [.8, 1.]], 10000, 45))
        normal = stats.norm.ppf((np.asarray(list(result["samples"].values())).T - 1.) / 9.)
        self.assertAlmostEqual(np.corrcoef(normal.T)[0, 1], .8, delta=.025)
        perfect = UC.sample_parameters(M.WellTestParams(), specs, [[1., -1.], [-1., 1.]], 20)
        np.testing.assert_allclose(np.sum(list(perfect["samples"].values()), axis=0), 11., atol=1e-12)

    def test_invalid_inputs_are_explicit_errors(self):
        p = M.WellTestParams()
        specs = {n: {"distribution": "Uniform", "low": 1., "high": 10.} for n in ("k", "h", "xf")}
        for matrix in ([[1, .9, .9], [.9, 1, -.9], [.9, -.9, 1]],
                       [[1, .3, 0], [.2, 1, 0], [0, 0, 1]], np.zeros((3, 3))):
            with self.assertRaises(ValueError):
                UC.sample_parameters(p, specs, matrix)
        for row in ({"distribution": "Uniform", "low": 10, "high": 1},
                    {"distribution": "Log-uniform", "low": -1, "high": 2},
                    {"distribution": "Triangular", "low": 1, "high": 2, "mode": 3},
                    {"distribution": "Truncated normal", "low": 1, "high": 2, "mean": 1.5, "sd": 0}):
            with self.assertRaises(ValueError):
                UC.sample_parameters(p, {"k": row}, [[1.]])
        with self.assertRaisesRegex(ValueError, "additional skin"):
            UC.sample_parameters(M.WellTestParams(model="fracture"),
                                 {"S": {"distribution": "Uniform", "low": -2., "high": 1.}}, [[1.]])

    def test_forward_envelope_is_not_a_refit(self):
        specs = {"k": {"distribution": "Uniform", "low": 30., "high": 70.}}
        with patch("engine.analysis.fit_buildup", side_effect=AssertionError("Must not refit")):
            mc = UC.sample_parameters(M.WellTestParams(), specs, [[1.]], 10)
            envelope = viz.monte_carlo_envelope(M.WellTestParams(), 1000., 24.,
                                              np.geomspace(.01, 10., 20), mc["samples"])
        self.assertEqual(envelope["count"], 10)
        for quantity in ("pressure", "dp", "der"):
            bounds = envelope[quantity]
            mask = np.isfinite(bounds["p10"])
            self.assertTrue(np.all(bounds["p10"][mask] <= bounds["p50"][mask]))
            self.assertTrue(np.all(bounds["p50"][mask] <= bounds["p90"][mask]))

    def test_ui_sampling_and_validation(self):
        app = AppTest.from_function(sampling_app, default_timeout=120).run()
        tag = "uncertainty_0_Field_homogeneous_none"
        app.number_input(key=f"{tag}_k_low").set_value(60.).run()
        app.number_input(key=f"{tag}_k_high").set_value(40.).run()
        app.button(key=f"{tag}_run").click().run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertTrue(app.error)
        self.assertNotIn("mc", app.session_state)
        app.number_input(key=f"{tag}_k_low").set_value(40.).run()
        app.number_input(key=f"{tag}_k_high").set_value(60.).run()
        app.selectbox(key=f"{tag}_k_dist").set_value("Truncated normal").run()
        app.selectbox(key=f"{tag}_S_dist").set_value("Triangular").run()
        app.checkbox(key=f"{tag}_correlated").check().run()
        app.number_input(key=f"{tag}_runs").set_value(5).run()
        import pandas as pd
        import streamlit as st
        matrix = pd.DataFrame([[1., .7], [.7, 1.]], index=["k", "S"], columns=["k", "S"])
        with patch.object(st, "data_editor", return_value=matrix):
            app.button(key=f"{tag}_run").click().run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(app.session_state["mc"]["mode"], "parameter_sampling")
        self.assertEqual(app.session_state["mc"]["envelope"]["count"], 5)
        self.assertEqual(app.session_state["mc"]["specs"]["k"]["distribution"], "Truncated normal")
        self.assertEqual(app.session_state["mc"]["correlation"], [[1., .7], [.7, 1.]])
        self.assertTrue(any("Achieved parameter Pearson" in m.value for m in app.markdown))

    def test_metric_inputs_convert_to_engine_units(self):
        app = AppTest.from_function(sampling_app, default_timeout=120).run()
        app.radio(key="test_units").set_value("Metric").run()
        tag = "uncertainty_0_Metric_homogeneous_none"
        app.multiselect(key=f"{tag}_names").set_value(["h"]).run()
        app.number_input(key=f"{tag}_h_low").set_value(10.).run()
        app.number_input(key=f"{tag}_h_high").set_value(20.).run()
        app.number_input(key=f"{tag}_runs").set_value(5).run()
        app.button(key=f"{tag}_run").click().run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertAlmostEqual(app.session_state["mc"]["specs"]["h"]["low"], 10. / .3048)
        self.assertAlmostEqual(app.session_state["mc"]["specs"]["h"]["high"], 20. / .3048)

    def test_interpretation_envelopes_use_sampled_results_and_keep_last_fit(self):
        app = AppTest.from_function(interpretation_sampling_app, default_timeout=120).run()
        app.radio(key="interp_mc_mode_0").set_value("Parameter distributions (no refitting)").run()
        tag = "uncertainty_0_Field_homogeneous_none"
        app.number_input(key=f"{tag}_runs").set_value(5).run()
        app.button(key=f"{tag}_run").click().run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertTrue(any("assumed parameter distributions" in c.value for c in app.caption))
        self.assertEqual(app.session_state["last_fit"]["params"]["k"], 50.)
        original = app.session_state["mc"]["samples"]
        app.number_input(key=f"{tag}_k_low").set_value(35.).run()
        self.assertEqual(app.session_state["mc"]["samples"], original)
        self.assertEqual(app.session_state["mc"]["envelope"]["count"], 5)


if __name__ == "__main__":
    unittest.main()
