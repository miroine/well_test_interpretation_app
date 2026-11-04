import pathlib
import sys
import unittest

import numpy as np
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from engine import qc, units as U
from ui import figures


def qc_interpretation_app():
    import numpy as np
    import streamlit as st
    from engine import models as M
    from ui import session, tab_data, tab_interp
    session.initialize(st.session_state)
    if st.session_state["dataset"] is None:
        st.session_state["dataset"] = {
            "name": "Pressure filter test", "fluid": "Oil", "tp": 2., "q_user": 100.,
            "dt": np.arange(1., 7.), "pws": np.arange(101., 107.), "pwf_shutin": 100.,
            "guess": M.WellTestParams().to_dict(), "gas": None,
            "history": {"t": np.arange(9.), "p": np.array([110., 105., 100., 101., 102., 103., 104., 105., 106.]),
                        "rate": np.array([100., 100., 0., 0., 0., 0., 0., 0., 0.])}}
    tab_data.render({"units": "Field", "fluid": "Oil"})
    tab_interp.render({"units": "Field", "fluid": "Oil"})


class QCInterpretationTests(unittest.TestCase):
    def test_pressure_plot_uses_only_kept_readings_and_preserves_rates(self):
        ds = {"fluid": "Oil", "tp": 2., "dt": np.arange(1., 7.), "pws": np.arange(101., 107.),
              "history": {"t": np.arange(9.), "p": np.array([110., 105., 100., 101., 102., 103., 104., 105., 106.]),
                          "rate": np.array([100., 100., 0., 0., 0., 0., 0., 0., 0.])}}
        filtered = qc.filter_response(ds, [], 0., 8., pressure_range=(102., 105.))
        for system in ("Field", "Metric"):
            fig = figures.interpretation_response(filtered, [], system)
            for trace in fig.data:
                if trace.yaxis != "y2":
                    pressure = U.to_field(np.asarray(trace.y), "pressure", system)
                    self.assertTrue(np.all((pressure >= 102. - 1e-10) & (pressure <= 105. + 1e-10)), trace.name)
            rate = next(trace for trace in fig.data if trace.name == "Recorded surface rate")
            np.testing.assert_array_equal(rate.x, ds["history"]["t"])
            np.testing.assert_array_equal(rate.y, U.from_field(ds["history"]["rate"], "rate", system))
        restored = qc.restore_response(filtered)
        plot = figures.interpretation_response(restored, [], "Field")
        self.assertTrue(any(110. in trace.y for trace in plot.data if trace.yaxis != "y2"))

    def test_filter_applied_in_ui_reaches_interpretation_diagnostics(self):
        app = AppTest.from_function(qc_interpretation_app, default_timeout=120).run()
        self.assertFalse(app.exception)
        app.checkbox(key="qc_pressure_filter_0").check().run()
        app.number_input(key="qc_low_0_Field").set_value(102.).run()
        app.number_input(key="qc_high_0_Field").set_value(105.).run()
        app.checkbox(key="qc_drop_0").uncheck().run()
        next(b for b in app.button if b.label == "Apply QC to the active dataset").click().run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        np.testing.assert_array_equal(app.session_state["dataset"]["pws"], [102., 103., 104., 105.])
        self.assertEqual(len(app.session_state["interp_eng"]["diag"]["ta"]), 4)
        next(b for b in app.button if b.label == "Restore original response (undo all QC)").click().run()
        self.assertEqual(len(app.session_state["dataset"]["pws"]), 6)
        self.assertFalse(app.exception)

    def test_manual_exclusions_and_time_window_are_not_reintroduced(self):
        ds = {"fluid": "Gas", "tp": 2., "dt": np.arange(1., 9.), "pws": np.arange(101., 109.)}
        filtered = qc.filter_response(ds, [2], 4., 9., ppc=4)
        plot = figures.interpretation_response(filtered, [], "Field")
        kept = next(t for t in plot.data if t.name == "Kept pressure data")
        np.testing.assert_array_equal(kept.x, [4., 6., 7., 8., 9.])
        self.assertNotIn(103., kept.y)
        active = next(t for t in plot.data if t.name == "Active buildup data")
        np.testing.assert_array_equal(active.y, filtered["pws"])


if __name__ == "__main__":
    unittest.main()
