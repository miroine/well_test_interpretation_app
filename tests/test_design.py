"""Numerical, chart and callback regression tests for the design timeline."""
import pathlib
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from engine import analysis, models, units, viz
from ui import design_timeline, figures


class DesignedTestCurves(unittest.TestCase):
    def setUp(self):
        self.p = models.WellTestParams(k=50, C=0.005, S=0)
        self.q, self.tp, self.dt = 1200.0, 200.0, 100.0
        self.curves = viz.designed_test(self.p, self.q, self.tp, self.dt, n=80)

    def test_timeline_matches_solver_and_schedule(self):
        timeline = self.curves["timeline"]
        expected = models.pwf_history(self.p, timeline["t"], [(0, self.q), (self.tp, 0)])
        np.testing.assert_allclose(timeline["pressure"], expected)
        self.assertTrue(np.all(np.diff(timeline["t"]) > 0))
        self.assertEqual(timeline["t"][-1], self.tp + self.dt)
        np.testing.assert_equal(timeline["rate_t"], [0, 200, 200, 300])
        np.testing.assert_equal(timeline["rate"], [1200, 1200, 0, 0])
        self.assertLess(self.curves["drawdown"]["pressure"][-1], self.p.pi)
        self.assertGreater(self.curves["buildup"]["pws"][-1],
                           self.curves["drawdown"]["pressure"][-1])

    def test_drawdown_pressure_and_derivative_use_unit_response(self):
        dd = self.curves["drawdown"]
        dp, der = models.unit_rate_response(self.p, dd["t"])
        np.testing.assert_allclose(dd["dp"], dp * self.q)
        np.testing.assert_allclose(dd["der"], der * self.q)

    def test_buildup_derivative_matches_selected_coordinate(self):
        bu = self.curves["buildup"]
        np.testing.assert_allclose(bu["ta"], self.tp * bu["dt"] / (self.tp + bu["dt"]))
        np.testing.assert_allclose(bu["der_elapsed"],
                                   analysis.bourdet_derivative(bu["dt"], bu["dp"]), equal_nan=True)
        self.assertGreater(np.nanmedian(bu["der"][-15:]),
                           np.nanmedian(bu["der_elapsed"][-15:]))

    def test_radial_derivatives_superpose(self):
        dd, bu = self.curves["drawdown"], self.curves["buildup"]
        plateau = 0.5 * models.p_scale(self.p) * self.q
        for curve, xkey in ((dd, "t"), (bu, "ta")):
            mask = (curve[xkey] > 5) & (curve[xkey] < 20)
            self.assertLess(abs(np.nanmedian(curve["der"][mask]) / plateau - 1), 0.1)

    def test_zero_rate_and_short_period_are_finite(self):
        curves = viz.designed_test(self.p, 0, 1, 0.1, n=20)
        np.testing.assert_allclose(curves["timeline"]["pressure"], self.p.pi)
        self.assertTrue(np.all(np.isfinite(curves["timeline"]["t"])))

    def test_invalid_inputs_are_rejected(self):
        for q, tp, dt in ((-1, 200, 100), (1, 0, 1), (1, 1, 0), (1, np.inf, 1)):
            with self.subTest(q=q, tp=tp, dt=dt), self.assertRaises(ValueError):
                viz.designed_test(self.p, q, tp, dt)

    def test_overlay_toggles_units_and_axes(self):
        selected = ["Drawdown derivative", "Buildup pressure change"]
        fig = figures.design_overlay(
            self.curves, "bar", lambda v: units.from_field(v, "pressure", "Metric"),
            selected, equivalent=False)
        self.assertEqual([t.name for t in fig.data], selected)
        self.assertEqual(fig.layout.xaxis.type, "log")
        self.assertIn("Elapsed time within", fig.layout.xaxis.title.text)
        self.assertIn("[bar]", fig.layout.yaxis.title.text)
        dd = self.curves["drawdown"]
        mask = np.isfinite(dd["der"]) & (dd["der"] > 0)
        np.testing.assert_allclose(fig.data[0].y, units.from_field(dd["der"][mask], "pressure", "Metric"))
        eq = figures.design_overlay(self.curves, "psi", lambda v: v,
                                     ["Buildup derivative"], equivalent=True)
        mask = np.isfinite(self.curves["buildup"]["der"]) & (self.curves["buildup"]["der"] > 0)
        np.testing.assert_allclose(eq.data[0].x, self.curves["buildup"]["ta"][mask])


class TimingEdits(unittest.TestCase):
    def test_valid_edit_and_limits(self):
        self.assertEqual(design_timeline.validate_edit(
            {"tp": 250, "dt_end": 80, "base": [200, 100]}, 200, 100), (250.0, 80.0))
        self.assertEqual(design_timeline.validate_edit(
            {"tp": 1, "dt_end": 0.1, "base": [1, 0.1]}, 1, 0.1), (1.0, 0.1))

    def test_invalid_or_stale_edits_rejected(self):
        edits = (None, {}, {"tp": True, "dt_end": 10, "base": [200, 100]},
                 {"tp": float("nan"), "dt_end": 10, "base": [200, 100]},
                 {"tp": 200, "dt_end": 0, "base": [200, 100]},
                 {"tp": 200001, "dt_end": 10, "base": [200, 100]},
                 {"tp": 200, "dt_end": 10, "base": [199, 100]})
        for edit in edits:
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                design_timeline.validate_edit(edit, 200, 100)

    def test_callback_syncs_widgets_and_keeps_surface_gas_rate(self):
        curves = viz.designed_test(models.WellTestParams(), 300, 200, 100, n=10)
        state = {"design_tp_0": 200.0, "design_dtend_0": 100.0}
        with patch.object(design_timeline.st, "session_state", state), \
                patch.object(design_timeline, "_timeline") as mount:
            design_timeline.render(curves, 200, 100, 8000, "Metric", "Gas", 0)
            self.assertAlmostEqual(mount.call_args.kwargs["data"]["rate"], 8000 * 28.316847)
            self.assertEqual(mount.call_args.kwargs["data"]["rate_unit"], "Sm³/d")
            state["design_timeline_0"] = {"timing": {"tp": 230, "dt_end": 120, "base": [200, 100]}}
            mount.call_args.kwargs["on_timing_change"]()
            self.assertEqual(state["design_tp_0"], 230)
            self.assertEqual(state["design_dtend_0"], 120)
            state["design_timeline_0"]["timing"]["base"] = [199, 100]
            mount.call_args.kwargs["on_timing_change"]()
            self.assertIn("_design_timing_error", state)
            self.assertEqual(state["design_tp_0"], 230)


if __name__ == "__main__":
    unittest.main()
