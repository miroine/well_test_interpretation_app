import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from engine import analysis as A, models as M, viz, units as U
from ui import figures as F


class MonteCarloEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.p = M.WellTestParams()
        self.dt = np.geomspace(.001, 24., 40)
        self.samples = {"k": [30., 60., 90., 120.], "S": [5., 2., 0., -1.]}

    def test_exact_paired_response_percentiles(self):
        result = viz.monte_carlo_envelope(self.p, 1000., 48., self.dt, self.samples)
        pressure, dp, derivative = [], [], []
        for k, skin in zip(self.samples["k"], self.samples["S"]):
            member = M.WellTestParams(**{**self.p.to_dict(), "k": k, "S": skin})
            values = M.pwf_history(member, result["t"], M.simple_buildup_schedule(1000., 48.))
            pressure.append(values)
            delta = values[-len(self.dt):] - values[-len(self.dt) - 1]
            dp.append(delta)
            derivative.append(A.bourdet_derivative(result["ta"], delta, .1))
        for name, rows in (("pressure", pressure), ("dp", dp), ("der", derivative)):
            rows = np.asarray(rows)
            valid = np.isfinite(rows).all(axis=0)
            expected = np.percentile(rows[:, valid], [10, 50, 90], axis=0)
            actual = np.asarray([result[name][k] for k in ("p10", "p50", "p90")])
            np.testing.assert_allclose(actual[:, valid], expected)
            self.assertTrue(np.all(actual[0, valid] <= actual[1, valid]))
            self.assertTrue(np.all(actual[1, valid] <= actual[2, valid]))
        self.assertEqual(result["count"], 4)

    def test_identical_parameters_produce_zero_width_band(self):
        result = viz.monte_carlo_envelope(self.p, 1000., 48., self.dt, {"k": [60.] * 3})
        for name in ("pressure", "dp", "der"):
            np.testing.assert_allclose(result[name]["p10"], result[name]["p90"], equal_nan=True)

    def test_malformed_samples_raise_explicit_error(self):
        for samples in ({}, {"k": [1., 2.]}, {"k": [1., 2., 3.], "S": [1., 2.]},
                        {"k": [1., np.nan, 3.]}, {"unknown": [1., 2., 3.]}):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                viz.monte_carlo_envelope(self.p, 1000., 48., self.dt, samples)

    def test_metric_band_conversion_and_log_gaps(self):
        bounds = {"p10": np.array([10., -1., 30.]), "p50": np.array([20., 2., 40.]),
                  "p90": np.array([30., 3., 50.])}
        fig = F.line_figure("Band", "time", "pressure", [])
        F.add_percentile_band(fig, [1., 2., 3.], bounds, "dp", "Metric", "rgba(34,81,255,0.18)", log=True)
        self.assertEqual(fig.data[1].fill, "tonexty")
        for trace, key in zip(fig.data, ("p10", "p90", "p50")):
            np.testing.assert_allclose(np.asarray(trace.y)[[0, 2]], bounds[key][[0, 2]] * U.PSI_BAR)
            self.assertTrue(np.isnan(trace.y[1]))
            self.assertFalse(trace.connectgaps)


if __name__ == "__main__":
    unittest.main()
