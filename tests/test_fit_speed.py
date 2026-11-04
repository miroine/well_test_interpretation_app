import pathlib
import sys
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from engine import analysis as A, models as M, multirate as MR


class FitSpeedTests(unittest.TestCase):
    def test_prepared_pressure_matches_all_models_and_boundaries(self):
        times = np.r_[0., .001, .1, 1., 10., 48., 48., 48.001, 48.1, 49., 60., 100.]
        schedule = [(0., 1000.), (10., 600.), (48., 0.), (80., 500.)]
        cases = [M.WellTestParams(model=model) for model in M.MODEL_TYPES]
        cases += [M.WellTestParams(boundary=boundary) for boundary in M.BOUNDARY_TYPES]
        cases += [M.WellTestParams(model=model, boundary=boundary)
                  for model in ("dual_pss", "dual_slab")
                  for boundary in M.CLOSED_BOUNDARIES]
        for p in cases:
            with self.subTest(model=p.model, boundary=p.boundary):
                evaluate = M.prepare_pwf_history(times, schedule)
                for trial in (p, replace(p, C=p.C * 2., S=0., pi=4200.),
                              replace(p, k=p.k * 1.1)):
                    np.testing.assert_array_equal(evaluate(trial), M.pwf_history(trial, times, schedule))

    def test_cache_reuses_only_reservoir_invariant_trials_and_is_bounded(self):
        p = M.WellTestParams()
        evaluate = M.prepare_pwf_history([.1, 1., 10.], [(0., 1000.)])
        with patch.object(M, "laplace_function", wraps=M.laplace_function) as factory:
            evaluate(p)
            evaluate(replace(p, k=60.))
            evaluate(replace(p, S=3., C=.05, pi=4200.))
            self.assertEqual(factory.call_count, 2)
            evaluate(replace(p, re=3000.))
            self.assertEqual(factory.call_count, 3)
            evaluate(replace(p, k=60.))
            self.assertEqual(factory.call_count, 4)

    def test_start_deduplication_preserves_first_and_distinct_bounded_vectors(self):
        p = M.WellTestParams()
        starts = [p, replace(p, pi=4500.), replace(p, k=60.),
                  replace(p, k=1e6), replace(p, k=2e6)]
        actual = A._unique_starts(starts, ["k", "S", "C"])
        self.assertEqual([s.k for s in actual], [50., 60., 1e6])
        self.assertIs(actual[0], p)

    def test_buildup_does_not_repeat_an_identical_start(self):
        p = M.WellTestParams()
        data = M.synthetic_buildup(p, 1000., 48., 24., n=30)
        events = []
        A.fit_buildup(p, 1000., 48., data["dt"], data["pws"], ["re"],
                      pwf_shutin=data["pwf_at_shutin"], multistart=False, max_nfev=2,
                      progress=lambda *args: events.append(args))
        self.assertEqual(events[-1][:2], (1, 1))

    def test_fitting_results_unchanged_with_prepared_pressure(self):
        truth = M.WellTestParams(k=65., S=3., C=.02)
        guess = replace(truth, k=45., S=1., C=.01)
        data = M.synthetic_buildup(truth, 1000., 48., 24., n=40)
        schedule = [(0., 1000.), (48., 0.)]
        times = np.r_[np.geomspace(.001, 47., 30), 48. + data["dt"]]
        pressure = M.pwf_history(truth, times, schedule)

        def buildup():
            return A.fit_buildup(guess, 1000., 48., data["dt"], data["pws"],
                                 ["k", "S", "C"], pwf_shutin=data["pwf_at_shutin"],
                                 max_nfev=10, multistart=False)

        def multirate():
            return MR.fit_multirate(guess, schedule, times, pressure, ["k", "S", "C"],
                                    max_nfev=10, multistart=False)

        for fit in (buildup, multirate):
            actual, info = fit()
            with patch.object(M, "prepare_pwf_history",
                              side_effect=lambda t, s: lambda p: M.pwf_history(p, t, s)):
                expected, old_info = fit()
            self.assertEqual(actual.to_dict(), expected.to_dict())
            self.assertEqual(info, old_info)


if __name__ == "__main__":
    unittest.main()
