import pathlib
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from engine import analysis as A, models as M, multirate as MR
from ui.calculation import FitProgress


class FittingProgressTests(unittest.TestCase):
    def setUp(self):
        self.p = M.WellTestParams()
        self.dt = np.geomspace(.01, 24., 30)
        self.t = np.concatenate([self.dt, 48. + self.dt])
        self.schedule = [(0., 1000.), (48., 0.)]

    def test_buildup_progress_tracks_actual_attempts_without_changing_result(self):
        pw = M.pwf_history(self.p, 48. + self.dt, self.schedule)
        pw0 = M.pwf_history(self.p, [48. * (1.-1e-9)], self.schedule)[0]
        events = []
        fitted, info = A.fit_buildup(self.p, 1000., 48., self.dt, pw, ["k"],
                                    pwf_shutin=pw0, multistart=False, max_nfev=5,
                                    progress=lambda *args: events.append(args))
        baseline, _ = A.fit_buildup(self.p, 1000., 48., self.dt, pw, ["k"],
                                    pwf_shutin=pw0, multistart=False, max_nfev=5)
        self.assertEqual(fitted.to_dict(), baseline.to_dict())
        self.assertEqual(events[0][:2], (0, 0))
        self.assertEqual(events[-1][:2], (2, 2))
        self.assertEqual(info["failed_starts"], 0)

    def test_multirate_progress_tracks_attempts(self):
        events = []
        pw = M.pwf_history(self.p, self.t, self.schedule)
        _, info = MR.fit_multirate(self.p, self.schedule, self.t, pw, ["k"],
                                   multistart=False, max_nfev=5,
                                   progress=lambda *args: events.append(args))
        self.assertEqual(events[-1][:2], (1, 1))
        self.assertEqual(info["failed_starts"], 0)

    def test_all_failed_attempts_raise_explicit_error(self):
        pw = M.pwf_history(self.p, 48. + self.dt, self.schedule)
        with patch.object(A, "_fit_once", side_effect=ValueError("Invalid trial geometry")):
            with self.assertRaisesRegex(ValueError, "No fitting attempt succeeded.*Invalid trial geometry"):
                A.fit_buildup(self.p, 1000., 48., self.dt, pw, ["k"], multistart=False)
        with patch.object(MR, "least_squares", side_effect=ValueError("Invalid trial geometry")):
            with self.assertRaisesRegex(ValueError, "No multi-rate fitting attempt succeeded"):
                MR.fit_multirate(self.p, self.schedule, self.t,
                                 M.pwf_history(self.p, self.t, self.schedule), ["k"], multistart=False)

    def test_calculation_panel_only_reaches_100_on_return(self):
        with patch("ui.calculation.st.progress") as widget, patch("ui.calculation.st.markdown"):
            with FitProgress("Test") as progress:
                progress(1, 2, "Completed attempt 1")
                progress(2, 2, "Completed attempt 2")
            values = [call.args[0] for call in widget.return_value.progress.call_args_list]
            self.assertEqual(values, [.5, .99, 1.])
            widget.reset_mock()
            with self.assertRaisesRegex(ValueError, "Failed"):
                with FitProgress("Test"):
                    raise ValueError("Failed")
            widget.return_value.empty.assert_called_once()
            widget.return_value.progress.assert_not_called()


if __name__ == "__main__":
    unittest.main()
