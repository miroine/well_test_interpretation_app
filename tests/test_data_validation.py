import pathlib
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from engine import qc, units, project


class GaugeValidationTests(unittest.TestCase):
    def test_nonfinite_values_are_removed_and_reported(self):
        data, issues = qc.clean_gauge(pd.DataFrame({
            "time": [1, 2, np.inf, 3, 4], "pressure": [100, 101, 102, -np.inf, 104],
        }), "time", "pressure")
        self.assertTrue(np.isfinite(data.to_numpy()).all())
        self.assertEqual(len(data), 3)
        self.assertTrue(issues)

    def test_distinct_columns_required(self):
        with self.assertRaisesRegex(ValueError, "different"):
            qc.clean_gauge(pd.DataFrame({"t": [1, 2, 3]}), "t", "t")

    def test_invalid_buildups_are_rejected(self):
        for t, p in (([], []), ([1, 2], [100, 101]), ([0, 1, 2], [100, 101, 102]),
                     ([1, 1, 2], [100, 101, 102]), ([2, 1, 3], [100, 101, 102]),
                     ([1, 2, 3], [100, np.inf, 102]), ([1, 2, 3], [100, 101])):
            with self.subTest(t=t), self.assertRaises(ValueError):
                qc.validate_buildup(t, p)

    def test_clean_metric_record_preserves_field_values(self):
        frame = pd.DataFrame({"t": [3, 1, 2, 2], "p": [210, 200, 202, 204]})
        data, issues = qc.clean_gauge(frame, "t", "p")
        pressure = units.to_field(data["p"].to_numpy(), "pressure", "Metric")
        qc.validate_buildup(data["t"].to_numpy(), pressure)
        np.testing.assert_allclose(units.from_field(pressure, "pressure", "Metric"), [200, 203, 210])
        self.assertEqual(len(issues), 2)

    def test_all_unit_kinds_roundtrip(self):
        for kind in units.KINDS:
            for system in ("Field", "Metric"):
                with self.subTest(kind=kind, system=system):
                    values = np.array([0.1, 1.0, 100.0])
                    np.testing.assert_allclose(
                        units.to_field(units.from_field(values, kind, system), kind, system), values)

    def history_dataset(self):
        return {"tp": 2.0, "q_user": 100.0, "dt": np.arange(1., 7.),
                "pws": np.arange(101., 107.), "history": {
                    "t": np.array([0., 1., 2., 2., 3., 4., 5., 6., 7., 8.]),
                    "p": np.array([110., 105., 100., 100., 101., 102., 103., 104., 105., 106.]),
                    "rate": np.array([100., 100., 100., 0., 0., 0., 0., 0., 0., 0.])}}

    def test_history_cleaning_preserves_both_sides_of_rate_event(self):
        frame = pd.DataFrame({"t": [3, 0, 2, 2, 4, 5, 6, np.inf],
                              "p": [101, 110, 100, 100, 102, 103, 104, 0],
                              "q": [0, 100, 100, 0, 0, 0, 0, 0]})
        history, issues = qc.clean_history(frame, "t", "p", "q")
        np.testing.assert_array_equal(history["t"], [0, 2, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(history["rate"], [100, 100, 0, 0, 0, 0, 0])
        self.assertTrue(issues)
        with self.assertRaisesRegex(ValueError, "different"):
            qc.clean_history(frame, "t", "p", "p")

    def test_manual_exclusions_ranges_restore_and_project_roundtrip(self):
        ds = self.history_dataset()
        filtered = qc.filter_response(ds, [1, 5], 0, 8)
        np.testing.assert_array_equal(filtered["dt"], [1, 3, 4, 5, 6])
        np.testing.assert_array_equal(filtered["pws"], [101, 103, 104, 105, 106])
        np.testing.assert_array_equal(qc.response_source(filtered)["t"], ds["history"]["t"])
        self.assertEqual(filtered["qc_excluded"], [1, 5])
        narrowed = qc.filter_response(filtered, [], 4, 8, pressure_range=(102, 105))
        np.testing.assert_array_equal(narrowed["dt"], [2, 3, 4, 5])
        _, state = project.loads(project.dumps({}, {"dataset": narrowed}))
        restored = qc.restore_response(state["dataset"])
        self.assertNotIn("qc_source", restored)
        np.testing.assert_array_equal(restored["pws"], ds["pws"])
        np.testing.assert_array_equal(restored["history"]["rate"], ds["history"]["rate"])

    def test_invalid_qc_does_not_mutate_source(self):
        ds = self.history_dataset()
        for kwargs in ({"excluded": list(range(10)), "start": 0, "end": 8},
                       {"excluded": [90], "start": 0, "end": 8},
                       {"excluded": [], "start": 8, "end": 0}):
            with self.assertRaises(ValueError):
                qc.filter_response(ds, **kwargs)
            self.assertNotIn("qc_source", ds)
        ds["history"]["rate"][-1] = 100
        with self.assertRaisesRegex(ValueError, "non-zero"):
            qc.filter_response(ds, [], 0, 8)

    def test_legacy_buildup_and_resampling_use_original_readings(self):
        ds = self.history_dataset()
        del ds["history"]
        filtered = qc.filter_response(ds, [1], 0, 8)
        np.testing.assert_array_equal(filtered["dt"], [1, 3, 4, 5, 6])
        resampled = qc.filter_response(filtered, [], 0, 8, ppc=40)
        np.testing.assert_allclose(resampled["pws"], ds["pws"])


if __name__ == "__main__":
    unittest.main()
