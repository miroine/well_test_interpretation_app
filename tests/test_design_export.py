"""Export values, event semantics and Eclipse record layout."""
import csv
import io
import pathlib
import shlex
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from engine import design_export as E, models, units as U, viz


def production_items(schedule):
    record = schedule.split("WCONPROD\n", 1)[1].split("/", 1)[0]
    items = []
    for token in shlex.split(record):
        if token.endswith("*"):
            items.extend([None] * int(token[:-1]))
        else:
            items.append(token)
    return items


class DesignExports(unittest.TestCase):
    def setUp(self):
        self.q, self.tp, self.dt = 1500.0, 72.0, 36.0
        self.curves = viz.designed_test(models.WellTestParams(), self.q, self.tp, self.dt)

    def test_csv_pressure_and_both_sides_of_shut_in(self):
        rows = list(csv.DictReader(io.StringIO(E.design_csv(
            self.curves, self.q, self.tp, self.dt, "Field", "Oil"))))
        self.assertEqual(len(rows), len(self.curves["timeline"]["t"]) + 1)
        event = [row for row in rows if float(row["time_hr"]) == self.tp]
        self.assertEqual([row["phase"] for row in event], ["drawdown", "buildup"])
        self.assertEqual([float(row["surface_rate_stb_per_day"]) for row in event], [self.q, 0])
        self.assertEqual(event[0]["bottom_hole_pressure_psi"], event[1]["bottom_hole_pressure_psi"])
        self.assertEqual(float(rows[0]["time_hr"]), 0)
        self.assertEqual(float(rows[-1]["time_hr"]), self.tp + self.dt)
        self.assertEqual(float(rows[-1]["shut_in_elapsed_hr"]), self.dt)
        original = [row for row in rows if not (row["phase"] == "buildup" and float(row["time_hr"]) == self.tp)]
        np.testing.assert_allclose([float(row["bottom_hole_pressure_psi"]) for row in original],
                                   self.curves["timeline"]["pressure"], rtol=1e-14)

    def test_csv_units_and_gas_pressure_label(self):
        for fluid, system, rate_unit in (("Oil", "Metric", "m3_per_day"),
                                        ("Gas", "Field", "Mscf_per_day"),
                                        ("Gas", "Metric", "Sm3_per_day")):
            with self.subTest(fluid=fluid, system=system):
                rows = list(csv.DictReader(io.StringIO(E.design_csv(
                    self.curves, self.q, self.tp, self.dt, system, fluid))))
                expected_q = U.from_field(self.q, "gasrate" if fluid == "Gas" else "rate", system)
                self.assertAlmostEqual(float(rows[0][f"surface_rate_{rate_unit}"]), expected_q)
                quantity = "normalized_pseudo_pressure" if fluid == "Gas" else "bottom_hole_pressure"
                self.assertAlmostEqual(float(rows[0][f"{quantity}_{U.label('pressure', system)}"]),
                                       U.from_field(4000.0, "pressure", system))

    def test_eclipse_surface_control_positions_and_units(self):
        for fluid in ("Oil", "Gas"):
            for system in ("Field", "Metric"):
                with self.subTest(fluid=fluid, system=system):
                    schedule = E.eclipse_schedule("WELL-A", self.q, self.tp, self.dt, system, fluid, 100.0)
                    items = production_items(schedule)
                    self.assertEqual(items[:3], ["WELL-A", "OPEN", "ORAT" if fluid == "Oil" else "GRAT"])
                    rate_index = 3 if fluid == "Oil" else 5
                    self.assertAlmostEqual(float(items[rate_index]),
                                           U.from_field(self.q, "rate" if fluid == "Oil" else "gasrate", system))
                    self.assertAlmostEqual(float(items[8]), U.from_field(100, "pressure", system))
                    self.assertEqual(len(items), 9)
                    steps = [float(block.split("/")[0]) for block in schedule.split("TSTEP\n")[1:]]
                    self.assertEqual(steps, [3, 1.5])
                    self.assertIn("'WELL-A' 'SHUT' /", schedule)
                    self.assertNotIn("\nSCHEDULE\n", schedule)
                    self.assertNotIn("\nEND\n", schedule)
                    self.assertNotIn("WELSPECS\n", schedule)
                    self.assertNotIn("COMPDAT\n", schedule)

    def test_eclipse_fractional_hours_preserve_duration(self):
        schedule = E.eclipse_schedule("PROD", 1200, 1.25, 0.1, "Field", "Oil", 14.7)
        steps = [float(block.split("/")[0]) for block in schedule.split("TSTEP\n")[1:]]
        np.testing.assert_allclose(np.array(steps) * 24, [1.25, 0.1], rtol=1e-14)

    def test_zero_rate_stays_shut(self):
        schedule = E.eclipse_schedule("PROD", 0, self.tp, self.dt, "Field", "Gas", 14.7)
        self.assertNotIn("WCONPROD\n", schedule)
        self.assertNotIn("'OPEN'", schedule)
        self.assertEqual(schedule.count("'SHUT'"), 2)

    def test_invalid_export_inputs_are_explicit_errors(self):
        for well in ("", "TOOLONGNAME", "A' /\nEND", "A*", "1WELL"):
            with self.subTest(well=well), self.assertRaises(ValueError):
                E.eclipse_schedule(well, self.q, self.tp, self.dt, "Field", "Oil", 14.7)
        for bhp in (0, -1, np.inf, np.nan):
            with self.subTest(bhp=bhp), self.assertRaises(ValueError):
                E.eclipse_schedule("PROD", self.q, self.tp, self.dt, "Field", "Oil", bhp)
        for q, tp, dt, system, fluid in ((-1, 72, 36, "Field", "Oil"),
                                       (1, 0, 36, "Field", "Oil"),
                                       (1, 72, np.inf, "Field", "Oil"),
                                       (1, 72, 36, "Other", "Oil"),
                                       (1, 72, 36, "Field", "Water")):
            with self.subTest(q=q, tp=tp, system=system, fluid=fluid), self.assertRaises(ValueError):
                E.design_csv(self.curves, q, tp, dt, system, fluid)


if __name__ == "__main__":
    unittest.main()
