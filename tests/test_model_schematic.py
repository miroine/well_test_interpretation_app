"""Schematic payload coverage and unit/solver consistency."""
import json
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from engine import models, units, viz
from ui import helpers, model_schematic


class ModelSchematicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.curves = viz.designed_test(models.WellTestParams(), 1500, 200, 100, n=20)

    def test_all_model_boundary_payloads_are_serializable(self):
        for model in models.MODEL_TYPES:
            for boundary in models.BOUNDARY_TYPES:
                with self.subTest(model=model, boundary=boundary):
                    p = models.WellTestParams(model=model, boundary=boundary)
                    data = model_schematic.payload(p, self.curves, 200, 100, 1500, "Field", "Oil")
                    json.dumps(data, allow_nan=False)
                    self.assertEqual(data["model_label"], helpers.MODEL_LABELS[model])
                    self.assertEqual(data["boundary_label"], helpers.BOUNDARY_LABELS[boundary])
                    self.assertEqual(data["params"], p.to_dict())
                    self.assertEqual(data["time"][-1], data["tp"] + data["dt_end"])

    def test_display_units_convert_pressure_rate_and_dimensions(self):
        p = models.WellTestParams(model="fracture", xf=100)
        data = model_schematic.payload(p, self.curves, 200, 100, 1500, "Metric", "Oil")
        np.testing.assert_allclose(data["pressure"],
                                   units.from_field(self.curves["timeline"]["pressure"], "pressure", "Metric"))
        self.assertAlmostEqual(data["rate"], 1500 * units.STB_M3)
        self.assertTrue(any("30.48 m" in text for text in data["details"]))
        self.assertEqual(data["params"]["xf"], 100)

    def test_gas_surface_rate_and_pressure_are_explicit(self):
        data = model_schematic.payload(models.WellTestParams(), self.curves, 200, 100,
                                       8000, "Metric", "Gas")
        self.assertAlmostEqual(data["rate"], 8000 * 28.316847)
        self.assertEqual(data["rate_unit"], "Sm³/d")
        self.assertEqual(data["pressure_label"], "Normalized pseudo-pressure")

    def test_timing_and_pressure_are_existing_solver_samples(self):
        p = models.WellTestParams()
        curves = viz.designed_test(p, 0, 72, 36, n=10)
        data = model_schematic.payload(p, curves, 72, 36, 0, "Field", "Oil")
        self.assertEqual((data["tp"], data["dt_end"]), (72, 36))
        self.assertEqual(data["time"][-1], 108)
        np.testing.assert_allclose(data["pressure"], p.pi)
        self.assertEqual(data["rate"], 0)


if __name__ == "__main__":
    unittest.main()
