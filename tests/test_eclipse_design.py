"""Numerical mapping and standalone Eclipse deck regression checks."""
import csv
import io
import json
import pathlib
import shlex
import sys
import unittest
import zipfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from engine import design_export as E, eclipse_design as D, gas as G, models as M, units as U, viz


def expand(block):
    values = []
    for token in block.split():
        if "*" in token:
            count, value = token.split("*")
            values.extend([float(value)] * int(count))
        else:
            values.append(float(token))
    return np.array(values)


def keyword_array(deck, keyword):
    return expand(deck.split("\n" + keyword + "\n", 1)[1].split("/", 1)[0])


class EclipseDesignDecks(unittest.TestCase):
    def setUp(self):
        self.p = M.WellTestParams(C=0)
        self.setup = D.DeckSetup(nx=9, ny=9, nz=3, report_points_per_period=12)

    def test_metric_deck_converts_every_dimensional_surface(self):
        for fluid in ("Oil", "Gas"):
            gas = {"T_F": 220, "gamma_g": 0.68, "cf": 4e-6} if fluid == "Gas" else None
            with self.subTest(fluid=fluid):
                field, _ = D.build_deck(self.p, self.setup, "TEST", 100, 72, 36, fluid, 100, gas)
                metric, meta = D.build_deck(self.p, self.setup, "TEST", 100, 72, 36, fluid, 100, gas, system="Metric")
                self.assertIn("\nMETRIC\n", metric)
                self.assertNotIn("\nFIELD\n", metric)
                self.assertEqual(meta["deck_units"], "METRIC")
                for keyword in ("DX", "DY", "DZ", "TOPS"):
                    np.testing.assert_allclose(keyword_array(metric, keyword),
                                               keyword_array(field, keyword) * U.FT_M, rtol=2e-11)
                np.testing.assert_allclose(keyword_array(metric, "PRESSURE"),
                                           keyword_array(field, "PRESSURE") * U.PSI_BAR, rtol=2e-11)
                for keyword in ("PERMX", "PERMY", "PERMZ", "PORO", "ACTNUM", "PVTNUM"):
                    np.testing.assert_array_equal(keyword_array(metric, keyword), keyword_array(field, keyword))
                for keyword, index, factor in (("DENSITY", 0, 16.01846337396),
                                                ("DENSITY", 1, 16.01846337396),
                                                ("DENSITY", 2, 16.01846337396),
                                                ("ROCK", 0, U.PSI_BAR), ("ROCK", 1, 1 / U.PSI_BAR)):
                    a = keyword_array(field, keyword)
                    b = keyword_array(metric, keyword)
                    self.assertAlmostEqual(b[index] / a[index], factor, places=9)
                if fluid == "Oil":
                    a, b = keyword_array(field, "PVCDO"), keyword_array(metric, "PVCDO")
                    np.testing.assert_allclose(b, a * [U.PSI_BAR, 1, 1 / U.PSI_BAR, 1, 1], rtol=2e-11)
                else:
                    a = keyword_array(field, "PVDG").reshape(-1, 3)
                    b = keyword_array(metric, "PVDG").reshape(-1, 3)
                    np.testing.assert_allclose(b, a * [U.PSI_BAR, U.STB_M3 / 28.316847, 1], rtol=2e-11)
                a = field.split("\nWELSPECS\n")[1].split("/")[0].split()
                b = metric.split("\nWELSPECS\n")[1].split("/")[0].split()
                self.assertAlmostEqual(float(b[4]), float(a[4]) * U.FT_M)
                a = field.split("\nCOMPDAT\n")[1].split("/")[0].split()
                b = metric.split("\nCOMPDAT\n")[1].split("/")[0].split()
                self.assertAlmostEqual(float(b[8]), float(a[8]) * U.FT_M)
                def production_items(deck):
                    items = []
                    for token in shlex.split(deck.split("WCONPROD\n", 1)[1].split("/", 1)[0]):
                        items.extend([None] * int(token[:-1]) if token.endswith("*") else [token])
                    return items
                a, b = production_items(field), production_items(metric)
                idx = 3 if fluid == "Oil" else 5
                factor = U.STB_M3 if fluid == "Oil" else 28.316847
                self.assertAlmostEqual(float(b[idx]), float(a[idx]) * factor)
                self.assertAlmostEqual(float(b[8]), float(a[8]) * U.PSI_BAR)
                np.testing.assert_array_equal(keyword_array(metric, "TSTEP"), keyword_array(field, "TSTEP"))

    def test_metric_package_has_consistent_reference_and_includes(self):
        curves = viz.designed_test(self.p, 100, 72, 36, n=20)
        _, package, _ = D.deck_package(self.p, self.setup, "TEST", curves, 100, 72, 36,
                                      "Oil", 100, system="Metric")
        with zipfile.ZipFile(io.BytesIO(package)) as z:
            self.assertIn("synthetic_bhp_bar", z.read("predicted_bhp.csv").decode())
            self.assertEqual(json.loads(z.read("comparison.json"))["units"], "Metric")
            self.assertEqual(json.loads(z.read("model.json"))["deck_units"], "METRIC")
            self.assertIn("METRIC deck", z.read("USAGE.txt").decode())
            self.assertIn("METRIC units", z.read("schedule.INC").decode())

    def test_standalone_sections_arrays_controls_and_elapsed_time(self):
        deck, meta = D.build_deck(self.p, self.setup, "TEST", 100, 72, 36, "Oil", 100)
        sections = ["RUNSPEC", "GRID", "PROPS", "REGIONS", "SOLUTION", "SUMMARY", "SCHEDULE", "END"]
        positions = [deck.index("\n" + kw + "\n") for kw in sections]
        self.assertEqual(positions, sorted(positions))
        n = int(np.prod(meta["grid_shape"]))
        for keyword in ("DX", "DY", "DZ", "PERMX", "PERMY", "PERMZ", "PORO", "ACTNUM", "PVTNUM", "PRESSURE", "SOIL"):
            self.assertEqual(len(keyword_array(deck, keyword)), n, keyword)
        for keyword in ("DX", "DY", "DZ", "PERMX", "PERMY", "PERMZ", "PORO", "PRESSURE"):
            self.assertTrue(np.all(keyword_array(deck, keyword) > 0), keyword)
        steps = [expand(block.split("/", 1)[0]) for block in deck.split("TSTEP\n")[1:]]
        self.assertEqual([len(x) for x in steps], [12, 12])
        self.assertAlmostEqual(sum(steps[0]) * 24, 72, places=8)
        self.assertAlmostEqual(sum(steps[1]) * 24, 36, places=8)
        self.assertTrue(all(np.all(x > 0) for x in steps))
        self.assertIn("'ORAT' 100 4* 100", deck)
        self.assertEqual(deck.count("\nSCHEDULE\n"), 1)
        self.assertIn("\nWBHP\n", deck)
        self.assertIn("\nBPR\n", deck)
        self.assertIn("C", meta["approximated_source_parameters"])

    def test_all_models_and_boundaries_have_resolved_active_completions(self):
        for model in M.MODEL_TYPES:
            for boundary in M.BOUNDARY_TYPES:
                with self.subTest(model=model, boundary=boundary):
                    p = M.WellTestParams(model=model, boundary=boundary, Lw=600)
                    grid = D.grid_model(p, self.setup, "Oil")
                    for i, j, k, direction in grid["connections"]:
                        self.assertTrue(grid["active"][k, j, i])
                        self.assertFalse(grid["buffer"][k, j, i])
                        self.assertIn(direction, ("X", "Z"))
                    self.assertTrue(np.all(np.diff(grid["x_edges"]) > 0))
                    self.assertTrue(np.all(np.diff(grid["z_edges"]) > 0))
                    self.assertEqual(grid["perm"].size, np.prod(grid["shape"]))

    def test_fractures_preserve_geometry_conductivity_and_completion_count(self):
        p = M.WellTestParams(model="mfhw", n_frac=4, f_space=250, xf=100, FcD=10)
        grid = D.grid_model(p, self.setup, "Oil")
        self.assertEqual(len(grid["connections"]), p.n_frac)
        expected_x = (np.arange(p.n_frac) - (p.n_frac - 1) / 2) * p.f_space
        x = (grid["x_edges"][:-1] + grid["x_edges"][1:]) / 2
        np.testing.assert_allclose([x[i] for i, j, k, _ in grid["connections"]], expected_x, atol=1e-10)
        np.testing.assert_allclose(grid["perm"][grid["fracture"]],
                                   p.FcD * p.k * p.xf / self.setup.fracture_width_ft)
        self.assertTrue(np.all(grid["poro"][grid["fracture"]] == self.setup.fracture_porosity))
        for i, j, k, _ in grid["connections"]:
            self.assertTrue(grid["fracture"][k, j, i])

    def test_horizontal_and_limited_entry_geometry(self):
        p = M.WellTestParams(model="horizontal", Lw=1000, zw_h=0.25, kv_kh=0.07)
        grid = D.grid_model(p, self.setup, "Oil")
        z = (grid["z_edges"][:-1] + grid["z_edges"][1:]) / 2
        for i, j, k, _ in grid["connections"]:
            self.assertAlmostEqual(z[k], p.zw_h * p.h)
        np.testing.assert_allclose(grid["permz"], grid["perm"] * 0.07)
        p = M.WellTestParams(model="limited_entry", perf_frac=0.2, perf_pos=0.3)
        grid = D.grid_model(p, self.setup, "Oil")
        self.assertAlmostEqual(sum(grid["dz"][k, j, i] for i, j, k, _ in grid["connections"]), p.h * p.perf_frac)

    def test_unstimulated_zero_skin_has_valid_near_well_cells(self):
        for model in ("homogeneous", "fracture_fc", "mfhw", "limited_entry", "horizontal"):
            with self.subTest(model=model):
                grid = D.grid_model(M.WellTestParams(model=model, S=0), self.setup, "Oil")
                self.assertTrue(grid["connections"])
        with self.assertRaisesRegex(ValueError, "skin too negative"):
            D.grid_model(M.WellTestParams(S=-7), self.setup, "Oil")

    def test_composite_mobility_and_storage_mapping(self):
        p = M.WellTestParams(model="composite3", M=2, D=3, M3=4, D3=2)
        grid = D.grid_model(p, self.setup, "Oil")
        for region, mobility, diffusivity in ((1, 1, 1), (2, 2, 3), (3, 4, 2)):
            np.testing.assert_allclose(grid["perm"][grid["region"] == region], p.k / mobility)
            self.assertAlmostEqual(grid["ct"][region - 1], p.ct * diffusivity / mobility)
        gas = {"T_F": 220, "gamma_g": 0.68, "cf": 4e-6}
        grid = D.grid_model(p, self.setup, "Gas", gas)
        np.testing.assert_allclose(grid["poro"][grid["region"] == 2], p.phi * 3 / 2)

    def test_boundaries_and_buffer_are_explicit(self):
        for boundary in ("fault", "channel", "corner", "circle", "circle_cp", "rectangle", "const_p_fault"):
            p = M.WellTestParams(boundary=boundary)
            g = D.grid_model(p, self.setup, "Oil")
            if boundary in ("circle", "circle_cp"):
                self.assertTrue(np.any(~g["active"]))
                self.assertAlmostEqual(g["x_edges"][-1], p.re)
            if boundary == "channel":
                self.assertEqual((g["x_edges"][0], g["x_edges"][-1]), (-p.d1, p.d2))
            if boundary in ("const_p_fault", "circle_cp"):
                self.assertTrue(g["buffer"].any())
                np.testing.assert_equal(g["pv_mult"][g["buffer"]], self.setup.boundary_pv_multiplier)
            else:
                self.assertFalse(g["buffer"].any())

    def test_dual_porosity_is_not_misrepresented_as_exact(self):
        for model in ("dual_pss", "dual_slab"):
            p = M.WellTestParams(model=model)
            deck, meta = D.build_deck(p, self.setup, "PROD", 10, 72, 36, "Oil", 14.7)
            self.assertIn("does NOT reproduce", deck)
            self.assertIn("omega", meta["approximated_source_parameters"])
            self.assertNotIn("\nDUALPORO\n", deck)
            self.assertEqual(meta["source_design"]["omega"], p.omega)

    def test_gas_pvt_uses_physical_field_units(self):
        gas = {"T_F": 220, "gamma_g": 0.68, "cf": 4e-6}
        p = M.WellTestParams(pi=4000)
        deck, meta = D.build_deck(p, self.setup, "GAS1", 6000, 72, 36, "Gas", 14.7, gas)
        rows = np.array([list(map(float, line.split())) for line in
                         deck.split("\nPVDG\n", 1)[1].split("/", 1)[0].strip().splitlines()])
        row = rows[np.argmin(abs(rows[:, 0] - p.pi))]
        self.assertEqual(row[0], p.pi)
        props = G.gas_props(p.pi, gas["T_F"], gas["gamma_g"])
        self.assertAlmostEqual(row[1], 1000 * props["Bg"], places=8)
        self.assertIn("'GRAT' 2* 6000 2*", deck)
        self.assertEqual(meta["deck_units"], "FIELD")

    def test_package_contains_deck_source_inputs_and_comparison(self):
        curves = viz.designed_test(self.p, 100, 72, 36, n=20)
        deck, package, meta = D.deck_package(self.p, self.setup, "TEST", curves, 100, 72, 36, "Oil", 100)
        with zipfile.ZipFile(io.BytesIO(package)) as z:
            self.assertEqual(set(z.namelist()), {"TEST.DATA", "model.json", "comparison.json", "schedule.INC",
                                                "summary.INC", "predicted_bhp.csv", "USAGE.txt"})
            self.assertEqual(z.read("TEST.DATA").decode(), deck)
            self.assertEqual(json.loads(z.read("model.json"))["grid_shape"], meta["grid_shape"])
            self.assertIn("synthetic_bhp_psi", z.read("predicted_bhp.csv").decode())

    def test_invalid_or_unresolved_inputs_fail_explicitly(self):
        for setup, p in ((D.DeckSetup(half_extent_ft=10), M.WellTestParams(model="mfhw")),
                         (D.DeckSetup(rock_compressibility=1), self.p),
                         (D.DeckSetup(nx=0), self.p),
                         (self.setup, M.WellTestParams(model="limited_entry", perf_pos=0.99)),
                         (self.setup, M.WellTestParams(model="composite", D=0.001, M=100))):
            with self.subTest(setup=setup, p=p), self.assertRaises(ValueError):
                D.grid_model(p, setup, "Oil")


class PressureComparison(unittest.TestCase):
    def test_nonphysical_oil_prediction_is_not_labelled_valid_bhp(self):
        curves = {"timeline": {"t": np.array([0, 72, 108]), "pressure": np.array([4000, -100, 3500])}}
        rows = list(csv.DictReader(io.StringIO(E.comparison_csv(
            curves, 100, 72, 36, "Field", "Oil", 4000))))
        self.assertEqual(rows[1]["synthetic_bhp_psi"], "")
        self.assertEqual(rows[1]["reference_status"], "nonphysical_oil_prediction")
        self.assertEqual(rows[1]["bottom_hole_pressure_psi"], "-100")

    def test_oil_reference_is_synthetic_not_a_control(self):
        p = M.WellTestParams()
        curves = viz.designed_test(p, 100, 72, 36, n=20)
        files = E.comparison_files("TEST", curves, 100, 72, 36, "Metric", "Oil", 100, p.pi)
        rows = list(csv.DictReader(io.StringIO(files["predicted_bhp.csv"])))
        self.assertEqual(rows[0]["synthetic_bhp_bar"], rows[0]["bottom_hole_pressure_bar"])
        self.assertEqual(rows[0]["reference_status"], "synthetic_prediction")
        self.assertIn("WBHP\n  'TEST' /", files["summary.INC"])
        self.assertNotIn("WCONHIST", files["schedule.INC"])
        self.assertIn("'ORAT'", files["schedule.INC"])
        steps = [expand(chunk.split("/", 1)[0]) for chunk in files["schedule.INC"].split("TSTEP\n")[1:]]
        self.assertEqual(len(steps[0]), len(curves["drawdown"]["t"]))
        self.assertAlmostEqual(sum(steps[0]) * 24, 72, places=8)
        self.assertAlmostEqual(sum(steps[1]) * 24, 36, places=8)

    def test_gas_inverse_never_silently_clamps_out_of_range(self):
        gas = {"T_F": 220, "gamma_g": 0.68}
        mapping = G.NormalizedPseudoPressure(4000, 220, 0.68, n=800)
        curves = {"timeline": {"t": np.array([0, 72, 73, 108]),
                               "pressure": np.array([4000, float(mapping.forward(3000)), mapping.pn[0] - 100, 4000])}}
        text = E.comparison_csv(curves, 6000, 72, 36, "Field", "Gas", 4000, gas)
        rows = list(csv.DictReader(io.StringIO(text)))
        self.assertAlmostEqual(float(rows[1]["synthetic_bhp_psi"]), 3000, places=7)
        invalid = next(row for row in rows if row["time_hr"] == "73")
        self.assertEqual(invalid["synthetic_bhp_psi"], "")
        self.assertEqual(invalid["reference_status"], "outside_gas_inverse_range")
        self.assertIn("normalized_pseudo_pressure_psi", rows[0])


if __name__ == "__main__":
    unittest.main()
