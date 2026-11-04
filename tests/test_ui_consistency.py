import pathlib
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from engine import analysis, models, viz
from ui import figures, helpers, report_ui, tab_interp, tab_info, theme


class InterfaceConsistencyTests(unittest.TestCase):
    def test_documentation_replaces_roadmap_with_implemented_formulas(self):
        with patch.object(tab_info.st, "markdown") as markdown:
            tab_info.documentation()
        text = markdown.call_args.args[0]
        for expected in ("Documentation & formulas", "0.0002637", "0.8936", "162.6",
                         "Bourdet", "P10/P50/P90", "FIELD/METRIC", "rate-superposition"):
            self.assertIn(expected, text)
        source = (pathlib.Path(__file__).resolve().parents[1] / "app.py").read_text()
        self.assertIn('"Documentation & formulas"', source)
        self.assertNotIn('"Roadmap"', source)
        self.assertIn("tab_info.documentation()", source)

    def test_every_default_fit_parameter_is_selectable(self):
        options = set(tab_interp.LABELS) & set(analysis.FITTABLE)
        for model in helpers.MODEL_LABELS:
            for boundary in helpers.BOUNDARY_LABELS:
                with self.subTest(model=model, boundary=boundary):
                    self.assertLessEqual(set(helpers.fit_free_params(model, boundary)), options)

    def test_takeaway_renders_text_not_markup(self):
        with patch.object(theme.st, "markdown") as markdown:
            theme.takeaway('Well <A> & partners', label="<Result>")
        content = markdown.call_args.args[0]
        self.assertIn("Well &lt;A&gt; &amp; partners", content)
        self.assertIn("&lt;Result&gt;", content)

    def test_chart_has_explicit_readable_house_style(self):
        fig = figures.line_figure("Pressure", "Time", "Pressure", [("Measured", [1, 2], [10, 11])])
        self.assertEqual(fig.layout.font.color, theme.NAVY)
        self.assertEqual(fig.layout.paper_bgcolor, "#fff")
        self.assertEqual(fig.data[0].line.color, theme.BLUE)
        self.assertEqual(fig.layout.xaxis.type, "linear")

    def test_typecurve_highlight_uses_plotly_boolean(self):
        x = np.array([1.0, 10.0])
        fig = figures.typecurve_figure({1.0: (x, x, x), 100.0: (x, x, x)}, highlight=1.0)
        self.assertIs(fig.data[0].showlegend, True)
        self.assertIs(fig.data[2].showlegend, False)

    def test_typecurve_log_traces_preserve_invalid_gaps(self):
        x = np.array([.01, .1, 1., 10.])
        pressure = np.array([.01, -2., 1., 2.])
        derivative = np.array([.01, .1, np.inf, .5])
        fig = figures.typecurve_figure({1.: (x, pressure, derivative)})
        self.assertTrue(np.isnan(fig.data[0].y[1]))
        self.assertTrue(np.isnan(fig.data[1].y[2]))
        self.assertFalse(fig.data[0].connectgaps)
        self.assertFalse(fig.data[1].connectgaps)
        np.testing.assert_array_equal(fig.data[0].x, x)

    def test_report_does_not_present_wrong_fluid_interpretation(self):
        meta = {"well": "W", "field": "F", "date": "2026-11-04",
                "analyst": "", "units": "Metric", "fluid": "Gas"}
        state = {"dataset": {"name": "Oil test", "fluid": "Oil"},
                 "last_fit": {"params": {}}, "horner": {"k": 999}}
        html = report_ui.build(meta, state, "Interpretation + simulation")
        self.assertIn("Dataset fluid mismatch", html)
        self.assertNotIn("Analytical regression", html)
        self.assertNotIn("Semilog analysis:", html)

    def test_report_reuses_current_design_without_resolving(self):
        meta = {"well": "W", "field": "F", "date": "2026-11-04",
                "analyst": "", "units": "Field", "fluid": "Oil"}
        params = models.WellTestParams().to_dict()
        state = {"design_eng": {"params": params, "q": 1500, "tp": 200, "dt_end": 100},
                 "design_params": params}
        curves = viz.designed_test(models.WellTestParams(), 1500, 200, 100)["buildup"]
        expected = report_ui.build(meta, state, "Test design only")
        with patch.object(viz, "model_curves", side_effect=AssertionError("Duplicate solve")):
            actual = report_ui.build(meta, state, "Test design only", design_curves=curves)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
