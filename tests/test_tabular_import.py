import io
import pathlib
import sys
import struct
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine import tabular as T, project, units as U
from ui import import_data


def data_app():
    import streamlit as st
    from ui import session, tab_data
    session.initialize(st.session_state)
    tab_data.render({"units": "Metric", "fluid": "Oil"})


def rate_app():
    import streamlit as st
    from ui import session, tab_multirate
    session.initialize(st.session_state)
    tab_multirate.render({"units": "Metric", "fluid": "Oil"})


def rate_selector_app():
    import pandas as pd
    import streamlit as st
    from ui import import_data
    result = import_data.rate_input(pd.DataFrame({"time": [0], "pressure": [100]}),
                                     "selector", "Field")
    st.session_state["rate_result"] = result


def upload(content, name):
    value = io.BytesIO(content)
    value.name = name
    return value


class TableParsingTests(unittest.TestCase):
    def test_preamble_whitespace_duration_rates(self):
        frame = T.read_table(b"Duration   Rate\n(hours)  (units)\n10 800\n5 1100\n30 0\n",
                             "rates.txt", skiprows=2, header=False)
        schedule, end = T.rate_schedule(frame.iloc[:, 0], frame.iloc[:, 1], durations=True)
        self.assertEqual(schedule, [(0., 800.), (10., 1100.), (15., 0.)])
        self.assertEqual(end, 45.)
        np.testing.assert_array_equal(T.rates_at([0, 10, 14, 15, 45], schedule, end), [800, 1100, 1100, 0, 0])
        with self.assertRaisesRegex(ValueError, "cover"):
            T.rates_at([46], schedule, end)

    def test_delimiters_and_decimal_comma(self):
        for delimiter, sep in (("Comma", ","), ("Tab", "\t"), ("Semicolon", ";"), ("Whitespace", "   ")):
            text = f"skip this\nTime{sep}Pressure\n1{sep}100\n2{sep}101\n"
            frame = T.read_table(text.encode(), "gauge.dat", 1, True, delimiter)
            self.assertEqual(frame["Pressure"].tolist(), [100, 101])
        frame = T.read_table(b"Time;Pressure\n1,5;100,2\n", "gauge.csv", decimal=",")
        self.assertEqual(frame.iloc[0].tolist(), [1.5, 100.2])

    def test_xlsx_worksheet_skip_and_header(self):
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            pd.DataFrame([["Gauge", None], ["Time", "Pressure"], [0, 100], [1, 101]]).to_excel(
                writer, index=False, header=False, sheet_name="Pressure")
            pd.DataFrame([[2, 0], [3, 10]]).to_excel(writer, index=False, header=False, sheet_name="Rates")
        frame = T.read_table(out.getvalue(), "test.xlsx", 1, True, sheet="Pressure")
        self.assertEqual(frame["Pressure"].tolist(), [100, 101])
        self.assertEqual(T.read_table(out.getvalue(), "test.xlsx", header=False, sheet="Rates").shape, (2, 2))

    def test_legacy_xls_reader_routing(self):
        # A minimal, real BIFF2 worksheet exercises xlrd without a writer dependency.
        def record(code, value):
            return struct.pack("<HH", code, len(value)) + value
        content = record(0x0009, struct.pack("<HH", 2, 0x10))
        for row, values in enumerate(((0, 100), (1, 101), (2, 102))):
            for col, value in enumerate(values):
                content += record(0x0002, struct.pack("<HH3sH", row, col, bytes(3), value))
        content += record(0x000A, b"")
        frame = T.read_table(content, "test.xls", skiprows=1, header=False)
        self.assertEqual(frame.iloc[:, 1].tolist(), [101, 102])

    def test_time_units_and_calendar_origin(self):
        for unit, values in (("Hours", [0, 1]), ("Minutes", [0, 60]), ("Seconds", [0, 3600]), ("Days", [0, 1 / 24])):
            np.testing.assert_allclose(T.elapsed_hours(values, unit), [0, 1])
        np.testing.assert_array_equal(T.elapsed_hours(
            pd.Series(["2026-01-01 02:00", "2026-01-01 03:00"]),
            "Calendar date/time", "2026-01-01 00:00"), [2, 3])
        np.testing.assert_array_equal(T.elapsed_hours(
            ["02/01/2026 01:00", "02/01/2026 02:00"], "Calendar date/time",
            "02/01/2026 00:00", "%d/%m/%Y %H:%M", True), [1, 2])
        with self.assertRaisesRegex(ValueError, "shared test-start"):
            T.elapsed_hours(["2026-01-01"], "Calendar date/time")

    def test_pressure_durations_accumulate_before_mapping_and_rate_alignment(self):
        np.testing.assert_array_equal(T.pressure_hours([0, 12, 12, 24], durations=True), [0, 12, 24, 48])
        np.testing.assert_array_equal(T.pressure_hours([30, 30, 60], "Minutes", durations=True), [.5, 1, 2])
        np.testing.assert_array_equal(T.pressure_hours([3, 1, 2]), [3, 1, 2])
        frame = pd.DataFrame({"interval": [0, 1, 1, 2], "p": [200, 190, 180, 185]})
        mapped = import_data.mapped_frame(frame, "interval", "p", {"unit": "Hours", "durations": True},
                                         {"schedule": [(0., 100.), (2., 0.)], "end": 5.})
        np.testing.assert_array_equal(mapped["time"], [0, 1, 2, 4])
        np.testing.assert_array_equal(mapped["rate"], [100, 100, 0, 0])
        for intervals in ([0, 0, 1], [-1, 1], [1, -1], [1, float("nan")]):
            with self.assertRaises(ValueError):
                T.pressure_hours(intervals, durations=True)
        with self.assertRaisesRegex(ValueError, "numeric time units"):
            T.pressure_hours(["2026-01-01"], "Calendar date/time", durations=True)

    def test_invalid_times_rates_and_layout_report_errors(self):
        for times, rates, durations in (([0, 0], [1, 0], False), ([1], [1], False),
                                        ([0], [1], True), ([0], [-1], False),
                                        ([0], [np.nan], False)):
            with self.assertRaises(ValueError):
                T.rate_schedule(times, rates, durations)
        for values in ([1, ""], [1, np.inf]):
            with self.assertRaises(ValueError):
                T.elapsed_hours(values)
        with self.assertRaises(ValueError):
            T.read_table(b"one\ncolumn\n", "p.txt")
        for extension in ("xlsx", "xls"):
            with self.assertRaisesRegex(ValueError, "Cannot read table"):
                T.read_table(b"not an Excel workbook", f"broken.{extension}")
        with self.assertRaises(ValueError):
            import_data.mapped_frame(pd.DataFrame({"x": [1], "y": [2]}), "x", "x", {"unit": "Hours"})

    def test_unsampled_rate_events_preserved_and_shutin_validated(self):
        schedule = [(0., 800.), (1., 900.), (2., 0.)]
        self.assertEqual(T.final_shutin_rate(schedule, 2.), 900.)
        with self.assertRaisesRegex(ValueError, "shut in"):
            T.final_shutin_rate(schedule + [(2.5, 100.), (2.6, 0.)], 2.)
        with self.assertRaisesRegex(ValueError, "positive"):
            T.final_shutin_rate(schedule + [(3., 0.)], 3.)
        self.assertEqual(T.final_shutin_rate([(0., 800.), (.1 + .2, 0.)], .3), 800.)


class ImportJourneys(unittest.TestCase):
    def check(self, app):
        app.run()
        self.assertFalse(app.exception, [e.message for e in app.exception])

    def test_pressure_duration_mode_imports_accumulated_reading_times(self):
        with patch.object(st, "file_uploader", side_effect=lambda *a, **k: upload(
                b"interval,pressure\n30,100\n30,101\n60,102\n", "pressure.csv")):
            app = AppTest.from_function(data_app, default_timeout=120).run()
            app.radio(key="data_src_0").set_value("Upload files")
            self.check(app)
            app.radio(key="qc_import_pressure_time_meaning").set_value("Durations between readings")
            app.selectbox(key="qc_import_time_unit").select("Minutes")
            self.check(app)
            self.assertNotIn("Calendar date/time", app.selectbox(key="qc_import_time_unit").options)
            next(b for b in app.button if b.label == "Use this file").click()
            self.check(app)
        self.assertFalse(app.error, [e.value for e in app.error])
        np.testing.assert_array_equal(app.session_state["dataset"]["dt"], [.5, 1., 2.])

    def test_repeated_durations_can_switch_from_actual_times_without_reupload(self):
        content = b"Duration Liquid-rate\n(hr) (STB/D)\n12 1000\n12 1200\n12 1400\n48 1500\n72 0\n"
        with patch.object(st, "file_uploader", side_effect=lambda *a, **k: upload(content, "rates.txt")):
            app = AppTest.from_function(rate_selector_app, default_timeout=120).run()
            app.radio(key="selector_source").set_value("Separate rate file")
            self.check(app)
            app.number_input(key="selector_upload_skip").set_value(2)
            app.checkbox(key="selector_upload_header").uncheck()
            self.check(app)
            self.assertTrue(any("select Duration" in e.value for e in app.error))
            self.assertIn("Interpret rate time", app.radio(key="selector_meaning").label)
            app.radio(key="selector_meaning").set_value("Period durations")
            self.check(app)
            self.assertFalse(app.error, [e.value for e in app.error])
            result = app.session_state["rate_result"]
            self.assertEqual(result["schedule"], [(0., 1000.), (12., 1200.), (24., 1400.),
                                                  (36., 1500.), (84., 0.)])
            self.assertEqual(result["end"], 156.)
            app.radio(key="selector_meaning").set_value("Period start times")
            self.check(app)
            self.assertIsNone(app.session_state["rate_result"])

    def test_separate_duration_rate_file_and_metric_pressure(self):
        pressure = b"Gauge export\nTime Pressure\n0 200\n1 190\n2 180\n3 181\n4 182\n5 183\n6 184\n"
        rates = b"Duration Rate\nhr units\n2 100\n5 0\n"
        def uploader(*args, **kwargs):
            return upload(rates, "rates.txt") if kwargs["key"] == "qc_rates_upload" else upload(pressure, "pressure.dat")
        with patch.object(st, "file_uploader", side_effect=uploader):
            app = AppTest.from_function(data_app, default_timeout=120).run()
            app.radio(key="data_src_0").set_value("Upload files")
            self.check(app)
            app.number_input(key="qc_pressure_upload_skip").set_value(1)
            app.checkbox(key="qc_upload_full").check()
            self.check(app)
            app.radio(key="qc_rates_source").set_value("Separate rate file")
            self.check(app)
            app.number_input(key="qc_rates_upload_skip").set_value(2)
            app.checkbox(key="qc_rates_upload_header").uncheck()
            app.radio(key="qc_rates_meaning").set_value("Period durations")
            self.check(app)
            self.assertTrue(any(m.label == "Flow period before shut-in [hr]" and m.value == "2"
                                for m in app.metric))
            next(b for b in app.button if b.label == "Use this file").click()
            self.check(app)
        self.assertFalse(app.error, [e.value for e in app.error])
        ds = app.session_state["dataset"]
        np.testing.assert_array_equal(ds["dt"], [1, 2, 3, 4])
        np.testing.assert_allclose(ds["history"]["rate"], U.to_field(np.array([100, 100, 0, 0, 0, 0, 0]), "rate", "Metric"))
        self.assertEqual([s[0] for s in ds["rate_schedule"]], [0., 2.])
        _, restored = project.loads(project.dumps({}, app.session_state.filtered_state))
        self.assertEqual(restored["dataset"]["rate_schedule"], ds["rate_schedule"])

    def test_multirate_manual_table_uses_unsampled_event_times(self):
        def uploader(*args, **kwargs):
            return upload(b"time;pressure\n0;200\n1;190\n3;180\n4;181\n5;182\n", "pressure.csv")
        original_editor = st.data_editor
        def editor(data, **kwargs):
            if kwargs.get("key") == "mr_rates_table_start":
                return pd.DataFrame({"Time": [0., 2.], "Rate": [100., 0.]})
            return original_editor(data, **kwargs)
        with patch.object(st, "file_uploader", side_effect=uploader), patch.object(st, "data_editor", side_effect=editor):
            app = AppTest.from_function(rate_app, default_timeout=120).run()
            app.radio(key="mr_src_0").set_value("Upload pressure and rates")
            self.check(app)
            app.radio(key="mr_rates_source").set_value("Enter rate table")
            self.check(app)
            next(b for b in app.button if b.label == "Use this file").click()
            self.check(app)
        self.assertFalse(app.error, [e.value for e in app.error])
        mr = app.session_state["mr"]
        self.assertEqual(mr["schedule"][1], (2., 0.))
        np.testing.assert_array_equal(mr["t"], [1, 3, 4, 5])
        self.assertAlmostEqual(mr["schedule"][0][1], float(U.to_field(100., "rate", "Metric")))

    def test_calendar_pressure_preview_and_buildup_import(self):
        def uploader(*args, **kwargs):
            return upload(b"stamp,pressure\n2026-01-01 01:00,100\n2026-01-01 02:00,101\n2026-01-01 03:00,102\n", "gauge.tsv")
        with patch.object(st, "file_uploader", side_effect=uploader):
            app = AppTest.from_function(data_app, default_timeout=120).run()
            app.radio(key="data_src_0").set_value("Upload files")
            self.check(app)
            app.selectbox(key="qc_import_time_unit").select("Calendar date/time")
            self.check(app)
            app.text_input(key="qc_import_origin").set_value("2026-01-01 00:00")
            self.check(app)
            next(b for b in app.button if b.label == "Use this file").click()
            self.check(app)
        np.testing.assert_array_equal(app.session_state["dataset"]["dt"], [1, 2, 3])

    def test_invalid_rate_coverage_does_not_replace_active_record(self):
        original_editor = st.data_editor
        def editor(data, **kwargs):
            if kwargs.get("key") == "mr_rates_table_duration":
                return pd.DataFrame({"Time": [1., 1.], "Rate": [100., 0.]})
            return original_editor(data, **kwargs)
        with patch.object(st, "file_uploader", return_value=upload(
                b"time,pressure\n0,200\n1,190\n3,191\n4,192\n5,193\n", "pressure.csv")), \
                patch.object(st, "data_editor", side_effect=editor):
            app = AppTest.from_function(rate_app, default_timeout=120).run()
            app.radio(key="mr_src_0").set_value("Upload pressure and rates")
            self.check(app)
            app.radio(key="mr_rates_source").set_value("Enter rate table")
            self.check(app)
            app.radio(key="mr_rates_meaning").set_value("Period durations")
            self.check(app)
            next(b for b in app.button if b.label == "Use this file").click()
            self.check(app)
        self.assertTrue(any("does not cover" in e.value for e in app.error))
        self.assertIsNone(app.session_state["mr"])

    def test_alignment_drag_updates_imported_shutin_and_schedule_not_pressure(self):
        pressure = b"time,pressure\n0,200\n1,190\n2,180\n3,181\n4,182\n5,183\n6,184\n"
        def uploader(*args, **kwargs):
            return upload(b"duration,rate\n2,100\n5,0\n", "rates.csv") if kwargs["key"] == "qc_rates_upload" else upload(pressure, "pressure.csv")
        with patch.object(st, "file_uploader", side_effect=uploader):
            app = AppTest.from_function(data_app, default_timeout=120).run()
            app.radio(key="data_src_0").set_value("Upload files")
            self.check(app)
            app.checkbox(key="qc_upload_full").check()
            self.check(app)
            app.radio(key="qc_rates_source").set_value("Separate rate file")
            self.check(app)
            app.radio(key="qc_rates_meaning").set_value("Period durations")
            self.check(app)
            draft = app.session_state["qc_import_alignment"]
            state = {"draft": draft}
            import_data.apply_alignment(state, "draft", {"base": draft["schedule"], "index": 1, "start": 2.5})
            app.session_state["qc_import_alignment"] = state["draft"]
            self.check(app)
            self.assertTrue(any(m.label == "Flow period before shut-in [hr]" and m.value == "2.5"
                                for m in app.metric))
            next(b for b in app.button if b.label == "Use this file").click()
            self.check(app)
        ds = app.session_state["dataset"]
        self.assertEqual(ds["tp"], 2.5)
        self.assertEqual(ds["rate_schedule"][1], (2.5, 0.))
        np.testing.assert_array_equal(ds["dt"], [.5, 1.5, 2.5, 3.5])
        np.testing.assert_allclose(ds["history"]["p"], U.to_field(np.array([200,190,180,181,182,183,184]), "pressure", "Metric"))
        with self.assertRaisesRegex(ValueError, "changed"):
            import_data.apply_alignment(state, "draft", {"base": draft["schedule"], "index": 1, "start": 3.})
        with self.assertRaisesRegex(ValueError, "ordered"):
            import_data.apply_alignment(state, "draft", {"base": state["draft"]["schedule"], "index": 1, "start": 8.})


if __name__ == "__main__":
    unittest.main()
