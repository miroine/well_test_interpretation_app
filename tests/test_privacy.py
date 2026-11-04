import ast
import pathlib
import sys
import tomllib
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine import laplace, report
from ui import privacy, session


class PrivacyTests(unittest.TestCase):
    def test_no_shared_user_data_caching(self):
        for path in [ROOT / "app.py", *(ROOT / "ui").glob("*.py")]:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    self.assertNotIn(node.attr, ("cache_data", "cache_resource"), str(path))
        self.assertFalse(hasattr(laplace._rect_images, "cache_info"))
        with (ROOT / ".streamlit" / "config.toml").open("rb") as source:
            self.assertIs(tomllib.load(source)["browser"]["gatherUsageStats"], False)

    def test_clear_removes_all_session_owned_data_and_preserves_other_sessions(self):
        first = {"dataset": {"name": "Private"}, "meta_well": "Private",
                 "proj_up": b"private upload", "_design_eclipse_deck": "private",
                 "custom_widget": "private"}
        other = {"dataset": {"name": "Other user"}}
        session.clear(first)
        self.assertEqual(first, {})
        self.assertEqual(other["dataset"]["name"], "Other user")
        session.initialize(first)
        self.assertIsNone(first["dataset"])

    def test_notice_and_report_disclaimer(self):
        with patch.object(privacy.st, "info") as info, \
                patch.object(privacy.st, "warning") as warning, \
                patch.object(privacy.st, "expander"), \
                patch.object(privacy.st, "markdown") as markdown:
            privacy.render()
        self.assertIn("no shared user-data cache", info.call_args.args[0])
        self.assertIn("Merouane Hamdani", warning.call_args.args[0])
        self.assertIn("MIT License", warning.call_args.args[0])
        self.assertIn("educational purposes only", warning.call_args.args[0])
        self.assertIn("outside the app's control", markdown.call_args.args[0])
        self.assertIn("For educational purposes only", report.build_html("Test", {}, []))

    def test_real_app_clear_resets_sensitive_widgets_and_project(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
        self.assertFalse(app.exception)
        app.text_input(key="meta_well").set_value("PRIVATE-WELL").run()
        app.session_state["dataset"] = {"name": "private"}
        app.session_state["_private_test_marker"] = b"private"
        app.button(key="privacy_clear").click().run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(app.text_input(key="meta_well").value, "Well-A")
        self.assertIsNone(app.session_state["dataset"])
        self.assertNotIn("_private_test_marker", app.session_state.filtered_state)
        self.assertTrue(any("Merouane Hamdani" in w.value for w in app.warning))
        self.assertTrue(any("session memory only" in i.value for i in app.info))


if __name__ == "__main__":
    unittest.main()
