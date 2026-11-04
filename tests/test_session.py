import datetime
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from engine import project
from ui import session


class ProjectLifecycleTests(unittest.TestCase):
    def test_roundtrip_replaces_results_and_metadata(self):
        state = {}
        session.initialize(state)
        state.update(sim={"old": True}, horner={"k": 999}, fits=[{"old": True}], ver=4)
        text = project.dumps(
            {"well": "New well", "fluid": "Gas", "units": "Metric", "date": "2026-11-04"},
            {"fits": [], "design_eng": {"tp": 24.0}, "fmb": {"fit": {"N": 2e7}},
             "interp_model": {"params": {"k": 85.0}, "gas": None}},
        )
        meta, saved = project.loads(text)
        session.restore(state, meta, saved)
        self.assertIsNone(state["sim"])
        self.assertIsNone(state["horner"])
        self.assertEqual(state["fits"], [])
        self.assertEqual(state["design_eng"]["tp"], 24.0)
        self.assertEqual(state["fmb"]["fit"]["N"], 2e7)
        self.assertEqual(state["interp_model"]["params"]["k"], 85.0)
        self.assertEqual(state["meta_units"], "Metric")
        self.assertEqual(state["meta_date"], datetime.date(2026, 11, 4))
        self.assertEqual(state["ver"], 5)

    def test_dataset_replacement_invalidates_derived_results(self):
        state = {}
        session.initialize(state)
        for key in session.DATASET_RESULTS:
            state[key] = {"stale": True}
        state["_interp_pending"] = {"params": {"k": 999}}
        session.replace_dataset(state, {"name": "new"})
        self.assertEqual(state["dataset"]["name"], "new")
        self.assertTrue(all(state[key] is None for key in session.DATASET_RESULTS))
        self.assertNotIn("_interp_pending", state)
        self.assertEqual(state["ver"], 1)

    def test_invalid_metadata_does_not_partially_replace_state(self):
        state = {"dataset": {"name": "original"}, "ver": 3}
        with self.assertRaises(ValueError):
            session.restore(state, {"fluid": "Water"}, {})
        self.assertEqual(state, {"dataset": {"name": "original"}, "ver": 3})

    def test_malformed_projects_are_rejected(self):
        for value in ([], None, {"version": "3", "state": {}}, {"state": []},
                      {"version": project.VERSION + 1, "state": {}}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                project.loads(json.dumps(value))

    def test_unknown_state_is_not_restored(self):
        _, state = project.loads('{"version": 3, "state": {"ver": -1, "meta_units": "bad", "fits": []}}')
        self.assertEqual(state, {"fits": []})


if __name__ == "__main__":
    unittest.main()
