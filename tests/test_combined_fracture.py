import pathlib
import sys
import unittest

import numpy as np
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from engine import models as M, laplace as L, multirate as MR, project, eclipse_design as E
from ui import helpers as H


def combined_app():
    import numpy as np
    import streamlit as st
    from ui import session, tab_multirate
    from engine import models as M
    session.initialize(st.session_state)
    if not st.session_state["dataset"]:
        p = M.WellTestParams(model="fracture_dual_pss")
        times = np.r_[0., 1., 2., 3., 4., 5., 6.]
        schedule = [(0., 1000.), (2., 1500.), (3., 0.)]
        st.session_state["dataset"] = {"name": "Measured test", "fluid": "Oil", "guess": p.to_dict(),
            "history": {"t": times, "p": M.pwf_history(p, times, schedule),
                        "rate": np.array([1000., 1000., 1500., 0., 0., 0., 0.])},
            "rate_schedule": schedule}
    tab_multirate.render({"fluid": "Oil", "units": "Field"})


class CombinedFractureTests(unittest.TestCase):
    def test_kernel_converts_lambda_reference_length(self):
        p = M.WellTestParams(model="fracture_dual_pss", rw=.5, xf=100., lam=2e-6)
        u = np.geomspace(1e-5, 1e5, 80)
        expected = L.pws_fracture(u, lambda v: L.f_dual_pss(v, p.omega, p.lam * (p.xf / p.rw) ** 2),
                                  M.CD_of(p), p.S)
        np.testing.assert_array_equal(M.laplace_function(p)(u), expected)
        self.assertEqual(M._scale_length(p), p.xf)

    def test_equilibrated_limit_matches_homogeneous_fracture(self):
        combined = M.WellTestParams(model="fracture_dual_pss", lam=1e8)
        radial = H.clone(combined, model="fracture")
        t = np.geomspace(.001, 100, 40)
        np.testing.assert_allclose(M.pwf_history(combined, t, [(0., 1000.)]),
                                   M.pwf_history(radial, t, [(0., 1000.)]), atol=1e-4, rtol=0)

    def test_prepared_response_and_additional_skin_guard(self):
        p = M.WellTestParams(model="fracture_dual_pss", S=.2, omega=.08, lam=6e-6)
        t = np.geomspace(.001, 50., 40)
        schedule = [(0., 1000.), (12., 1500.), (24., 0.)]
        evaluate = M.prepare_pwf_history(t, schedule)
        np.testing.assert_allclose(evaluate(p), M.pwf_history(p, t, schedule), atol=1e-8, rtol=0)
        negative = H.clone(p, S=-1.)
        for response in (evaluate, lambda params: M.pwf_history(params, t, schedule)):
            with self.assertRaisesRegex(ValueError, "non-negative additional skin"):
                response(negative)

    def test_dataset_transfer_preserves_independent_schedule_and_qc(self):
        p = M.WellTestParams(model="fracture_dual_pss")
        ds = {"name": "Test", "fluid": "Oil",
              "history": {"t": np.array([0., 1., 2., 2., 3., 4., 5.]),
                          "p": np.array([4000., 3900., 3800., 3801., 3850., 3900., 3950.]),
                          "rate": np.array([1000., 1000., 1000., 0., 0., 0., 0.])},
              "rate_schedule": [(0., 1000.), (1.5, 1500.), (2., 0.)], "rate_end": 6.}
        mr = MR.from_dataset(ds, p)
        self.assertEqual(mr["schedule"], ds["rate_schedule"])
        np.testing.assert_array_equal(mr["t"], [1, 2, 3, 4, 5])
        self.assertEqual(mr["pwf"][1], 3801.)
        ds["qc_source"] = {"history": ds["history"]}
        ds["qc_excluded"] = [5]
        ds["qc_window"] = [0., 5.]
        mr = MR.from_dataset(ds, p)
        np.testing.assert_array_equal(mr["t"], [1, 2, 3, 5])
        _, state = project.loads(project.dumps({}, {"mr": mr}))
        self.assertEqual(state["mr"]["guess"]["model"], "fracture_dual_pss")

    def test_combined_specs_and_export_limitations(self):
        free = H.fit_free_params("fracture_dual_pss", "none")
        self.assertTrue({"xf", "omega", "lam", "S"} <= set(free))
        p = M.WellTestParams(model="fracture_dual_pss")
        notes = E.assumptions(p, "Oil")
        self.assertIn("DUAL-POROSITY LIMITATION", " ".join(notes))
        self.assertIn("Infinite fracture conductivity", " ".join(notes))

    def test_active_record_and_forward_model_in_real_ui(self):
        app = AppTest.from_function(combined_app, default_timeout=120).run()
        app.radio(key="mr_src_0").set_value("Active Data & QC record").run()
        app.button(key="mr_from_dataset_0").click().run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(app.session_state["mr"]["name"], "Measured test")
        self.assertTrue({"xf", "omega", "lam"} <= set(app.multiselect(
            key="mr_free_1_fracture_dual_pss_none").value))
        app.button(key="mr_generate_1").click().run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertIsNotNone(app.session_state["mr"]["working_model"])
        self.assertTrue(any("Whole-history pressure RMS" in c.value for c in app.caption))


if __name__ == "__main__":
    unittest.main()
