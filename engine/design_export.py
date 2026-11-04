"""Portable exports of an analytical test design; no additional solver calls."""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile

import numpy as np

from . import units as U
from . import gas as G


def _validate(q, tp, dt_end, system, fluid):
    if system not in ("Field", "Metric") or fluid not in ("Oil", "Gas"):
        raise ValueError("Select Field or Metric units and Oil or Gas fluid.")
    if not all(np.isfinite(v) for v in (q, tp, dt_end)) or q < 0 or tp <= 0 or dt_end <= 0:
        raise ValueError("Surface rate must be finite and non-negative; both durations must be positive.")


def design_csv(curves, q_user, tp, dt_end, system, fluid):
    """Export chronological samples, including both sides of the shut-in event.

    q_user is the surface rate in field units, not the gas reservoir-equivalent rate.
    """
    _validate(q_user, tp, dt_end, system, fluid)
    timeline = curves["timeline"]
    t = np.asarray(timeline["t"], float)
    pressure = U.from_field(np.asarray(timeline["pressure"], float), "pressure", system)
    q = float(U.from_field(q_user, "gasrate" if fluid == "Gas" else "rate", system))
    quantity = "normalized_pseudo_pressure" if fluid == "Gas" else "bottom_hole_pressure"
    rate_unit = {("Oil", "Field"): "stb_per_day", ("Oil", "Metric"): "m3_per_day",
                 ("Gas", "Field"): "Mscf_per_day", ("Gas", "Metric"): "Sm3_per_day"}[fluid, system]
    out = io.StringIO(newline="")
    writer = csv.writer(out)
    writer.writerow(["time_hr", "phase", "shut_in_elapsed_hr",
                     f"surface_rate_{rate_unit}",
                     f"{quantity}_{U.label('pressure', system)}"])
    for time, value in zip(t, pressure):
        phase = "drawdown" if time <= tp else "buildup"
        writer.writerow([f"{time:.15g}", phase, f"{max(0.0, time - tp):.15g}",
                         f"{q if phase == 'drawdown' else 0.0:.15g}", f"{value:.15g}"])
        if time == tp:
            writer.writerow([f"{time:.15g}", "buildup", "0", "0", f"{value:.15g}"])
    return out.getvalue()


def eclipse_schedule(well, q_user, tp, dt_end, system, fluid, minimum_bhp):
    """SCHEDULE-section INCLUDE for a pre-existing, completed producer.

    Surface rate and minimum_bhp are supplied in field units; output follows
    the host deck's selected FIELD or METRIC convention. TSTEP is always days.
    """
    _validate(q_user, tp, dt_end, system, fluid)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,7}", well):
        raise ValueError("Eclipse well name must be 1-8 characters: start with a letter, then letters, digits, _ or -.")
    if not np.isfinite(minimum_bhp) or minimum_bhp <= 0:
        raise ValueError("Minimum BHP must be finite and positive.")
    rate = float(U.from_field(q_user, "gasrate" if fluid == "Gas" else "rate", system))
    bhp = float(U.from_field(minimum_bhp, "pressure", system))
    control = f"'ORAT' {rate:.15g} 4*" if fluid == "Oil" else f"'GRAT' 2* {rate:.15g} 2*"
    opening = (f"WELOPEN\n  '{well}' 'OPEN' /\n/\n"
               f"WCONPROD\n  '{well}' 'OPEN' {control} {bhp:.15g} /\n/\n") if rate > 0 else (
                   f"WELOPEN\n  '{well}' 'SHUT' /\n/\n")
    return (
        "-- Well Test Workbench: designed flow / shut-in test\n"
        f"-- Host deck must use {'FIELD' if system == 'Field' else 'METRIC'} units. TSTEP values are days.\n"
        "-- INCLUDE inside SCHEDULE at the intended test start; do not add another SCHEDULE keyword.\n"
        "-- Well and completions must already exist. No grid, PVT or fracture geometry is exported.\n"
        "-- Minimum BHP is a simulator constraint, not the analytical predicted pressure.\n"
        "-- BHP constraints can prevent the requested rate being achieved.\n"
        "-- Set simulator timestep/report controls as needed to resolve pressure transients.\n"
        + opening
        + f"TSTEP\n  {tp / 24:.15g} /\n"
        + f"WELOPEN\n  '{well}' 'SHUT' /\n/\n"
        + f"TSTEP\n  {dt_end / 24:.15g} /\n"
        + "-- End of test: well remains SHUT; host schedule may continue.\n"
    )


def comparison_csv(curves, q_user, tp, dt_end, system, fluid, pi, gas=None):
    """Synthetic reference BHP, never measured data or a simulator control."""
    rows = list(csv.DictReader(io.StringIO(design_csv(curves, q_user, tp, dt_end, system, fluid))))
    quantity = "normalized_pseudo_pressure" if fluid == "Gas" else "bottom_hole_pressure"
    pressure_key = f"{quantity}_{U.label('pressure', system)}"
    predicted = np.array([float(row[pressure_key]) for row in rows])
    physical = predicted.copy()
    valid = np.isfinite(predicted) & (predicted > 0) if fluid == "Oil" else np.ones(len(rows), dtype=bool)
    if fluid == "Gas":
        if gas is None:
            raise ValueError("Gas temperature and specific gravity are required for synthetic BHP comparison.")
        if not np.isfinite(pi) or pi <= 14.7:
            raise ValueError("Gas BHP inversion requires initial pressure above 14.7 psia.")
        mapping = G.NormalizedPseudoPressure(pi, gas["T_F"], gas["gamma_g"], n=800)
        pn = U.to_field(predicted, "pressure", system)
        valid = (pn >= mapping.pn[0]) & (pn <= mapping.pn[-1])
        physical = U.from_field(mapping.inverse(pn), "pressure", system)
    bhp_key = f"synthetic_bhp_{U.label('pressure', system)}"
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]) + [bhp_key, "reference_status"])
    writer.writeheader()
    for row, value, ok in zip(rows, physical, valid):
        row[bhp_key] = f"{value:.15g}" if ok else ""
        row["reference_status"] = ("synthetic_prediction" if fluid == "Oil" else
                                   "synthetic_gas_pseudopressure_inverse") if ok else (
                                       "nonphysical_oil_prediction" if fluid == "Oil" else "outside_gas_inverse_range")
        writer.writerow(row)
    return out.getvalue()


def comparison_files(well, curves, q_user, tp, dt_end, system, fluid, minimum_bhp, pi, gas=None):
    schedule = eclipse_schedule(well, q_user, tp, dt_end, system, fluid, minimum_bhp)
    prefix, flow, shut = schedule.split("TSTEP\n")
    t = np.asarray(curves["timeline"]["t"], float)
    flow_steps = np.diff(np.concatenate([[0.0], t[(t > 0) & (t <= tp)]])) / 24
    shut_steps = np.diff(np.concatenate([[tp], t[t > tp]])) / 24
    def steps(values):
        return "TSTEP\n  " + "\n  ".join(" ".join(f"{v:.15g}" for v in values[i:i + 8])
                                        for i in range(0, len(values), 8)) + " /\n"
    schedule = (prefix + steps(flow_steps) + flow.split("/", 1)[1] +
                steps(shut_steps) + shut.split("/", 1)[1])
    reference = comparison_csv(curves, q_user, tp, dt_end, system, fluid, pi, gas)
    rate_keyword = "WGPR" if fluid == "Gas" else "WOPR"
    summary = (f"-- INCLUDE inside the host SUMMARY section, not SCHEDULE.\n"
               f"WBHP\n  '{well}' /\n{rate_keyword}\n  '{well}' /\n")
    manifest = {
        "reference_kind": "synthetic analytical prediction; not measured field data",
        "well": well, "units": system, "fluid": fluid,
        "flow_hr": tp, "shut_in_hr": dt_end,
        "instructions": [
            "Include summary.INC inside SUMMARY and schedule.INC inside SCHEDULE at test start.",
            "BHP reference is comparison data only; it is not WCONHIST or a BHP control.",
            "WBHP reports simulator well BHP. Verify its datum matches the analytical well reference.",
            "Reference time_hr is relative to test start; simulator TIME is days from host-deck start.",
            "Subtract the host test-start TIME and multiply by 24 before aligning results.",
            "Shut-in timestamp occurs twice to preserve both sides of the rate discontinuity.",
            "The comparison schedule reports at the analytical reference sample times in both periods.",
            "SHUT closes the simulator well; shut-well WBHP is simulator-dependent. Inspect "
            "near-well block pressure if WBHP is unavailable or frozen during shut-in.",
            "Gas BHP uses the inverse normalized-pseudo-pressure table, not a new gas-flow solve. "
            "Out-of-range predictions have blank BHP and an explicit reference_status; no extrapolation.",
        ],
    }
    return {"schedule.INC": schedule, "summary.INC": summary, "predicted_bhp.csv": reference,
            "comparison.json": json.dumps(manifest, indent=2, allow_nan=False)}


def zip_files(files):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return out.getvalue()
