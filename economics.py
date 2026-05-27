"""
modules/economics.py
--------------------
Well-test economic exposure calculator.
Operational cost + deferred production revenue loss.
Pure-math (returns dicts); UI consumes the output.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class EconomicsInputs:
    rig_day_rate: float        # USD / day
    total_days: float          # days (mob + test + demob)
    fixed_service_fees: float  # USD (logging, gauges, slickline, lab, etc.)
    deferred_rate: float       # STB/D — production lost during shut-in / test
    test_duration_hr: float    # hr   — period during which production is deferred
    oil_price: float           # USD / STB


def compute_costs(e: EconomicsInputs) -> dict:
    """
    Operational cost  = rig_day_rate * total_days + fixed_service_fees
    Deferred revenue  = deferred_rate * (test_duration_hr / 24) * oil_price
    """
    op_cost = e.rig_day_rate * e.total_days + e.fixed_service_fees
    deferred_bbl = e.deferred_rate * (e.test_duration_hr / 24.0)
    deferred_loss = deferred_bbl * e.oil_price
    total = op_cost + deferred_loss

    return {
        "rig_cost": e.rig_day_rate * e.total_days,
        "service_fees": e.fixed_service_fees,
        "operational_cost": op_cost,
        "deferred_bbl": deferred_bbl,
        "deferred_loss": deferred_loss,
        "total_exposure": total,
        "breakdown": {
            "Rig Day Rate": e.rig_day_rate * e.total_days,
            "Service Fees": e.fixed_service_fees,
            "Deferred Production Revenue": deferred_loss,
        },
    }
