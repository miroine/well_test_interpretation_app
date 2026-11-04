"""Shared session lifecycle for project loading and dataset replacement."""
from __future__ import annotations

import datetime

from engine import project

DEFAULTS = {
    "ver": 0, "dataset": None, "fits": [], "design_params": None, "design_eng": None,
    "last_fit": None, "interp_eng": None, "interp_model": None, "tc_match": None, "mr": None, "deconv": None,
    "horner": None, "fmb": None, "sim": None, "sim_fit": None, "sim_inputs": None,
    "sim_flow": None, "areal": None, "interf": None, "wbfx": None, "gauge": None,
    "mc": None, "sens": None, "ipr_inputs": None, "fmb_inputs": None,
}
DATASET_RESULTS = ("last_fit", "interp_eng", "interp_model", "horner", "tc_match", "mc", "sens")


def clear(state):
    """Drop this session's project, results, metadata and widget references."""
    for key in list(state):
        del state[key]


def initialize(state):
    for key, value in DEFAULTS.items():
        state.setdefault(key, [] if isinstance(value, list) else value)


def replace_dataset(state, dataset):
    state["dataset"] = dataset
    for key in DATASET_RESULTS:
        state[key] = None
    state.pop("_interp_pending", None)
    state["ver"] += 1


def restore(state, meta, saved):
    """Replace project-owned state; never merge old results into a new project."""
    metadata = {"well": "Well-A", "field": "Field-X", "analyst": "", "fluid": "Oil",
                "units": "Field", "date": datetime.date.today()}
    for key in ("well", "field", "analyst"):
        if key in meta:
            metadata[key] = str(meta[key])
    for key, options in (("fluid", ("Oil", "Gas")), ("units", ("Field", "Metric"))):
        if key in meta:
            if meta[key] not in options:
                raise ValueError(f"Invalid project {key}: {meta[key]}")
            metadata[key] = meta[key]
    if "date" in meta:
        metadata["date"] = datetime.date.fromisoformat(meta["date"])
    version = state.get("ver", 0) + 1
    state.pop("_interp_pending", None)
    for key, default in DEFAULTS.items():
        value = saved.get(key) if key in project.KEYS else None
        state[key] = value if value is not None else ([] if isinstance(default, list) else default)
    state["ver"] = version
    for key, value in metadata.items():
        state[f"meta_{key}"] = value
