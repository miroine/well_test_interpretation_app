"""Project save/open (JSON). Numpy arrays are stored as lists."""
from __future__ import annotations

import json

import numpy as np

VERSION = 3
KEYS = ("dataset", "fits", "design_params", "design_eng", "last_fit", "interp_model", "mr", "deconv",
        "ipr_inputs", "fmb_inputs", "fmb", "sim_inputs", "sim", "sim_fit")


def _enc(o):
    if isinstance(o, np.ndarray):
        return {"__nd__": o.tolist()}
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, tuple):
        return {"__tuple__": [_enc(x) for x in o]}
    if isinstance(o, dict):
        return {k: _enc(v) for k, v in o.items()}
    if isinstance(o, (list,)):
        return [_enc(x) for x in o]
    return o


def _dec(o):
    if isinstance(o, dict):
        if "__nd__" in o:
            return np.asarray(o["__nd__"], float)
        if "__tuple__" in o:
            return tuple(_dec(x) for x in o["__tuple__"])
        return {k: _dec(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_dec(x) for x in o]
    return o


_RES_KEYS = ("t", "pwf", "qo", "qg", "qw", "pavg", "Np", "Gp", "Wp", "GOR", "WC", "qset", "iters")


def pack_sim(sim):
    """Simulator run (config objects + history + snapshots) -> JSON-friendly dict."""
    if not sim:
        return None
    import dataclasses
    res = sim["res"]
    return {"name": sim.get("name"), "cfg": sim["cfg"].to_dict(), "rp": dataclasses.asdict(sim["rp"]),
            "pvt": dataclasses.asdict(sim["pvt"]), "sched": [list(s) for s in sim["sched"]], "t_end": sim["t_end"],
            "res": {**{k: np.asarray(res[k], float) for k in _RES_KEYS if k in res},
                    "snapshots": res["snapshots"], "inplace0": list(res["inplace0"]),
                    "inplace": list(res["inplace"]), "steps": res["steps"]}}


def unpack_sim(d):
    if not d:
        return None
    from .blackoil import PVTConfig, RelPerm, SimConfig
    res = dict(d["res"])
    res["inplace0"], res["inplace"] = tuple(res["inplace0"]), tuple(res["inplace"])
    return {"name": d.get("name"), "cfg": SimConfig.from_dict(d["cfg"]), "rp": RelPerm(**d["rp"]),
            "pvt": PVTConfig(**d["pvt"]), "sched": [tuple(s) for s in d["sched"]], "t_end": d["t_end"], "res": res}


def pack_sim_fit(fit):
    if not fit:
        return None
    import dataclasses
    keep = ("t", "pwf", "GOR", "WC", "qset")
    return {"obs": {k: np.asarray(v, float) for k, v in fit["obs"].items()},
            "curve": {k: np.asarray(fit["curve"][k], float) for k in keep if k in fit["curve"]},
            "info": {k: v for k, v in fit["info"].items() if k != "trace"},
            "cfg": fit["cfg"].to_dict(), "rp": dataclasses.asdict(fit["rp"])}


def unpack_sim_fit(d):
    if not d:
        return None
    from .blackoil import RelPerm, SimConfig
    return {"obs": d["obs"], "curve": d["curve"], "info": {**d["info"], "trace": []},
            "cfg": SimConfig.from_dict(d["cfg"]), "rp": RelPerm(**d["rp"])}


def dumps(meta: dict, state: dict) -> str:
    st_ = {k: state.get(k) for k in KEYS}
    st_["sim"] = pack_sim(st_["sim"])
    st_["sim_fit"] = pack_sim_fit(st_["sim_fit"])
    payload = {"version": VERSION, "meta": {k: str(v) for k, v in meta.items()},
               "state": {k: _enc(st_[k]) for k in KEYS}}
    return json.dumps(payload, indent=1)


def loads(text: str):
    d = json.loads(text)
    if not isinstance(d, dict):
        raise ValueError("project file must contain a JSON object")
    version = d.get("version", 0)
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise ValueError("project version must be a non-negative integer")
    if version > VERSION:
        raise ValueError("project file was written by a newer version")
    if not isinstance(d.get("meta", {}), dict) or not isinstance(d.get("state"), dict):
        raise ValueError("project must contain metadata and a state object")
    state = {k: _dec(v) for k, v in d["state"].items() if k in KEYS}
    if state.get("sim"):
        state["sim"] = unpack_sim(state["sim"])
    if state.get("sim_fit"):
        state["sim_fit"] = unpack_sim_fit(state["sim_fit"])
    return d.get("meta", {}), state
