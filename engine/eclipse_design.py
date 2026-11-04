"""Standalone Eclipse-100-style single-phase design decks.

Numerical representations and their limits are exported alongside source inputs.
No analytical solution is imposed on the simulator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json

import numpy as np

from . import design_export as E
from . import gas as G
from . import models as M
from . import units as U


@dataclass
class DeckSetup:
    nx: int = 31
    ny: int = 31
    nz: int = 5
    half_extent_ft: float = 5000.0
    top_depth_ft: float = 8000.0
    rock_compressibility: float = 3e-6
    oil_density_lbft3: float = 53.0
    fracture_width_ft: float = 0.02
    fracture_porosity: float = 0.2
    infinite_fracture_fcd: float = 1e4
    boundary_pv_multiplier: float = 1e6
    report_points_per_period: int = 60


def assumptions(p, fluid, system="Field"):
    notes = [
        "Single-phase Cartesian finite-volume approximation, not an exact translation of the Laplace solution.",
        (f"Deck uses {system.upper()} units; time is days, lengths "
         f"{'ft' if system == 'Field' else 'm'}, pressure {'psia' if system == 'Field' else 'bara'}."),
        "Grid faces are sealed unless explicitly represented by a pressure-buffer rim.",
        "An infinite reservoir is truncated at the chosen grid extent; check boundary-distance and mesh convergence.",
        "Conventional well coupling does not reproduce analytical wellbore storage C. Early storage-time matching is not claimed.",
        "Skin is assigned in COMPDAT; grid-scale well indices and discretization can change near-well response.",
        "Initial pressure is hydrostatically adjusted around the well datum using reference fluid density.",
        "SHUT well WBHP may be frozen or absent. BPR at the first connected cell is also requested for shut-in comparison.",
        "Not executed in Eclipse or OPM here. Check parser acceptance, material balance, timestep and grid convergence before use.",
    ]
    if p.model in ("dual_pss", "dual_slab", "fracture_dual_pss"):
        notes.append(
            "DUAL-POROSITY LIMITATION: this is a late-time equilibrated single-continuum surrogate using total phi, ct and k. "
            "It does NOT reproduce matrix/fracture exchange, omega, lambda or the transient slab/PSS derivative dip. "
            "omega and lambda are retained as source metadata only; no DUALPORO claim is made.")
    if p.model in ("composite", "composite3"):
        notes.append(
            "Composite interfaces are stair-stepped cell-centre radial regions. k_outer=k_inner/M; "
            "storage_outer/storage_inner=D/M. Oil uses regional fluid compressibility at common porosity; "
            "gas uses regional porosity at common gas PVT. Transition geometry is mesh-dependent.")
    if p.model in (*M.INFINITE_FRACTURE_MODELS, "fracture_fc", "mfhw"):
        notes.append(
            "Fractures are explicit thin high-permeability strips with kf=FcD*k*xf/width; "
            "edges are inserted in the Cartesian grid. Finite width/porosity adds fracture storage.")
        if p.model in M.INFINITE_FRACTURE_MODELS:
            notes.append("Infinite fracture conductivity is approximated by the user-selected large finite FcD.")
    if p.model == "mfhw":
        notes.append(
            "Parallel transverse strips share a horizontal well connected only at fracture centres; "
            "the lateral has no pressure loss and no background-matrix completions.")
    if p.model in ("horizontal", "limited_entry"):
        notes.append("Layer edges honour the selected lateral depth/perforation interval; vertical permeability uses kv/kh.")
    if p.boundary in ("circle", "circle_cp"):
        notes.append("Circular reservoir edges use ACTNUM cell-centre masking: a stair-stepped approximation to radius re.")
    if p.boundary in ("const_p_fault", "circle_cp") or (
            p.model == "horizontal" and "const_p" in (p.top_bc, p.bot_bc)):
        notes.append(
            "Constant-pressure faces use a high-pore-volume buffer, NOT an exact pressure constraint or multiphase gas cap/aquifer. "
            "Pressure can drift; test the buffer multiplier and monitor the boundary cells.")
    if p.boundary in M.CLOSED_BOUNDARIES and p.model not in ("homogeneous", "dual_pss", "dual_slab"):
        notes.append(
            "The analytical engine ignores this closed boundary for the selected model, but this numerical deck represents it. "
            "The two cases therefore have intentionally different outer-boundary physics.")
    if fluid == "Gas":
        notes.append(
            "Dry-gas PVDG is tabulated from the current DAK/Lee gas correlations. Numerical pressure is physical BHP; "
            "the analytical gas reference is an inverse pseudo-pressure approximation.")
    else:
        notes.append(
            "Oil is single-phase PVCDO with constant reference viscosity/B. Fluid compressibility=regional ct minus rock "
            "compressibility. No solution gas, water, relperm or multiphase contacts are inferred.")
    return notes


def _axis(lo, hi, count, features, rw):
    if not lo < 0 < hi:
        raise ValueError("Well must lie strictly inside the export grid.")
    near = min(-lo, hi, max(rw * 8, 1.0))
    inner = min(-lo, hi)
    log = np.geomspace(near, inner, max(3, count // 2))
    edges = [lo, hi, -near / 2, near / 2, *(-log), *log, *features]
    return np.unique([float(x) for x in edges if lo <= x <= hi])


def _array(keyword, values, integer=False):
    values = np.asarray(values).ravel()
    fmt = (lambda x: str(int(x))) if integer else (lambda x: f"{float(x):.12g}")
    if np.all(values == values[0]):
        return f"{keyword}\n  {len(values)}*{fmt(values[0])} /\n"
    lines = ["  " + " ".join(fmt(x) for x in values[i:i + 8]) for i in range(0, len(values), 8)]
    return keyword + "\n" + "\n".join(lines) + " /\n"


def _period_steps(duration_hr, count):
    edges = np.geomspace(min(1e-3, duration_hr / 1000), duration_hr, count)
    return np.diff(np.concatenate([[0.0], edges])) / 24


def grid_model(p: M.WellTestParams, setup: DeckSetup, fluid, gas=None):
    """Field-unit arrays in Eclipse order (I fastest, K slowest)."""
    c = setup
    numeric = list(asdict(c).values())
    if not all(np.isfinite(v) for v in numeric):
        raise ValueError("All deck setup values must be finite.")
    for name in ("nx", "ny", "nz", "report_points_per_period"):
        value = getattr(c, name)
        if not isinstance(value, int) or value < (1 if name == "nz" else 3):
            raise ValueError(f"{name} must be an integer with sufficient cells/reporting points.")
    if c.nx > 151 or c.ny > 151 or c.nz > 50 or c.report_points_per_period > 500:
        raise ValueError("Export setup exceeds supported grid or reporting limits.")
    if min(c.half_extent_ft, c.fracture_width_ft, c.oil_density_lbft3, c.infinite_fracture_fcd) <= 0:
        raise ValueError("Grid extent, fracture width/conductivity and oil density must be positive.")
    if c.top_depth_ft < 0 or c.rock_compressibility < 0 or c.boundary_pv_multiplier < 1:
        raise ValueError("Depth/rock compressibility must be non-negative and boundary multiplier at least one.")
    if not 0 < c.fracture_porosity < 1:
        raise ValueError("Fracture porosity must lie between zero and one.")
    if p.model not in M.MODEL_TYPES or p.boundary not in M.BOUNDARY_TYPES or fluid not in ("Oil", "Gas"):
        raise ValueError("Unsupported design model, boundary or fluid.")
    if not all(np.isfinite(v) for v in p.to_dict().values() if isinstance(v, (int, float))):
        raise ValueError("All numerical design inputs must be finite.")
    if min(p.k, p.h, p.ct, p.mu, p.B, p.rw, p.pi, p.kv_kh) <= 0 or not 0 < p.phi < 1:
        raise ValueError("Design rock/fluid properties must be positive, with porosity between zero and one.")
    extent = c.half_extent_ft
    xmin, xmax, ymin, ymax = -extent, extent, -extent, extent
    if p.boundary in ("fault", "const_p_fault", "corner"):
        xmax = p.d1
        if p.boundary == "corner":
            ymax = p.d2
    elif p.boundary == "channel":
        xmin, xmax = -p.d1, p.d2
    elif p.boundary == "rectangle":
        xmin, xmax, ymin, ymax = -p.xw, p.xe - p.xw, -p.yw, p.ye - p.yw
    elif p.boundary in ("circle", "circle_cp"):
        xmin = ymin = -p.re
        xmax = ymax = p.re

    xf, yf = [], []
    if p.boundary in ("circle", "circle_cp"):
        xf.extend([-0.75 * p.re, 0.75 * p.re])
        yf.extend([-0.75 * p.re, 0.75 * p.re])
    if p.model in ("composite", "composite3"):
        for radius in ([p.r_inner, p.r_mid] if p.model == "composite3" else [p.r_inner]):
            xf.extend([-radius, -radius / 2, radius / 2, radius])
            yf.extend([-radius, -radius / 2, radius / 2, radius])
    fracture_x = []
    if p.model in (*M.INFINITE_FRACTURE_MODELS, "fracture_fc", "mfhw"):
        if p.xf <= 0 or p.FcD <= 0 or p.f_space <= 0 or not 1 <= p.n_frac <= 12 or int(p.n_frac) != p.n_frac:
            raise ValueError("Fracture dimensions, conductivity, spacing and integer count must be valid.")
        if c.fracture_width_ft >= p.f_space or c.fracture_width_ft >= p.xf:
            raise ValueError("Fracture width must be smaller than half-length and spacing.")
        n = int(p.n_frac) if p.model == "mfhw" else 1
        fracture_x = (np.arange(n) - (n - 1) / 2) * p.f_space
        fracture_edges = [v + offset for v in fracture_x for offset in (-c.fracture_width_ft / 2, c.fracture_width_ft / 2)]
        if min(fracture_edges) <= xmin or max(fracture_edges) >= xmax or p.xf >= min(-ymin, ymax):
            raise ValueError("The export domain must contain all fracture tips and centres strictly inside its boundary.")
        xf.extend(fracture_edges)
        yf.extend([-p.xf, p.xf])
    if p.model == "horizontal":
        if p.Lw <= 0 or not 0 < p.zw_h < 1 or p.Lw / 2 >= min(-xmin, xmax):
            raise ValueError("Horizontal well must fit strictly inside the domain, with 0 < zw/h < 1.")
        xf += [-p.Lw / 2, p.Lw / 2]
    xe = _axis(xmin, xmax, c.nx, xf, p.rw)
    ye = _axis(ymin, ymax, c.ny, yf, p.rw)
    ze = np.linspace(0, p.h, c.nz + 1)
    if p.model == "limited_entry":
        low, high = (p.perf_pos - p.perf_frac / 2) * p.h, (p.perf_pos + p.perf_frac / 2) * p.h
        if not 0 <= low < high <= p.h:
            raise ValueError("The full perforated interval must lie within the reservoir thickness.")
        ze = np.unique(np.concatenate([ze, [low, high]]))
    zc = (ze[:-1] + ze[1:]) / 2
    if p.model == "horizontal":
        # Put the lateral in a layer centred at its selected depth.
        depth = p.zw_h * p.h
        dz = min(p.h / c.nz, 2 * min(depth, p.h - depth)) / 2
        ze = np.unique(np.concatenate([ze[(ze < depth - dz) | (ze > depth + dz)], [depth - dz, depth + dz]]))
        zc = (ze[:-1] + ze[1:]) / 2
    nx, ny, nz = len(xe) - 1, len(ye) - 1, len(ze) - 1
    if nx * ny * nz > 200000:
        raise ValueError("Refined grid exceeds 200,000 cells. Reduce base grid counts.")
    x = (xe[:-1] + xe[1:]) / 2
    y = (ye[:-1] + ye[1:]) / 2
    z, yy, xx = np.meshgrid(zc, y, x, indexing="ij")
    shape = xx.shape
    radius = np.hypot(xx, yy)
    active = np.ones(shape, bool)
    if p.boundary in ("circle", "circle_cp"):
        active = radius <= p.re
    region = np.zeros(shape, int)
    ratios = [(1.0, 1.0)]
    if p.model in ("composite", "composite3"):
        if min(p.r_inner, p.M, p.D) <= 0:
            raise ValueError("Composite radii, mobility and diffusivity ratios must be positive.")
        region[radius >= p.r_inner] = 1
        ratios.append((p.M, p.D))
        if p.model == "composite3":
            if p.r_mid <= p.r_inner or min(p.M3, p.D3) <= 0:
                raise ValueError("Three-zone outer interface must exceed the inner radius; all ratios must be positive.")
            region[radius >= p.r_mid] = 2
            ratios.append((p.M3, p.D3))
        for r in range(len(ratios)):
            if not np.any((region == r) & active):
                raise ValueError("Grid does not resolve every composite region. Increase extent or refine the grid.")
    perm = np.full(shape, p.k)
    poro = np.full(shape, p.phi)
    ct = []
    for r, (mobility, diffusivity) in enumerate(ratios):
        perm[region == r] = p.k / mobility
        ct.append(p.ct * diffusivity / mobility)
        if fluid == "Gas":
            poro[region == r] = p.phi * diffusivity / mobility
    if np.any((poro <= 0) | (poro >= 1)):
        raise ValueError("Gas composite D/M mapping gives porosity outside (0,1). Revise ratios or use a custom deck.")
    if fluid == "Oil" and any(v <= c.rock_compressibility for v in ct):
        raise ValueError("Rock compressibility must be below total ct in every composite region.")
    fracture = np.zeros(shape, bool)
    if len(fracture_x):
        for location in fracture_x:
            fracture |= (np.abs(xx - location) < c.fracture_width_ft / 2) & (np.abs(yy) < p.xf)
        fcd = c.infinite_fracture_fcd if p.model in M.INFINITE_FRACTURE_MODELS else p.FcD
        perm[fracture] = fcd * p.k * p.xf / c.fracture_width_ft
        poro[fracture] = c.fracture_porosity
    permz = perm * (p.kv_kh if p.model in ("horizontal", "limited_entry") else 1.0)
    pv_mult = np.ones(shape)
    buffer = np.zeros(shape, bool)
    if p.boundary == "const_p_fault":
        buffer[:, :, -1] = True
    elif p.boundary == "circle_cp":
        # Face-neighbours of inactive cells, plus active cells on the box perimeter.
        padded = np.pad(active, 1, constant_values=False)
        neighbours = (padded[1:-1, 1:-1, :-2] & padded[1:-1, 1:-1, 2:] &
                      padded[1:-1, :-2, 1:-1] & padded[1:-1, 2:, 1:-1])
        buffer = active & ~neighbours
    if p.model == "horizontal":
        if p.top_bc == "const_p":
            buffer[0] = True
        if p.bot_bc == "const_p":
            buffer[-1] = True
    pv_mult[buffer] = c.boundary_pv_multiplier
    i0, j0 = int(np.argmin(abs(x))), int(np.argmin(abs(y)))
    connections = []
    if p.model in ("horizontal", "mfhw"):
        k0 = int(np.argmin(abs(zc - (p.zw_h * p.h if p.model == "horizontal" else p.h / 2))))
        indices = np.flatnonzero(abs(x) < p.Lw / 2) if p.model == "horizontal" else [
            int(np.argmin(abs(x - location))) for location in fracture_x]
        connections = [(int(i), j0, k0, "X") for i in indices]
    else:
        layers = range(nz) if p.model != "limited_entry" else np.flatnonzero((zc > low) & (zc < high))
        connections = [(i0, j0, int(k), "Z") for k in layers]
    if not connections or any(not active[k, j, i] for i, j, k, _ in connections):
        raise ValueError("No valid active well completion. Check grid, geometry and boundary inputs.")
    if any(buffer[k, j, i] for i, j, k, _ in connections):
        raise ValueError("Well completions intersect the pressure buffer. Refine layers or move the boundary/lateral.")
    for i, j, k, direction in connections:
        if direction == "Z":
            a, b, ratio = np.diff(xe)[i], np.diff(ye)[j], 1.0
        else:
            a, b = np.diff(ye)[j], np.diff(ze)[k]
            ratio = permz[k, j, i] / perm[k, j, i]
        radius = 0.28 * np.sqrt(np.sqrt(ratio) * a * a + b * b / np.sqrt(ratio)) / (
            ratio ** 0.25 + ratio ** -0.25)
        if radius <= p.rw or np.log(radius / p.rw) + p.S <= 0:
            raise ValueError("Completion cell is too small, or skin too negative, for a positive conventional well index. "
                             "Revise well/grid inputs or use a custom stimulated-well representation.")
    ref_depth = c.top_depth_ft + (p.zw_h * p.h if p.model == "horizontal" else p.h / 2)
    if fluid == "Gas":
        if gas is None or gas["T_F"] <= -459.67 or gas["gamma_g"] <= 0:
            raise ValueError("Valid gas temperature and specific gravity are required.")
        props = G.gas_props(p.pi, gas["T_F"], gas["gamma_g"])
        density = 0.0764 * gas["gamma_g"] / (props["Bg"] * 5.615)
    else:
        density = c.oil_density_lbft3 / p.B
    pressure = p.pi + (c.top_depth_ft + z - ref_depth) * density / 144
    if np.any(pressure <= 0):
        raise ValueError("Hydrostatic initialization gives non-positive pressure.")
    return {
        "shape": (nx, ny, nz), "dx": np.broadcast_to(np.diff(xe)[None, None, :], shape),
        "dy": np.broadcast_to(np.diff(ye)[None, :, None], shape),
        "dz": np.broadcast_to(np.diff(ze)[:, None, None], shape),
        "active": active, "poro": poro, "perm": perm, "permz": permz, "region": region + 1,
        "pressure": pressure, "pv_mult": pv_mult, "ct": ct, "connections": connections,
        "ref_depth": ref_depth, "fracture": fracture, "buffer": buffer,
        "x_edges": xe, "y_edges": ye, "z_edges": ze,
    }


def build_deck(p, setup, well, q_user, tp, dt_end, fluid, minimum_bhp, gas=None, *, system="Field"):
    """Return a deck in selected units; input params/setup remain field-unit engine values."""
    E.eclipse_schedule(well, q_user, tp, dt_end, system, fluid, minimum_bhp)
    def convert(value, kind):
        return U.from_field(value, kind, system)
    density_factor = 1.0 if system == "Field" else 16.01846337396
    gas_b_factor = 1000.0 if system == "Field" else U.STB_M3 / (U.KINDS["gasrate"][2] / 1000)
    g = grid_model(p, setup, fluid, gas)
    nx, ny, nz = g["shape"]
    n = nx * ny * nz
    nreg = len(g["ct"])
    notes = assumptions(p, fluid, system)
    out = ["-- Well Test Workbench: numerical approximation of an analytical test design\n"]
    out.extend(f"-- {line}\n" for line in notes)
    out.append(f"RUNSPEC\nTITLE\n  Well test design - {p.model}\n{system.upper()}\n{fluid.upper()}\n"
               f"DIMENS\n  {nx} {ny} {nz} /\nTABDIMS\n  1 {nreg} 120 120 /\n"
               f"WELLDIMS\n  1 {len(g['connections'])} 1 1 /\nSTART\n  1 'JAN' 2000 /\n\nGRID\n")
    for key in ("dx", "dy", "dz"):
        out.append(_array(key.upper(), convert(g[key], "length")))
    out.append(_array("TOPS", np.full(nx * ny, convert(setup.top_depth_ft, "length"))))
    for keyword, key, integer in (("PERMX", "perm", False), ("PERMY", "perm", False),
                                  ("PERMZ", "permz", False), ("PORO", "poro", False),
                                  ("ACTNUM", "active", True), ("MULTPV", "pv_mult", False)):
        out.append(_array(keyword, g[key], integer))
    out.append("\nPROPS\n")
    gas_density = 0.0764 * gas["gamma_g"] if gas is not None and fluid == "Gas" else 0.06
    out.append("DENSITY\n")
    out.extend(f"  {setup.oil_density_lbft3 * density_factor:.12g} {62.4 * density_factor:.12g} {gas_density * density_factor:.12g} /\n" for _ in range(nreg))
    out.append("ROCK\n")
    out.extend(f"  {convert(p.pi, 'pressure'):.12g} {convert(setup.rock_compressibility, 'ct'):.12g} /\n" for _ in range(nreg))
    if fluid == "Oil":
        out.append("PVCDO\n")
        out.extend(f"  {convert(p.pi, 'pressure'):.12g} {p.B:.12g} {convert(ct - setup.rock_compressibility, 'ct'):.12g} {p.mu:.12g} 0 /\n"
                   for ct in g["ct"])
    else:
        pressures = np.unique(np.concatenate([np.geomspace(14.7, max(p.pi * 2, 1000), 100), [p.pi]]))
        out.append("PVDG\n")
        for _ in range(nreg):
            for pressure in pressures:
                gp = G.gas_props(float(pressure), gas["T_F"], gas["gamma_g"])
                out.append(f"  {convert(pressure, 'pressure'):.12g} {gas_b_factor * gp['Bg']:.12g} {gp['mu']:.12g}\n")
            out.append("/\n")
    out.append("\nREGIONS\n" + _array("PVTNUM", g["region"], True))
    out.append("\nSOLUTION\n" + _array("PRESSURE", convert(g["pressure"], "pressure")))
    saturation = "SOIL" if fluid == "Oil" else "SGAS"
    out.append(f"{saturation}\n  {n}*1 /\n")
    i, j, k, _ = g["connections"][0]
    summary = (f"WBHP\n  '{well}' /\n{'WOPR' if fluid == 'Oil' else 'WGPR'}\n  '{well}' /\n"
               f"BPR\n  {i + 1} {j + 1} {k + 1} /\n/\n")
    out.append("\nSUMMARY\n" + summary)
    out.append(f"\nSCHEDULE\nWELSPECS\n  '{well}' 'TEST' {i + 1} {j + 1} {convert(g['ref_depth'], 'length'):.12g} '{fluid.upper()}' /\n/\n")
    out.append("COMPDAT\n")
    for i, j, k, direction in g["connections"]:
        out.append(f"  '{well}' {i + 1} {j + 1} {k + 1} {k + 1} 'OPEN' 1 1* {convert(2 * p.rw, 'length'):.12g} 1* {p.S:.12g} 1* '{direction}' /\n")
    out.append("/\n")
    controls = E.eclipse_schedule(well, q_user, tp, dt_end, system, fluid, minimum_bhp)
    # Keep controls but replace the two period-boundary TSTEPs with reporting steps.
    prefix, flow, shut = controls.split("TSTEP\n")
    out.append(prefix)
    out.append(_array("TSTEP", _period_steps(tp, setup.report_points_per_period)))
    out.append(flow.split("/", 1)[1])
    out.append(_array("TSTEP", _period_steps(dt_end, setup.report_points_per_period)))
    out.append(shut.split("/", 1)[1])
    out.append("END\n")
    metadata = {
        "source_design": p.to_dict(), "source_design_units": "FIELD",
        "setup_field_units": asdict(setup), "gas": gas,
        "well": well, "fluid": fluid, "deck_units": system.upper(), "initial_date": "2000-01-01",
        "surface_rate_field": q_user, "flow_hr": tp, "shut_in_hr": dt_end,
        "minimum_bhp_psia": minimum_bhp, "grid_shape": [nx, ny, nz], "active_cells": int(g["active"].sum()),
        "completion_count": len(g["connections"]), "well_datum_ft": g["ref_depth"],
        "bpr_probe_ijk": [int(v) + 1 for v in g["connections"][0][:3]],
        "approximated_source_parameters": (["omega", "lam"] if p.model in ("dual_pss", "dual_slab", "fracture_dual_pss") else []) + ["C"],
        "assumptions_and_limits": notes,
        "validation": "Generated and structurally tested only; not simulator-executed or history-matched.",
    }
    return "".join(out), metadata


def deck_package(p, setup, well, curves, q_user, tp, dt_end, fluid, minimum_bhp, gas=None, *, system="Field"):
    deck, metadata = build_deck(p, setup, well, q_user, tp, dt_end, fluid, minimum_bhp, gas, system=system)
    files = E.comparison_files(well, curves, q_user, tp, dt_end, system, fluid, minimum_bhp, p.pi, gas)
    files[f"{well}.DATA"] = deck
    files["model.json"] = json.dumps(metadata, indent=2, allow_nan=False)
    files["USAGE.txt"] = (
        f"Run {well}.DATA as a standalone {system.upper()} deck. It already contains its controls and summary requests.\n"
        "Do NOT include schedule.INC or summary.INC again in that deck; those files are for other host decks.\n"
        "Inspect model.json and the .DATA header for approximations before running.\n"
        "predicted_bhp.csv is synthetic comparison data, NOT observed field data or a well control.\n"
        "TIME is days from 1 JAN 2000; reference time_hr is hours from test start (=TIME*24 here).\n"
        "The CSV has two sides at shut-in; align by phase, not time alone.\n"
        "SHUT-WELL WARNING: WBHP may freeze or be absent while shut. BPR at the probe in model.json "
        "reports near-well cell pressure, not exact sandface pressure.\n"
        "Validate deck parsing, material balance, grid, buffer and timestep convergence in your simulator.\n"
    )
    return deck, E.zip_files(files), metadata
