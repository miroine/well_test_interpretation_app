"""2D areal (plan-view) single-phase numerical model for well testing.

Slightly compressible fluid, constant properties, fully implicit finite volumes on a graded
Cartesian grid.  What it adds over the analytical models:

* any reservoir outline (polygon), sealed outside, optional constant-pressure edges
* faults as line segments with a transmissibility multiplier (0 = sealing, 0..1 = leaky)
* zones (polygons) with permeability / porosity multipliers - lenses, channels, tight areas
* several wells, each with its own rate schedule, skin, wellbore storage (interference, pulse)
* kx/ky anisotropy

Field units: k md, h ft, mu cp, ct 1/psi, B rb/stb, q stb/d, p psi, t hr, lengths ft.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

DARCY = 0.001127          # rb/d/psi  per  md ft2 / (cp ft) ... transmissibility constant


# ------------------------------------------------------------------ configuration
@dataclass
class ArealWell:
    name: str = "W1"
    x: float = 0.0
    y: float = 0.0
    rw: float = 0.354
    skin: float = 0.0
    C: float = 0.0                          # bbl/psi
    schedule: list = field(default_factory=lambda: [(0.0, 1000.0)])   # (start hr, stb/d), + producer


@dataclass
class ArealConfig:
    k: float = 50.0
    ky_kx: float = 1.0
    h: float = 50.0
    phi: float = 0.18
    mu: float = 1.0
    ct: float = 1.5e-5
    B: float = 1.2
    pi: float = 4000.0
    outline: list = field(default_factory=lambda: [(-3000.0, -3000.0), (3000.0, -3000.0),
                                                   (3000.0, 3000.0), (-3000.0, 3000.0)])
    const_p_edges: list = field(default_factory=list)          # subset of "W","E","S","N"
    faults: list = field(default_factory=list)                 # dicts: x1,y1,x2,y2,mult
    zones: list = field(default_factory=list)                  # dicts: poly,kmult,phimult
    wells: list = field(default_factory=lambda: [ArealWell()])
    d0: float = 15.0               # smallest cell near wells (ft)
    growth: float = 0.20           # spacing growth per unit distance from a well
    dmax_frac: float = 0.04        # largest cell as fraction of the domain size

    def to_dict(self):
        d = asdict(self)
        d["wells"] = [asdict(w) for w in self.wells]
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d["wells"] = [ArealWell(**{**w, "schedule": [tuple(s) for s in w["schedule"]]}) for w in d.get("wells", [])]
        d["outline"] = [tuple(p) for p in d.get("outline", [])]
        names = set(ArealConfig.__dataclass_fields__)
        return ArealConfig(**{k: v for k, v in d.items() if k in names})


# ---------------------------------------------------------------------- the grid
def graded_axis(lo, hi, centres, d0, growth, dmax):
    """Cell edges on [lo, hi] with a cell centred on every entry of `centres` and spacing growing away."""
    centres = sorted(set(float(c) for c in centres))
    if not centres:
        n = max(int(np.ceil((hi - lo) / dmax)), 4)
        return np.linspace(lo, hi, n + 1)
    mids = [lo] + [0.5 * (a + b) for a, b in zip(centres[:-1], centres[1:])] + [hi]
    edges = []
    for c, a, b in zip(centres, mids[:-1], mids[1:]):
        right = [c + d0 / 2]
        d = d0
        while right[-1] < b - 1e-9:
            d = min(dmax, d * (1.0 + growth) if d < dmax else dmax)
            right.append(right[-1] + d)
        right = [x for x in right if x < b - 0.25 * d0]
        right.append(b)
        left = [c - d0 / 2]
        d = d0
        while left[-1] > a + 1e-9:
            d = min(dmax, d * (1.0 + growth) if d < dmax else dmax)
            left.append(left[-1] - d)
        left = [x for x in left if x > a + 0.25 * d0]
        left.append(a)
        edges += left + right
    edges = np.unique(np.round(edges, 9))
    return edges


def _in_polygon(px, py, poly):
    """Vectorised even-odd point-in-polygon."""
    poly = np.asarray(poly, float)
    inside = np.zeros(px.shape, bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = (yi > py) != (yj > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (xj - xi) * (py - yi) / (yj - yi) + xi
        inside ^= cond & (px < xint)
        j = i
    return inside


def _segments_cross(ax, ay, bx, by, cx, cy, dx, dy):
    """Proper intersection of segments AB (arrays) and CD (scalars)."""
    def orient(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)
    o1 = orient(ax, ay, bx, by, cx, cy)
    o2 = orient(ax, ay, bx, by, dx, dy)
    o3 = orient(cx, cy, dx, dy, ax, ay)
    o4 = orient(cx, cy, dx, dy, bx, by)
    return (o1 * o2 < 0) & (o3 * o4 < 0)


class Grid:
    def __init__(self, cfg: ArealConfig):
        poly = np.asarray(cfg.outline, float)
        xlo, xhi = poly[:, 0].min(), poly[:, 0].max()
        ylo, yhi = poly[:, 1].min(), poly[:, 1].max()
        dmax = cfg.dmax_frac * max(xhi - xlo, yhi - ylo)
        wx = [w.x for w in cfg.wells]
        wy = [w.y for w in cfg.wells]
        self.xe = graded_axis(xlo, xhi, wx, cfg.d0, cfg.growth, dmax)
        self.ye = graded_axis(ylo, yhi, wy, cfg.d0, cfg.growth, dmax)
        self.xc = 0.5 * (self.xe[:-1] + self.xe[1:])
        self.yc = 0.5 * (self.ye[:-1] + self.ye[1:])
        self.dx = np.diff(self.xe)
        self.dy = np.diff(self.ye)
        self.nx, self.ny = len(self.xc), len(self.yc)
        X, Y = np.meshgrid(self.xc, self.yc, indexing="ij")
        self.X, self.Y = X, Y
        self.active = _in_polygon(X, Y, poly)
        # property fields
        kx = np.full((self.nx, self.ny), cfg.k)
        pm = np.ones((self.nx, self.ny))
        for z in cfg.zones:
            m = _in_polygon(X, Y, z["poly"])
            kx[m] *= z.get("kmult", 1.0)
            pm[m] *= z.get("phimult", 1.0)
        self.kx, self.ky, self.phi = kx, kx * cfg.ky_kx, cfg.phi * pm
        # fault multipliers on faces
        self.fx = np.ones((self.nx - 1, self.ny))
        self.fy = np.ones((self.nx, self.ny - 1))
        for f in cfg.faults:
            m = float(f.get("mult", 0.0))
            x1, y1, x2, y2 = f["x1"], f["y1"], f["x2"], f["y2"]
            ax, ay = X[:-1, :], Y[:-1, :]
            bx, by = X[1:, :], Y[1:, :]
            self.fx = np.where(_segments_cross(ax, ay, bx, by, x1, y1, x2, y2), self.fx * m, self.fx)
            ax, ay = X[:, :-1], Y[:, :-1]
            bx, by = X[:, 1:], Y[:, 1:]
            self.fy = np.where(_segments_cross(ax, ay, bx, by, x1, y1, x2, y2), self.fy * m, self.fy)
        self.index = -np.ones((self.nx, self.ny), int)
        self.index[self.active] = np.arange(int(self.active.sum()))
        self.n = int(self.active.sum())

    def cell_of(self, x, y):
        i = int(np.clip(np.searchsorted(self.xe, x) - 1, 0, self.nx - 1))
        j = int(np.clip(np.searchsorted(self.ye, y) - 1, 0, self.ny - 1))
        return i, j


def _harm(a, b):
    return 2.0 * a * b / np.maximum(a + b, 1e-30)


def assemble(cfg: ArealConfig, g: Grid):
    """Conductance matrix L (graph Laplacian), pore-volume vector V (rb per psi), edge Dirichlet term."""
    I = g.index
    rows, cols, vals = [], [], []
    area_x = g.dy[None, :] * cfg.h          # face area for x-faces (per j)
    area_y = g.dx[:, None] * cfg.h
    # x faces between (i,j) and (i+1,j)
    kx_h = _harm(g.kx[:-1, :], g.kx[1:, :])
    dist = 0.5 * (g.dx[:-1, None] + g.dx[1:, None])
    Tx = DARCY * kx_h * area_x / (cfg.mu * dist) * g.fx
    ky_h = _harm(g.ky[:, :-1], g.ky[:, 1:])
    disty = 0.5 * (g.dy[None, :-1] + g.dy[None, 1:])
    Ty = DARCY * ky_h * area_y / (cfg.mu * disty) * g.fy
    for T, a, b in ((Tx, I[:-1, :], I[1:, :]), (Ty, I[:, :-1], I[:, 1:])):
        ok = (a >= 0) & (b >= 0) & (T > 0)
        ia, ib, t = a[ok], b[ok], T[ok]
        rows += [ia, ib, ia, ib]
        cols += [ia, ib, ib, ia]
        vals += [t, t, -t, -t]
    rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    L = sp.csr_matrix((vals, (rows, cols)), shape=(g.n, g.n))
    vol = (g.dx[:, None] * g.dy[None, :] * cfg.h)
    V = (vol * g.phi * cfg.ct / 5.615)[g.active]
    # constant-pressure edges: half-cell conductance to an external node at pi
    bdiag = np.zeros(g.n)
    for side in cfg.const_p_edges:
        if side in ("W", "E"):
            for j in range(g.ny):
                col = np.where(g.active[:, j])[0]
                if len(col) == 0:
                    continue
                i = col[0] if side == "W" else col[-1]
                T = DARCY * g.kx[i, j] * (g.dy[j] * cfg.h) / (cfg.mu * 0.5 * g.dx[i])
                bdiag[I[i, j]] += T
        else:
            for i in range(g.nx):
                col = np.where(g.active[i, :])[0]
                if len(col) == 0:
                    continue
                j = col[0] if side == "S" else col[-1]
                T = DARCY * g.ky[i, j] * (g.dx[i] * cfg.h) / (cfg.mu * 0.5 * g.dy[j])
                bdiag[I[i, j]] += T
    return L, V, bdiag


def well_index(cfg: ArealConfig, g: Grid, w: ArealWell):
    i, j = g.cell_of(w.x, w.y)
    if not g.active[i, j]:
        raise ValueError(f"well {w.name} lies outside the reservoir outline")
    kx, ky = g.kx[i, j], g.ky[i, j]
    dx, dy = g.dx[i], g.dy[j]
    ro = 0.28 * np.sqrt(np.sqrt(ky / kx) * dx ** 2 + np.sqrt(kx / ky) * dy ** 2) / (
        (ky / kx) ** 0.25 + (kx / ky) ** 0.25)
    kk = np.sqrt(kx * ky)
    WI = 2 * np.pi * DARCY * kk * cfg.h / (cfg.mu * (np.log(ro / w.rw) + w.skin))
    return i, j, WI


def _rate_at(schedule, t):
    r = 0.0
    for ts, q in schedule:
        if t >= ts - 1e-12:
            r = q
    return r


def time_grid(cfg: ArealConfig, t_end, dt0=1e-3, per_decade=10, snap=None):
    """Geometric steps restarted after every rate change of any well."""
    events = sorted({0.0} | {float(ts) for w in cfg.wells for ts, _ in w.schedule if ts < t_end} | {float(t_end)})
    times = [0.0]
    r = 10 ** (1.0 / per_decade)
    for a, b in zip(events[:-1], events[1:]):
        t = a
        dt = min(dt0, (b - a) / 4)
        while t < b - 1e-12:
            t = min(b, t + dt)
            times.append(t)
            dt *= r
            # never take a step larger than a tenth of the phase length
            dt = min(dt, max((b - a) / 10.0, dt0))
    if snap:
        times = sorted(set(times) | {float(s) for s in snap if 0 < s <= t_end})
    return np.array(times)


# ------------------------------------------------------------------- the solver
def run(cfg: ArealConfig, t_end: float, dt0: float = 1e-3, per_decade: int = 10, snap_times=None):
    g = Grid(cfg)
    L, V, bdiag = assemble(cfg, g)
    nw = len(cfg.wells)
    wells = [well_index(cfg, g, w) for w in cfg.wells]
    wc = [g.index[i, j] for i, j, _ in wells]
    times = time_grid(cfg, t_end, dt0, per_decade, snap_times)
    n = g.n
    p = np.full(n, cfg.pi)
    pwf = np.full(nw, cfg.pi)
    out = {"t": [0.0], "pwf": [pwf.copy()], "q": [np.zeros(nw)], "pavg": [cfg.pi]}
    snaps = {}
    snap_set = {round(float(s), 9) for s in (snap_times or [])}
    last_dt = None
    lu = None
    Lb = L + sp.diags(bdiag)
    for k in range(1, len(times)):
        t0, t1 = times[k - 1], times[k]
        dt = (t1 - t0) / 24.0                               # days
        if last_dt is None or abs(dt - last_dt) > 1e-14 * max(dt, 1):
            A = sp.lil_matrix((n + nw, n + nw))
            A[:n, :n] = (Lb + sp.diags(V / dt)).tolil()
            for m, ((i, j, WI), w) in enumerate(zip(wells, cfg.wells)):
                c = wc[m]
                A[c, c] += WI
                A[c, n + m] = -WI
                A[n + m, c] = -WI
                A[n + m, n + m] = WI + w.C / dt
            lu = spl.splu(A.tocsc())
            last_dt = dt
        rhs = np.zeros(n + nw)
        rhs[:n] = V / dt * p + bdiag * cfg.pi
        tm = 0.5 * (t0 + t1)
        qs = np.zeros(nw)
        for m, w in enumerate(cfg.wells):
            qs[m] = _rate_at(w.schedule, tm)
            rhs[n + m] = -qs[m] * cfg.B + w.C / dt * pwf[m]
        sol = lu.solve(rhs)
        p, pwf = sol[:n], sol[n:]
        out["t"].append(t1)
        out["pwf"].append(pwf.copy())
        out["q"].append(qs)
        out["pavg"].append(float((V * p).sum() / V.sum()))
        if round(float(t1), 9) in snap_set:
            snaps[float(t1)] = full_field(g, p)
    res = {"t": np.array(out["t"]), "pwf": np.array(out["pwf"]), "q": np.array(out["q"]),
           "pavg": np.array(out["pavg"]), "names": [w.name for w in cfg.wells],
           "xc": g.xc, "yc": g.yc, "xe": g.xe, "ye": g.ye, "active": g.active, "snaps": snaps,
           "nx": g.nx, "ny": g.ny, "n": g.n, "fx": g.fx, "fy": g.fy, "kx": g.kx, "final": full_field(g, p)}
    return res


def full_field(g: Grid, p):
    f = np.full((g.nx, g.ny), np.nan)
    f[g.active] = p
    return f


def drawdown_response(res, well=0):
    """Return t (>0), dp = pi - pwf, for one well."""
    t = res["t"][1:]
    return t, res["pwf"][0, well] - res["pwf"][1:, well]
