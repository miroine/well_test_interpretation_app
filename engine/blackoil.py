"""Fully implicit black-oil r-z (radial-vertical) simulator for single-well tests.

Field units. Three phases (oil, gas, water), three components (stock-tank oil,
surface gas, water). Unknowns per cell: p, Sw and x3 = Sg (saturated cell) or
Rs (undersaturated cell) -- variable switching. Plus the flowing well pressure.
Jacobian by forward differences with 5-colour cell colouring (sparse), Newton
iteration with damping, adaptive time steps. No capillary pressure.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from . import gas as G

GRAV_CONV = 144.0          # psi per (lb/ft3 * ft)


# ----------------------------------------------------------------------- PVT
@dataclass
class PVTConfig:
    api: float = 35.0
    gas_gravity: float = 0.75
    T_F: float = 200.0
    co: float = 1.2e-5          # undersaturated oil compressibility [1/psi]
    cw: float = 3.0e-6
    Bw0: float = 1.02           # rb/stb at p_ref
    muw: float = 0.5
    gamma_w: float = 1.05
    p_ref: float = 4000.0
    p_max: float = 12000.0


class BlackOilPVT:
    """Standing / Beggs-Robinson oil, DAK gas; tabulated and linearly interpolated."""

    def __init__(self, cfg: PVTConfig, n: int = 180):
        self.cfg = cfg
        c = cfg
        self.go = 141.5 / (c.api + 131.5)
        self.co, self.cw, self.Bw0, self.muw = c.co, c.cw, c.Bw0, c.muw
        self.p_ref = c.p_ref
        p = np.linspace(14.7, max(c.p_max, 2000.0), n)
        self.p = p
        a = 10 ** (0.0125 * c.api - 0.00091 * c.T_F)
        self._a = a
        self.rs = c.gas_gravity * ((p / 18.2 + 1.4) * a) ** 1.2048
        F = self.rs * math.sqrt(c.gas_gravity / self.go) + 1.25 * c.T_F
        self.bo = 0.9759 + 0.000120 * F ** 1.2
        T = c.T_F
        mud = 10 ** (10 ** (3.0324 - 0.02023 * c.api) * T ** -1.163) - 1.0
        A = 10.715 * (self.rs + 100.0) ** -0.515
        B = 5.44 * (self.rs + 150.0) ** -0.338
        self.muo = A * mud ** B
        self.dmu = 0.001 * (0.024 * self.muo ** 1.6 + 0.038 * self.muo ** 0.56)
        self.bg = np.empty(n)
        self.mug = np.empty(n)
        for i, pi_ in enumerate(p):
            gp = G.gas_props(pi_, c.T_F, c.gas_gravity)
            self.bg[i], self.mug[i] = gp["Bg"], gp["mu"]
        self.rho_o_sc = 62.4 * self.go
        self.rho_w_sc = 62.4 * c.gamma_w

    def rs_sat(self, p):
        return np.interp(p, self.p, self.rs)

    def pb_of_rs(self, rs):
        return np.interp(rs, self.rs, self.p)

    def bo_sat(self, p):
        return np.interp(p, self.p, self.bo)

    def muo_sat(self, p):
        return np.interp(p, self.p, self.muo)

    def dmu_dp(self, p):
        return np.interp(p, self.p, self.dmu)

    def bg_(self, p):
        return np.interp(p, self.p, self.bg)

    def mug_(self, p):
        return np.interp(p, self.p, self.mug)


# ------------------------------------------------------------------ relperm
@dataclass
class RelPerm:
    swc: float = 0.2
    sorw: float = 0.2
    sorg: float = 0.1
    sgc: float = 0.03
    krw_max: float = 0.4
    krg_max: float = 0.8
    krocw: float = 0.9
    nw: float = 2.0
    ng: float = 2.0
    now: float = 2.5
    nog: float = 2.5

    def kr(self, sw, sg):
        """Corey curves, Stone II three-phase oil. Returns krw, krg, kro."""
        sw = np.asarray(sw, float)
        sg = np.asarray(sg, float)
        ew = np.clip((sw - self.swc) / (1 - self.swc - self.sorw), 0, 1)
        eg = np.clip((sg - self.sgc) / (1 - self.swc - self.sorg - self.sgc), 0, 1)
        krw = self.krw_max * ew ** self.nw
        krg = self.krg_max * eg ** self.ng
        krow = self.krocw * np.clip((1 - sw - self.sorw) / (1 - self.swc - self.sorw), 0, 1) ** self.now
        krog = self.krocw * np.clip((1 - sg - self.swc - self.sorg) / (1 - self.swc - self.sorg), 0, 1) ** self.nog
        kro = self.krocw * ((krow / self.krocw + krw) * (krog / self.krocw + krg) - (krw + krg))
        return krw, krg, np.clip(kro, 0.0, self.krocw)


# ------------------------------------------------------------------- config
@dataclass
class SimConfig:
    # grid
    rw: float = 0.354
    re: float = 3000.0
    nr: int = 22
    nz: int = 10
    h: float = 100.0              # gross thickness [ft]
    top_depth: float = 8000.0     # [ft TVD]
    # rock
    k: float = 100.0              # horizontal permeability [md]
    kv_kh: float = 0.1
    phi: float = 0.2
    cr: float = 4.0e-6
    gamma: float = 0.0            # permeability modulus [1/psi]: k = k0 exp(-gamma (p_init - p))
    k_layers: tuple | None = None  # optional per-layer multipliers (length nz)
    # well
    skin: float = 0.0
    perf_from: float = 0.0        # fraction of thickness from top
    perf_to: float = 1.0
    C: float = 0.0                # wellbore storage [bbl/psi]
    wb_grad: float | None = None  # wellbore fluid gradient [psi/ft]; None = oil
    # initial state
    p_datum: float = 4000.0
    datum_depth: float | None = None   # None = mid perforation
    goc: float | None = None      # depth of gas-oil contact (None = no gas cap)
    woc: float | None = None      # depth of water-oil contact (None = no water leg)
    Rsi: float = 300.0            # solution GOR in the oil zone if no gas cap [scf/stb]
    outer_pv_mult: float = 1.0    # >1 mimics aquifer / pressure support at re

    def to_dict(self):
        d = dataclasses.asdict(self)
        if d.get("k_layers") is not None:
            d["k_layers"] = list(d["k_layers"])
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        if d.get("k_layers") is not None:
            d["k_layers"] = tuple(d["k_layers"])
        names = {f.name for f in dataclasses.fields(SimConfig)}
        return SimConfig(**{k: v for k, v in d.items() if k in names})


@dataclass
class Props:
    sg: np.ndarray
    rs: np.ndarray
    bo: np.ndarray
    muo: np.ndarray
    bg: np.ndarray
    mug: np.ndarray
    bw: np.ndarray
    lam_o: np.ndarray
    lam_g: np.ndarray
    lam_w: np.ndarray
    rho_o: np.ndarray
    rho_g: np.ndarray
    rho_w: np.ndarray


class NewtonFailure(RuntimeError):
    pass


# ---------------------------------------------------------------- simulator
class BlackOilSim:
    def __init__(self, cfg: SimConfig, pvt: PVTConfig | None = None, rp: RelPerm | None = None):
        self.cfg = cfg
        self.pvt_cfg = pvt or PVTConfig(p_ref=cfg.p_datum, p_max=max(2.5 * cfg.p_datum, 6000.0))
        self.pvt = BlackOilPVT(self.pvt_cfg)
        self.rp = rp or RelPerm()
        self._build_grid()
        self._init_state()

    # ------------------------------------------------------------- geometry
    def _build_grid(self):
        c = self.cfg
        nr, nz = c.nr, c.nz
        self.N = nr * nz
        edges = c.rw * (c.re / c.rw) ** (np.arange(nr + 1) / nr)
        self.r_edges = edges
        self.r_c = np.sqrt(edges[:-1] * edges[1:])
        self.dz = c.h / nz
        self.zc = c.top_depth + (np.arange(nz) + 0.5) * self.dz
        mult = np.ones(nz) if c.k_layers is None else np.asarray(c.k_layers, float)
        self.kh = c.k * mult
        self.kv = c.k * c.kv_kh * mult
        I, J = np.meshgrid(np.arange(nr), np.arange(nz), indexing="ij")
        self.I, self.J = I.ravel(), J.ravel()           # cell n = i*nz + j
        idx = lambda i, j: i * nz + j
        area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        vol = area[self.I] * self.dz
        self.cellvol = vol
        mpv = np.ones(self.N)
        mpv[self.I == nr - 1] = c.outer_pv_mult
        self.pv0 = vol * c.phi * mpv                     # ft3
        self.Dc = self.zc[self.J]
        # connections
        ia, ib, T, dD = [], [], [], []
        for i in range(nr - 1):
            f = 0.00708 / math.log(self.r_c[i + 1] / self.r_c[i])
            for j in range(nz):
                ia.append(idx(i, j)); ib.append(idx(i + 1, j))
                T.append(f * self.kh[j] * self.dz); dD.append(0.0)
        for i in range(nr):
            for j in range(nz - 1):
                kvh = 2 * self.kv[j] * self.kv[j + 1] / (self.kv[j] + self.kv[j + 1])
                ia.append(idx(i, j)); ib.append(idx(i, j + 1))
                T.append(0.001127 * kvh * area[i] / self.dz); dD.append(-self.dz)
        self.ia, self.ib = np.array(ia), np.array(ib)
        self.T, self.dD = np.array(T), np.array(dD)
        # neighbour table for colouring: self, i-1, i+1, j-1, j+1
        nbr = -np.ones((self.N, 5), int)
        nbr[:, 0] = np.arange(self.N)
        for n in range(self.N):
            i, j = self.I[n], self.J[n]
            if i > 0: nbr[n, 1] = idx(i - 1, j)
            if i < nr - 1: nbr[n, 2] = idx(i + 1, j)
            if j > 0: nbr[n, 3] = idx(i, j - 1)
            if j < nz - 1: nbr[n, 4] = idx(i, j + 1)
        self.nbr = nbr
        col = (self.I + 2 * self.J) % 5
        self.colors = [np.where(col == k)[0] for k in range(5)]
        # well completion
        top = c.perf_from * c.h
        bot = c.perf_to * c.h
        lo = np.arange(nz) * self.dz
        overlap = np.clip(np.minimum(lo + self.dz, bot) - np.maximum(lo, top), 0, self.dz) / self.dz
        wj = np.where(overlap > 1e-9)[0]
        if len(wj) == 0:
            raise ValueError("no layer is perforated")
        self.well_cells = np.array([idx(0, j) for j in wj])
        wi_den = math.log(self.r_c[0] / c.rw) + c.skin
        if wi_den <= 0.05:
            raise ValueError("skin too negative for this grid")
        self.WI = 0.00708 * self.kh[wj] * self.dz * overlap[wj] / wi_den
        self.D_ref = (c.datum_depth if c.datum_depth is not None
                      else c.top_depth + 0.5 * (top + bot))

    # ------------------------------------------------------------ properties
    def props(self, p, sw, x3, sat) -> Props:
        v = self.pvt
        sg = np.where(sat, x3, 0.0)
        rs = np.where(sat, v.rs_sat(p), x3)
        pb = np.where(sat, p, v.pb_of_rs(x3))
        bo = v.bo_sat(pb) * np.exp(-v.co * (p - pb))
        muo = v.muo_sat(pb) + v.dmu_dp(pb) * (p - pb)
        bg, mug = v.bg_(p), v.mug_(p)
        bw = v.Bw0 * np.exp(-v.cw * (p - v.p_ref))
        krw, krg, kro = self.rp.kr(sw, sg)
        gg = self.pvt_cfg.gas_gravity
        return Props(
            sg=sg, rs=rs, bo=bo, muo=muo, bg=bg, mug=mug, bw=bw,
            lam_o=kro / (muo * bo), lam_g=krg / (mug * bg), lam_w=krw / (v.muw * bw),
            rho_o=(v.rho_o_sc + 0.0136 * rs * gg) / bo,
            rho_g=0.0764 * gg / (bg * 5.615),
            rho_w=v.rho_w_sc / bw)

    # ------------------------------------------------------ initial equilibrium
    def _density_profile(self, p, D, rs_oil):
        """Phase density at pressure p, depth D for the zone D lies in."""
        c = self.cfg
        v = self.pvt
        gg = self.pvt_cfg.gas_gravity
        if c.goc is not None and D < c.goc:
            bg = float(v.bg_(p))
            return 0.0764 * gg / (bg * 5.615)
        if c.woc is not None and D > c.woc:
            return v.rho_w_sc / (v.Bw0 * math.exp(-v.cw * (p - v.p_ref)))
        rs = min(rs_oil, float(v.rs_sat(p)))
        pb = float(v.pb_of_rs(rs))
        bo = float(v.bo_sat(pb)) * math.exp(-v.co * (p - pb))
        return (v.rho_o_sc + 0.0136 * rs * gg) / bo

    def _column(self, rs_oil):
        c = self.cfg
        Dd = self.D_ref
        ds = 1.0
        Dtop, Dbot = c.top_depth, c.top_depth + c.h
        grid_up = np.arange(Dd, Dtop - ds, -ds)
        grid_dn = np.arange(Dd, Dbot + ds, ds)

        def integrate(grid, sign):
            pp = [c.p_datum]
            for D in grid[:-1]:
                rho = self._density_profile(pp[-1], D + sign * 0.5 * ds * -1, rs_oil)
                pp.append(pp[-1] + sign * rho * ds / GRAV_CONV)
            return np.array(pp)

        p_up = integrate(grid_up, -1.0)
        p_dn = integrate(grid_dn, +1.0)
        Dall = np.concatenate([grid_up[::-1], grid_dn[1:]])
        pall = np.concatenate([p_up[::-1], p_dn[1:]])
        return Dall, pall

    def _init_state(self):
        c, rp, v = self.cfg, self.rp, self.pvt
        rs_oil = c.Rsi
        for _ in range(4):
            Dall, pall = self._column(rs_oil)
            if c.goc is not None and c.goc > c.top_depth:
                p_goc = float(np.interp(c.goc, Dall, pall))
                rs_oil = float(v.rs_sat(p_goc))
            else:
                break
        self.rs_oil = rs_oil
        pj = np.interp(self.zc, Dall, pall)
        nz = c.nz
        sw_j = np.full(nz, rp.swc)
        x3_j = np.zeros(nz)
        sat_j = np.zeros(nz, bool)
        for j in range(nz):
            D = self.zc[j]
            if c.goc is not None and D < c.goc:
                x3_j[j] = 1 - rp.swc - rp.sorg
                sat_j[j] = True
            elif c.woc is not None and D > c.woc:
                sw_j[j] = 1 - rp.sorw
                x3_j[j] = min(rs_oil, float(v.rs_sat(pj[j])))
            else:
                rs_s = float(v.rs_sat(pj[j]))
                if rs_oil >= rs_s:
                    sat_j[j], x3_j[j] = True, 0.0
                else:
                    x3_j[j] = rs_oil
        N = self.N
        u = np.empty(3 * N + 1)
        u[0:3 * N:3] = pj[self.J]
        u[1:3 * N:3] = sw_j[self.J]
        u[2:3 * N:3] = x3_j[self.J]
        self.p_init = pj[self.J].copy()
        self.pi_ref = self.p_init.copy()
        u[3 * N] = float(np.interp(self.D_ref, Dall, pall))
        self.u0 = u
        self.sat0 = sat_j[self.J].copy()
        if c.wb_grad is None:
            P = self.props(u[0:3 * N:3][self.well_cells], u[1:3 * N:3][self.well_cells],
                           u[2:3 * N:3][self.well_cells], self.sat0[self.well_cells])
            self.wb_grad = float(np.mean(P.rho_o)) / GRAV_CONV
        else:
            self.wb_grad = c.wb_grad
        P = self.props(u[0:3 * N:3][self.well_cells], u[1:3 * N:3][self.well_cells],
                       u[2:3 * N:3][self.well_cells], self.sat0[self.well_cells])
        self.Bo_ref = float(np.mean(P.bo))
        self.gscale = max(c.Rsi, self.rs_oil, 200.0)
        self.oip0, self.gip0, self.wip0 = self.inplace(u, self.sat0)

    # ----------------------------------------------------------- inventories
    def _masses(self, u, sat):
        N = self.N
        p, sw, x3 = u[0:3 * N:3], u[1:3 * N:3], u[2:3 * N:3]
        P = self.props(p, sw, x3, sat)
        so = 1 - sw - P.sg
        pvr = self.pv0 * np.exp(self.cfg.cr * (p - self.pi_ref)) / 5.615
        Mo = pvr * so / P.bo
        Mg = pvr * (P.sg / P.bg + so * P.rs / P.bo)
        Mw = pvr * sw / P.bw
        return Mo, Mg, Mw, P

    def inplace(self, u, sat):
        Mo, Mg, Mw, _ = self._masses(u, sat)
        return float(Mo.sum()), float(Mg.sum()), float(Mw.sum())

    # --------------------------------------------------------------- residual
    def _cell_res(self, u, sat, dt, old, ctrl_pwf=None):
        N = self.N
        pwf = u[3 * N]
        Mo, Mg, Mw, P = self._masses(u, sat)
        Mg = Mg / self.gscale
        p = u[0:3 * N:3]
        Ro = (Mo - old[0]) / dt
        Rg = (Mg - old[1]) / dt
        Rw = (Mw - old[2]) / dt
        ia, ib, dD = self.ia, self.ib, self.dD
        km = self._km(p)
        T = self.T * 0.5 * (km[ia] + km[ib])
        dp = p[ia] - p[ib]
        do = dp - 0.5 * (P.rho_o[ia] + P.rho_o[ib]) * dD / GRAV_CONV
        dg = dp - 0.5 * (P.rho_g[ia] + P.rho_g[ib]) * dD / GRAV_CONV
        dw = dp - 0.5 * (P.rho_w[ia] + P.rho_w[ib]) * dD / GRAV_CONV
        uo = np.where(do > 0, ia, ib)
        ug = np.where(dg > 0, ia, ib)
        uw = np.where(dw > 0, ia, ib)
        fo = T * P.lam_o[uo] * do
        fg = (T * P.lam_g[ug] * dg + P.rs[uo] * fo) / self.gscale
        fw = T * P.lam_w[uw] * dw
        Ro = Ro + np.bincount(ia, fo, N) - np.bincount(ib, fo, N)
        Rg = Rg + np.bincount(ia, fg, N) - np.bincount(ib, fg, N)
        Rw = Rw + np.bincount(ia, fw, N) - np.bincount(ib, fw, N)
        wc = self.well_cells
        dpw = p[wc] - (pwf + self.wb_grad * (self.Dc[wc] - self.D_ref))
        WIk = self.WI * km[wc]
        qo = WIk * P.lam_o[wc] * dpw
        qg = WIk * (P.lam_g[wc] + P.rs[wc] * P.lam_o[wc]) * dpw / self.gscale
        qw = WIk * P.lam_w[wc] * dpw
        Ro[wc] += qo
        Rg[wc] += qg
        Rw[wc] += qw
        R = np.empty(3 * N)
        R[0::3], R[1::3], R[2::3] = Ro, Rg, Rw
        return R

    def _km(self, p):
        g = self.cfg.gamma
        if g <= 0:
            return np.ones(self.N)
        return np.exp(-g * np.maximum(self.p_init - p, 0.0))

    def well_rates(self, u, sat):
        """Sandface surface rates: qo [stb/d], qg [scf/d], qw [stb/d] (production +)."""
        N = self.N
        wc = self.well_cells
        p = u[0:3 * N:3][wc]
        P = self.props(p, u[1:3 * N:3][wc], u[2:3 * N:3][wc], sat[wc])
        dpw = p - (u[3 * N] + self.wb_grad * (self.Dc[wc] - self.D_ref))
        WIk = self.WI * self._km(u[0:3 * N:3])[wc]
        qo = WIk * P.lam_o * dpw
        qg = WIk * (P.lam_g + P.rs * P.lam_o) * dpw
        qw = WIk * P.lam_w * dpw
        return float(qo.sum()), float(qg.sum()), float(qw.sum())

    def _well_res(self, u, sat, dt, pwf_old, ctrl):
        N = self.N
        if ctrl[0] == "p":
            return u[3 * N] - ctrl[1]
        qo = self.well_rates(u, sat)[0]
        C = self.cfg.C
        stor = (C / self.Bo_ref) * (u[3 * N] - pwf_old) / dt if C > 0 else 0.0
        return stor - qo + ctrl[1]

    # --------------------------------------------------------------- Jacobian
    def _eps(self, sat):
        e = np.empty(3 * self.N)
        e[0::3] = 0.02
        e[1::3] = 1e-5
        e[2::3] = np.where(sat, 1e-5, 0.05)
        return e

    def _jacobian(self, u, sat, dt, old, pwf_old, ctrl, R0, Rw0):
        N = self.N
        n = 3 * N + 1
        eps = self._eps(sat)
        rows, cols, vals = [], [], []
        for col_cells in self.colors:
            for vv in range(3):
                idx = 3 * col_cells + vv
                up = u.copy()
                up[idx] += eps[idx]
                dR = (self._cell_res(up, sat, dt, old) - R0)
                nb = self.nbr[col_cells]                      # (m,5)
                for s in range(5):
                    m = nb[:, s]
                    ok = m >= 0
                    if not ok.any():
                        continue
                    mm, cc, ee = m[ok], idx[ok], eps[idx][ok]
                    for e in range(3):
                        r = 3 * mm + e
                        rows.append(r); cols.append(cc); vals.append(dR[r] / ee)
        # pwf column
        up = u.copy(); up[3 * N] += 0.02
        dR = (self._cell_res(up, sat, dt, old) - R0) / 0.02
        nzr = np.where(dR != 0.0)[0]
        rows.append(nzr); cols.append(np.full(len(nzr), 3 * N)); vals.append(dR[nzr])
        # well row
        for k in range(3 * N, 3 * N + 1):
            pass
        for c_ in self.well_cells:
            for vv in range(3):
                idx = 3 * c_ + vv
                up = u.copy(); up[idx] += eps[idx]
                d = (self._well_res(up, sat, dt, pwf_old, ctrl) - Rw0) / eps[idx]
                rows.append(np.array([3 * N])); cols.append(np.array([idx])); vals.append(np.array([d]))
        up = u.copy(); up[3 * N] += 0.02
        d = (self._well_res(up, sat, dt, pwf_old, ctrl) - Rw0) / 0.02
        rows.append(np.array([3 * N])); cols.append(np.array([3 * N])); vals.append(np.array([d]))
        J = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(n, n)).tocsc()
        return J

    # ------------------------------------------------------------------ Newton
    def _switch(self, u, sat):
        N = self.N
        p = u[0:3 * N:3]
        x3 = u[2:3 * N:3]
        rs_s = self.pvt.rs_sat(p)
        to_unsat = sat & (x3 < 0.0)
        to_sat = (~sat) & (x3 > rs_s)
        changed = bool(to_unsat.any() or to_sat.any())
        if changed:
            x3 = x3.copy()
            x3[to_unsat] = rs_s[to_unsat]
            x3[to_sat] = 0.0
            sat = sat.copy()
            sat[to_unsat] = False
            sat[to_sat] = True
            u = u.copy()
            u[2:3 * N:3] = x3
        return u, sat, changed

    def _clip(self, u, sat):
        N = self.N
        sw = u[1:3 * N:3]
        x3 = u[2:3 * N:3]
        np.clip(sw, 1e-4, 1.0 - 1e-4, out=sw)
        room = 1.0 - sw
        x3[:] = np.where(sat, np.minimum(x3, room), np.maximum(x3, 0.0))
        return u

    def _newton(self, u_old, sat_old, dt, ctrl, maxit=14):
        N = self.N
        old = self._masses(u_old, sat_old)[:3]
        old = (old[0], old[1] / self.gscale, old[2])
        pwf_old = u_old[3 * N]
        u, sat = u_old.copy(), sat_old.copy()
        for it in range(1, maxit + 1):
            R0 = self._cell_res(u, sat, dt, old)
            Rw0 = self._well_res(u, sat, dt, pwf_old, ctrl)
            J = self._jacobian(u, sat, dt, old, pwf_old, ctrl, R0, Rw0)
            rhs = -np.concatenate([R0, [Rw0]])
            try:
                du = spsolve(J, rhs)
            except Exception as exc:                         # pragma: no cover
                raise NewtonFailure(str(exc))
            if not np.all(np.isfinite(du)):
                raise NewtonFailure("non-finite update")
            dp_, ds_ = du[0:3 * N:3], du[1:3 * N:3]
            dx_ = du[2:3 * N:3]
            dxs = np.where(sat, np.abs(dx_), 0.0)
            dxr = np.where(sat, 0.0, np.abs(dx_))
            alpha = min(1.0, 400.0 / max(np.abs(dp_).max(), 1e-12), 400.0 / max(abs(du[3 * N]), 1e-12),
                        0.2 / max(np.abs(ds_).max(), 1e-12), 0.2 / max(dxs.max(), 1e-12),
                        200.0 / max(dxr.max(), 1e-12))
            u = u + alpha * du
            u = self._clip(u, sat)
            u, sat, sw_changed = self._switch(u, sat)
            small = (np.abs(dp_).max() < 0.02 and abs(du[3 * N]) < 0.02 and np.abs(ds_).max() < 2e-5
                     and dxs.max() < 2e-5 and dxr.max() < 0.05)
            if alpha >= 1.0 and small and not sw_changed:
                return u, sat, it
        raise NewtonFailure("no convergence")

    # --------------------------------------------------------------- run loop
    @staticmethod
    def _norm_schedule(schedule):
        out = []
        for e in schedule:
            if len(e) == 2:
                out.append((float(e[0]), "q", float(e[1])))
            else:
                out.append((float(e[0]), str(e[1]), float(e[2])))
        out.sort(key=lambda x: x[0])
        if not out or out[0][0] > 0:
            raise ValueError("schedule must start at t=0")
        return out

    def run(self, schedule, t_end, snap_times=(), dt0=2e-4, growth=1.35, dt_max=None,
            dp_target=120.0, ds_target=0.05, max_steps=4000):
        """Run the well schedule. Times in hours, rates stb/d, bhp psi.

        schedule  [(t_hr, qo_stb_d), ...] or [(t_hr, 'bhp', psi), ...]
        Returns dict with history arrays and snapshots.
        """
        sch = self._norm_schedule(schedule)
        N = self.N
        events = [s[0] / 24.0 for s in sch]
        t_end_d = t_end / 24.0
        stops = sorted(set([t_end_d] + [e for e in events if 0 < e < t_end_d] +
                           [s / 24.0 for s in snap_times if 0 < s <= t_end]))
        dt0_d = dt0 / 24.0
        dt_max_d = (dt_max or t_end) / 24.0
        u, sat = self.u0.copy(), self.sat0.copy()
        t = 0.0
        dt_cur = dt0_d
        hist = {k: [] for k in ("t", "pwf", "qo", "qg", "qw", "pavg", "Np", "Gp", "Wp", "iters", "qset")}
        snaps = []
        Np = Gp = Wp = 0.0
        snap_left = sorted(s / 24.0 for s in snap_times)

        def record(t_, u_, sat_, its=0, qset=0.0):
            qo, qg, qw = self.well_rates(u_, sat_)
            hist["t"].append(t_ * 24.0); hist["pwf"].append(float(u_[3 * N]))
            hist["qo"].append(qo); hist["qg"].append(qg); hist["qw"].append(qw)
            Mo, Mg, Mw, _ = self._masses(u_, sat_)
            pv = self.pv0
            hist["pavg"].append(float((u_[0:3 * N:3] * pv).sum() / pv.sum()))
            hist["Np"].append(Np); hist["Gp"].append(Gp); hist["Wp"].append(Wp)
            hist["iters"].append(its)
            hist["qset"].append(qset)

        def snapshot(t_, u_, sat_):
            P = self.props(u_[0:3 * N:3], u_[1:3 * N:3], u_[2:3 * N:3], sat_)
            shp = (self.cfg.nr, self.cfg.nz)
            snaps.append({"t_hr": t_ * 24.0, "p": u_[0:3 * N:3].reshape(shp).copy(),
                          "sw": u_[1:3 * N:3].reshape(shp).copy(), "sg": P.sg.reshape(shp).copy(),
                          "rs": P.rs.reshape(shp).copy()})

        record(0.0, u, sat, 0, 1.0 if sch[0][1] == 'p' or sch[0][2] > 0 else 0.0)
        snapshot(0.0, u, sat)
        steps = 0
        while t < t_end_d - 1e-12:
            ctrl = next((s[1:] for s in reversed(sch) if s[0] / 24.0 <= t + 1e-12), sch[0][1:])
            nxt = min(s for s in stops if s > t + 1e-12)
            dt = min(dt_cur, nxt - t, dt_max_d)
            if nxt - t - dt < 0.2 * dt:                      # avoid a tiny trailing step
                dt = nxt - t
            while True:
                try:
                    un, sn, its = self._newton(u, sat, dt, ctrl)
                    break
                except NewtonFailure:
                    dt *= 0.4
                    dt_cur = min(dt_cur, dt)             # do not retry the failing size next step
                    if dt < 1e-9:
                        raise NewtonFailure(f"time step collapsed at t={t * 24:.4g} h")
            qo, qg, qw = self.well_rates(un, sn)
            Np += qo * dt; Gp += qg * dt; Wp += qw * dt
            dpmax = float(np.abs(un[0:3 * N:3] - u[0:3 * N:3]).max())
            dsmax = float(np.abs(un[1:3 * N:3] - u[1:3 * N:3]).max())
            u, sat = un, sn
            t += dt
            steps += 1
            if steps > max_steps:
                raise NewtonFailure("max_steps exceeded")
            record(t, u, sat, its, ctrl[1] if ctrl[0] == 'q' else 1.0)
            while snap_left and t >= snap_left[0] - 1e-12:
                snapshot(t, u, sat)
                snap_left.pop(0)
            if any(abs(t - e) < 1e-12 for e in events):      # schedule change: restart small
                dt_cur = dt0_d
            else:
                f = min(growth, dp_target / max(dpmax, 1e-9), ds_target / max(dsmax, 1e-9))
                if its > 8:
                    f = min(f, 0.7)
                dt_cur = max(min(dt_cur * max(f, 0.5), dt_max_d), dt0_d * 0.1)
        if not snaps or abs(snaps[-1]["t_hr"] - t_end) > 1e-6:
            snapshot(t, u, sat)
        out = {k: np.array(v) for k, v in hist.items()}
        out["snapshots"] = snaps
        out["u_final"], out["sat_final"] = u, sat
        flowing = out["qset"] > 0
        out["GOR"] = np.where(flowing & (out["qo"] > 1e-9), out["qg"] / np.maximum(out["qo"], 1e-9), np.nan)
        out["WC"] = np.where(flowing & (out["qo"] + out["qw"] > 1e-9),
                             out["qw"] / np.maximum(out["qo"] + out["qw"], 1e-9), np.nan)
        out["inplace0"] = (self.oip0, self.gip0, self.wip0)
        out["inplace"] = self.inplace(u, sat)
        out["steps"] = steps
        return out
