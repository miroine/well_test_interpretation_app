"""Laplace-space well-test solutions with Stehfest inversion (field-unit engine).

Dimensionless groups (length scale Ls = rw, or xf for fractured wells):
    tD = 0.0002637 k t / (phi mu ct Ls^2)          t in hours
    pD = k h dp / (141.2 q B mu)                   constant-rate unit response
    CD = 0.8936 C / (phi ct h Ls^2)

Every ``pws_*`` function returns the Laplace-space *sandface* pressure for a
unit constant surface rate INCLUDING skin and wellbore storage, so that

    pD(tD)               = invert(pws)
    tD * dpD/dtD (deriv) = tD * invert(u * pws)

Scaled Bessel functions are used throughout so that the small-time (large-u)
end of the Stehfest sum never overflows.
"""
from __future__ import annotations

from functools import lru_cache
from math import factorial

import numpy as np
from scipy.linalg import solve
from scipy.special import i0e, i1e, iti0k0, k0, k0e, k1e

# ------------------------------------------------------------------ Stehfest


@lru_cache(maxsize=None)
def stehfest_coeffs(N: int = 12) -> np.ndarray:
    V = np.zeros(N)
    h = N // 2
    for i in range(1, N + 1):
        s = 0.0
        for k in range((i + 1) // 2, min(i, h) + 1):
            s += (k ** h * factorial(2 * k)) / (
                factorial(h - k) * factorial(k) * factorial(k - 1)
                * factorial(i - k) * factorial(2 * k - i)
            )
        V[i - 1] = (-1) ** (i + h) * s
    return V


def invert(F, t, N: int = 12) -> np.ndarray:
    """Invert Laplace function F(u) (vectorised in u) at times t."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    V = stehfest_coeffs(N)
    ln2 = np.log(2.0)
    out = np.zeros_like(t)
    for i, Vi in enumerate(V, start=1):
        out += Vi * F(i * ln2 / t)
    return out * ln2 / t


# --------------------------------------------------------- reservoir kernels


def f_homogeneous(u):
    return np.ones_like(u)


def f_dual_pss(u, omega, lam):
    """Warren-Root pseudo-steady-state matrix-fracture transfer."""
    return (omega * (1 - omega) * u + lam) / ((1 - omega) * u + lam)


def f_dual_slab(u, omega, lam):
    """Transient slab-matrix (Kazemi / de Swaan) transfer."""
    x = np.sqrt(3.0 * (1 - omega) * u / lam)
    return omega + np.sqrt(lam * (1 - omega) / (3.0 * u)) * np.tanh(x)


# ------------------------------------------------------------ small helpers


def _k_ratio(d, z):
    """K0(d z) / K1(z) for d >= 1, computed without under/overflow."""
    return k0e(d * z) / k1e(z) * np.exp(-(d - 1.0) * z)


def _add_storage_skin(pD_u, u, CD, S):
    """pD_u = unit-rate Laplace pressure without skin/storage (infinite or not)."""
    ps = pD_u + S / u
    return ps / (1.0 + CD * u * u * ps)


# -------------------------------------------------- infinite / bounded radial


def image_terms(u, fu, images):
    """Sum of +/- K0(d z) / K1(z) over image wells.

    images: list of (distance_D, sign); distance is well-to-image, in rw units.
    Returns the value to be divided by (u z) -> use together with 1/(u z).
    """
    z = np.sqrt(u * fu)
    out = np.zeros_like(u)
    for d, sign in images:
        out += sign * _k_ratio(d, z)
    return out


def boundary_images(boundary: dict | None):
    """Convert a boundary description to image-well list [(distance, sign)].

    boundary kinds
      {'type': 'none'}
      {'type': 'fault', 'd': L, 'sealing': True}             one linear fault
      {'type': 'channel', 'd1': L1, 'd2': L2}                 two parallel seals
      {'type': 'corner', 'd1': L1, 'd2': L2}                  90 deg sealing corner
      {'type': 'const_p_parallel', 'd1':..., 'd2':...}        two const-p lines
    Distances are in length-scale (rw) units.
    """
    if not boundary or boundary.get("type", "none") == "none":
        return []
    t = boundary["type"]
    if t == "fault":
        s = 1.0 if boundary.get("sealing", True) else -1.0
        return [(2.0 * boundary["d"], s)]
    if t == "corner":
        a, b = boundary["d1"], boundary["d2"]
        return [(2 * a, 1.0), (2 * b, 1.0), (2 * np.hypot(a, b), 1.0)]
    if t in ("channel", "const_p_parallel"):
        a, b = boundary["d1"], boundary["d2"]
        L = a + b
        odd_sign = 1.0 if t == "channel" else -1.0   # one reflection flips sign
        imgs = []
        for n in range(1, 60):                       # even reflections: +-2nL
            imgs.append((2 * n * L, 1.0))
            imgs.append((2 * n * L, 1.0))
        for n in range(-60, 60):                     # odd reflections
            imgs.append((abs(2 * n * L + 2 * b), odd_sign))
        return [(d, s) for d, s in imgs if d >= 1.0]
    raise ValueError(f"unknown boundary type {t!r}")


def pws_radial(u, kernel_f, CD, S, boundary=None):
    """Homogeneous / dual-porosity radial flow, finite wellbore, any boundary."""
    fu = kernel_f(u)
    z = np.sqrt(u * fu)
    p0 = k0e(z) / (u * z * k1e(z))
    imgs = boundary_images(boundary)
    if imgs:
        p0 = p0 + image_terms(u, fu, imgs) / (u * z)
    return _add_storage_skin(p0, u, CD, S)


# ----------------------------------------------------------- radial composite


def pws_composite(u, kernel_f, CD, S, rD, M, Dratio, boundary=None):
    """Two-zone radial composite.

    rD     inner-zone radius / rw (> 1)
    M      mobility ratio (k/mu)_inner / (k/mu)_outer
    Dratio diffusivity ratio eta_inner / eta_outer
    Outer zone is homogeneous and infinite; inner zone uses kernel_f
    (homogeneous or dual porosity).

    boundary  optional image-well boundary (same dict as pws_radial). Images are
              placed in the outer zone: each contributes M * K0(d z2) / (u z1 K1(z1))
              (outer-zone line source, outer mobility 1/M). Valid when the images lie
              well outside the inner zone (distance to boundary > r_inner); with
              M = D = 1 it reduces exactly to pws_radial.
    """
    fu = kernel_f(u)
    z1 = np.sqrt(u * fu)
    z2 = np.sqrt(u / Dratio)
    a, b = z2 * rD, z1 * rD
    r = z2 * k1e(a) / k0e(a)
    num = M * z1 * k1e(b) - r * k0e(b)
    den = M * z1 * i1e(b) + r * i0e(b)
    beta_e = num / den                                  # beta * exp(2 b)
    w = beta_e * np.exp(-2.0 * z1 * (rD - 1.0))         # beta * exp(2 z1)
    p0 = (k0e(z1) + w * i0e(z1)) / (u * z1 * (k1e(z1) - w * i1e(z1)))
    imgs = boundary_images(boundary)
    if imgs:
        pref = M / (u * z1 * k1e(z1))
        for d, sign in imgs:
            ex = np.minimum(z1 - d * z2, 700.0)
            p0 = p0 + sign * pref * k0e(d * z2) * np.exp(ex)
    return _add_storage_skin(p0, u, CD, S)


# ----------------------------------------------------------- fractured wells


def pws_fracture(u, kernel_f, CD, S, x_obs=0.732, boundary=None):
    """Uniform-flux fracture evaluated at x_obs (0.732 -> infinite-conductivity
    equivalent).  All lengths are in units of fracture half-length xf."""
    fu = kernel_f(u)
    z = np.sqrt(u * fu)
    p0 = (iti0k0(z * (1 + x_obs))[1] + iti0k0(z * (1 - x_obs))[1]) / (2.0 * u * z)
    imgs = boundary_images(boundary)
    for d, sign in imgs:
        p0 = p0 + sign * k0e(d * z) * np.exp(-d * z) / u * 1.0
    return _add_storage_skin(p0, u, CD, S)


# ------------------------------------------------------------- limited entry


def pws_limited_entry(u, kernel_f, CD, S, hD, b_over_h, zw_over_h, boundary=None,
                      n_modes=60):
    """Partially penetrating well in a slab with sealed top/bottom.

    hD        (h/rw) * sqrt(kh/kv)
    b_over_h  perforated fraction of h
    zw_over_h perforation-centre depth from the top as fraction of h
    The perforation is uniform-flux and observed at its midpoint.
    """
    fu = kernel_f(u)
    z0 = np.sqrt(u * fu)
    z1c = zw_over_h - b_over_h / 2.0
    z2c = zw_over_h + b_over_h / 2.0
    s = k0e(z0) / (z0 * k1e(z0))
    for n in range(1, n_modes + 1):
        eps = n * np.pi / hD
        zn = np.sqrt(u * fu + eps * eps)
        coef = (2.0 / (n * np.pi * b_over_h)) * (
            np.sin(n * np.pi * z2c) - np.sin(n * np.pi * z1c)
        ) * np.cos(n * np.pi * zw_over_h)
        s = s + coef * k0e(zn) / (zn * k1e(zn))
    p0 = s / u
    imgs = boundary_images(boundary)
    if imgs:
        p0 = p0 + sum(sg * k0e(d * z0) * np.exp(-d * z0) for d, sg in imgs) / u
    return _add_storage_skin(p0, u, CD, S)


# ------------------------------------------------------------ horizontal wells


CUT = 28.0


def _bc_sign(bc):
    if bc in ("sealed", "no_flow"):
        return 1.0
    if bc in ("const_p", "constant_pressure", "gas_cap", "aquifer"):
        return -1.0
    raise ValueError(f"unknown top/bottom boundary {bc!r}")


def pws_horizontal(u, kernel_f, CD, S, LD, hD, zw_over_h, top="sealed", bottom="sealed",
                   x_obs=0.732, n_gl=32):
    """Horizontal well of length LD (units of rw) in a slab, uniform-flux line source.

    hD         (h/rw) * sqrt(kh/kv)   (vertical coordinate scaled to the isotropic space)
    zw_over_h  well depth from the top as a fraction of h
    top/bottom 'sealed' or 'const_p' (gas cap / aquifer acting as a constant-pressure plane)
    Solved by images in z of the finite line source (exact for the plane boundaries). Each image
    contributes  integral exp(-z d cosh v) dv  over the well length; observed at x = x_obs * LD/2
    (0.732 is the uniform-flux equivalent of the infinite-conductivity condition) and one wellbore
    radius off the line.
    """
    u = np.asarray(u, float)
    shp = u.shape
    uf = u.ravel()
    fu = kernel_f(uf)
    z = np.sqrt(uf * fu)
    st_, sb_ = _bc_sign(top), _bc_sign(bottom)
    zwD = zw_over_h * hD
    xo = x_obs * LD / 2.0
    s_lo, s_hi = -LD / 2.0 - xo, LD / 2.0 - xo
    gx, gw = np.polynomial.legendre.leggauss(n_gl)
    order = np.argsort(z)
    p0 = np.empty_like(uf)
    chunk = 48
    for i0 in range(0, len(order), chunk):
        idx = order[i0:i0 + chunk]
        zc = z[idx]
        zmax_dist = CUT / max(zc.min(), 1e-12)
        M = int(min(np.ceil(zmax_dist / (2.0 * hD)) + 1, 1200))
        m = np.arange(-M, M + 1)
        sA = np.where(m % 2 == 0, 1.0, st_ * sb_)
        zpos = np.concatenate([zwD + 2 * m * hD, -zwD + 2 * m * hD])
        sg = np.concatenate([sA, st_ * sA])
        d = np.sqrt(1.0 + (zpos - zwD) ** 2)
        keep = d < zmax_dist
        d, sg = d[keep], sg[keep]
        dd = d[None, :]
        arg = zc[:, None] * dd
        v1, v2 = np.arcsinh(s_lo / dd), np.arcsinh(s_hi / dd)
        vc = np.arccosh(1.0 + CUT / np.maximum(arg, 1e-12))
        vlo, vhi = np.maximum(v1, -vc), np.minimum(v2, vc)
        width = np.maximum(vhi - vlo, 0.0)
        mid = 0.5 * (vhi + vlo)
        acc = np.zeros_like(arg)
        for xg, wg in zip(gx, gw):
            acc += wg * np.exp(-arg * np.cosh(mid + 0.5 * width * xg))
        integ = 0.5 * width * acc
        p0[idx] = 0.5 * hD * (integ * sg[None, :]).sum(axis=1) / (uf[idx] * LD)
    return _add_storage_skin(p0.reshape(shp), u, CD, S)


# ----------------------------------------------- finite-conductivity fractures


def _I_signed(x):
    """Odd antiderivative of K0: sign(x) * integral_0^|x| K0."""
    ax = np.abs(x)
    return np.sign(x) * iti0k0(ax)[1]


def _frac_geometry(nseg):
    edges = np.linspace(0.0, 1.0, nseg + 1)
    xc = 0.5 * (edges[:-1] + edges[1:])
    dx = 1.0 / nseg
    # g[i, j] = integral_0^{xc_i} of the cumulative flux of unit-flux segment j
    a, b = edges[:-1][None, :], edges[1:][None, :]
    xi = xc[:, None]
    g = np.where(xi <= a, 0.0, np.where(xi <= b, 0.5 * (xi - a) ** 2, dx * (xi - b) + 0.5 * dx ** 2))
    return edges, xc, dx, g


def _self_matrix(z, edges, xc):
    """A[u, i, j] = integral over segment j and its mirror of K0(z |xc_i - x'|)."""
    zz = z[:, None, None]
    a, b = edges[:-1][None, None, :], edges[1:][None, None, :]
    xi = xc[None, :, None]
    direct = _I_signed(zz * (b - xi)) - _I_signed(zz * (a - xi))
    mirror = _I_signed(zz * (b + xi)) - _I_signed(zz * (a + xi))
    return (direct + mirror) / zz


def pws_fracture_fc(u, kernel_f, CD, S, FcD, boundary=None, nseg=24):
    """Finite-conductivity vertical fracture (Cinco-Ley), lengths in xf.

    FcD = kf wf / (k xf).  The fracture is split into nseg uniform-flux elements per wing; the
    pressure drop along the fracture follows from the cumulative flow towards the well.
    """
    return pws_mfhw(u, kernel_f, CD, S, FcD, 1, 1.0, boundary, nseg)


def pws_mfhw(u, kernel_f, CD, S, FcD, n_frac, spacing_D, boundary=None, nseg=10, ngq=6):
    """n_frac identical transverse finite-conductivity fractures on a horizontal well.

    The fractures are parallel, spaced spacing_D (in xf) along the well, all connected to the
    same wellbore pressure at their centres (infinite-conductivity lateral).  Plan-view model
    (vertical flow ignored, as for a fully penetrating fracture).
    """
    u = np.atleast_1d(np.asarray(u, float))
    shp = u.shape
    uf = u.ravel()
    fu = kernel_f(uf)
    z = np.sqrt(uf * fu)
    edges, xc, dx, g = _frac_geometry(nseg)
    nu, N, n = len(uf), nseg, int(n_frac)
    A0 = _self_matrix(z, edges, xc)                              # (nu, N, N)
    gx, gw = np.polynomial.legendre.leggauss(ngq)
    # quadrature points in every element (right-wing coordinates) and mirrored
    xq = (0.5 * (edges[:-1] + edges[1:])[:, None] + 0.5 * dx * gx[None, :]).ravel()   # (N*ngq)
    wq = np.tile(0.5 * dx * gw, N)
    Afull = np.zeros((nu, n * N + 1, n * N + 1))
    cf = 2.0 * np.pi / FcD
    drop = cf * (xc[:, None] * dx - g)[None]
    blocks = [A0 + drop]
    # Uniform spacing makes every interaction depend only on fracture separation.
    for separation in range(1, n):
        dy = separation * spacing_D
        r1 = np.sqrt((xc[:, None] - xq[None, :]) ** 2 + dy ** 2)
        r2 = np.sqrt((xc[:, None] + xq[None, :]) ** 2 + dy ** 2)
        kk = (k0(z[:, None, None] * r1[None]) + k0(z[:, None, None] * r2[None])) * wq[None, None, :]
        blocks.append(kk.reshape(nu, N, N, ngq).sum(axis=3))
    for k in range(n):
        for m in range(n):
            Afull[:, k * N:(k + 1) * N, m * N:(m + 1) * N] = blocks[abs(k - m)]
    Afull[:, :n * N, n * N] = -1.0
    Afull[:, n * N, :n * N] = dx
    rhs = np.zeros((nu, n * N + 1))
    rhs[:, n * N] = 1.0 / (2.0 * uf)
    # Small dense systems are faster through SciPy's LAPACK driver than a
    # threaded NumPy batch; individual solves also support SciPy < 1.16.
    sol = np.empty_like(rhs)
    for j, (a, b) in enumerate(zip(Afull, rhs)):
        sol[j] = solve(a, b, assume_a="gen")
    p0 = sol[:, n * N].reshape(shp)
    u_ = uf.reshape(shp)
    for d, sign in boundary_images(boundary):
        p0 = p0 + sign * k0e(d * z.reshape(shp)) * np.exp(-d * z.reshape(shp)) / u_
    return _add_storage_skin(p0, u_ if shp else uf, CD, S)


# ------------------------------------------------------------ closed reservoirs


def pws_circle(u, kernel_f, CD, S, reD, outer="sealed"):
    """Well at the centre of a circle of radius reD (units of rw); sealed or constant pressure."""
    fu = kernel_f(u)
    z = np.sqrt(u * fu)
    X = z * reD
    E = np.exp(-2.0 * np.minimum(X - z, 350.0))
    if outer == "sealed":
        num = k0e(z) * i1e(X) + k1e(X) * i0e(z) * E
        den = u * z * (k1e(z) * i1e(X) - k1e(X) * i1e(z) * E)
    elif outer in ("const_p", "constant_pressure"):
        num = k0e(z) * i0e(X) - k0e(X) * i0e(z) * E
        den = u * z * (k1e(z) * i0e(X) + k0e(X) * i1e(z) * E)
    else:
        raise ValueError(outer)
    return _add_storage_skin(num / den, u, CD, S)


def _rect_images(xe, ye, xw, yw, sx, sy, rmax):
    """Image lattice of a well at (xw, yw) in [0,xe]x[0,ye]; signed, sorted by distance."""
    mx = int(np.ceil(rmax / (2 * xe))) + 1
    my = int(np.ceil(rmax / (2 * ye))) + 1
    m = np.arange(-mx, mx + 1)
    n = np.arange(-my, my + 1)
    xs = np.concatenate([2 * m * xe + xw, 2 * m * xe - xw])
    sxs = np.concatenate([np.ones(len(m)), np.full(len(m), sx)])
    ys = np.concatenate([2 * n * ye + yw, 2 * n * ye - yw])
    sys_ = np.concatenate([np.ones(len(n)), np.full(len(n), sy)])
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    sg = np.outer(sxs, sys_)
    d = np.hypot(X - xw, Y - yw)
    keep = (d > 1e-9) & (d <= rmax)
    d, sg = d[keep], sg[keep]
    o = np.argsort(d)
    return d[o], sg[o]


def pws_rectangle(u, kernel_f, CD, S, xeD, yeD, xwD, ywD, sides_x="sealed", sides_y="sealed",
                  ncell=3.5):
    """Well at (xwD, ywD) in a rectangle [0,xeD]x[0,yeD] (units of rw), by image lattice.

    Images closer than rmax are summed explicitly; the lattice beyond rmax is replaced by its
    continuum average (exact tail of the K0 integral), which carries the pseudo-steady state.
    sides_x/sides_y: 'sealed' or 'const_p' for the pair of opposite sides.
    """
    sx = 1.0 if sides_x == "sealed" else -1.0
    sy = 1.0 if sides_y == "sealed" else -1.0
    u = np.asarray(u, float)
    shp = u.shape
    uf = u.ravel()
    fu = kernel_f(uf)
    z = np.sqrt(uf * fu)
    rmax = ncell * max(xeD, yeD) * 2.0
    d, sg = _rect_images(float(xeD), float(yeD), float(xwD), float(ywD), sx, sy, float(rmax))
    A = xeD * yeD
    tot = np.empty_like(uf)
    chunk = max(1, int(4e6 // max(len(d), 1)))
    for i0 in range(0, len(uf), chunk):
        zc = z[i0:i0 + chunk, None]
        arg = zc * d[None, :]
        kk = np.where(arg < 700.0, k0e(arg) * np.exp(-np.minimum(arg, 700.0)), 0.0)
        tot[i0:i0 + chunk] = (kk * sg[None, :]).sum(axis=1)
    if sx > 0 and sy > 0:
        x = z * rmax
        tail = np.where(x < 700.0, 2.0 * np.pi / (A * z ** 2) * x * k1e(x) * np.exp(-np.minimum(x, 700.0)), 0.0)
        tot = tot + tail
    p0 = (k0e(z) + tot) / (uf * z * k1e(z))
    return _add_storage_skin(p0.reshape(shp), u, CD, S)


# --------------------------------------------------------------- three-zone composite


def pws_composite3(u, kernel_f, CD, S, r1D, r2D, M12, D12, M13, D13):
    """Three-zone radial composite (inner / middle / outer infinite).

    r1D, r2D   zone radii in rw;  M12, M13 mobility ratios inner/middle, inner/outer;
    D12, D13   diffusivity ratios inner/middle, inner/outer.
    """
    fu = kernel_f(u)
    z1 = np.sqrt(u * fu)
    z2 = np.sqrt(u / D12)
    z3 = np.sqrt(u / D13)
    m2, m3 = 1.0 / M12, 1.0 / M13
    # outer zone: Z = p/q at r2
    a3 = z3 * r2D
    Z = k0e(a3) / (m3 * z3 * r2D * k1e(a3))

    def step(Zout, m, z, rin, rout):
        xo, xi = z * rout, z * rin
        c = m * z * rout
        be = -(k0e(xo) - Zout * c * k1e(xo)) / (i0e(xo) + Zout * c * i1e(xo))
        w = be * np.exp(-2.0 * np.minimum(xo - xi, 350.0))
        return (k0e(xi) + w * i0e(xi)) / (m * z * rin * (k1e(xi) - w * i1e(xi)))

    Z = step(Z, m2, z2, r1D, r2D)          # at r1 from the middle zone
    Zw = step(Z, 1.0, z1, 1.0, r1D)        # at rw from the inner zone
    return _add_storage_skin(Zw / u, u, CD, S)


# ----------------------------------------------------------------- interference


def pws_interference(u, kernel_f, rD, CD=0.0, S=0.0):
    """Pressure at distance rD (units of rw) from an active well with storage and skin.

    The observation point sees the *sandface* rate of the active well, 1/u - CD u pw(u).
    """
    fu = kernel_f(u)
    z = np.sqrt(u * fu)
    pw0 = k0e(z) / (u * z * k1e(z)) + S / u
    pw = pw0 / (1.0 + CD * u * u * pw0)
    qsf = 1.0 - CD * u * u * pw                       # = u * sandface rate (Laplace)
    obs = k0e(rD * z) * np.exp(-(rD - 1.0) * z) / (u * z * k1e(z))
    return qsf * obs


# --------------------------------------------------------------- convenience


def unit_response(pws, tD, N: int = 12):
    """Return (pD, tD*dpD/dtD) for a Laplace sandface function pws(u)."""
    tD = np.atleast_1d(np.asarray(tD, dtype=float))
    ln2 = np.log(2.0)
    pressure, derivative = np.zeros_like(tD), np.zeros_like(tD)
    for i, Vi in enumerate(stehfest_coeffs(N), start=1):
        u = i * ln2 / tD
        value = pws(u)
        pressure += Vi * value
        derivative += Vi * (u * value)
    pD = pressure * ln2 / tD
    dD = tD * (derivative * ln2 / tD)
    return pD, dD
