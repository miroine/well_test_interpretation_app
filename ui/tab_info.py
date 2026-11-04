"""User documentation, implemented formulas and method notes."""
from __future__ import annotations

import streamlit as st

def documentation():
    st.markdown(r"""
## Documentation & formulas

This reference describes the implemented calculations, not a simulator certification.
Equations below use **field units internally**; the sidebar converts displayed
inputs/results to Metric. Natural logarithms are written $\ln$; semilog slopes
use $\log_{10}$. Pressure is absolute unless explicitly labelled as a change.

### 1. Workflow and units

| Workspace | Recommended use |
|---|---|
| Test design | Select reservoir/well/boundary, generate predicted drawdown and buildup, then export or create synthetic data. |
| Data & QC | Inspect pressure/rate history; exclude points or filter time/pressure; apply QC. Restore originals to undo. |
| Interpretation | Generate model without fitting; edit parameters to move the Working model; Fit to data loads fitted values. Last fit remains a snapshot. |
| Type curve | Match homogeneous storage/skin curves; check assumptions before interpreting shifts. |
| Multi-rate | Edit period starts, overlay local dp/derivative, or use advanced rate-superposition diagnostics; fit the whole record. |
| Deconvolution | Recover a regularised unit-rate response from pressure and rate history. |
| Simulator / Advanced | Numerical or specialised models; verify discretisation, convergence and assumptions separately. |
| IPR / FMB | Stabilised deliverability or tank-model material-balance analysis, not transient type-curve fitting. |

| Symbol | Meaning | Internal units |
|---|---|---|
| $k,h,\phi$ | Permeability, net thickness, porosity | md, ft, fraction |
| $\mu,c_t,B$ | Viscosity, total compressibility, formation volume factor | cp, psi$^{-1}$, rb/stb |
| $q,t,p$ | Oil surface rate, elapsed time, pressure | stb/day, hr, psi |
| $C,S,r_w$ | Wellbore storage, skin, well radius | bbl/psi, dimensionless, ft |
| $r_e,x_f$ | Reservoir radius, fracture half-length | ft |

Metric conversions: 1 ft = 0.3048 m; 1 psi = 1/14.5037738 bar;
1 stb = 0.158987295 m3; 1 Mscf = 28.316847 Sm3.
Compressibility converts inversely with pressure. Eclipse timesteps are **days**,
although workbench test times are hours.

### 2. Dimensionless groups and pressure scaling

$$
t_D=\frac{0.0002637\,k\,t}{\phi\mu c_t L_s^2},\qquad
C_D=\frac{0.8936\,C}{\phi c_t h L_s^2}
$$
$$
p_D=\frac{k h\,\Delta p}{141.2\,q B\mu},\qquad
r_{\rm inv}=\sqrt{\frac{k t}{948\,\phi\mu c_t}}
$$

$L_s=r_w$ for radial/other well models; fractured models use $L_s=x_f$.
Radius of investigation is an estimate, not a sharp physical boundary.

### 3. Laplace-space response, storage and skin

For the finite-radius homogeneous well, $u$ is Laplace frequency,
$z=\sqrt{u}$ and $K_0,K_1$ are modified Bessel functions:

$$
\bar p_0(u)=\frac{K_0(z)}{u z K_1(z)},\qquad
\bar p_w(u)=
\frac{\bar p_0(u)+S/u}{1+C_Du^2[\bar p_0(u)+S/u]} .
$$
$$
p_D(t_D)=\mathcal L^{-1}\{\bar p_w\},\qquad
p_D'=t_D\,\mathcal L^{-1}\{u\bar p_w\}.
$$

The implementation uses 12-term Stehfest inversion. The model derivative uses
the transformed derivative, not finite differences, but numerical inversion
still has finite precision. Early storage tends to $p_D=p_D'=t_D/C_D$;
infinite-acting radial flow tends to $p_D'=0.5$.

### 4. Reservoir models and boundaries

Dual porosity replaces $z$ by $\sqrt{u f(u)}$:

$$
f_{\rm PSS}(u)=
\frac{\omega(1-\omega)u+\lambda}{(1-\omega)u+\lambda}
$$
$$
f_{\rm slab}(u)=\omega+
\sqrt{\frac{\lambda(1-\omega)}{3u}}\,
\tanh\!\sqrt{\frac{3(1-\omega)u}{\lambda}} .
$$

Here $\omega$ is storativity ratio and $\lambda$ interporosity flow parameter.
Composite models enforce pressure/flux continuity between radial zones;
three-zone models repeat this across two interfaces. Fracture conductivity is
$F_{cD}=k_f w_f/(k x_f)$. Finite-conductivity and multi-fracture models solve
coupled fracture elements; horizontal/limited-entry models account for vertical
geometry and anisotropy through their specialised kernels.

Sealing boundaries impose zero normal flux; constant-pressure boundaries impose
zero pressure change at the ideal boundary. Fault/channel/corner models use
signed image wells. Circular boundaries use finite-reservoir Bessel solutions;
rectangles use an image lattice with a continuum tail. Select **Outer boundary**
to choose the condition; regression varies numerical geometry such as $r_e$,
$d_1,d_2$ or $x_e,y_e$, not the categorical boundary type.

### 5. Buildup diagnostics and Horner analysis

For flow duration $t_p$ and time since shut-in $\Delta t$:

$$
t_a=\frac{t_p\Delta t}{t_p+\Delta t},\qquad
H=\frac{t_p+\Delta t}{\Delta t},\qquad
\Delta p=p_{ws}-p_{wf,\rm shut}.
$$

The Bourdet derivative of data uses $x=\ln(t_a)$ and bracketing points $j<i<k$:

$$
p_i'=\frac{
\frac{p_i-p_j}{x_i-x_j}(x_k-x_i)+
\frac{p_k-p_i}{x_k-x_i}(x_i-x_j)}
{x_k-x_j}.
$$

The smoothing control sets the minimum log-time neighbourhood. Endpoints are
undefined; non-positive values cannot be displayed on logarithmic axes.
Horner fits $p_{ws}=m\log_{10}H+b$ within the chosen radial-flow window:

$$
k=\frac{162.6\,q B\mu}{|m|h}
$$
$$
S=1.1513\left[
\frac{p_{\rm 1hr}-p_{wf,\rm shut}}{|m|}
-\log_{10}\!\left(\frac{k}{\phi\mu c_t r_w^2}\right)+3.2275
\right].
$$

Here $p_{\rm 1hr}$ is the fitted pressure at $\Delta t=1$ hour. Window choice
matters; storage, boundaries and multiphase effects can invalidate radial estimates.

### 6. Variable rates and period overlays

For rate steps $\Delta q_j=q_j-q_{j-1}$, with initial rate zero,
and unit-rate pressure-drop response $g$:

$$
p(t)=p_i-\sum_{t_j<t}\Delta q_j\,g(t-t_j).
$$

The default period view plots local pressure drop after a rate increase, or
pressure rise after a decrease/shut-in, and its derivative against log elapsed
period time. Boundary pressure is interpolated from readings if necessary;
the first period uses initial pressure. Optional normalization divides both
curves by $|\Delta q_n|$. Overlaying periods **does not remove earlier rate effects**.

The advanced rate-superposition view uses

$$
\tau_n(t)=\sum_{j=0}^{n}\frac{\Delta q_j}{\Delta q_n}\ln(t-t_j),
\qquad g_n=\frac{p_i-p(t)}{\Delta q_n},
\qquad g_n'=\frac{d g_n}{d\tau_n}.
$$

It displays $|g_n|$ and the signed derivative on positive log axes. Its pressure
offset differs from local buildup dp. Deconvolution instead estimates a
non-negative unit-response derivative using bounded least squares with a
curvature penalty (linear mode), or log-derivative nodes (nonlinear mode).
Regularisation trades detail against stability; it is not unique recovery.

### 7. Type-curve matching

$$
G=C_D e^{2S},\quad x=s_x t,\quad p_D=s_y\Delta p,\quad
k=\frac{141.2\,q B\mu\,s_y}{h},\quad
C=\frac{0.0002637\,k h}{0.8936\,\mu\,s_x},\quad
S=\tfrac12\ln(G/C_D).
$$

The reference family uses $C_D=100$ for $G\ge100$, and $C_D=G,S=0$ below
100 to avoid nonphysical negative lumped-skin early-time poles.
Finite-radius similarity in $G$ is approximate; verify low-G matches in
Interpretation. Buildup matching uses Agarwal time in place of elapsed drawdown time.

### 8. Regression and uncertainty

Buildup regression uses bounded least squares with relative dp residuals and
weighted log-derivative residuals:

$$
r_p=\frac{\Delta p_m-\Delta p_o}{\max(|\Delta p_o|,1\,{\rm psi})},
\qquad
r_d=w_d\,[\ln\max(p_m',10^{-6})-\ln\max(p_o',10^{-6})].
$$

Only valid observed diagnostic locations enter the fit. Whole-record multi-rate
regression uses a separately scaled absolute-pressure objective plus its
superposition derivative. A small RMS does not establish uniqueness.
Jacobian intervals are local linear approximations; sensitivity varies one
parameter at a time. Monte Carlo refits synthetic noisy copies of the fitted
response, using the observed residual standard deviation.

Response envelopes propagate **paired parameter sets**, then take pointwise
P10/P50/P90. P10-P90 is a central 80% numerical-percentile range, not a
simultaneous confidence band or measurement-noise prediction interval.
The median curve need not represent one parameter set. Rates remain prescribed;
there is no rate uncertainty band. Drawdown may be extrapolation from buildup-only fitting.

### 9. Gas, deliverability and numerical exports

Gas transient calculations use DAK Z and Lee viscosity, with initial-pressure
properties in an oil-equivalent normalized pseudo-pressure calculation.
Gas plots and imports labelled normalized pseudo-pressure are **not raw gauge BHP**.
Synthetic inverse-BHP exports flag out-of-range values rather than clamp them.

Oil IPR uses $q=J(p_r-p_{wf})$ above bubble point and Vogel below:

$$
q/q_{\max}=1-0.2(p_{wf}/p_r)-0.8(p_{wf}/p_r)^2,\qquad
J=\frac{k h}{141.2 B\mu[\ln(r_e/r_w)-0.75+S]}.
$$

Gas deliverability supports Fetkovich and quadratic Forchheimer/LIT fits to
stabilised points. FMB uses tank-model regression. The black-oil simulator is
a fully implicit finite-volume model, separate from the analytical kernels;
grid/timestep convergence, material balance and calibration must be checked.

CSV follows sidebar units. Eclipse full decks follow FIELD/METRIC, but are
numerical approximations: analytical storage is omitted; dual porosity is a
late-time single-continuum surrogate; fractures are finite-width strips;
constant-pressure boundaries use high-pore-volume buffers. Simulator execution
and history matching are **not certified** by successful export.

### 10. Limits and source map

Stehfest inversion and long-time rectangle approximations lose accuracy at
extreme dimensionless times (around $10^8$ and beyond). Composite boundaries
use outer-zone images; distant point-image approximations apply to fracture
and limited-entry boundaries. Horizontal/multi-fracture parameters can trade off.
Animation shading is conceptual, not a computed spatial flow field.

Implementation modules: `engine/models.py` (scales/superposition),
`engine/laplace.py` (kernels/inversion), `engine/analysis.py` (derivatives/fitting),
`engine/typecurve.py`, `engine/multirate.py`, `engine/deconv.py`,
`engine/gas.py`, `engine/ipr.py`, `engine/fmb.py`,
`engine/blackoil.py` and `engine/eclipse_design.py`.
Use **How it works** for a compact methods overview.
""")


def how_it_works():
    st.markdown("""
#### How the calculations work

**Backend (Python).** Each reservoir model is written as a Laplace-space solution for a constant-rate well
with wellbore storage and skin (Everdingen-Hurst). Time-domain pressure comes from the Stehfest inversion
(N = 12); the derivative `t dp/dt` is obtained by inverting `u * p(u)`, avoiding time-domain
differencing but retaining numerical inversion error. Multi-rate histories use superposition.

| Model | Method |
|---|---|
| Homogeneous | Finite-radius well, `K0(z) / (u z K1(z))` |
| Dual porosity | Warren-Root pseudo-steady or transient slab transfer function f(u) |
| Radial composite | Two zones with mobility and diffusivity ratios; flux and pressure continuity at the interface |
| Fractured | Uniform-flux fracture evaluated at x = 0.732 xf (infinite-conductivity equivalent) |
| Limited entry | Slab with sealed top/bottom, cosine-mode expansion (60 modes), uniform-flux perforation |
| Horizontal well | Line source with image sources in z (sealed or constant-pressure top/bottom), 32-point quadrature per image |
| Finite-conductivity fracture | Cinco-Ley: uniform-flux elements plus pressure drop along the fracture; also n transverse fractures on a horizontal well |
| Closed reservoir | Circle exact in Bessel functions; rectangle by image lattice plus continuum tail |
| Three-zone composite | Impedance recursion across two interfaces |
| Boundaries | Image wells: sealing/constant-pressure fault, parallel channel, 90 deg corner |
| Gas | Dranchuk-Abou-Kassem Z, Lee-Gonzalez-Eakin viscosity, normalized pseudo-pressure |

**Variable rate.** Superposition diagnostics divide the pressure drop by the last rate change and plot it
against the superposition time, so radial flow again gives a derivative plateau of 0.5 (times the pressure scale).
Deconvolution solves a bounded linear least-squares problem for the non-negative derivative of the unit-rate
response on log-spaced bins with a curvature penalty.

**IPR and FMB.** Oil: linear above the bubble point and Vogel below. Gas: Fetkovich (log-log) and Forchheimer / LIT
(quadratic) from stabilised points. Flowing material balance fits in-place volume and productivity by tank-model regression.

**Simulator.** Black-oil r-z model: Standing / Beggs-Robinson / DAK PVT, Corey relative permeability with Stone II oil,
log-spaced radial cells, fully implicit Newton with a colour-compressed finite-difference Jacobian, variable switching
between saturated (Sg) and undersaturated (Rs) cells, hydrostatic initialisation with GOC / WOC, Peaceman-style well
index with skin, wellbore storage on the oil phase, gravity-segregated upstream weighting. No capillary pressure; the
well crossflow is not upwinded. Calibration runs the simulator inside a bounded least-squares loop.

**Frontend.** Streamlit for the interface and Plotly for the plots. The buildup animation is plain
JavaScript canvas code driven by a JSON payload from the Python engine, so every number on screen
comes from the same solver used for interpretation.

**Verification.** The engine is tested against: the line-source exponential-integral limit, derivative
stabilisation at 0.5, unit-slope storage, the double-slope fault response, constant-pressure decline,
composite limits (M = D = 1), full-penetration limits for limited entry, Horner permeability recovery,
and parameter recovery by regression.

**Advanced tab.** The areal model is a fully implicit single-phase finite-volume code on a graded grid with Peaceman wells;
interference uses the line-source kernel with the active well's sandface rate; the wellbore module convolves the reservoir
response step by step so storage and skin may change with time and rate; gauge tools include a hand-written Daubechies-4
wavelet filter.

**Known limitations.** Horizontal and multi-fracture fits are slow and weakly identifiable (length and kv/kh trade off); the
rectangle lattice and the Stehfest inversion lose accuracy beyond about 1e8 dimensionless time; stress sensitivity is
reversible; the multi-fracture model is plan-view (no vertical flow); the areal model is single-phase.
 Composite-model boundaries use an outer-zone image approximation; fracture and limited-entry
boundary effects use a point-source image approximation (valid when the image is several fracture
lengths away); gas uses constant compressibility-viscosity evaluated at initial pressure with normalized
pseudo-pressure; the pressure map in the animation uses the homogeneous/dual-porosity radial shape.
""")
