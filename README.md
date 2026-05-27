# Aquila — Pressure Transient Analysis

**Author:** Merouane Hamdani &nbsp;·&nbsp; **License:** MIT &nbsp;·&nbsp; **Version:** 1.0

> **Disclaimer.** Aquila is provided *as is*, without warranty of any kind.
> Outputs are intended to assist — not replace — the work of a qualified
> petroleum engineer. The visual identity is *inspired by* publicly-visible
> corporate design conventions; Aquila is **not** produced by, nor affiliated
> with, Equinor ASA, and the Equinor name, logo and wordmark are not used.

---

Modular Streamlit application for **Pressure Transient Analysis** —
well-test design, interpretation, and uncertainty modeling, with an
Equinor-inspired corporate aesthetic and full Field/SI unit support.

Supports **oil, water, and gas wells**. Gas wells use pseudo-pressure
m(p) and pseudo-time transforms with built-in PVT correlations
(Dranchuk-Abou-Kassem Z-factor, Lee-Gonzalez-Eakin viscosity, Sutton +
Wichert-Aziz pseudo-criticals).

## Flow regimes supported

  Regime                     | Oil/Water | Gas
  ---                        | ---       | ---
  Radial flow (IARF)         | ✓         | ✓
  Linear flow (fracture)     | ✓         | ✓
  Closed boundary (PSS)      | ✓         | ✓
  Two-layer commingled       | ✓         | ✓
  Double-permeability        | TODO      | TODO

## Layout

```
pta_app/
├── app.py                  ← Streamlit UI orchestrator (thin)
├── styles.py               ← Centralised theming / CSS / Plotly layout
├── requirements.txt
└── modules/
    ├── __init__.py
    ├── engine.py           ← Physics: line-source, Fair-Schulz, Bourdet,
    │                         regression (radial, linear, PSS).
    │                         Liquid + gas wrappers; closed-boundary
    │                         simulator.
    ├── pvt.py              ← PVT: Z-factor (DAK), μg (LGE), pseudo-
    │                         pressure m(p), pseudo-time tₐ.
    ├── layered.py          ← Multi-layer: commingled (no-crossflow)
    │                         forward sim and inverse fit. Stehfest
    │                         infrastructure ready for double-perm.
    ├── laplace.py          ← Stehfest numerical Laplace inversion
    │                         (validated to 0.30% on K₀(√s)/s benchmark).
    ├── economics.py        ← Cost & deferred-revenue calculator
    ├── uncertainty.py      ← Monte Carlo (oil & gas) + KDE/CDF
    ├── units.py            ← Field ↔ SI conversions (exact factors)
    └── reporter.py         ← FPDF publication-ready PDF builder
```

Math is fully isolated from Streamlit, so each module is unit-testable and
re-usable from notebooks or batch pipelines.

## Fluid handling

Choose **Oil / Gas / Water** in the sidebar. The engine internally:
- For liquid (oil/water): operates on ΔP and Δt directly.
- For gas: transforms pressure → m(p) = 2∫p/(μZ)dp, time → tₐ. The same
  line-source / Fair-Schulz / Bourdet / IARF machinery applies in
  pseudo-variables; only the slope constant changes (162.6 → 1637).

Gas wells expose a PVT panel for γg, Tres, and CO₂/H₂S/N₂ composition.
The Wichert-Aziz correction is applied automatically when acid gases are
present.

## Closed boundaries

Tab 1 — toggle "Add a closed circular drainage boundary at radius re".
The forward simulator adds a pseudo-steady-state ramp that takes over
after the boundary is felt. On a log-log Bourdet derivative, this shows
as a unit-slope rise at late time.

Tab 2 — pick the "Closed Boundary (PSS)" flow regime. The regression
extracts the PSS slope m* (or m_g* for gas) and computes drainage area
and equivalent drainage radius.

## Multi-layer (commingled)

Tab 1 — expander "Multi-Layer Forward Simulation" lets you specify two
layers (k₁,h₁,φ₁ and k₂,h₂,φ₂) and visualises the response against the
equivalent single-layer kh_total. The qualitative signature is a step
DOWN in the Bourdet derivative as the slower layer joins the flow.

Tab 2 — expander "Multi-Layer Fit" runs an inverse fit on the selected
buildup to recover kh_total and an indicative κ. Without PLT rate data
the inverse problem is non-unique; the routine reports aggregates only
and surfaces an explicit warning.

## Coming soon

Double-permeability with crossflow (Bourdet 1985). The Stehfest Laplace-
inversion infrastructure is already in place and validated against five
known transforms including the radial-diffusion benchmark to 0.30%.
What's missing is a verified reference for the Laplace-space transfer
function f(s; κ, ω, λ) — pending verification against published type-
curves or commercial PTA software output.

## Unit handling

The engine is **always** internal-canonical in oil-field units. The UI is
the only place that converts:
```
user input  →  units.to_field()    →  engine
engine      →  units.from_field()  →  display
```
A single radio in the sidebar switches every input field, every axis label,
every metric, and the PDF report between Field and SI without touching the
physics. Conversion factors are derived from exact SI definitions:

| Quantity              | Field unit | SI unit  | Notes                          |
|-----------------------|------------|----------|--------------------------------|
| permeability k        | mD         | mD       | same — PTA convention          |
| net pay h, rw         | ft         | m        | 1 ft = 0.3048 m (exact)        |
| rate q                | STB/D      | m³/d     | 1 bbl = 0.158987294928 m³      |
| skin s, porosity φ    | –          | –        | dimensionless                  |
| wellbore storage C    | bbl/psi    | m³/bar   |                                |
| viscosity μ           | cp         | mPa·s    | numerically identical          |
| FVF B                 | rb/STB     | rm³/m³   | numerically identical          |
| compressibility cₜ    | 1/psi      | 1/bar    | 1 psi = 6894.757293168 Pa      |
| pressure P, ΔP        | psi        | bar      |                                |
| time t                | hr         | hr       | same — universal PTA convention|

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tabs

1. **Design & Economics** — forward simulation (Fair-Schulz + radial line-source),
   log-log diagnostic plot, wellbore-storage-clearance warning, cost donut chart.
2. **Interpretation** — CSV ingest with decimation, dynamic event detection,
   superposed buildup derivatives, visual time-window regression for IARF and
   linear-flow regimes.
3. **Uncertainty & Report** — Monte Carlo (≥1 000 iters) with triangular
   distributions on h and μ, KDE PDF + CDF expectation curve with P90/P50/P10,
   and a one-click branded PDF report tagged with the active unit system.

## Conventions

* Petroleum probability convention: P90 = conservative (low), P50 = best,
  P10 = optimistic (high).
* CSVs uploaded for interpretation are assumed to be in the **currently
  selected** unit system; switch to the matching unit system **before**
  uploading.


---

## License

This software is licensed under the [MIT License](LICENSE).

See the `LICENSE` file for full terms, the additional engineering-use
disclaimer, and the trademark / affiliation notice (Aquila is NOT
affiliated with Equinor ASA).

## Citation

If you use Aquila in published work, please cite as:

> Hamdani, M. (2026). *Aquila — Pressure Transient Analysis Workbench*
> (Version 1.0) [Computer software]. MIT License.

The underlying analytical methods are referenced in `docs/Aquila_Methods.pdf`.

## Contact

Author: **Merouane Hamdani**
