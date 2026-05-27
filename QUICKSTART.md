# Aquila — Quick-start guide

**Author:** Merouane Hamdani &nbsp;·&nbsp; **License:** MIT &nbsp;·&nbsp; **Version:** 1.0

Welcome. This document walks a first-time user from launch to a finished
PDF report in about ten minutes.

> **Note.** Aquila's visual identity is *inspired by* Equinor ASA's
> publicly-visible corporate design language. The project is not produced
> by, nor affiliated with, Equinor ASA.

---

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`. The
Aquila logo and version banner appear at the top of every page.

---

## Anatomy of the interface

```
┌──────────────────────────────────────────────────────────────────┐
│                   AQUILA  ·  PRESSURE TRANSIENT ANALYSIS    v1.0 │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [Sidebar]                  [Main area]                          │
│                                                                  │
│  Project name               ┌─ Tab 1 · Design & Economics ─┐     │
│  ─────────                  │                              │     │
│  Unit system    Field ▼     │   Forward simulation         │     │
│  Fluid          Oil ▼       │   Cost / deferred revenue    │     │
│                             │   Multi-layer (commingled)   │     │
│  Gas PVT  (if gas)          │                              │     │
│                             └──────────────────────────────┘     │
│  Reservoir defaults                                              │
│  ─────────                  ┌─ Tab 2 · Interpretation ─────┐     │
│   μ, B, φ, ct, rw, Pi       │   CSV upload / demo data     │     │
│                             │   Buildup auto-detection     │     │
│                             │   Regression (radial/linear/ │     │
│                             │     PSS, both fluids)        │     │
│                             │   📊 Model vs Data panel     │     │
│                             │   Multi-layer fit            │     │
│                             └──────────────────────────────┘     │
│                                                                  │
│                             ┌─ Tab 3 · Uncertainty & Report ┐    │
│                             │   Monte Carlo on permeability │    │
│                             │   Tornado sensitivity         │    │
│                             │   PDF report download         │    │
│                             └───────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Walkthrough: a 10-minute first analysis

Goal: load the demo dataset, identify k and skin from a buildup, run
Monte Carlo, and download a PDF report.

### 1. Set up the project context (30 s)

In the **sidebar**:
- Leave **Unit system** as `Field` (psi, ft, STB/D).
- Leave **Fluid type** as `Oil`.
- Leave the reservoir defaults at their values (μ = 1 cp, B = 1.2, φ = 0.15).

### 2. Tab 1 — Design a candidate well (1 min)

You should see a log-log diagnostic plot of ΔP and t·dP/dt for a default
50-mD well producing 2500 STB/D for 72 hr.

- Confirm the **clearance-time warning** is green (✓ duration exceeds
  estimated WBS clearance). If it's red, increase the duration.
- Look at the **Total exposure** metric — this is what the test costs
  including deferred revenue.

Tip: try setting **Skin** to `5` and watch the early-time response
shift up. The plateau height is unaffected (skin doesn't change kh).

### 3. Tab 2 — Interpret the data (3 min)

- Don't upload anything; the app loads a synthetic 2-flow / 2-buildup
  demo dataset automatically.
- You'll see the **raw pressure & rate** plot at the top — two flow
  periods (2500 → 2200 STB/D) separated by buildups.
- Scroll to the **log-log buildup derivatives** plot — two buildup
  signatures should be visible.
- In the **Visual time-window regression** section:
  - Buildup `#1` is selected by default.
  - Leave **Flow regime** on `Radial Flow (IARF)`.
  - Drag the **regression window** so its left edge is around 1 hr
    (clear of the wellbore-storage hump) and right edge near 15 hr.
  - You should see `k ≈ 45 mD` and `Skin ≈ +2` recovered — these match
    the parameters used to generate the demo data.
- The **Window overlay** plot below shows your window plus a horizontal
  IARF plateau line.

### 4. Model-vs-Data four-panel view (2 min)

Click **📊 Model vs Data — full PTA workspace** to expand it.

You'll see four panels in one view:
- **Top-left**: log-log ΔP & t·dP/dt — data as markers, model as dashed
  lines. A good match means the dashes track the markers across all
  log-decades.
- **Top-right**: Horner plot ΔP vs (t_p + Δt)/Δt — the IARF slope.
- **Bottom-left**: full raw pressure history.
- **Bottom-right**: the rate schedule.

Tweak `Model k`, `Model skin`, or `Model WBS C` and watch the dashed
overlay update. The **Match quality** metric reports the log-residual:
< 0.05 is Excellent, < 0.15 is Good.

### 5. Tab 3 — Monte Carlo uncertainty (2 min)

- The slope from Tab 2 has been captured automatically.
- Set **Iterations** to 5000 (default).
- The triangular distributions for `h` (50/75/100 ft) and `μ`
  (0.8/1.0/1.3 cp) are pre-populated. Adjust if your case differs.
- Click **Run Monte Carlo**.

You'll see:
- P90 / P50 / P10 metric tiles (petroleum convention: P90 = conservative).
- A KDE PDF + CDF plot.
- A **tornado chart** showing which input drives most of the spread
  (usually h, since the response is k ∝ 1/h).

### 6. Download the PDF report (1 min)

Scroll to the bottom of Tab 3 and click **Generate PDF report**.

The PDF includes:
- Project header with the active unit-system badge
- All design inputs and economics
- The interpretation result table
- The Monte Carlo P-tiles + a summary chart
- A footer with timestamp

---

## Working with your own data

The CSV format Aquila expects:

```csv
time,pressure,rate
0.000,5000.0,2500.0
0.083,4892.1,2500.0
0.167,4815.3,2500.0
...
30.000,3812.4,0.0     ← buildup starts when rate = 0
30.083,3920.5,0.0
...
```

- **`time`** in hours (or in whatever time unit your gauges export — the
  app reads it as-is, so make sure it's hours).
- **`pressure`** in **psi** if the unit system is `Field`, **bar** if `SI`.
- **`rate`** in **STB/D** (Field) / **m³/d** (SI) for oil, **Mscf/D** /
  **10³ m³/d** for gas.

Buildups are detected wherever the rate drops by more than 1 % of the
maximum rate (with a sensible minimum floor). If your data is very noisy,
increase the **rate threshold** by increasing the decimation slider.

---

## Common situations and what to do

**Q: My buildup derivative isn't flat anywhere — I can't find IARF.**
A: Either (a) wellbore storage hasn't cleared (look at the early hump —
extend test duration), or (b) you're seeing boundary effects (try the
`Closed Boundary (PSS)` regime — late-time derivative rises with unit
slope). Both regimes are now selectable.

**Q: My well is gas, not oil.**
A: Switch **Fluid type** to `Gas` in the sidebar. A new **Gas PVT** panel
appears — set γg, reservoir temperature, and CO₂/H₂S/N₂ if applicable.
Everything downstream automatically switches to pseudo-pressure m(p)
analysis. Axis labels, slope units, and the slope constant (1637 vs 162.6)
all update.

**Q: My well has two clearly different layers.**
A: In Tab 1, open the **Multi-Layer Forward Simulation** expander to
see what a two-layer commingled response looks like. In Tab 2, after
running a regular regression, open **Multi-Layer Fit — Commingled** for
an inverse fit. *Caveat: without PLT rate-allocation data the kh split
between layers is non-unique; the fit reports only aggregates.*

**Q: I want to overlay the model on my data interactively.**
A: That's the **📊 Model vs Data** four-panel expander in Tab 2.

**Q: How do I cite the methods?**
A: See `docs/Aquila_Methods.pdf` (or `Aquila_Methods.tex` if you want to
recompile). Key references are Bourdet (1985), Lee–Wattenbarger (gas),
Dranchuk–Abou-Kassem (Z-factor), Lee-Gonzalez-Eakin (μg).

---

## Keyboard shortcuts

Streamlit native:
- `R` — rerun the app
- `C` — clear cached data
- `Esc` — close any open dialog

---

## Where to look when something goes wrong

| Symptom                                  | Where to check                                 |
| ---                                      | ---                                            |
| App won't start                          | `pip install -r requirements.txt`              |
| Logo doesn't render                      | `assets/aquila_logo.svg` present?              |
| "Buildup not detected"                   | Decimation too high, or rate signal too noisy  |
| "Non-physical (negative) slope"          | Regression window crosses a regime transition  |
| Gas analysis fails                       | PVT panel — check γg, T, composition are sane  |
| Monte Carlo says "Min ≤ Mode ≤ Max"      | Re-order your triangular bounds                |

If you need to escalate, the math layer is fully isolated from the UI —
every module under `modules/` has unit tests in its docstring and can be
exercised directly from a notebook to triangulate where the problem is.
