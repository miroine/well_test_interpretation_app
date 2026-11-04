# Well Test Workbench (Streamlit + Python engine + JavaScript animation)

Pressure-transient analysis app modelled on the reference Well Test Workbench, built for
Python/Streamlit with a JavaScript canvas animation of buildups.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Layout
```
engine/   pure-Python numerics (no Streamlit): laplace, models, analysis, typecurve, gas, qc, samples, viz, units
ui/       Streamlit tabs, input forms, Plotly figures, animation.py (HTML/JS canvas)
tests/    engine tests + ui_smoke.py (stand-in Streamlit/Plotly smoke test)
```
Engine runs in field units internally; the UI converts for Metric display.

## Interactive test design

The **00 Test design** tab includes a chronological rate/pressure timeline with
shaded drawdown and buildup periods. This feature requires **Streamlit 1.57 or newer**
for its bidirectional component API; no JavaScript build step, network assets or
additional plotting package is needed.

The **Reservoir model schematic** above the timeline updates with the selected
model and boundary. It illustrates homogeneous, dual-porosity, radial composite,
fracture, multi-fractured horizontal, horizontal and limited-entry geometry.
Plan views show the selected lateral boundary; horizontal/limited-entry models
use a cross-section with top/bottom conditions or the perforated interval.
Lateral boundaries on those cross-sections are listed but not drawn.
Dimension and model-parameter labels follow the sidebar units.

Use **Play test**, **Pause test**, **Restart**, playback speed and the time slider
to follow the designed drawdown and buildup. Playback starts paused and stops when
the view is hidden or the design is changed. Its time, surface rate and interpolated
bottom-hole pressure come from the current designed test. Gas readouts explicitly
use normalized pseudo-pressure. Geometry is **not to scale**, and animated shading
and flow dots are conceptual, **not a computed spatial pressure/velocity field**.
The numerical equations are unchanged; playback does not modify the design.

- Drag the blue **Start shut-in** handle to change flowing duration while keeping
  the shut-in duration unchanged. The test end moves with it.
- Drag the green **End test** handle to change only the shut-in duration.
- On release, the solver recalculates pressure, diagnostics and design predictions.
  During the drag, shading and rate show a preview; the previous pressure curve
  is dimmed rather than presented as a recalculated result.
- Numeric duration inputs remain available for precise edits. Focus a handle and
  use arrow keys for 0.1-hour edits, or Shift + arrow keys for 10-hour edits.
  Escape cancels a drag. Valid ranges are 1-100,000 hours flowing and
  0.1-100,000 hours shut-in.
- Under **Superpose drawdown and buildup**, independently toggle either period's
  pressure change and derivative. Solid curves are pressure change; dashed curves
  are derivatives. Blue denotes drawdown, green buildup.
- Choose **Agarwal equivalent time** (flow: elapsed time; buildup:
  `tp * dt / (tp + dt)`) or **Elapsed time** within each period. The buildup
  derivative is calculated against the selected log-time coordinate. No amplitude
  shifting or forced curve matching is applied; storage/boundary effects need not
  coincide. Non-positive values cannot be shown on logarithmic axes.

Pressure and surface rate follow the sidebar units and fluid selection. Gas
pressure remains normalized pseudo-pressure, consistent with the existing analytical
engine, and gas surface rate is not the reservoir-equivalent rate used internally.
Edited durations are used by synthetic-test generation, design reports, animation
and project save/open. Overlay visibility/alignment are display preferences, not
saved engineering parameters.

### In-app documentation

The former Roadmap tab is now **Documentation & formulas**: a workflow guide,
symbol/unit tables, dimensionless groups, Laplace/storage/skin equations,
reservoir/boundary methods, Bourdet/Horner diagnostics, variable-rate overlays,
type-curve scaling, regression/Monte Carlo interpretation and export limitations.
Formulas follow the field-unit engine conventions. **How it works** remains the
compact methods overview.

### Interpretation: generate a model, then fit

The Gringarten type-curve family in **03 Type curve** uses a non-negative-skin
finite-radius reference: CD=100 for G>=100, and CD=G with zero skin for G<100,
where G=CD exp(2S). This avoids the positive-real Laplace pole and nonphysical
early-time spikes produced by negative lumped skin with a fixed CD reference.
Both pressure and logarithmic derivative approach pD=tD/CD in early storage.
Plotting and auto-match use the same reference curves. The G similarity is
approximate for finite-radius early times, particularly at low G; validate
low-G matches against the actual parameters in Interpretation. This does not
change the analytical well solver or its negative-skin parameter support.
The family uses denser log-time sampling; invalid log-axis values remain gaps,
not lines joined across removed points.

In **02 Interpretation**, edit the reservoir, well, model and boundary parameters,
then click **Generate model** to calculate a forward response without regression.
After the first generation, edits automatically update the **Working model**
pressure-change and derivative curves. Repeated unchanged responses are cached,
so changing display units does not rerun the solver.

**Fit to data** separately runs automatic regression using the selected free
parameters. The fitted values are loaded back into the parameter controls.
Interpretation and Multi-rate fitting display a calculation panel with a
coffee animation (respecting reduced-motion preferences) and a progress bar.
Progress counts actual completed multi-start attempts, not estimated elapsed
time; it can remain at one percentage during a long attempt or initial screening.
A persistent **Model fitted** success label identifies converged results.
Non-converged results are labelled provisional; failed attempts are reported,
and an entirely failed fit produces an explicit error, not a success label.
Subsequent manual edits move the Working model while **Last fit** remains an
explicitly labelled regression snapshot; it is not silently refitted. The result
table retains the actual starting values for that regression.

Fitting reuses reservoir solutions within a bounded, per-fit cache when only
skin, wellbore storage or initial pressure changes. Skin and storage are applied
in Laplace space before inversion, preserving the original numerical response.
Identical bounded starting vectors are fitted only once. All observations,
distinct starting points, residual weights and solver tolerances are retained.
Complex horizontal/fractured models can still take substantially longer than
radial models. Fit fewer uncertain parameters or turn off **Multi-start** in
Interpretation for a quicker local fit (less robust to local minima). The costly
horizontal, multi-fractured and finite-conductivity fracture buildup models
already omit the starting grid and cap each attempt at 60 solver evaluations;
an analytical estimate and the entered starting values can still give two
distinct attempts.

The **Pressure and rate: data and model** chart compares original pressure
readings, the active QC buildup readings, the Working model and the Last fit on
elapsed test time. Pressure is predicted absolutely from initial pressure, not
shifted to force agreement. Each model's rate trace is its prescribed
single-rate/shut-in control, **not a fitted rate**; recorded rates are overlaid
when available. Gas pressure uses normalized pseudo-pressure. Use **Multi-rate**
for variable-rate interpretation.

Working parameters survive unit changes and project save/open. Replacing the
dataset or applying/restoring QC clears both the working response and the old
fit, so neither is reused against changed data.

After **Run Monte Carlo** in the uncertainty expander, the diagnostic and
pressure/rate plots show shaded **P10-P90** response envelopes and dotted **P50**
curves. Toggle **Show Monte Carlo P10-P90 response envelope** to hide/show them.
Each returned Monte Carlo parameter row is propagated as a complete joint model:
the envelope is not calculated by combining independent parameter percentiles.
Blue shading covers dp/absolute pressure; orange covers derivative.

P10 and P90 mean the numerical 10th and 90th percentiles at each time, a central
80% pointwise range (not petroleum reserves exceedance notation). These are
parameter-response bands, not simultaneous confidence bands or measurement-noise
prediction intervals. The median at each time need not be one realizable model.
Drawdown is an extrapolation when regression used buildup only. Rates are fixed
controls, so they have no Monte Carlo band. Bands remain tied to **Last fit**
while manually editing the Working model; a new fit or dataset/QC change clears
them. Field/Metric display converts the bounds without repeating the ensemble
solver. At least three returned parameter sets are required; incomplete refits
are reported. Model and regression bias are not corrected by these bands.

### Full response and reversible Data & QC

In **Multi-rate**, the rate-history and recorded-pressure plots share shaded,
numbered flow/shut-in periods. **Diagnose period** emphasizes the same interval
on both plots. Drag a period-start marker on either plot and release to apply;
the first start is fixed at zero, and markers cannot cross neighbouring starts
or the record end. Keyboard arrows move by 0.1 hour (Shift: 1 hour); Escape
cancels a drag. Edits change only schedule timings, not recorded pressure or
period rates. Regression and deconvolution results are cleared, diagnostics use
the edited schedule, and project save/open retains it.

The default multi-rate diagnostic is **Period pressure change and derivative**:
select **Periods to superpose** to overlay any combination on elapsed time since
each start. Circle markers are local dp, diamonds its Bourdet derivative;
each period has one color, with solid dp/dashed derivative lines for a fitted
whole-history model when available. Rate increases use local pressure drop,
rate decreases/shut-ins use pressure rise. Initial pressure anchors the first
period; later boundary pressures are interpolated from readings if necessary.
Optionally normalize both quantities by the absolute rate change to compare
unequal-rate steps. Negative/non-finite values remain gaps on the log plot.
This is an observed-period comparison, not deconvolution: earlier rate effects
remain and curves are not forced to coincide. **Rate-superposition diagnostic**
retains the advanced rate-weighted log-time view and its normalized units;
its pressure offset is not local buildup dp.

Multi-rate regression selects fit parameters from the active reservoir model
and boundary, matching Interpretation defaults. A closed or constant-pressure
circle includes **Reservoir radius (re)**; channels/corners include both boundary
distances, and rectangles include their dimensions. Boundary type remains a
model choice under **Outer boundary**, not a continuous regression variable.
Fit selectors and result tables label dimensional parameters in project units.

Generated tests retain their full drawdown and buildup pressure response plus the
surface-rate schedule. In **01 Data & QC**, inspect the full-resolution response
on linear elapsed test time, zoom/pan, or box/lasso-select points on either
pressure or rate. Click **Mark selected points for removal**, then **Apply QC to
the active dataset**. The expandable point-ID selector also supports exact edits
and unmarking individual points.

Use the kept time window and optional pressure range together with manual
exclusions, outlier removal and log-time resampling. Only positive-time buildup
readings enter interpretation; at least three must remain. Recorded rates are
preserved as context, not silently rewritten by pressure QC. Excluded pressure
points remain visible in red. **Restore original response (undo all QC)** brings
back the original readings, including after project save/open. Applying or
restoring QC invalidates old analysis results.

For measured full histories, choose **Upload CSV** and enable **Upload full test
history (pressure and rate)**. Map elapsed time in hours, pressure and surface
rate in the sidebar units, and specify the flow duration/start of the final
shut-in. Duplicate timestamps at a rate change retain both sides of the event.
The last non-zero pre-shut-in rate supplies the interpretation rate; post-shut-in
rates must be zero. Variable flowing rates are flagged: use **Multi-rate** for
rate-history-aware interpretation rather than the single-rate buildup workflow.
Legacy/sample buildup-only records do not invent drawdown pressure; their
separate rate plot is explicitly labelled as a specified schedule.
Synthetic gas pressure is labelled normalized pseudo-pressure, not measured BHP.
Gas CSV imports likewise expect normalized pseudo-pressure in the selected
pressure-equivalent units, matching the existing analytical interpretation;
raw gas gauge BHP must be converted before import.

### Multi-fractured horizontal well performance

The solver reuses equal-separation fracture interactions and shares each Laplace
evaluation between pressure and derivative inversion. The same quadrature,
fracture segmentation, time sampling and Stehfest order are retained.
Small dense fracture systems use SciPy's general LAPACK solver.
Reports reuse the current design curves rather than solving the design again.

The separate **Buildup animation** tab now computes its full spatial-animation
payload only when opened; it is cached until the model, rate or timing changes.
The model schematic in Test design still plays locally without extra solves.
Switching tabs triggers a Streamlit rerun, with design calculations cached.

On the approved Windows environment, default four-fracture design calculations
fell from about 15.6 s to 2.5 s. A fresh browser session selected the model in
3.7 s and applied a flowing-period edit in 2.8 s, with no Streamlit exceptions.
The 12-fracture design calculation took about 7.8 s; larger designs still need
noticeable computation. These are local measurements, not guaranteed limits.
Restart a running Streamlit server after updating the engine to load the changes.

### Export the designed test

Under **Use this design**, **Download designed test (CSV)** exports the current
noise-free drawdown and buildup prediction without another solver run:

- `time_hr`: hours from the start of flow.
- `phase`: `drawdown` or `buildup`.
- `shut_in_elapsed_hr`: zero during flow; hours since shut-in during buildup.
- `surface_rate_...`: oil in stb/day or m3/day; gas in Mscf/day or Sm3/day,
  following the sidebar units. Gas exports the surface rate, not the internal
  reservoir-equivalent rate.
- `bottom_hole_pressure_psi` / `_bar` for oil, or
  `normalized_pseudo_pressure_psi` / `_bar` for gas. The latter is not physical BHP.

The shut-in timestamp appears twice: the last flowing sample and the first
zero-rate sample have the same continuous pressure. This preserves the rate
discontinuity. Software requiring strictly increasing times must treat these
as event sides rather than independent elapsed-time readings.

Expand **Export Eclipse schedule INCLUDE**, enter the existing simulator well
name (1-8 characters, starting with a letter) and minimum BHP, then download
the `.INC` file. Inside the host deck's existing `SCHEDULE` section, at the desired
test start, insert:

```text
INCLUDE
  'PROD_test_schedule.INC' /
```

Match the host deck's `FIELD` or `METRIC` units to the sidebar. The snippet uses
`WCONPROD` with `ORAT` for oil or `GRAT` for gas, followed by `WELOPEN ... SHUT`.
Two `TSTEP` records advance the flowing and shut-in periods, converting hours to
days. Zero-rate tests stay shut throughout. The well remains shut at test end;
there is no `END` keyword, so the host schedule may continue.

This is **not a standalone reservoir deck**: the well, completions, PVT, grid
and any fractures must already exist. Predicted analytical pressure is not imposed
as a simulator constraint. Minimum BHP can prevent the requested rate being
achieved. Set timestep/reporting controls in the host deck to resolve the
transient; the two exported steps only define period boundaries.
Record layout is checked against the OPM Eclipse-compatible keyword definitions;
execution in Eclipse/OPM is not verified in this environment.

#### BHP comparison package

**Download schedule + predicted BHP comparison (.ZIP)** adds:

- `schedule.INC`: rate-controlled schedule with reporting steps at the analytical
  samples, then shut-in. Predicted BHP is **not** a simulator constraint.
- `summary.INC`: insert in `SUMMARY` to request `WBHP` and `WOPR`/`WGPR`.
- `predicted_bhp.csv`: the original prediction columns plus `synthetic_bhp_psi`
  or `_bar` and `reference_status`. These are **synthetic predictions, not measured data**.
- `comparison.json`: unit, timing, datum and alignment instructions.

For gas, synthetic BHP is obtained by inverting the existing normalized
pseudo-pressure table at the current temperature/gravity. This is not a new
gas-flow simulation. Values outside that table's physical pressure range have
blank BHP and `outside_gas_inverse_range`; the exporter does not silently clamp
or extrapolate them.
Non-positive/non-finite oil predictions likewise have blank synthetic BHP and
`nonphysical_oil_prediction`; the raw analytical pressure column is retained.

Simulator `TIME` is in days from the host deck start; subtract the host test-start
time and multiply by 24 before aligning with reference `time_hr`. Eclipse may
freeze or omit `WBHP` for a `SHUT` well. Check the simulator's shut-in semantics
and near-well block pressure rather than treating a frozen `WBHP` as recovery.

#### Standalone numerical .DATA model

Expand **Export full Eclipse .DATA model**. Provide grid refinement/extent,
reservoir top depth, rock compressibility and reference oil density. Fracture
models also expose physical fracture width/porosity and, for the infinite-
conductivity model, the finite conductivity proxy. Boundary buffer pore volume
and reporting density are editable. Accept the displayed approximation list,
then select **Build Eclipse .DATA model**.

The `.DATA` download contains `RUNSPEC`, Cartesian `GRID`, single-phase `PROPS`,
`REGIONS`, pressure initialization, `SUMMARY`, completions and the entire test
`SCHEDULE`. It follows the project **Field/Metric** selection: Metric decks use
`METRIC`, metres, bar absolute, kg/m3, surface m3/day and gas Bg in reservoir
m3/standard m3. Field decks retain `FIELD`, feet, psia and their field-unit PVT.
Compressibilities convert with inverse pressure; permeability remains md and
timesteps remain days in both systems. Export form values follow the same units
and preserve their physical values when switching systems.
Feature edges refine the base
grid, so final cell counts differ from the refinement controls. Exports above
200,000 cells are rejected. Negative/unresolved well indices, out-of-domain
geometry and inconsistent fluid compressibilities are reported as errors.

The model ZIP includes that same `.DATA`, `model.json` with source parameters,
setup and limitations, `predicted_bhp.csv`, reusable INCLUDE files and `USAGE.txt`.
The `.DATA` already embeds controls and summary requests: **do not INCLUDE them
again**. Its start is 1 JAN 2000, with the designed test beginning at day zero.
Change the start date in your simulator workflow if needed.
Changing design/setup inputs hides old downloads until the model is rebuilt.

| Analytical model / boundary | Exported numerical representation |
|---|---|
| Homogeneous | Single-continuum Cartesian reservoir with source k, phi, h and fluid inputs. |
| Dual PSS / transient slab | **Late-time equilibrated single-continuum surrogate only.** omega/lambda and the exchange/derivative dip are not simulated; they are retained as source metadata. No `DUALPORO` equivalence is claimed. |
| Two-/three-zone composite | Stair-stepped radial regions; k and storage map the selected mobility/diffusivity ratios. Oil varies regional compressibility; gas varies porosity, rejecting nonphysical mappings. |
| Infinite-/finite-conductivity fracture | Thin explicit high-k strip, `kf = FcD * k * xf / width`. Infinite conductivity uses an editable large finite proxy. |
| Multi-fractured horizontal | Parallel transverse strips at source spacing, with a common lossless lateral connected at fracture centres. |
| Horizontal | X-directed lateral completions at selected relative depth, with kv/kh. |
| Limited entry | Vertical completions only through the specified perforation interval, with kv/kh. |
| Fault / channel / corner / rectangle | Sealed domain faces placed at the specified distances/dimensions. |
| Circle | Cell-centre `ACTNUM` radius mask; stair-stepped outer face. |
| Constant-pressure fault/circle or horizontal top/bottom | Large-pore-volume pressure-buffer cells, **not an exact pressure constraint or multiphase cap/aquifer**. |
| Infinite acting | Finite truncation at the selected far-field extent; convergence must be checked. |

**Important scientific limits:** conventional well coupling does not reproduce
analytical wellbore storage `C`. Thin strips add fracture storage; coarse grids,
well indices and buffers alter transient response. Oil uses single-phase `PVCDO`
with fluid compressibility `ct - cr`, not inferred live-oil PVT. Gas uses
correlated dry-gas `PVDG`. The export is a numerical approximation requiring
mesh/timestep/boundary checks, not a certified behavior-equivalent conversion.
For closed boundaries that the analytical engine ignores for certain models,
the numerical deck represents the selected boundary and explicitly warns that
the reference therefore uses different physics.

The full model ZIP uses the same units for its deck, schedule INCLUDE and BHP
reference CSV. `model.json` identifies `deck_units`; source engine values are
retained for reproducibility under explicitly field-labelled metadata
(`source_design_units`, `setup_field_units`, `minimum_bhp_psia`, etc.).
Changing units invalidates previously built downloads; build again before exporting.

Local validation found Eclipse 2025.3 installed. Two approved tiny-case attempts,
including the installed launcher's standard licence configuration, stopped with
**licence error -5 before grid execution**. Runner exit zero did not mean success:
the opt-in validator checks the simulator `PRT` error summary and returned failure.
No successful simulator execution or numerical history match is claimed.
`tests/validate_eclipse_design.py` can be rerun explicitly with a working licence;
normal test discovery never checks out a simulator licence.

## Tests
```text
python run_tests.py
python run_tests.py test_session test_data_validation test_ui_consistency test_streamlit_app
```
The runner discovers all `tests/test_*.py` suites, then runs `ui_smoke.py` in a separate
process. It exits nonzero if any suite fails. `run_tests.sh` calls the same runner.
Use an interpreter with the dependencies in `requirements.txt` installed.

On Windows, with an existing approved environment:
```powershell
& 'C:\Appl\py_venv_314\.venv\Scripts\python.exe' run_tests.py
& 'C:\Appl\py_venv_314\.venv\Scripts\python.exe' -m streamlit run app.py
```

Tests cover numerical reference cases, unit conversions, project round trips, input
validation, chart properties and session invalidation. `test_streamlit_app.py` uses
the real Streamlit test runtime for sample loading, type-curve matching, unit/fluid
switching, new model/boundary defaults and project upload/restoration. The stand-in
smoke test additionally exercises expensive fitting and simulation flows, but does
not prove browser rendering.
`test_design.py` checks the schedule against the pressure solver, drawdown and
buildup derivatives, radial-flow superposition, unit conversion, curve selection
and validation/callbacks for timing edits. Real Streamlit tests also check that
edited timing survives unit switching, project serialization and synthetic generation.
`test_mfhw_performance.py` compares optimized fracture matrices against direct
pairwise assembly for 1, 4 and 12 fractures and checks deterministic solver-call
counts, pressure/derivative parity and empty batches. UI regressions verify that
display-only edits reuse curves, timing edits invalidate them, reports avoid
duplicate solves and the full buildup animation is generated only when opened.
`test_design_export.py` checks CSV event sides, pressure/rate units, gas labelling,
Eclipse oil/gas record positions, fractional-hour durations, zero rate and invalid
inputs. Real Streamlit tests also verify downloads follow timing, fluid and unit edits.
`test_eclipse_design.py` checks all model/boundary grid mappings, fracture
conductivity and placement, composite storage, completion intervals, pressure
initialization, PVT units, reporting times, package contents and gas inverse-range
flags. Real Streamlit tests cover acknowledgement, building an MFHW deck, comparison
downloads and stale-deck invalidation after timing edits.

## Status
Delivered: 00 Test design | 01 Data & QC | 02 Interpretation | 03 Type curve | 04 Compare | Multi-rate/DST |
05 Deconvolution (linear regularised variant) | 07 IPR (oil composite/Vogel, gas Fetkovich + LIT) |
08 FMB (tank-model regression, oil and gas) | Buildup animation | project save/open (JSON) |
printable HTML report (Print to PDF).
Models: homogeneous, dual porosity (PSS / slab), radial composite, infinite-conductivity fracture,
limited entry; boundaries: sealing fault, constant-pressure boundary, channel, 90 deg corner; gas via
normalized pseudo-pressure.

06 Simulator: fully implicit three-phase r-z black-oil model (engine/blackoil.py) with automatic
calibration (engine/calibrate.py) and presets: single-phase check, DST near a gas cap, gas coning, water coning
(engine/sim_presets.py). Validated against the analytical homogeneous model (~1% on drawdown dp, buildup within
a few psi) and closed-system mass balance (1e-6). Simplifications: no capillary pressure, well crossflow not
upwinded, oil-phase wellbore storage, single well, field units only in this tab.

## Known limitations
* Boundaries combined with the composite model use outer-zone images (approximate when the boundary is within ~1.2 x r_inner).
* Fracture / limited-entry boundaries use a point-image approximation (images several xf away).
* Regression: oil, noisy-gauge and sealing-fault samples recover truth within a few percent; the
  dual-porosity sample can stop in a local solution (k ~ +10 %, skin off by ~1) - start omega/lambda
  near plausible values and compare alternatives in tab 04.
* Deconvolution is a linear non-negative-derivative formulation, not von Schroeter's nonlinear method.
* Flowing material balance assumes boundary-dominated flow, a volumetric tank and constant J.
* Multi-rate analysis is oil only in this release (gas uses the buildup path).
* Animation pressure map uses the homogeneous / dual-porosity radial shape with the chosen model's sandface rate.
* Project files preserve the analytical dataset, comparison fits, test design, multi-rate/deconvolution,
  FMB results and simulator results. Advanced-tab runs and widget-only settings (including IPR inputs)
  are not persisted; project files are not a complete session snapshot.
* Browser validation and regression tests do not independently certify the physical validity of every model.

## Added in v3.1
Consulting-style theme (ui/theme.py); key-takeaway banners on every tab; simulator unit support; simulator results
saved in the project file; GOR / water-cut calibration; boundaries with the composite model; dual-porosity start
screening (omega, lambda); OPM Flow deck export + Flow CSV overlay (engine/opm_deck.py, unverified against a live Flow
run); combined 'Interpretation + simulation' report.

## Added in v4 (Saphir-oriented)

* Confidence intervals, sensitivity curves and Monte Carlo on every analytical fit.
* Nonlinear deconvolution (ln-derivative parameterisation, exponential-integral kernel).
* New analytical models: horizontal well (gas cap / aquifer), finite-conductivity fracture, multi-fractured horizontal well,
  closed circle / rectangle, three-zone radial composite.
* Tab **09 Advanced**: 2D areal numerical model (polygon outline, sealing or leaky faults, permeability zones, several
  wells), interference and pulse tests, changing storage / rate-dependent skin / stress-sensitive permeability, large gauge
  records (wavelet denoising, smart reduction, event detection, temperature).
* Black-oil simulator: stress-sensitive permeability (`gamma`).
* Engine modules: `laplace.py` (new kernels), `areal.py`, `interference.py`, `wellbore.py`, `reduce.py`.
* Tests: `run_tests.py` automatically discovers regression suites; `run_tests.sh` is a wrapper.

Known limits are listed in the "How it works" tab. As before, Streamlit and Plotly were not available when the package was
built originally. The current audit adds real Streamlit integration coverage and browser checks.

## Interface and consistency audit

The interface uses a McKinsey-inspired navy/white palette, restrained blue accents,
local serif editorial headings and clean sans-serif controls. No logos, proprietary
fonts or affiliation are implied. Reports and charts retain the same palette; warning
and data-series colours remain distinct.

The audit corrected project metadata restoration, stale results after project/data
replacement, missing default fit choices for newer models/boundaries, invalid gauge
imports, Metric IPR labels and a real Plotly type-curve highlighting crash. Technical
terms and numerical solvers are unchanged. See [AUDIT.md](AUDIT.md) for findings,
verification evidence, remaining limits and a manual release checklist.

The audit's full run passed 28 of 30 suites. Existing dual-porosity recovery and
Monte Carlo median-recovery tests failed in the approved environment; the numerical
solvers and test thresholds were left unchanged. This is **not a release sign-off**.
