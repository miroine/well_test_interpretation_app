# Application consistency and release audit

Date: 4 November 2026

## Scope

Reviewed the Streamlit entry point, project persistence and session lifecycle, gauge
import/QC, analytical fit controls, chart builders, IPR display units, reports,
theme and test execution. Numerical reference suites provide regression evidence;
this is not an independent reservoir-engineering certification, penetration test or
dependency vulnerability scan.

## Findings and changes

| Priority | Finding | Resolution |
|---|---|---|
| High | Three-zone composite and closed-boundary defaults included fit parameters missing from the UI options, causing Streamlit errors. | Added the missing labels/options; tested all model/boundary default combinations and real widget selection. |
| High | Real Plotly rejected NumPy booleans in highlighted type curves, crashing after Auto-match. Stand-in charts did not catch it. | Convert the highlight flag to a Python boolean; added real Plotly and Streamlit regressions. |
| High | Opening a project ignored well/fluid/unit metadata and retained old results wherever the new project contained null or missing values. | Load before widgets, restore metadata, and replace project-owned state instead of merging sessions. |
| Medium | Files with the same name and size could be mistaken for an already-open project. | Identify uploaded content by SHA-256 rather than name/size. |
| Medium | Test-design timing/rate/gas inputs and FMB results were incompletely restored. | Persist the relevant state and initialize design widgets from the saved values. |
| Medium | CSV imports could retain infinity, accept the same time/pressure column, or crash on empty positive-time data. | Remove/report non-finite rows; reject invalid column choices and require at least three finite, increasing, positive-time readings. |
| Medium | New datasets/QC could retain type-curve and uncertainty results from the previous data. A new fit could retain old sensitivity/Monte Carlo output. | Centralize dataset invalidation and clear fit-dependent uncertainty when refitting. |
| Medium | Cached sensitivity figures kept old pressure units after switching to Metric. | Cache field-unit numerical results and rebuild the chart using the current display units. |
| Medium | Oil datasets could be interpreted using gas settings, or vice versa. | Warn and stop interpretation until the sidebar fluid matches the dataset. |
| Low | Metric oil IPR summary/theoretical productivity labels remained in field units. | Convert both values and labels consistently with the selected unit system. |
| Low | Broad sidebar colour overrides reduced control readability; heading styles were overridden by Streamlit; metric values could truncate. | Use a light sidebar with navy text, explicitly scoped serif headings, visible focus states and wrapping metric values. |
| Low | Documentation and the shell script enumerated incomplete/outdated test lists. | Add a cross-platform discovery runner and make the shell entry point call it. |

Saved comparison alternatives remain available when changing datasets. They are
intentionally not deleted by the dataset invalidation helper.

## Visual direction

McKinsey-inspired, not an official McKinsey design or affiliation:

- Deep navy and white with restrained blue emphasis.
- Local Georgia editorial headings; local sans-serif controls and chart labels.
- No downloaded fonts, logos or proprietary assets.
- Consistent chart/report palette and distinct diagnostic data colours.
- Polished navigation guidance and result text; technical terminology retained.

## Validation

Approved interpreter: `C:\Appl\py_venv_314\.venv\Scripts\python.exe`.
The default Python lacked NumPy, and Windows policy blocked local environment
creation. The incomplete local environment was removed; the approved interpreter
was used instead.

New test suites:

- `tests/test_session.py`: project replacement, metadata, malformed envelopes,
  allowed state keys and dataset-result invalidation.
- `tests/test_data_validation.py`: non-finite input, duplicate-column selection,
  invalid buildup arrays, Metric cleaning and every unit-kind round trip.
- `tests/test_ui_consistency.py`: all default fit choices, safe text rendering,
  chart styling and actual Plotly highlight flags.
- `tests/test_streamlit_app.py`: real Streamlit sample/Auto-match flows,
  unit/fluid switching, newer model defaults, sensitivity units/refitting and
  project upload with widget restoration.

The browser check used the real local Streamlit server and integrated Playwright
tools, not mocked chart objects. Sample loading and Auto-match completed with no
Streamlit exceptions. The title's computed font was Georgia. Metric IPR displayed
productivity in m3/d/bar, and no visible metric-value overflow was detected.
Startup health endpoint: HTTP 200, body `ok`.
The 390-pixel viewport also rendered without Streamlit exceptions or document-level
horizontal overflow. This is a spot check, not full mobile certification.

| Validation | Result | Python exit code |
|---|---|---|
| Full discovery run: `python run_tests.py` | **FAIL: 28 suites passed, 2 failed** | 1 |
| Final targeted run: session, data validation, UI consistency, real Streamlit, IPR/project and simulator persistence | PASS: 6 suites; all 19 newly added test methods passed | 0 |
| Final `ui_smoke` rerun after sensitivity/report fixes | PASS: fitting, uncertainty, project/report, simulator/calibration and advanced workflows | 0 |
| Real browser: sample loading, Auto-match, Metric IPR, typography, metric overflow and narrow viewport | PASS for checked flows | Not applicable: browser tool API |
| Pylance syntax checks: entry point, session helpers, theme and interpretation | PASS | Not applicable: editor tool API |

PowerShell's execution wrapper emits a constrained-language `SetShouldExit`
error even when Python succeeds. The table records the explicitly captured
`$LASTEXITCODE` from Python, not that wrapper error.

### Open numerical release blockers

These are failures in existing numerical tests and unchanged fitting/model code.
No solver changes or relaxed assertions were used to make the audit appear green.

| Test | Observed | Required |
|---|---|---|
| `test_dual_fit.test_dual_porosity_screening_recovers_omega_lambda` | Fitted permeability 43.673 md for synthetic truth 30 md: about 45.6% high. | Permeability error below 15%. |
| `test_uncertainty.test_monte_carlo_brackets_truth` | Monte Carlo median permeability 133.085 md for truth 65 md: about 104.7% high. P10/P90: 61.882 / 228.940 md, 8 runs. Initial fit was 63.325 md. | Median permeability error below 10%, with ordered percentiles. |

**Overall release verdict: FAIL / not release-ready.** The interface and consistency
fixes passed their targeted checks, but the two numerical failures require a separate
scientific-fitting investigation before a release can be approved. The cause of
these numerical failures has not been established by this audit.

## Remaining release checks and limits

### Follow-up: design comparison and standalone Eclipse export

Added synthetic BHP comparison packages with rate-controlled schedules, `WBHP`
summary requests and explicit relative-time/datum instructions. Gas BHP uses
inverse normalized pseudo-pressure mapping; out-of-range values are blank and
flagged rather than silently clamped. No predicted pressure is imposed as a
simulator control or presented as a field measurement.

Standalone `.DATA` exports follow project Field/Metric units and provide explicit setup for grid, depth, fluid,
fracture and boundary-buffer parameters. Tests cover each model/boundary mapping,
including MFHW strip positions/conductivity and completions, plus reporting
durations, source metadata, package contents and real Streamlit build/invalidation.
The exported approximations require acknowledgement. In particular, dual-porosity
exports are late-time single-continuum surrogates (no omega/lambda exchange), and
analytical wellbore storage is not reproduced. See the README's mapping table.

An installed Eclipse 2025.3 runner was found. With explicit user permission, tiny
oil-deck validation was attempted twice, including the configured launcher licence
environment. Both stopped at **licence error -5 before grid execution**. The runner
returned zero despite the simulator error; the new opt-in validation script checks
the PRT error summary and correctly reports failure. Simulator execution remains
**UNVERIFIED**, not a passing release gate. No licence was used by normal tests.

Metric-export follow-up: converted geometry/datum/diameter, initialization and
control pressures, surface rates, densities, ROCK/PVCDO compressibilities and gas
PVDG volume factors. Comparison files and usage metadata use the selected units.
Setup controls preserve physical values across unit changes; unit changes
invalidate prior builds. Field/Metric numerical parity and real widget
round-trip tests cover these conversions. Source metadata remains explicitly
field-labelled for reproducibility.

### Follow-up: multi-fractured horizontal design lag

Profiling the default four-fracture design found approximately 15.6 s in design
curves, 22.8 s in the eagerly rendered hidden buildup animation, and 10.3 s in
duplicate report curves. Equal-separation fracture blocks and pressure/derivative
Laplace evaluations are now shared; small dense systems use SciPy's general
LAPACK solver. No discretization, inversion order or physical equations changed.
Reports reuse the displayed design buildup, including its short-shut-in sampling.
The detailed buildup animation is generated only while its tab is selected.

Measured design-only computation after optimization: about 2.5 s for four
fractures and 7.8 s for 12. A restarted browser session rendered four-fracture
selection in 3.7 s and a timing edit in 2.8 s without exceptions. The animation
was absent while hidden and rendered correctly when its tab was selected.
Ten affected suites passed, including numerical references, direct pairwise
matrix parity, bounded solver calls, real Streamlit interactions and smoke flows.
These targeted results do not resolve or supersede the numerical release
blockers above.

- Streamlit reports deprecations for `use_container_width` and the HTML component
  API. They still execute in the tested environment; a future compatibility
  update should migrate them with an explicit minimum-version policy.
- Dependencies have minimum versions, not a fully locked reproducible environment.
- OPM Flow deck execution remains unverified against a live OPM installation.
- Advanced-tab runs and widget-only settings (including IPR inputs) are not a
  complete persisted session. Project save/open does not claim to save every UI control.
- Browser checks cover selected journeys, not exhaustive responsive/accessibility
  certification or visual regression across every device and tab.
- Existing physical approximations and model-identifiability limits remain; see
  the README and the app's How it works tab.

## Manual release checklist

- [ ] Load oil and gas samples; check diagnostics and fit recovery against revealed truth.
- [ ] Switch Field/Metric; compare pressure, storage, rate, IPR and report units.
- [ ] Select three-zone composite, closed circle and closed rectangle; run a fit.
- [ ] Run Auto-match and refine shifts; verify the curves, legend and metrics.
- [ ] Compute sensitivity/Monte Carlo, refit, then replace/QC the dataset; confirm old results disappear.
- [ ] Import a CSV containing blanks, infinity, duplicates and non-positive times; confirm cleaning messages or a clear rejection.
- [ ] Save a populated project, open an empty project, then reopen the saved project; confirm metadata and results.
- [ ] Re-upload different project content with the same filename and byte count; confirm it reloads.
- [ ] Export reports for all scopes; print to PDF and check labels, chart colours and page breaks.
- [ ] Run/calibrate the simulator and repeat in Metric; review balance and calibration diagnostics.
- [ ] Check animation playback and advanced workflows on the target workstation.
- [ ] Check keyboard navigation, narrow-screen layout and long well/model names before release.
