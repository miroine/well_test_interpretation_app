"""
modules/units.py
----------------
Unit system handling for the PTA Workbench.

Strategy
--------
The physics engine in `engine.py` is **always** in OILFIELD UNITS internally:
    q [STB/D], B [rb/STB], μ [cp], k [mD], h [ft], t [hr], P [psi],
    rw [ft], ct [1/psi], C [bbl/psi]

The UI calls `to_field(value, quantity, system)` on every input on the way in,
and `from_field(value, quantity, system)` on every output on the way out.
This isolates all unit handling in one well-tested layer and keeps the
engine constants (162.6, 70.6, 948, 0.0002637, etc.) untouched.

Conversion factors are derived from EXACT SI base unit definitions:
    1 ft  = 0.3048 m                  (international foot, exact)
    1 psi = 6894.757293168 Pa         (1 lbf/in², conventional value)
    1 bar = 100 000 Pa                (exact, by definition)
    1 bbl = 0.158987294928 m³         (42 US gal × 3.785411784 L/gal, exact)
    1 cp  = 1 mPa·s                   (exact, by definition)

We deliberately keep PTA-conventional units for both systems rather than going
to strict SI base units; this matches how engineers actually report results:

    quantity              FIELD              SI (engineering)
    ------------------    ----------------   --------------------
    permeability k        mD                 mD          (same)
    net pay h             ft                 m
    rate q                STB/D              m³/d
    skin s                –                  –           (dimensionless)
    wellbore storage C    bbl/psi            m³/bar
    viscosity μ           cp                 mPa·s       (numerically identical)
    FVF B                 rb/STB             rm³/m³      (numerically identical)
    porosity φ            –                  –           (dimensionless)
    compressibility cₜ    1/psi              1/bar
    wellbore radius rw    ft                 m
    pressure P            psi                bar
    pressure drop ΔP      psi                bar
    time t                hr                 hr          (same)
    oil price             USD/STB            USD/m³
    money                 USD                USD         (same)
"""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass


class UnitSystem(str, Enum):
    FIELD = "Field"
    SI = "SI"


# =====================================================================
#  Exact factors (derived from SI definitions)
# =====================================================================
M_PER_FT = 0.3048                       # exact
PA_PER_PSI = 6894.757293168             # exact conventional value
M3_PER_BBL = 42.0 * 3.785411784 * 1e-3  # exact (US barrel = 42 US gal)

# Derived
FT_PER_M = 1.0 / M_PER_FT                       # 3.28084...
BBL_PER_M3 = 1.0 / M3_PER_BBL                   # 6.28981...
PSI_PER_BAR = 1e5 / PA_PER_PSI                  # 14.50377...
BAR_PER_PSI = 1.0 / PSI_PER_BAR                 # 0.06894757...
PSI_PER_KPA = 1e3 / PA_PER_PSI                  # 0.14503774...
# Wellbore storage: bbl/psi vs m³/bar
#   C[bbl/psi] = C[m³/bar] * (bbl/m³) / (psi/bar)
BBLPSI_PER_M3BAR = BBL_PER_M3 / PSI_PER_BAR    # 0.43366718...


# =====================================================================
#  Per-quantity factor table
#
#  Each entry contains:
#     to_field    multiplier such that  value_field = value_SI * to_field
#     from_field  multiplier such that  value_SI    = value_field * from_field
#     field_unit  display label in field units
#     si_unit     display label in SI engineering units
#     fmt         default format string for display
# =====================================================================
@dataclass(frozen=True)
class _UnitDef:
    to_field: float
    from_field: float
    field_unit: str
    si_unit: str
    fmt: str = "{:.3f}"


_TABLE: dict[str, _UnitDef] = {
    # --- Reservoir / well geometry ---
    "length":          _UnitDef(FT_PER_M,        M_PER_FT,         "ft",     "m",       "{:.2f}"),
    "height":          _UnitDef(FT_PER_M,        M_PER_FT,         "ft",     "m",       "{:.1f}"),
    "wellbore_radius": _UnitDef(FT_PER_M,        M_PER_FT,         "ft",     "m",       "{:.3f}"),
    # --- Pressure ---
    "pressure":        _UnitDef(PSI_PER_BAR,     BAR_PER_PSI,      "psi",    "bar",     "{:.1f}"),
    "delta_pressure":  _UnitDef(PSI_PER_BAR,     BAR_PER_PSI,      "psi",    "bar",     "{:.2f}"),
    # --- Liquid flow rate (oil/water, surface conditions) ---
    "rate":            _UnitDef(BBL_PER_M3,      M3_PER_BBL,       "STB/D",  "m³/d",    "{:.1f}"),
    # --- Wellbore storage ---
    "wbs":             _UnitDef(BBLPSI_PER_M3BAR, 1.0 / BBLPSI_PER_M3BAR,
                                "bbl/psi", "m³/bar", "{:.4f}"),
    # --- Compressibility (inverse pressure) ---
    "compressibility": _UnitDef(1.0 / PSI_PER_BAR, PSI_PER_BAR,    "1/psi",  "1/bar",   "{:.3e}"),
    # --- Permeability (mD in both systems by PTA convention) ---
    "permeability":    _UnitDef(1.0, 1.0,                          "mD",     "mD",      "{:.2f}"),
    # --- Viscosity (cp ≡ mPa·s exactly) ---
    "viscosity":       _UnitDef(1.0, 1.0,                          "cp",     "mPa·s",   "{:.2f}"),
    # --- FVF (rb/STB ≡ rm³/m³ numerically) ---
    "fvf":             _UnitDef(1.0, 1.0,                          "rb/STB", "rm³/m³",  "{:.3f}"),
    # --- Dimensionless quantities ---
    "porosity":        _UnitDef(1.0, 1.0, "–", "–", "{:.3f}"),
    "skin":            _UnitDef(1.0, 1.0, "–", "–", "{:.2f}"),
    # --- Time (hours in both systems) ---
    "time":            _UnitDef(1.0, 1.0, "hr", "hr", "{:.2f}"),
    # --- Money (USD in both systems) ---
    "money":           _UnitDef(1.0, 1.0, "USD", "USD", "{:,.0f}"),
    # --- Oil price (USD/STB ↔ USD/m³) ---
    "oil_price":       _UnitDef(1.0 / BBL_PER_M3,  BBL_PER_M3,
                                "USD/STB", "USD/m³", "{:.2f}"),
    # --- Fracture-capacity product xf·√k (ft·√mD vs m·√mD) ---
    "frac_capacity":   _UnitDef(FT_PER_M, M_PER_FT, "ft·√mD", "m·√mD", "{:.2f}"),
    # --- Semilog slope (psi/cycle vs bar/cycle) ---
    "slope_semilog":   _UnitDef(PSI_PER_BAR, BAR_PER_PSI, "psi/cycle", "bar/cycle", "{:.2f}"),
    # --- Sqrt-time slope (psi/√hr vs bar/√hr) ---
    "slope_sqrt_t":    _UnitDef(PSI_PER_BAR, BAR_PER_PSI, "psi/√hr",   "bar/√hr",   "{:.2f}"),

    # ======================================================================
    #  Gas-specific units
    # ======================================================================
    # Gas rate: Mscf/D ↔ 10³ m³/d (E3m3/d in metric oilfield)
    # 1 Mscf = 1000 scf at standard conditions = 28.3168 m³
    # so 1 Mscf/D = 0.0283168 × 10³m³/D
    "gas_rate":        _UnitDef(1.0 / 0.0283168466,   0.0283168466,
                                "Mscf/D", "10³m³/d", "{:.1f}"),
    # Gas price: USD/Mscf ↔ USD/10³m³
    "gas_price":       _UnitDef(0.0283168466, 1.0 / 0.0283168466,
                                "USD/Mscf", "USD/10³m³", "{:.2f}"),
    # Pseudo-pressure m(p): psi²/cp ↔ kPa²/(mPa·s) ≡ kPa²/cp (since 1 cp = 1 mPa·s)
    # 1 psi = PSI_PER_KPA × 1 kPa = 0.145 kPa → 1 psi² = 0.0210 kPa²
    # value_psi² = value_kPa² × (PSI_PER_KPA)²
    "pseudo_pressure": _UnitDef((PSI_PER_KPA ** 2), 1.0 / (PSI_PER_KPA ** 2),
                                "psi²/cp", "kPa²/(mPa·s)", "{:.3e}"),
    # Pseudo-pressure semilog slope: same as pseudo-pressure unit per cycle
    "slope_psp":       _UnitDef((PSI_PER_KPA ** 2), 1.0 / (PSI_PER_KPA ** 2),
                                "psi²/cp/cycle", "kPa²/(mPa·s)/cycle", "{:.3e}"),
    # Temperature: °F ↔ °C  (NOTE: temperature uses an OFFSET, not a pure multiplier.
    # Pure ratio-based conversion is wrong; we handle temperature via custom helpers
    # below — this entry is here only for labelling.)
    "temperature":     _UnitDef(1.0, 1.0, "°F", "°C", "{:.1f}"),
    # Gas specific gravity (dimensionless)
    "gas_gravity":     _UnitDef(1.0, 1.0, "–", "–", "{:.3f}"),
}


# =====================================================================
#  Temperature requires an OFFSET conversion — keep a separate API.
# =====================================================================
def temperature_to_F(value: float, system: UnitSystem | str) -> float:
    """Convert a temperature in the user's system → °F (engine canonical)."""
    sysname = system.value if isinstance(system, UnitSystem) else str(system)
    if sysname == UnitSystem.FIELD.value:
        return float(value)
    # SI uses °C
    return float(value) * 9.0 / 5.0 + 32.0


def temperature_from_F(value_F: float, system: UnitSystem | str) -> float:
    """Convert a temperature in °F → user's system."""
    sysname = system.value if isinstance(system, UnitSystem) else str(system)
    if sysname == UnitSystem.FIELD.value:
        return float(value_F)
    return (float(value_F) - 32.0) * 5.0 / 9.0


# =====================================================================
#  Public API
# =====================================================================
def to_field(value: float, quantity: str, system: UnitSystem | str) -> float:
    """Convert `value` expressed in `system` to the engine's field-units form."""
    sysname = system.value if isinstance(system, UnitSystem) else str(system)
    if sysname == UnitSystem.FIELD.value:
        return float(value)
    d = _TABLE[quantity]
    return float(value) * d.to_field


def from_field(value: float, quantity: str, system: UnitSystem | str) -> float:
    """Convert `value` (field units) into the user's chosen `system`."""
    sysname = system.value if isinstance(system, UnitSystem) else str(system)
    if sysname == UnitSystem.FIELD.value:
        return float(value)
    d = _TABLE[quantity]
    return float(value) * d.from_field


def label(quantity: str, system: UnitSystem | str) -> str:
    """Return the unit label string for the given quantity in the given system."""
    sysname = system.value if isinstance(system, UnitSystem) else str(system)
    d = _TABLE[quantity]
    return d.field_unit if sysname == UnitSystem.FIELD.value else d.si_unit


def fmt(quantity: str, value: float, system: UnitSystem | str | None = None) -> str:
    """Format a numeric value using the quantity's default format, with unit label appended."""
    d = _TABLE[quantity]
    if system is None:
        return d.fmt.format(value)
    return f"{d.fmt.format(value)} {label(quantity, system)}"


def axis_title(name: str, quantity: str, system: UnitSystem | str) -> str:
    """Build a chart-axis title like 'Pressure (bar)' or 'Δt (hr)'."""
    return f"{name} ({label(quantity, system)})"
