"""Field <-> metric conversion table. Engine is always field units."""
from __future__ import annotations

FT_M = 0.3048
PSI_BAR = 1.0 / 14.5037738
STB_M3 = 0.158987295

# kind -> (field label, metric label, factor field->metric)
KINDS = {
    "length":   ("ft", "m", FT_M),
    "pressure": ("psi", "bar", PSI_BAR),
    "rate":     ("stb/d", "m³/d", STB_M3),
    "gasrate":  ("Mscf/d", "Sm³/d", 28.316847),      # 1 Mscf = 28.3168 Sm3 (x1000 scf)
    "ct":       ("1/psi", "1/bar", 1.0 / PSI_BAR),
    "storage":  ("bbl/psi", "m³/bar", STB_M3 / PSI_BAR),
    "temp":     ("°F", "°C", None),                  # special
    "gor":      ("scf/stb", "Sm³/Sm³", 0.178108),
    "time":     ("hr", "hr", 1.0),
    "perm":     ("md", "md", 1.0),
    "visc":     ("cp", "mPa·s", 1.0),
    "none":     ("", "", 1.0),
}


def label(kind: str, system: str) -> str:
    f, m, _ = KINDS[kind]
    return f if system == "Field" else m


def to_field(value: float, kind: str, system: str) -> float:
    if system == "Field":
        return value
    if kind == "temp":
        return value * 9.0 / 5.0 + 32.0
    return value / KINDS[kind][2]


def from_field(value, kind: str, system: str):
    if system == "Field":
        return value
    if kind == "temp":
        return (value - 32.0) * 5.0 / 9.0
    return value * KINDS[kind][2]
