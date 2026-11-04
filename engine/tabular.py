"""In-memory pressure/rate import, with explicit file layout and time semantics."""
from __future__ import annotations

import io
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import pandas as pd
from xlrd.biffh import XLRDError

TIME_FACTORS = {"Hours": 1., "Minutes": 1. / 60., "Seconds": 1. / 3600., "Days": 24.}
SEPARATORS = {"Auto": None, "Comma": ",", "Semicolon": ";", "Tab": "\t", "Whitespace": r"\s+"}
EXTENSIONS = ("csv", "txt", "dat", "tsv", "xlsx", "xls")


def decode_text(content, encoding="utf-8-sig"):
    try:
        return content.decode(encoding)
    except UnicodeError as exc:
        raise ValueError(f"Cannot decode this file as {encoding}. Select its text encoding.") from exc


def read_table(content, filename, skiprows=0, header=True, delimiter="Auto",
               decimal=".", encoding="utf-8-sig", sheet=0):
    """Skip physical lines/worksheet rows, then optionally consume one header row."""
    suffix = Path(filename).suffix.lower()
    if skiprows < 0:
        raise ValueError("Rows to skip must be non-negative.")
    try:
        if suffix in (".xlsx", ".xls"):
            frame = pd.read_excel(io.BytesIO(content), sheet_name=sheet, skiprows=skiprows,
                                  header=0 if header else None,
                                  engine="openpyxl" if suffix == ".xlsx" else "xlrd")
        elif suffix.lstrip(".") in EXTENSIONS:
            text = decode_text(content, encoding)
            sep = SEPARATORS[delimiter]
            if sep is None:
                # Sniff delimiter on the first retained nonempty line only; a
                # preamble with a different layout must not affect detection.
                line = next((s for s in text.splitlines()[skiprows:] if s.strip()), "")
                sep = next((s for s in ("\t", ";", ",") if s in line), r"\s+")
            frame = pd.read_csv(io.StringIO(text), skiprows=skiprows, header=0 if header else None,
                                sep=sep, engine="python", decimal=decimal)
        else:
            raise ValueError("Supported formats: CSV, TXT, DAT, TSV, XLSX and XLS.")
    except (ImportError, OSError, ValueError, pd.errors.ParserError, BadZipFile, XLRDError) as exc:
        raise ValueError(f"Cannot read table: {exc}") from exc
    if frame.empty or frame.shape[1] < 2:
        raise ValueError("At least two columns and one data row are required. Check rows to skip and delimiter.")
    frame.columns = [str(c) if header else f"Column {i + 1}" for i, c in enumerate(frame.columns)]
    return frame


def elapsed_hours(values, unit="Hours", origin=None, date_format=None, dayfirst=False):
    if unit == "Calendar date/time":
        if not origin:
            raise ValueError("Enter the shared test-start date/time for calendar timestamps.")
        try:
            times = pd.to_datetime(values, format=date_format or "mixed", dayfirst=dayfirst,
                                   errors="raise", utc=True)
            start = pd.to_datetime(origin, dayfirst=dayfirst, errors="raise", utc=True)
            out = np.asarray((times - start).total_seconds() if isinstance(times, pd.DatetimeIndex)
                             else (times - start).dt.total_seconds(), float) / 3600.
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"Invalid calendar time: {exc}") from exc
    else:
        out = np.asarray(pd.to_numeric(values, errors="coerce"), float) * TIME_FACTORS[unit]
    if not np.isfinite(out).all():
        raise ValueError("Time contains blank, invalid or non-finite values. Check data start and time format.")
    return out


def pressure_hours(values, unit="Hours", origin=None, date_format=None, dayfirst=False,
                   durations=False):
    """Reading timestamps, or intervals ending at each corresponding reading."""
    if durations and unit == "Calendar date/time":
        raise ValueError("Pressure durations require numeric time units, not calendar timestamps.")
    times = elapsed_hours(values, unit, origin, date_format, dayfirst)
    if durations:
        if not len(times) or times[0] < 0 or np.any(times[1:] <= 0):
            raise ValueError("Pressure durations must be positive after the first row; "
                             "the first duration may be zero for a reading at the origin.")
        times = np.cumsum(times)
        if not np.isfinite(times).all():
            raise ValueError("Accumulated pressure durations exceed the supported numeric range.")
    return times


def numeric(values, name):
    out = np.asarray(pd.to_numeric(values, errors="coerce"), float)
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains blank, non-numeric or non-finite values. Check data start and columns.")
    return out


def rate_schedule(times, rates, durations=False):
    times, rates = np.asarray(times, float), numeric(rates, "Rate")
    if not len(times) or times.shape != rates.shape or not np.isfinite(times).all():
        raise ValueError("Rate periods require matching, finite time and rate values.")
    if (rates < 0).any():
        raise ValueError("Rates must be non-negative production rates; zero denotes shut-in.")
    if durations:
        if (times <= 0).any():
            raise ValueError("Each rate-period duration must be positive.")
        starts = np.r_[0., np.cumsum(times)[:-1]]
        end = float(np.sum(times))
    else:
        starts, end = times, None
        if starts[0] != 0. or (np.diff(starts) <= 0).any():
            raise ValueError("Actual rate start times must start at zero and be strictly increasing. "
                             "If the column gives the length of each period (including repeated values), "
                             "select Duration of each period instead.")
    return [(float(t), float(q)) for t, q in zip(starts, rates)], end


def rates_at(times, schedule, end=None):
    """Right-continuous rate: the new rate applies exactly at a period start."""
    times = np.asarray(times, float)
    starts = np.array([s[0] for s in schedule])
    if (times < starts[0]).any() or (end is not None and (times > end + 1e-9).any()):
        raise ValueError("Rate history does not cover all pressure times. Extend the rate periods.")
    return np.array([s[1] for s in schedule])[np.searchsorted(starts, times, side="right") - 1]


def final_shutin_rate(schedule, shutin):
    """Validate the actual schedule, including events between pressure readings."""
    starts = np.array([s[0] for s in schedule])
    rates = np.array([s[1] for s in schedule])
    matches = np.flatnonzero(np.isclose(starts, shutin, rtol=0., atol=1e-9))
    if not len(matches):
        raise ValueError("The chosen shut-in time must match a zero-rate period start in the separate schedule.")
    boundary = matches[-1]
    before = rates[:boundary]
    if not len(before) or before[-1] <= 0:
        raise ValueError("The chosen shut-in must immediately follow a positive flowing period.")
    if np.any(rates[boundary:] != 0):
        raise ValueError("The separate schedule must be shut in at and after the chosen shut-in time. Use Multi-rate for later flow periods.")
    return float(before[-1])
