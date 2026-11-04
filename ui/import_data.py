"""Shared, uncached upload preview and rate-history editor."""
from __future__ import annotations

import io
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile

import pandas as pd
import streamlit as st
from xlrd.biffh import XLRDError

from engine import tabular as T, units as U
from . import multirate_timeline


def file_table(label, key):
    up = st.file_uploader(label, type=list(T.EXTENSIONS), key=key)
    if up is None:
        return None
    content = up.getvalue()
    excel = Path(up.name).suffix.lower() in (".xlsx", ".xls")
    try:
        sheet = 0
        if excel:
            with pd.ExcelFile(io.BytesIO(content), engine="openpyxl" if up.name.lower().endswith(".xlsx") else "xlrd") as book:
                sheet = st.selectbox("Worksheet", book.sheet_names, key=f"{key}_sheet")
                st.caption("Raw worksheet preview (row numbers are 1-based).")
                raw = book.parse(sheet, header=None, nrows=30)
                raw.index = range(1, len(raw) + 1)
                st.dataframe(raw.astype(str))
        encoding = "utf-8-sig"
        if not excel:
            encoding = st.selectbox("Text encoding", ["utf-8-sig", "utf-16", "cp1252"], key=f"{key}_encoding")
            lines = T.decode_text(content, encoding).splitlines()
            st.code("\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines[:30])), language="text")
        skip = st.number_input("Rows to skip before header/data", min_value=0, value=0, step=1,
                               key=f"{key}_skip",
                               help="Count physical lines (worksheet rows for Excel). Skip titles/units first. For the supplied KAPPA rate example: skip 2, then uncheck header.")
        header = st.checkbox("First remaining row contains column names", value=True, key=f"{key}_header")
        st.caption(f"First data row: {int(skip) + (2 if header else 1)} (1-based, before blank-line removal).")
        delimiter, decimal = "Auto", "."
        if not excel:
            delimiter = st.selectbox("Delimiter", list(T.SEPARATORS), key=f"{key}_delimiter")
            decimal = st.selectbox("Decimal separator", [".", ","], key=f"{key}_decimal")
        frame = T.read_table(content, up.name, int(skip), header, delimiter, decimal, encoding, sheet)
        st.dataframe(frame.head(20), hide_index=True)
        st.caption(f"{len(frame)} parsed rows. Preview shows up to 20; all rows are imported.")
        return up.name, frame
    except (ValueError, ImportError, OSError, BadZipFile, XLRDError) as exc:
        st.error(f"Cannot preview file: {exc}")
        return None


def time_options(key, origin=None, allow_calendar=True):
    choices = list(T.TIME_FACTORS) + (["Calendar date/time"] if allow_calendar else [])
    unit = st.selectbox("Time unit / format", choices, key=f"{key}_time_unit")
    fmt, dayfirst = None, False
    if unit == "Calendar date/time":
        origin = st.text_input("Shared test-start date/time", value=origin or "", key=f"{key}_origin",
                               help="Use the same origin for pressure and rates. Prefer ISO YYYY-MM-DD HH:MM:SS. Naive times are treated as UTC; do not mix local time zones.")
        fmt = st.text_input("Timestamp format (blank = mixed automatic)", key=f"{key}_date_format")
        dayfirst = st.checkbox("Day before month for ambiguous dates", key=f"{key}_dayfirst")
    return {"unit": unit, "origin": origin, "date_format": fmt, "dayfirst": dayfirst}


def pressure_columns(frame, key, fluid, system, full=False):
    cols = list(frame.columns)
    tc = st.selectbox("Elapsed test time" if full else "Time since shut-in", cols, key=f"{key}_time")
    pc = st.selectbox("Normalized pseudo-pressure" if fluid == "Gas" else f"Pressure [{U.label('pressure', system)}]",
                      cols, index=1, key=f"{key}_pressure")
    meaning = st.radio("Interpret pressure time values as",
                       ["Actual reading times", "Durations between readings"],
                       key=f"{key}_pressure_time_meaning")
    durations = meaning == "Durations between readings"
    if durations:
        st.caption("Each duration is the interval ending at that row's pressure reading. "
                   "Intervals are accumulated in file order: 0, 12, 12, 24 becomes 0, 12, 24, 48 hours. "
                   "The first interval is measured from test start (or shut-in for buildup-only data); "
                   "it may be zero. All later intervals must be positive.")
    else:
        st.caption("Use elapsed times at each reading (not intervals), or calendar timestamps with an explicit origin.")
    opts = {**time_options(key, allow_calendar=not durations), "durations": durations}
    try:
        converted = T.pressure_hours(frame[tc], **opts)
        st.dataframe(pd.DataFrame({"Elapsed reading time [hr]": converted[:20]}), hide_index=True)
    except ValueError as exc:
        st.error(f"Cannot interpret pressure times: {exc}")
    return tc, pc, opts


def rate_input(frame, key, system, kind="rate", origin=None):
    source = st.radio("Rate source", ["Pressure file column", "Separate rate file", "Enter rate table"],
                      key=f"{key}_source")
    ru = U.label(kind, system)
    if source == "Pressure file column":
        col = st.selectbox(f"Recorded surface rate [{ru}]", list(frame.columns),
                          index=min(2, len(frame.columns) - 1), key=f"{key}_column")
        return {"column": col}
    if source == "Separate rate file":
        result = file_table("Upload rate history", f"{key}_upload")
        if result is None:
            return None
        _, table = result
        tc = st.selectbox("Rate time / duration column", list(table.columns), key=f"{key}_time_column")
        qc = st.selectbox(f"Rate column [{ru}]", list(table.columns), index=1, key=f"{key}_rate_column")
        if tc == qc:
            st.error("Choose different time and rate columns.")
            return None
    st.markdown("**Does the rate time column contain durations or actual times?**")
    meanings = {
        "Period start times": "Actual time at the start of each period (elapsed time or calendar timestamp)",
        "Period durations": "Duration of each period (length of time at that rate)",
    }
    duration = st.radio("Interpret rate time values as", list(meanings), format_func=meanings.get,
                        key=f"{key}_meaning",
                        help="Durations are added together from test start. Actual times are period starts, not lengths or period end times.") == "Period durations"
    if duration:
        st.info("Duration mode: repeated values are allowed. For example, 12, 12, 12, 48, 72 hours "
                "become period starts 0, 12, 24, 36, 84 hours and a total duration of 156 hours.")
    else:
        st.caption("Actual-time mode: enter when each rate begins, e.g. 0, 12, 24, 36, 84 hours. "
                   "For calendar timestamps use the same test-start origin as the pressure data. "
                   "If your file lists 12, 12, 12, 48, 72, choose Duration instead.")
    opts = time_options(f"{key}_times", origin, allow_calendar=not duration)
    if source == "Enter rate table":
        st.caption("Add/delete rows or paste from a spreadsheet. Zero rate means shut-in; rates use the selected project units.")
        calendar = opts["unit"] == "Calendar date/time"
        initial = [origin or ""] if calendar else ([48., 24.] if duration else [0., 48.])
        table = st.data_editor(pd.DataFrame({"Time": initial, "Rate": [0.] if calendar else [1000., 0.]}),
                               num_rows="dynamic", key=f"{key}_table_{'calendar' if calendar else 'duration' if duration else 'start'}",
                               hide_index=True)
        tc, qc = "Time", "Rate"
    st.caption("Start times must begin at zero (the shared test origin); the last rate continues to the end of the pressure record. "
               "Durations define finite coverage and must cover the entire pressure record.")
    try:
        table = table.dropna(how="all")
        time = T.elapsed_hours(table[tc], **opts)
        rates = U.to_field(T.numeric(table[qc], "Rate"), kind, system)
        schedule, end = T.rate_schedule(time, rates, duration)
        st.markdown("**Interpreted rate schedule**")
        if end is not None:
            st.caption(f"Duration mode: {len(schedule)} periods, total duration {end:g} hr.")
        st.dataframe(pd.DataFrame({"Start [hr]": [s[0] for s in schedule],
                                   f"Rate [{ru}]": [U.from_field(s[1], kind, system) for s in schedule]}),
                     hide_index=True)
        return {"schedule": schedule, "end": end}
    except ValueError as exc:
        st.error(f"Cannot use rate history: {exc}")
        return None


def mapped_frame(frame, tc, pc, opts, rate=None):
    if tc == pc or (rate and "column" in rate and rate["column"] in (tc, pc)):
        raise ValueError("Choose different columns for time, pressure and rate.")
    mapped = pd.DataFrame({"time": T.pressure_hours(frame[tc], **opts),
                           "pressure": T.numeric(frame[pc], "Pressure")})
    if rate:
        mapped["rate"] = (T.numeric(frame[rate["column"]], "Rate") if "column" in rate
                          else T.rates_at(mapped["time"], rate["schedule"], rate["end"]))
        if (mapped["rate"] < 0).any():
            raise ValueError("Rates must be non-negative production rates; zero denotes shut-in.")
    return mapped


def apply_alignment(state, key, edit):
    draft = state[key]
    schedule = multirate_timeline.validate_edit(edit, draft["schedule"], draft["limit"])
    state[key] = {**draft, "schedule": schedule}


def alignment_preview(frame, tc, pc, opts, rate, key, system, kind="rate", buildup=False):
    """Preview uncommitted inputs before coverage/shut-in validation."""
    if not rate or "schedule" not in rate:
        return rate, None
    try:
        pressure = mapped_frame(frame, tc, pc, opts)
        if (pressure["time"] < 0).any() or pressure["time"].max() <= 0:
            raise ValueError("Preview requires non-negative pressure times and a positive record end.")
        pressure = pressure.sort_values("time", kind="stable")
    except ValueError as exc:
        st.error(f"Cannot preview synchronization: {exc}")
        return rate, None
    times = pressure["time"].to_numpy()
    values = U.to_field(pressure["pressure"].to_numpy(), "pressure", system)
    signature = hashlib.sha256(json.dumps(
        [times.tolist(), values.tolist(), rate["schedule"], rate["end"]]).encode()).hexdigest()
    state_key, widget_key = f"{key}_alignment", f"{key}_alignment_plot"
    limit = max(float(times[-1]), rate["end"] or 0., rate["schedule"][-1][0] * 1.05, 1.)
    if st.session_state.get(state_key, {}).get("signature") != signature:
        st.session_state[state_key] = {"signature": signature, "schedule": rate["schedule"], "limit": limit}
        st.session_state.pop(f"{key}_alignment_error", None)
    draft = st.session_state[state_key]
    schedule = draft["schedule"]
    shutins = [i for i in range(1, len(schedule))
               if schedule[i][1] == 0 and schedule[i - 1][1] > 0]
    selected = shutins[-1] if shutins else 0
    st.markdown("#### Preview and synchronize pressure / rate")
    st.caption("Before importing, drag a vertical period-start marker on either plot to align rate changes "
               "with the pressure response. Pressure readings and rate values stay unchanged. "
               "First start is fixed at zero. Arrow keys: 0.1 hr; Shift: 1 hr. "
               "Changing source data resets these draft edits; nothing is imported until Use this file.")
    if buildup and shutins:
        selected = st.selectbox("Shut-in boundary for buildup", shutins, index=len(shutins) - 1,
                                format_func=lambda i: f"Period {i + 1}: {schedule[i][0]:g} hr",
                                key=f"{key}_alignment_shutin_{signature[:12]}")
    def changed():
        try:
            apply_alignment(st.session_state, state_key, st.session_state[widget_key]["timing"])
        except ValueError as exc:
            st.session_state[f"{key}_alignment_error"] = str(exc)
        else:
            st.session_state.pop(f"{key}_alignment_error", None)
    if st.session_state.get(f"{key}_alignment_error"):
        st.error(st.session_state[f"{key}_alignment_error"])
    multirate_timeline.draw(schedule, times, values, system, selected, widget_key, changed,
                            kind=kind, end=limit)
    if st.button("Reset timing alignment", key=f"{key}_alignment_reset"):
        st.session_state.pop(state_key, None)
        st.rerun()
    if buildup and not shutins:
        st.warning("No flow-to-zero-rate boundary exists. Add a shut-in period to the rate file/table or use Multi-rate.")
    return {**rate, "schedule": schedule}, (schedule[selected][0] if buildup and shutins else None)
