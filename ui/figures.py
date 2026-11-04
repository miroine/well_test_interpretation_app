"""Plotly figure builders. Figures only - no calculations beyond array slicing."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from . import theme

BLUE, ORANGE, NAVY, GREY, GREEN, RED = "#2251FF", "#c2410c", "#051C2C", "#9bb0c8", "#0a7d4f", "#b3261e"
PALETTE = ["#2251FF", "#00A9F4", "#051C2C", "#0A7D6B", "#7A8CA0", "#c2410c"]


def _base(title, xt, yt, xlog=True, ylog=True, height=430):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(family="Georgia, serif", size=15, color=NAVY)),
        template="plotly_white", height=height, margin=dict(l=62, r=18, t=52, b=52),
        legend=dict(orientation="h", y=-0.22), font=dict(family="Inter, Arial, sans-serif", size=12),
        xaxis_title=xt, yaxis_title=yt)
    fig.update_xaxes(type="log" if xlog else "linear", showgrid=True, gridcolor="#e8edf4",
                     mirror=False, showline=True, linecolor=NAVY)
    fig.update_yaxes(type="log" if ylog else "linear", showgrid=True, gridcolor="#e8edf4",
                     mirror=False, showline=True, linecolor=NAVY)
    return theme.polish(fig)


def _pos(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    return x[m], y[m]


def diag_figure(title, xlabel, ylabel, data=None, models=None):
    """data: dict(x, dp, der) ; models: list of dict(name, x, dp, der)."""
    fig = _base(title, xlabel, ylabel)
    if data is not None:
        x, y = _pos(data["x"], data["dp"])
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="dp (data)",
                                 marker=dict(color=BLUE, size=6, opacity=0.8)))
        x, y = _pos(data["x"], data["der"])
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="derivative (data)",
                                 marker=dict(color=ORANGE, size=6, symbol="diamond", opacity=0.85)))
    for i, m in enumerate(models or []):
        col = PALETTE[(i + 2) % len(PALETTE)] if data is not None else PALETTE[i % len(PALETTE)]
        x, y = _pos(m["x"], m["dp"])
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"{m['name']} dp",
                                 line=dict(color=col, width=2.4)))
        x, y = _pos(m["x"], m["der"])
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"{m['name']} derivative",
                                 line=dict(color=col, width=2.4, dash="dash")))
    return fig


def horner_figure(title, ht, pws, ylabel, line=None, window=None):
    fig = _base(title, "Horner time (tp + dt) / dt", ylabel, xlog=True, ylog=False)
    fig.update_xaxes(autorange="reversed")
    fig.add_trace(go.Scatter(x=ht, y=pws, mode="markers", name="data",
                             marker=dict(color=BLUE, size=6, opacity=0.8)))
    if window is not None:
        sel = (ht >= window[0]) & (ht <= window[1])
        fig.add_trace(go.Scatter(x=np.asarray(ht)[sel], y=np.asarray(pws)[sel], mode="markers",
                                 name="fit window", marker=dict(color=ORANGE, size=8)))
    if line is not None:
        fig.add_trace(go.Scatter(x=line[0], y=line[1], mode="lines", name="radial-flow line",
                                 line=dict(color=RED, width=2.2)))
    return fig


def line_figure(title, xlabel, ylabel, series, xlog=False, ylog=False, vlines=None, height=380,
                hlines=None):
    """series: list of (name, x, y)."""
    fig = _base(title, xlabel, ylabel, xlog=xlog, ylog=ylog, height=height)
    for i, (name, x, y) in enumerate(series):
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name,
                                 line=dict(color=PALETTE[i % len(PALETTE)], width=2.4)))
    for v, label in (vlines or []):
        fig.add_vline(x=v, line_dash="dash", line_color=RED, annotation_text=label)
    for v, label in (hlines or []):
        fig.add_hline(y=v, line_dash="dash", line_color=RED, annotation_text=label)
    return fig


def qc_response(history, excluded, tp, system, fluid, pseudo=False):
    """Full-resolution selectable traces with stable original row IDs."""
    from engine import units as U
    kind = "gasrate" if fluid == "Gas" else "rate"
    ylabel = ("normalized pseudo-pressure" if pseudo else "pressure") + f" [{U.label('pressure', system)}]"
    fig = _base("Full pressure and rate response", "Elapsed test time [hr]", ylabel,
                xlog=False, ylog=False, height=500)
    t = history["t"]
    ids = np.arange(len(t))
    removed = np.isin(ids, excluded)
    for mask, name, color in ((~removed, "Pressure", BLUE), (removed, "Excluded pressure", RED)):
        fig.add_trace(go.Scatter(x=t[mask], y=U.from_field(history["p"][mask], "pressure", system),
                                 customdata=ids[mask], mode="markers", name=name,
                                 marker=dict(color=color, size=6, symbol="x" if name.startswith("Excluded") else "circle"),
                                 hovertemplate="Point %{customdata}<br>t=%{x:.6g} hr<br>p=%{y:.6g}<extra>%{fullData.name}</extra>"))
    fig.add_trace(go.Scatter(x=t, y=U.from_field(history["p"], "pressure", system), mode="lines",
                             name="Original pressure", line=dict(color=GREY, width=1), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=U.from_field(history["rate"], kind, system), customdata=ids,
                             mode="lines+markers", name="Recorded rate", yaxis="y2",
                             line=dict(color=GREEN, shape="hv"), marker=dict(size=4)))
    fig.add_vline(x=tp, line_dash="dash", line_color=ORANGE, annotation_text="Start shut-in")
    fig.update_layout(dragmode="lasso", yaxis2=dict(title=f"Rate [{U.label(kind, system)}]",
                      overlaying="y", side="right", showgrid=False), margin=dict(r=90))
    return fig


def interpretation_response(dataset, responses, system):
    """Compare absolute pressure predictions and prescribed surface-rate control."""
    from engine import qc, units as U
    history = qc.response_source(dataset)
    pseudo = dataset["fluid"] == "Gas"
    ylabel = ("Normalized pseudo-pressure" if pseudo else "Pressure") + f" [{U.label('pressure', system)}]"
    fig = _base("Pressure and rate: data and model", "Elapsed test time [hr]", ylabel,
                xlog=False, ylog=False, height=500)
    keep = np.ones(len(history["t"]), dtype=bool)
    if dataset.get("qc_source") is not None:
        keep[np.asarray(dataset.get("qc_excluded", []), int)] = False
        window = dataset.get("qc_window")
        if window:
            keep &= (history["t"] >= window[0]) & (history["t"] <= window[1])
    fig.add_trace(go.Scatter(x=history["t"][keep], y=U.from_field(history["p"][keep], "pressure", system),
                             mode="markers", name="Kept pressure data",
                             marker=dict(color=GREY, size=4)))
    fig.add_trace(go.Scatter(x=dataset["tp"] + np.asarray(dataset["dt"]),
                             y=U.from_field(np.asarray(dataset["pws"]), "pressure", system),
                             mode="markers", name="Active buildup data", marker=dict(color=BLUE, size=6)))
    kind = "gasrate" if pseudo else "rate"
    for i, response in enumerate(responses):
        fig.add_trace(go.Scatter(x=response["t"], y=U.from_field(response["pressure"], "pressure", system),
                                 mode="lines", name=response["name"] + " pressure",
                                 line=dict(color=GREEN if i == 0 else ORANGE, width=2.4,
                                           dash="solid" if i == 0 else "dash")))
        tp = response["tp"]
        fig.add_trace(go.Scatter(x=[0., tp, tp, float(response["t"][-1])],
                                 y=U.from_field(np.array([response["q_user"], response["q_user"], 0., 0.]), kind, system),
                                 name=response["name"] + " rate control (not fitted)", mode="lines", yaxis="y2",
                                 line=dict(color=GREEN if i == 0 else ORANGE, dash="dot")))
    if dataset.get("history") is not None:
        fig.add_trace(go.Scatter(x=history["t"], y=U.from_field(history["rate"], kind, system),
                                 mode="lines", name="Recorded surface rate", yaxis="y2",
                                 line=dict(color=NAVY, shape="hv")))
    fig.add_vline(x=dataset["tp"], line_dash="dash", line_color=GREY, annotation_text="Start shut-in")
    fig.update_layout(yaxis2=dict(title=f"Surface rate [{U.label(kind, system)}]",
                      overlaying="y", side="right", showgrid=False), margin=dict(r=90))
    return fig


def add_percentile_band(fig, x, bounds, name, system, color, log=False):
    """Add an ordered band and median without bridging invalid log-axis gaps."""
    from engine import units as U
    x = np.asarray(x, float)
    low, median, high = [np.asarray(U.from_field(bounds[k], "pressure", system), float)
                         for k in ("p10", "p50", "p90")]
    valid = np.isfinite(x) & np.isfinite(low) & np.isfinite(median) & np.isfinite(high)
    if log:
        valid &= (x > 0) & (low > 0) & (median > 0) & (high > 0)
    low, median, high = [np.where(valid, y, np.nan) for y in (low, median, high)]
    group = f"MC {name}"
    fig.add_trace(go.Scatter(x=x, y=low, mode="lines", name=f"MC {name} P10", legendgroup=group,
                             showlegend=False, line=dict(width=0), connectgaps=False))
    fig.add_trace(go.Scatter(x=x, y=high, mode="lines", name=f"MC {name} P10-P90", legendgroup=group,
                             fill="tonexty", fillcolor=color, line=dict(width=0), connectgaps=False))
    fig.add_trace(go.Scatter(x=x, y=median, mode="lines", name=f"MC {name} P50", legendgroup=group,
                             line=dict(color=NAVY, width=1.5, dash="dot"), connectgaps=False))


def design_overlay(curves, pressure_unit, convert, selected, equivalent=True):
    """Overlay both periods without shifting pressure amplitudes."""
    xlabel = ("Elapsed flow time / Agarwal buildup time [hr]" if equivalent
              else "Elapsed time within each period [hr]")
    fig = _base("Drawdown and buildup diagnostics", xlabel,
                f"Pressure change and derivative [{pressure_unit}]")
    dd, bu = curves["drawdown"], curves["buildup"]
    series = (
        ("Drawdown pressure change", dd["t"], dd["dp"], BLUE, "solid"),
        ("Drawdown derivative", dd["t"], dd["der"], BLUE, "dash"),
        ("Buildup pressure change", bu["ta"] if equivalent else bu["dt"], bu["dp"], GREEN, "solid"),
        ("Buildup derivative", bu["ta"] if equivalent else bu["dt"],
         bu["der"] if equivalent else bu["der_elapsed"], GREEN, "dash"),
    )
    for name, x, y, color, dash in series:
        if name in selected:
            x, y = _pos(x, convert(y))
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name,
                                     line=dict(color=color, width=2.4, dash=dash)))
    return fig


def typecurve_figure(fam, data_x=None, data_dp=None, data_der=None, highlight=None):
    fig = _base("Gringarten type curves (homogeneous, wellbore storage + skin)",
                "tD / CD   (data: time x match)", "pD and tD dpD/dtD   (data: dp x match)", height=500)
    for G, (x, pD, dD) in fam.items():
        hi = bool(highlight is not None and abs(np.log10(G) - np.log10(highlight)) < 0.5)
        col = RED if hi else "#c3cedd"
        w = 2.6 if hi else 1.0
        x = np.asarray(x, float)
        pD = np.where(np.isfinite(pD) & (np.asarray(pD) > 0) & (x > 0), pD, np.nan)
        dD = np.where(np.isfinite(dD) & (np.asarray(dD) > 0) & (x > 0), dD, np.nan)
        fig.add_trace(go.Scatter(x=x, y=pD, mode="lines", line=dict(color=col, width=w), connectgaps=False,
                                 name=f"CDe2S = {G:.0e}" if hi else None, showlegend=hi,
                                 hovertemplate=f"CDe2S={G:.1e}<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=dD, mode="lines", line=dict(color=col, width=w, dash="dash"), connectgaps=False,
                                 showlegend=False, hovertemplate=f"CDe2S={G:.1e}<extra></extra>"))
    if data_x is not None:
        x, y = _pos(data_x, data_dp)
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="dp (shifted data)",
                                 marker=dict(color=BLUE, size=6)))
        x, y = _pos(data_x, data_der)
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="derivative (shifted data)",
                                 marker=dict(color=ORANGE, size=6, symbol="diamond")))
    fig.update_xaxes(range=[-2, 7])
    fig.update_yaxes(range=[-2, 2])
    return fig
