"""Printable HTML report with inline SVG charts (no plotting dependency)."""
from __future__ import annotations

import html
import math

import numpy as np

NAVY, BLUE, ORANGE, GREY = "#051C2C", "#2251FF", "#c2410c", "#9bb0c8"


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    return html.escape(str(v))


def svg_chart(series, xlabel, ylabel, logx=True, logy=True, w=640, h=380):
    """series: list of dict(x, y, color, kind='points'|'line', name)."""
    pts = []
    for s in series:
        x, y = np.asarray(s["x"], float), np.asarray(s["y"], float)
        m = np.isfinite(x) & np.isfinite(y)
        if logx:
            m &= x > 0
        if logy:
            m &= y > 0
        pts.append((x[m], y[m], s))
    allx = np.concatenate([p[0] for p in pts]) if pts else np.array([1.0])
    ally = np.concatenate([p[1] for p in pts]) if pts else np.array([1.0])
    if allx.size == 0:
        return "<p>(no plottable data)</p>"
    tx = (lambda v: np.log10(v)) if logx else (lambda v: v)
    ty = (lambda v: np.log10(v)) if logy else (lambda v: v)
    x0, x1 = float(tx(allx).min()), float(tx(allx).max())
    y0, y1 = float(ty(ally).min()), float(ty(ally).max())
    if logx:
        x0, x1 = math.floor(x0), math.ceil(x1)
    if logy:
        y0, y1 = math.floor(y0), math.ceil(y1)
    if x1 == x0: x1 = x0 + 1
    if y1 == y0: y1 = y0 + 1
    L, R, T, B = 62, 14, 12, 46
    px = lambda v: L + (float(tx(v)) - x0) / (x1 - x0) * (w - L - R)
    py = lambda v: h - B - (float(ty(v)) - y0) / (y1 - y0) * (h - T - B)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" '
         f'font-family="Arial,sans-serif" font-size="11">',
         f'<rect x="{L}" y="{T}" width="{w-L-R}" height="{h-T-B}" fill="white" stroke="{GREY}"/>']
    def ticks(a, b, lg):
        if lg:
            return [(e, f"1e{e}" if abs(e) > 2 else f"{10.0**e:g}") for e in range(int(a), int(b) + 1)]
        step = (b - a) / 5
        return [(a + i * step, f"{a + i * step:.4g}") for i in range(6)]
    for e, lab in ticks(x0, x1, logx):
        X = L + (e - x0) / (x1 - x0) * (w - L - R)
        o.append(f'<line x1="{X:.1f}" y1="{T}" x2="{X:.1f}" y2="{h-B}" stroke="#e3e9f2"/>'
                 f'<text x="{X:.1f}" y="{h-B+14}" text-anchor="middle" fill="#33475b">{lab}</text>')
    for e, lab in ticks(y0, y1, logy):
        Y = h - B - (e - y0) / (y1 - y0) * (h - T - B)
        o.append(f'<line x1="{L}" y1="{Y:.1f}" x2="{w-R}" y2="{Y:.1f}" stroke="#e3e9f2"/>'
                 f'<text x="{L-6}" y="{Y+3:.1f}" text-anchor="end" fill="#33475b">{lab}</text>')
    for x, y, s in pts:
        col = s.get("color", BLUE)
        if s.get("kind", "points") == "line":
            d = " ".join(f"{'M' if i == 0 else 'L'}{px(a):.1f},{py(b):.1f}" for i, (a, b) in enumerate(zip(x, y)))
            dash = ' stroke-dasharray="6 4"' if s.get("dash") else ""
            o.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2"{dash}/>')
        else:
            for a, b in zip(x, y):
                o.append(f'<circle cx="{px(a):.1f}" cy="{py(b):.1f}" r="2.6" fill="{col}" fill-opacity="0.85"/>')
    lx = L + 8
    for s in series:
        o.append(f'<rect x="{lx}" y="{T+6}" width="10" height="10" fill="{s.get("color", BLUE)}"/>'
                 f'<text x="{lx+14}" y="{T+15}" fill="{NAVY}">{html.escape(s.get("name", ""))}</text>')
        lx += 14 + 7 * len(s.get("name", "")) + 16
    o.append(f'<text x="{(L + w - R) / 2}" y="{h-8}" text-anchor="middle" fill="{NAVY}">{html.escape(xlabel)}</text>')
    o.append(f'<text transform="translate(14,{(T + h - B) / 2}) rotate(-90)" text-anchor="middle" fill="{NAVY}">{html.escape(ylabel)}</text>')
    o.append("</svg>")
    return "".join(o)


def table(rows, header=None):
    if not rows:
        return ""
    keys = header or list(rows[0].keys())
    t = ["<table><thead><tr>" + "".join(f"<th>{html.escape(str(k))}</th>" for k in keys) + "</tr></thead><tbody>"]
    for r in rows:
        t.append("<tr>" + "".join(f"<td>{_fmt(r.get(k, ''))}</td>" for k in keys) + "</tr>")
    t.append("</tbody></table>")
    return "".join(t)


_CSS = """
@page{size:A4;margin:16mm}
body{font-family:Inter,Segoe UI,Arial,sans-serif;color:#051C2C;max-width:900px;margin:0 auto;padding:24px;line-height:1.45}
h1{font-family:Georgia,serif;font-size:26px;margin:0 0 2px 0;border-bottom:3px solid #2251FF;padding-bottom:8px;font-weight:600}
h2{font-family:Georgia,serif;font-size:17px;margin:26px 0 8px 0;color:#051C2C;border-left:4px solid #2251FF;padding-left:8px;font-weight:600}
.meta{color:#33475b;font-size:13px;margin:6px 0 18px 0}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:6px 0 10px 0}
th{background:#051C2C;color:#fff;text-align:left;padding:5px 8px;font-weight:600}
td{padding:5px 8px;border-bottom:1px solid #dbe3ee}
tr:nth-child(even) td{background:#F2F6FC}
.note{font-size:12px;color:#5b6f85}
figure{margin:8px 0;page-break-inside:avoid}
section{page-break-inside:avoid}
@media print{body{padding:0}}
"""


def build_html(title, meta: dict, sections: list) -> str:
    """sections: dict(heading, paras=[str], rows=[dict], svg=str, note=str)."""
    o = [f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{_CSS}</style></head><body>",
         f"<h1>{html.escape(title)}</h1>",
         "<div class='meta'>" + " &middot; ".join(f"{html.escape(str(k))}: <b>{html.escape(str(v))}</b>"
                                                  for k, v in meta.items() if str(v)) + "</div>"]
    for s in sections:
        o.append(f"<section><h2>{html.escape(s['heading'])}</h2>")
        for p in s.get("paras", []):
            o.append(f"<p>{html.escape(p)}</p>")
        if s.get("rows"):
            o.append(table(s["rows"]))
        if s.get("svg"):
            o.append(f"<figure>{s['svg']}</figure>")
        if s.get("note"):
            o.append(f"<p class='note'>{html.escape(s['note'])}</p>")
        o.append("</section>")
    o.append("</body></html>")
    return "".join(o)
