"""
modules/reporter.py
-------------------
Publication-ready PDF report builder. Uses FPDF (no system fonts required).
Returns raw PDF bytes so the Streamlit layer can stream a download.
"""
from __future__ import annotations
from io import BytesIO
from datetime import datetime
from fpdf import FPDF


# --- Equinor-inspired RGB tokens ---
RED = (255, 18, 67)
DARK = (0, 30, 43)
TEAL = (0, 112, 127)
GREY = (90, 107, 115)
LIGHT = (245, 245, 245)


class PTAReport(FPDF):
    """Branded report layout — header bar, footer with page number."""

    def header(self):
        # Top accent bar
        self.set_fill_color(*DARK)
        self.rect(0, 0, 210, 18, "F")
        self.set_fill_color(*RED)
        self.rect(0, 18, 210, 2, "F")

        self.set_xy(10, 5)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, "AQUILA  |  PRESSURE TRANSIENT ANALYSIS REPORT", ln=1)

        self.set_x(10)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(184, 196, 202)
        self.cell(0, 4, datetime.now().strftime("Generated  %Y-%m-%d  %H:%M") +
                       "   |   Author: Merouane Hamdani", ln=1)

        self.ln(10)
        self.set_text_color(*DARK)

    def footer(self):
        self.set_y(-15)
        self.set_draw_color(*RED)
        self.set_line_width(0.4)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*GREY)
        self.cell(0, 4, f"Aquila v1.0  ·  (c) 2026 Merouane Hamdani  ·  "
                        f"MIT License  ·  Not affiliated with Equinor ASA",
                  align="C", ln=1)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 4, f"Page {self.page_no()}", align="C")

    # ------- styled helpers -------
    def section_title(self, text: str):
        self.ln(2)
        self.set_fill_color(*DARK)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 7, f"  {text.upper()}", ln=1, fill=True)
        self.set_text_color(*DARK)
        self.ln(2)

    def kv_row(self, key: str, value: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*GREY)
        self.cell(70, 6, key)
        self.set_text_color(*DARK)
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 6, value, ln=1)

    def highlight_box(self, title: str, value: str, color=RED):
        # Sharp-edged metric tile
        x, y = self.get_x(), self.get_y()
        w, h = 60, 18
        self.set_fill_color(255, 255, 255)
        self.set_draw_color(*color)
        self.set_line_width(1.5)
        self.rect(x, y, w, h, "DF")
        # Left red bar
        self.set_fill_color(*color)
        self.rect(x, y, 2.5, h, "F")

        self.set_xy(x + 5, y + 2)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*GREY)
        self.cell(w - 5, 4, title.upper())
        self.set_xy(x + 5, y + 7)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*DARK)
        self.cell(w - 5, 8, value)
        self.set_xy(x + w + 3, y)


def build_report(
    project_name: str,
    well_inputs: dict,
    economics: dict,
    interpretation: dict | None,
    monte_carlo: dict | None,
    unit_system: str = "Field",
) -> bytes:
    """
    Compose the full PDF and return its bytes.

    Parameters
    ----------
    well_inputs, economics, interpretation : dicts of pre-formatted strings
        Keys already contain unit suffixes from the UI layer, e.g.
        "Permeability k (mD)" → "50.00".
    monte_carlo : dict | None
        Expected shape:
            {"n_iter": int, "P90": float, "P50": float, "P10": float,
             "mean": float, "std": float, "unit": str}
        Values are in the user's chosen unit system; `unit` is the label.
    unit_system : str
        "Field" or "SI" — shown on the cover page as a badge.
    """
    pdf = PTAReport(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # === Cover summary ===
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*DARK)
    pdf.cell(0, 10, project_name or "Untitled PTA Project", ln=1)
    pdf.set_draw_color(*RED)
    pdf.set_line_width(1)
    pdf.line(10, pdf.get_y(), 60, pdf.get_y())
    pdf.ln(4)

    # Unit-system badge
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(*RED)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(38, 5, f"  UNITS · {unit_system.upper()}", fill=True, ln=1)
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*GREY)
    pdf.multi_cell(0, 5,
                   "This report consolidates the operational design parameters, "
                   "interpretation outcomes, and probabilistic uncertainty ranges "
                   "produced by the PTA workbench. All values below are expressed "
                   f"in {unit_system} units.")
    pdf.ln(2)

    # === Operational Inputs ===
    pdf.section_title("Operational Inputs")
    for k, v in well_inputs.items():
        pdf.kv_row(k, v)

    # === Economics ===
    pdf.section_title("Economic Exposure")
    for k, v in economics.items():
        pdf.kv_row(k, v)

    # === Interpretation ===
    pdf.section_title("Interpretation Results")
    if interpretation:
        for k, v in interpretation.items():
            pdf.kv_row(k, v)
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(*GREY)
        pdf.cell(0, 6, "No interpretation has been run yet.", ln=1)

    # === Monte Carlo ===
    pdf.section_title("Probabilistic Permeability  (Monte Carlo)")
    if monte_carlo:
        unit = monte_carlo.get("unit", "mD")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*GREY)
        pdf.cell(0, 5, f"Iterations: {monte_carlo.get('n_iter', '-')}", ln=1)
        pdf.ln(3)

        pdf.highlight_box("P90  Conservative",
                          f"{monte_carlo['P90']:.2f} {unit}", color=TEAL)
        pdf.highlight_box("P50  Best Estimate",
                          f"{monte_carlo['P50']:.2f} {unit}", color=DARK)
        pdf.highlight_box("P10  Optimistic",
                          f"{monte_carlo['P10']:.2f} {unit}", color=RED)
        pdf.ln(22)

        pdf.kv_row(f"Mean (k, {unit})", f"{monte_carlo['mean']:.2f}")
        pdf.kv_row(f"Std. deviation ({unit})", f"{monte_carlo['std']:.2f}")
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(*GREY)
        pdf.cell(0, 6, "Monte Carlo has not been executed.", ln=1)

    # === Disclaimer ===
    pdf.ln(8)
    pdf.section_title("Disclaimer")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*GREY)
    disclaimer_text = (
        "This report is generated by Aquila, a software tool for pressure-"
        "transient analysis authored by Merouane Hamdani and distributed "
        "under the MIT License. The outputs are intended to assist, not "
        "replace, the work of a qualified petroleum engineer. PVT "
        "correlations and inversion routines are valid only within their "
        "published applicability ranges. Decisions on the basis of this "
        "output are the user's sole responsibility.\n\n"
        "Aquila is NOT affiliated with, endorsed by, or sponsored by "
        "Equinor ASA. The Equinor name, logo and wordmark are not used. "
        "The visual identity of this report is inspired by publicly-"
        "available corporate-design conventions for aesthetic purposes only."
    )
    pdf.multi_cell(0, 4, disclaimer_text)

    # Return bytes (FPDF returns either str or bytearray depending on version)
    out = pdf.output(dest="S")
    if isinstance(out, str):
        return out.encode("latin-1")
    return bytes(out)
