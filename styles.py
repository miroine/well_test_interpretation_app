"""
styles.py
---------
Centralised CSS/theming for the Aquila app.
Corporate aesthetic — INSPIRED BY (not affiliated with) Equinor ASA's
publicly-visible design language:
  - High-contrast light theme on a soft neutral background
  - Equinor Red accent (#FF1243)
  - Deep Sea Blue primary dark (#001E2B)
  - Teal secondary (#00707F)
  - Minimalist grey/white background (#F7F7F7)
  - Sharp edges throughout (no rounded corners on functional elements)
  - Geometric sans-serif typography with generous letter-spacing on labels

Author: Merouane Hamdani
"""

# --- Color tokens (also imported by plotting modules for consistency) ---
COLOR_RED = "#FF1243"
COLOR_DARK = "#001E2B"
COLOR_TEAL = "#00707F"
COLOR_BG = "#F7F7F7"
COLOR_WHITE = "#FFFFFF"
COLOR_BORDER = "#D9D9D9"
COLOR_TEXT_MUTED = "#5A6B73"
COLOR_SIDEBAR_TEXT = "#B8C4CA"


def get_plotly_layout(title: str = "") -> dict:
    """Reusable Plotly layout for branded charts."""
    return dict(
        title=dict(text=title, font=dict(color=COLOR_DARK, size=16, family="Inter, Arial, sans-serif")),
        paper_bgcolor=COLOR_WHITE,
        plot_bgcolor=COLOR_WHITE,
        font=dict(color=COLOR_DARK, family="Inter, Arial, sans-serif", size=12),
        xaxis=dict(
            gridcolor="#E5E5E5", zerolinecolor="#E5E5E5",
            linecolor=COLOR_DARK, mirror=True, ticks="outside",
        ),
        yaxis=dict(
            gridcolor="#E5E5E5", zerolinecolor="#E5E5E5",
            linecolor=COLOR_DARK, mirror=True, ticks="outside",
        ),
        legend=dict(
            bgcolor=COLOR_WHITE, bordercolor=COLOR_BORDER, borderwidth=1,
            font=dict(color=COLOR_DARK),
        ),
        margin=dict(l=60, r=30, t=60, b=60),
    )


def inject_css() -> str:
    """Return the full CSS block. Caller injects via st.markdown(unsafe_allow_html=True)."""
    return f"""
    <style>
        /* ---------- Global ---------- */
        .stApp {{
            background-color: {COLOR_BG};
            color: {COLOR_DARK};
        }}
        html, body, [class*="css"] {{
            font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
            color: {COLOR_DARK};
            -webkit-font-smoothing: antialiased;
        }}

        /* Red rule running across the top of the viewport — corporate signature */
        .stApp::before {{
            content: "";
            display: block;
            height: 3px;
            background: {COLOR_RED};
            width: 100%;
            position: sticky;
            top: 0;
            z-index: 999;
        }}

        /* ---------- Headings ---------- */
        h1, h2, h3, h4 {{
            color: {COLOR_DARK};
            font-weight: 600;
            letter-spacing: -0.01em;
        }}
        h1 {{ border-bottom: 3px solid {COLOR_RED}; padding-bottom: 0.4rem; }}
        h2 {{ border-bottom: 1px solid {COLOR_BORDER}; padding-bottom: 0.3rem; }}
        h3 {{ color: {COLOR_DARK}; }}
        /* Subheaders that follow a section divider get a teal accent */
        h4 {{ color: {COLOR_TEAL}; font-weight: 600; }}

        /* ---------- Branded banner (fallback) ---------- */
        .pta-banner {{
            background: {COLOR_DARK};
            color: {COLOR_WHITE};
            padding: 1.4rem 1.6rem;
            border-left: 6px solid {COLOR_RED};
            margin-bottom: 1.5rem;
        }}
        .pta-banner h1 {{
            color: {COLOR_WHITE};
            border-bottom: none;
            margin: 0;
            font-size: 1.6rem;
        }}
        .pta-banner p {{
            color: {COLOR_SIDEBAR_TEXT};
            margin: 0.25rem 0 0 0;
            font-size: 0.9rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}

        /* ---------- Metric cards (5px red left border) ---------- */
        [data-testid="stMetric"] {{
            background-color: {COLOR_WHITE};
            border: 1px solid {COLOR_BORDER};
            border-left: 5px solid {COLOR_RED};
            padding: 14px 18px;
            box-shadow: 0 1px 2px rgba(0, 30, 43, 0.04);
        }}
        [data-testid="stMetricLabel"] {{
            color: {COLOR_TEXT_MUTED};
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 600;
        }}
        [data-testid="stMetricValue"] {{
            color: {COLOR_DARK};
            font-weight: 700;
            font-size: 1.6rem;
        }}

        /* ---------- Buttons ---------- */
        .stButton > button {{
            background-color: {COLOR_RED};
            color: {COLOR_WHITE};
            border: none;
            border-radius: 0;            /* sharp edges */
            padding: 0.55rem 1.2rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            font-size: 0.85rem;
            transition: background-color 0.15s ease, transform 0.05s ease;
        }}
        .stButton > button:hover {{
            background-color: {COLOR_DARK};
            color: {COLOR_WHITE};
        }}
        .stButton > button:active {{
            transform: translateY(1px);
        }}
        .stDownloadButton > button {{
            background-color: {COLOR_DARK};
            color: {COLOR_WHITE};
            border: none;
            border-radius: 0;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .stDownloadButton > button:hover {{
            background-color: {COLOR_RED};
        }}

        /* ---------- Tabs ---------- */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0;
            border-bottom: 2px solid {COLOR_BORDER};
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: transparent;
            color: {COLOR_TEXT_MUTED};
            border-radius: 0;
            padding: 12px 22px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.82rem;
        }}
        .stTabs [aria-selected="true"] {{
            color: {COLOR_DARK} !important;
            border-bottom: 3px solid {COLOR_RED} !important;
            background-color: transparent !important;
        }}

        /* ---------- Sidebar ---------- */
        [data-testid="stSidebar"] {{
            background-color: {COLOR_DARK};
        }}
        [data-testid="stSidebar"] * {{
            color: {COLOR_WHITE};
        }}
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown {{
            color: {COLOR_SIDEBAR_TEXT} !important;
        }}
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
            color: {COLOR_WHITE} !important;
            border-bottom: 2px solid {COLOR_RED};
            padding-bottom: 0.3rem;
            letter-spacing: 0.02em;
        }}
        /* Sidebar caption (footer-y muted text) */
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
            color: {COLOR_SIDEBAR_TEXT} !important;
            font-size: 0.75rem;
            letter-spacing: 0.04em;
        }}

        /* ---------- Inputs ---------- */
        .stNumberInput input, .stTextInput input,
        .stSelectbox div[data-baseweb="select"],
        .stTextArea textarea {{
            border-radius: 0 !important;
            border: 1px solid {COLOR_BORDER} !important;
            transition: border-color 0.15s ease;
        }}
        .stNumberInput input:focus, .stTextInput input:focus,
        .stTextArea textarea:focus {{
            border-color: {COLOR_RED} !important;
            outline: none !important;
            box-shadow: 0 0 0 1px {COLOR_RED} !important;
        }}

        /* ---------- Sliders (Equinor-flavoured) ---------- */
        .stSlider [data-baseweb="slider"] [role="slider"] {{
            background-color: {COLOR_RED} !important;
            border: 2px solid {COLOR_DARK} !important;
            box-shadow: none !important;
        }}
        .stSlider [data-baseweb="slider"] > div > div {{
            background-color: {COLOR_RED} !important;
        }}

        /* ---------- Expanders ---------- */
        .streamlit-expanderHeader, [data-testid="stExpander"] details summary {{
            background-color: {COLOR_WHITE};
            border-left: 4px solid {COLOR_TEAL};
            font-weight: 600;
            letter-spacing: 0.02em;
            color: {COLOR_DARK};
        }}
        [data-testid="stExpander"] {{
            border: 1px solid {COLOR_BORDER};
            border-radius: 0;
        }}

        /* ---------- Alerts (sharp left border, no rounded corners) ---------- */
        .stAlert {{
            border-radius: 0;
            border-left-width: 5px;
        }}

        /* ---------- Divider rule ---------- */
        hr {{
            border: none;
            border-top: 1px solid {COLOR_BORDER};
            margin: 1.5rem 0;
        }}

        /* ---------- Radio buttons (segmented look when horizontal) ---------- */
        .stRadio [role="radiogroup"] {{
            gap: 0;
        }}
        .stRadio label {{
            letter-spacing: 0.02em;
        }}

        /* ---------- Code blocks (Equinor-toned syntax) ---------- */
        code {{
            background-color: rgba(0, 30, 43, 0.06);
            color: {COLOR_DARK};
            padding: 0 4px;
            border-radius: 0;
            font-size: 0.88em;
        }}

        /* ---------- Footer ---------- */
        .pta-footer {{
            color: {COLOR_TEXT_MUTED};
            font-size: 0.75rem;
            text-align: center;
            margin-top: 2rem;
            padding-top: 0.8rem;
            border-top: 1px solid {COLOR_BORDER};
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}
    </style>
    """


def banner_html(title: str, subtitle: str) -> str:
    """Branded banner block for the app header (fallback when logo SVG missing)."""
    return f"""
    <div class="pta-banner">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """
