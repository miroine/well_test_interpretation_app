"""McKinsey-inspired editorial styling, without logos or external font requests."""
from __future__ import annotations

import html

import streamlit as st

NAVY, BLUE, CYAN, TEAL, GREY, LIGHT, RULE = "#051C2C", "#2251FF", "#00A9F4", "#0A7D6B", "#7A8CA0", "#F2F6FC", "#D5DEE9"

CSS = f"""
<style>
  html, body, .stApp, [data-testid="stAppViewContainer"] {{ font-family: "Segoe UI", Arial, sans-serif; color: {NAVY}; }}
  .stApp {{ background: #fff; }}
  .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }}
  .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ font-family: Georgia, "Times New Roman", serif; color: {NAVY}; letter-spacing: -.01em; font-weight: 600; }}
  .stApp h1 {{ font-size: clamp(2rem, 3.2vw, 3.2rem); line-height: 1.12; border-bottom: 2px solid {NAVY};
      padding-bottom: .7rem; margin-bottom: .4rem; }}
  .stApp h4 {{ font-size: 1.15rem; margin-top: 1.1rem; }}
  [data-testid="stSidebar"] {{ background: {LIGHT}; border-right: 1px solid {RULE}; }}
  [data-testid="stSidebar"] h3 {{ color: {NAVY}; font-family: Georgia, serif; }}
  [data-testid="stSidebar"] hr {{ border-color: {RULE}; }}
  button[kind="primary"] {{ background: {NAVY}; color: #fff; border: 1px solid {NAVY}; border-radius: 2px;
      font-weight: 600; letter-spacing: .02em; }}
  button[kind="primary"]:hover {{ background: {BLUE}; color: #fff; border-color: {BLUE}; }}
  button[kind="primary"] p {{ color: #fff; }}
  button[kind="secondary"] {{ color: {NAVY}; background: #fff; border-radius: 2px; border: 1px solid {RULE}; }}
  button:focus-visible, a:focus-visible, input:focus-visible {{ outline: 2px solid {BLUE}; outline-offset: 3px; }}
  div[data-testid="stMetric"] {{ background: #fff; border: 0; border-top: 3px solid {BLUE}; border-radius: 0;
      padding: 8px 12px; box-shadow: 0 1px 2px rgba(5,28,44,.08); }}
  div[data-testid="stMetricLabel"] p {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; color: #4a5f75; }}
  .stApp div[data-testid="stMetricValue"] {{ font-family: Georgia, serif; color: {NAVY};
      font-size: clamp(1.3rem, 2.1vw, 2.1rem); }}
  div[data-testid="stMetricValue"] > div {{ white-space: normal; overflow-wrap: anywhere; text-overflow: clip; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid {RULE}; }}
  .stTabs [data-baseweb="tab"] {{ height: 2.4rem; padding: 0 .8rem; font-size: .8rem; text-transform: uppercase;
      letter-spacing: .02em; color: #4a5f75; white-space: nowrap; }}
  .stTabs [aria-selected="true"] {{ color: {NAVY}; font-weight: 700; }}
  .stTabs [data-baseweb="tab-highlight"] {{ background-color: {BLUE}; height: 3px; }}
  [data-testid="stExpander"] {{ border: 1px solid {RULE}; border-radius: 2px; }}
  .takeaway {{ border-left: 4px solid {BLUE}; background: {LIGHT}; padding: .6rem .9rem; margin: .4rem 0 .8rem 0;
      font-family: Georgia, serif; font-size: 1.05rem; color: {NAVY}; }}
  .takeaway small {{ display: block; font-family: Inter, Arial, sans-serif; font-size: .72rem; letter-spacing: .06em;
      text-transform: uppercase; color: #4a5f75; margin-bottom: 2px; }}
  .source {{ font-size: .72rem; color: {GREY}; margin-top: -.4rem; }}
</style>
"""


def apply():
    st.markdown(CSS, unsafe_allow_html=True)


def takeaway(text: str, label: str = "Key takeaway"):
    """Action-title banner: one sentence stating the conclusion of the analysis."""
    st.markdown(f'<div class="takeaway"><small>{html.escape(label)}</small>{html.escape(text)}</div>', unsafe_allow_html=True)


def polish(fig):
    """Apply the house chart style to any Plotly figure."""
    fig.update_layout(template="plotly_white", colorway=[BLUE, NAVY, TEAL, CYAN, GREY],
                      font=dict(family="Segoe UI, Arial, sans-serif", size=12, color=NAVY),
                      title=dict(x=0.01, font=dict(family="Georgia, serif", size=15, color=NAVY)),
                      plot_bgcolor="#fff", paper_bgcolor="#fff")
    fig.update_xaxes(showgrid=True, gridcolor="#E8EDF4", zeroline=False, showline=True, linecolor=NAVY, mirror=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E8EDF4", zeroline=False, showline=True, linecolor=NAVY, mirror=False)
    return fig
