"""Baseball-inspired color palette + Streamlit CSS injection."""
from __future__ import annotations

import streamlit as st


PALETTE = {
    "field_green": "#4CAF50",   # brighter green for buttons
    "dirt_tan": "#C49A6C",      # infield dirt
    "baseline_white": "#F5F5F0",
    "home_red": "#B0282C",      # MLB red
    "away_navy": "#0E1A40",     # classic navy
    "leather_brown": "#6B3F17",
    "chalk": "#E8E3D3",
}

CSS = f"""
<style>
  html, body, [data-testid="stAppViewContainer"] {{
    background: linear-gradient(180deg, {PALETTE['baseline_white']} 0%,
                                        {PALETTE['chalk']} 100%);
    color: {PALETTE['away_navy']};
  }}
  [data-testid="stHeader"] {{
    background: {PALETTE['field_green']};
    color: {PALETTE['baseline_white']};
  }}

  /* --- MAIN CONTENT AREA: force dark text everywhere --- */
  [data-testid="stMain"], [data-testid="stMain"] * {{
    color: {PALETTE['away_navy']};
  }}
  [data-testid="stMain"] label,
  [data-testid="stMain"] p,
  [data-testid="stMain"] span,
  [data-testid="stMain"] [data-testid="stMarkdownContainer"],
  [data-testid="stMain"] [data-testid="stWidgetLabel"],
  [data-testid="stMain"] [data-testid="stWidgetLabel"] p,
  [data-testid="stMain"] [data-testid="stMetricLabel"],
  [data-testid="stMain"] [data-testid="stMetricLabel"] p,
  [data-testid="stMain"] [data-testid="stMetricValue"],
  [data-testid="stMain"] [data-testid="stMetricDelta"],
  [data-testid="stMain"] [data-testid="stCaptionContainer"],
  [data-testid="stMain"] .stCaption,
  [data-testid="stMain"] .stRadio label,
  [data-testid="stMain"] .stCheckbox label,
  [data-testid="stMain"] .stSelectbox label,
  [data-testid="stMain"] .stTextInput label,
  [data-testid="stMain"] .stNumberInput label,
  [data-testid="stMain"] .stDateInput label,
  [data-testid="stMain"] .stExpander summary,
  [data-testid="stMain"] .stExpander p {{
    color: {PALETTE['away_navy']} !important;
  }}

  /* --- SIDEBAR: keep white text, but ONLY here --- */
  section[data-testid="stSidebar"] {{
    background: {PALETTE['away_navy']};
  }}
  section[data-testid="stSidebar"],
  section[data-testid="stSidebar"] *:not(button):not(.wb-badge) {{
    color: {PALETTE['baseline_white']};
  }}

  /* --- Buttons: red with navy text for better contrast --- */
  .stButton>button,
  .stForm button {{
    background: {PALETTE['home_red']} !important;
    color: {PALETTE['away_navy']} !important;
    border: 2px solid {PALETTE['away_navy']};
    border-radius: 999px;
    font-weight: 600;
    box-shadow: 0 4px 8px rgba(0,0,0,0.12);
  }}
  .stButton>button:hover,
  .stForm button:hover {{
    background: #ce2a34 !important;
    border-color: {PALETTE['away_navy']};
  }}

  /* --- Metric cards --- */
  div[data-testid="stMetric"], div[data-testid="metric-container"] {{
    background: {PALETTE['baseline_white']};
    border: 1px solid {PALETTE['dirt_tan']};
    padding: 12px;
    border-radius: 10px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.08);
  }}

  /* --- Form widgets (selectbox, text_input, number_input, date_input):
         Streamlit renders these on a dark BaseWeb surface; force white text
         and white placeholders so they're legible. --- */
  [data-testid="stMain"] [data-baseweb="select"],
  [data-testid="stMain"] [data-baseweb="select"] *,
  [data-testid="stMain"] [data-baseweb="input"],
  [data-testid="stMain"] [data-baseweb="input"] *,
  [data-testid="stMain"] [data-baseweb="textarea"],
  [data-testid="stMain"] [data-baseweb="textarea"] *,
  [data-testid="stMain"] [data-baseweb="popover"] li,
  [data-testid="stMain"] [data-baseweb="menu"] li {{
    color: {PALETTE['baseline_white']} !important;
  }}
  [data-testid="stMain"] [data-baseweb="select"] input::placeholder,
  [data-testid="stMain"] [data-baseweb="input"] input::placeholder,
  [data-testid="stMain"] [data-baseweb="textarea"] textarea::placeholder {{
    color: {PALETTE['chalk']} !important;
    opacity: 0.7;
  }}
  /* The open dropdown list (popover) — match navy bg with white items. */
  [data-testid="stMain"] [data-baseweb="popover"] ul,
  [data-testid="stMain"] [data-baseweb="menu"] ul {{
    background: {PALETTE['away_navy']} !important;
  }}
  [data-testid="stMain"] [data-baseweb="popover"] li[aria-selected="true"],
  [data-testid="stMain"] [data-baseweb="menu"] li[aria-selected="true"] {{
    background: {PALETTE['field_green']} !important;
  }}

  .wb-badge {{
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    background: {PALETTE['field_green']};
    color: {PALETTE['baseline_white']} !important;
    font-size: 0.8rem;
    margin-right: 6px;
  }}
  .wb-card {{
    border: 1px solid {PALETTE['dirt_tan']};
    border-left: 6px solid {PALETTE['home_red']};
    background: {PALETTE['baseline_white']};
    padding: 14px 16px;
    border-radius: 8px;
    margin-bottom: 10px;
  }}
  [data-testid="stMain"] h1,
  [data-testid="stMain"] h2,
  [data-testid="stMain"] h3 {{
    color: {PALETTE['field_green']} !important;
  }}

  /* --- st.json(): force a light background + dark, legible text --- */
  [data-testid="stMain"] [data-testid="stJson"],
  [data-testid="stMain"] [data-testid="stJson"] * {{
    background: {PALETTE['baseline_white']} !important;
    color: {PALETTE['away_navy']} !important;
  }}
  [data-testid="stMain"] [data-testid="stJson"] {{
    border: 1px solid {PALETTE['dirt_tan']};
    border-radius: 8px;
    padding: 8px 12px;
  }}
  /* Keys, strings, numbers, booleans — color-code while staying readable. */
  [data-testid="stMain"] [data-testid="stJson"] .key {{
    color: {PALETTE['home_red']} !important;
    font-weight: 600;
  }}
  [data-testid="stMain"] [data-testid="stJson"] .string {{
    color: {PALETTE['field_green']} !important;
  }}
  [data-testid="stMain"] [data-testid="stJson"] .number,
  [data-testid="stMain"] [data-testid="stJson"] .boolean {{
    color: {PALETTE['leather_brown']} !important;
  }}

  /* --- Code blocks (st.code / markdown ```): light bg, dark text --- */
  [data-testid="stMain"] pre,
  [data-testid="stMain"] code,
  [data-testid="stMain"] [data-testid="stCode"],
  [data-testid="stMain"] [data-testid="stCode"] * {{
    background: {PALETTE['baseline_white']} !important;
    color: {PALETTE['away_navy']} !important;
  }}
  [data-testid="stMain"] pre {{
    border: 1px solid {PALETTE['dirt_tan']};
    border-radius: 8px;
  }}

  /* --- Green page headers: force white text, overriding the dark-text rule --- */
  [data-testid="stMain"] .wb-header,
  [data-testid="stMain"] .wb-header *,
  [data-testid="stMain"] .wb-header p,
  [data-testid="stMain"] .wb-header span,
  [data-testid="stMain"] .wb-header div {{
    color: {PALETTE['baseline_white']} !important;
  }}
</style>
"""


def apply_theme():
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="wb-header" style="background:{PALETTE['field_green']};
             padding:18px 20px;border-radius:12px;
             border:3px solid {PALETTE['dirt_tan']};
             color:{PALETTE['baseline_white']};margin-bottom:14px;">
          <div style="font-size:1.8rem;font-weight:800;letter-spacing:0.5px;
                      color:{PALETTE['baseline_white']};">
            ⚾ {title}
          </div>
          <div style="opacity:0.9;color:{PALETTE['baseline_white']};">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
