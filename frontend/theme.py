"""MLB-inspired color palette and Streamlit CSS injection."""
from __future__ import annotations

from html import escape

import streamlit as st


PALETTE = {
    "mlb_navy": "#0A2240",
    "mlb_red": "#BF0D3E",
    "mlb_blue": "#1D428A",
    "field_green": "#2E7D32",
    "grass_light": "#EAF3EC",
    "dirt_tan": "#C9A66B",
    "baseline_white": "#FAFAF7",
    "card": "#FFFFFF",
    "chalk": "#EEF0F4",
    "ink": "#071426",
    "muted": "#5B677A",
    "line": "#D8DEE8",
    "warning": "#F4B740",
}

# Backward-compatible aliases used by older page code.
PALETTE["away_navy"] = PALETTE["mlb_navy"]
PALETTE["home_red"] = PALETTE["mlb_red"]
PALETTE["leather_brown"] = "#7A4A21"


CSS = f"""
<style>
  :root {{
    --wb-navy: {PALETTE['mlb_navy']};
    --wb-red: {PALETTE['mlb_red']};
    --wb-blue: {PALETTE['mlb_blue']};
    --wb-green: {PALETTE['field_green']};
    --wb-bg: {PALETTE['baseline_white']};
    --wb-card: {PALETTE['card']};
    --wb-ink: {PALETTE['ink']};
    --wb-muted: {PALETTE['muted']};
    --wb-line: {PALETTE['line']};
    --wb-dirt: {PALETTE['dirt_tan']};
  }}

  html, body, [data-testid="stAppViewContainer"] {{
    background:
      linear-gradient(180deg, rgba(10,34,64,0.045) 0%, rgba(250,250,247,0) 220px),
      var(--wb-bg);
    color: var(--wb-ink);
  }}

  [data-testid="stHeader"] {{
    background: var(--wb-navy);
    color: #fff;
    border-bottom: 3px solid var(--wb-red);
  }}

  [data-testid="stMain"] {{
    color: var(--wb-ink);
  }}

  [data-testid="stMain"] h1,
  [data-testid="stMain"] h2,
  [data-testid="stMain"] h3 {{
    color: var(--wb-navy) !important;
    letter-spacing: 0;
  }}

  [data-testid="stMain"] p,
  [data-testid="stMain"] span,
  [data-testid="stMain"] label,
  [data-testid="stMain"] [data-testid="stMarkdownContainer"],
  [data-testid="stMain"] [data-testid="stWidgetLabel"],
  [data-testid="stMain"] [data-testid="stCaptionContainer"],
  [data-testid="stMain"] [data-testid="stMetricLabel"],
  [data-testid="stMain"] [data-testid="stMetricValue"],
  [data-testid="stMain"] [data-testid="stMetricDelta"],
  [data-testid="stMain"] .stExpander summary {{
    color: var(--wb-ink) !important;
  }}

  section[data-testid="stSidebar"] {{
    background:
      linear-gradient(180deg, var(--wb-navy) 0%, #071A33 100%);
    border-right: 1px solid rgba(255,255,255,0.10);
  }}
  section[data-testid="stSidebar"],
  section[data-testid="stSidebar"] *:not(button):not(.wb-badge):not(code) {{
    color: #fff !important;
  }}
  section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
  section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {{
    color: rgba(255,255,255,0.72) !important;
  }}

  .stButton>button,
  .stForm button {{
    background: var(--wb-red) !important;
    color: #fff !important;
    border: 1px solid #8B0A2D !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    min-height: 42px;
    box-shadow: 0 2px 0 rgba(10,34,64,0.22);
    transition: transform .12s ease, background .12s ease, box-shadow .12s ease;
  }}
  .stButton>button:hover,
  .stForm button:hover {{
    background: #A80C36 !important;
    border-color: #710825 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(10,34,64,0.18);
  }}
  .stButton>button:focus,
  .stForm button:focus {{
    outline: 3px solid rgba(29,66,138,0.22) !important;
  }}

  div[data-testid="stMetric"], div[data-testid="metric-container"] {{
    background: var(--wb-card);
    border: 1px solid var(--wb-line);
    border-top: 4px solid var(--wb-red);
    padding: 14px 16px;
    border-radius: 8px;
    box-shadow: 0 8px 22px rgba(10,34,64,0.06);
  }}

  [data-testid="stMain"] [data-baseweb="select"],
  [data-testid="stMain"] [data-baseweb="input"],
  [data-testid="stMain"] [data-baseweb="textarea"] {{
    background: var(--wb-navy) !important;
    border-radius: 8px !important;
  }}
  [data-testid="stMain"] [data-baseweb="select"] *,
  [data-testid="stMain"] [data-baseweb="input"] *,
  [data-testid="stMain"] [data-baseweb="textarea"] * {{
    color: #fff !important;
  }}
  [data-testid="stMain"] [data-baseweb="select"] input::placeholder,
  [data-testid="stMain"] [data-baseweb="input"] input::placeholder,
  [data-testid="stMain"] [data-baseweb="textarea"] textarea::placeholder {{
    color: rgba(255,255,255,0.72) !important;
  }}
  [data-testid="stMain"] [data-baseweb="popover"] ul,
  [data-testid="stMain"] [data-baseweb="menu"] ul {{
    background: var(--wb-navy) !important;
  }}
  [data-testid="stMain"] [data-baseweb="popover"] li,
  [data-testid="stMain"] [data-baseweb="menu"] li {{
    color: #fff !important;
  }}
  [data-testid="stMain"] [data-baseweb="popover"] li[aria-selected="true"],
  [data-testid="stMain"] [data-baseweb="menu"] li[aria-selected="true"] {{
    background: var(--wb-blue) !important;
  }}

  [data-testid="stDataFrame"] {{
    background: var(--wb-card);
    border: 1px solid var(--wb-line);
    border-radius: 8px;
    box-shadow: 0 8px 22px rgba(10,34,64,0.05);
    overflow: hidden;
  }}
  [data-testid="stDataFrame"] div,
  [data-testid="stDataFrame"] span {{
    color: var(--wb-ink) !important;
  }}

  [data-testid="stMain"] [data-testid="stJson"],
  [data-testid="stMain"] [data-testid="stJson"] * {{
    background: var(--wb-card) !important;
    color: var(--wb-ink) !important;
  }}
  [data-testid="stMain"] [data-testid="stJson"] {{
    border: 1px solid var(--wb-line);
    border-radius: 8px;
    padding: 8px 12px;
  }}

  [data-testid="stMain"] pre,
  [data-testid="stMain"] code,
  [data-testid="stMain"] [data-testid="stCode"],
  [data-testid="stMain"] [data-testid="stCode"] * {{
    background: #F5F7FB !important;
    color: var(--wb-ink) !important;
  }}
  [data-testid="stMain"] pre {{
    border: 1px solid var(--wb-line);
    border-radius: 8px;
  }}

  .wb-header {{
    position: relative;
    background:
      linear-gradient(135deg, var(--wb-navy) 0%, #123C70 58%, var(--wb-blue) 100%);
    padding: 20px 24px;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.16);
    border-bottom: 4px solid var(--wb-red);
    color: #fff !important;
    margin: 0 0 18px 0;
    box-shadow: 0 12px 30px rgba(10,34,64,0.16);
    overflow: hidden;
  }}
  .wb-header:after {{
    content: "";
    position: absolute;
    inset: auto 0 0 0;
    height: 5px;
    background: linear-gradient(90deg, var(--wb-red), #fff, var(--wb-red));
    opacity: .75;
  }}
  [data-testid="stMain"] .wb-header,
  [data-testid="stMain"] .wb-header * {{
    color: #fff !important;
  }}
  .wb-title {{
    font-size: 1.9rem;
    line-height: 1.15;
    font-weight: 850;
    letter-spacing: 0;
  }}
  .wb-subtitle {{
    margin-top: 8px;
    max-width: 920px;
    color: rgba(255,255,255,0.82) !important;
    font-size: 1rem;
  }}

  .wb-card {{
    border: 1px solid var(--wb-line);
    border-left: 6px solid var(--wb-red);
    background: var(--wb-card);
    padding: 16px 18px;
    border-radius: 8px;
    margin-bottom: 12px;
    box-shadow: 0 8px 22px rgba(10,34,64,0.06);
  }}
  .wb-card h3 {{
    margin-top: 10px !important;
    color: var(--wb-navy) !important;
  }}
  .wb-badge {{
    display: inline-flex;
    align-items: center;
    min-height: 26px;
    padding: 4px 10px;
    border-radius: 999px;
    background: var(--wb-blue);
    color: #fff !important;
    font-size: 0.78rem;
    font-weight: 700;
    margin-right: 6px;
    border: 1px solid rgba(255,255,255,0.16);
  }}
  .wb-badge--red {{
    background: var(--wb-red);
  }}
  .wb-badge--green {{
    background: var(--wb-green);
  }}

  .wb-loading {{
    display: flex;
    align-items: center;
    gap: 14px;
    background: var(--wb-card);
    border: 1px solid var(--wb-line);
    border-left: 6px solid var(--wb-red);
    border-radius: 8px;
    padding: 16px 18px;
    margin: 10px 0 14px 0;
    box-shadow: 0 10px 24px rgba(10,34,64,0.08);
  }}
  .wb-loader {{
    width: 34px;
    height: 34px;
    border-radius: 999px;
    border: 4px solid rgba(29,66,138,0.18);
    border-top-color: var(--wb-red);
    border-right-color: var(--wb-blue);
    animation: wb-spin .8s linear infinite;
    flex: 0 0 auto;
  }}
  .wb-loading-title {{
    font-weight: 800;
    color: var(--wb-navy) !important;
    line-height: 1.2;
  }}
  .wb-loading-detail {{
    color: var(--wb-muted) !important;
    font-size: .92rem;
    margin-top: 2px;
  }}
  @keyframes wb-spin {{
    to {{ transform: rotate(360deg); }}
  }}
</style>
"""


def apply_theme():
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    safe_title = escape(title)
    safe_subtitle = escape(subtitle)
    st.markdown(
        f"""
        <div class="wb-header">
          <div class="wb-title">{safe_title}</div>
          <div class="wb-subtitle">{safe_subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
