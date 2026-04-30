"""Werbley's Squad - Streamlit entry point."""
from __future__ import annotations

import os

import streamlit as st

from frontend.components import llm_status_panel, provider_pill
from frontend.pages import draft as draft_page
from frontend.pages import home as home_page
from frontend.pages import model_lab as model_page
from frontend.pages import season_tracker as season_page
from frontend.pages import team_page
from frontend.pages import trade_analyzer as trade_page
from frontend.theme import apply_theme

st.set_page_config(
    page_title="Werbley's Squad - Fantasy Baseball Optimizer",
    page_icon="WB",
    layout="wide",
)
apply_theme()

with st.sidebar:
    st.markdown("### Werbley's Squad")
    st.caption("IBM Experiential AI Lab | 4-agent fantasy baseball optimizer")
    provider_pill()
    st.divider()
    page = st.radio(
        "Navigate",
        ["Home", "Draft Room", "Season Tracker", "Team Advisor", "Trade Analyzer", "Model Lab"],
    )
    skill = st.radio(
        "Skill level",
        ["beginner", "expert"],
        horizontal=True,
        key="skill_level",
    )
    st.divider()
    st.caption("Set `LLM_PROVIDER` in `.env` to `watsonx`, `custom`, or `mock`.")
    st.caption(f"Current: `{os.getenv('LLM_PROVIDER', 'mock')}`")
    llm_status_panel()

if page == "Home":
    home_page.render()
elif page == "Draft Room":
    draft_page.render()
elif page == "Season Tracker":
    season_page.render()
elif page == "Team Advisor":
    team_page.render()
elif page == "Trade Analyzer":
    trade_page.render()
elif page == "Model Lab":
    model_page.render()
