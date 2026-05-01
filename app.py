"""Werbley's Squad - Streamlit entry point."""
from __future__ import annotations

import os

import streamlit as st

from backend.config import CONFIG
from backend.data import espn_client
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


def _clear_league_state():
    for key in (
        "draft_state",
        "draft_bundle",
        "standings_table",
        "pipeline",
        "home_last_response",
    ):
        st.session_state.pop(key, None)
    try:
        espn_client.clear_runtime_caches()
    except Exception:
        pass


def _clean_runtime_espn_config(config: dict) -> dict:
    league_id = str(config.get("league_id", "")).strip()
    season = str(config.get("season", "2026")).strip() or "2026"
    swid = str(config.get("swid", "")).strip()
    s2 = str(config.get("s2", "")).strip()
    return {"league_id": league_id, "season": season, "swid": swid, "s2": s2}


def sidebar_league_connector():
    with st.expander("Connect ESPN league", expanded=False):
        saved = st.session_state.get("runtime_espn_config", {})
        st.caption("Paste your ESPN fantasy baseball league info for only this browser session.")
        with st.form("runtime_espn_form"):
            league_id = st.text_input(
                "League ID",
                value=str(saved.get("league_id", CONFIG.espn_league_id or "")),
                placeholder="1907954991",
            )
            season = st.text_input(
                "Season",
                value=str(saved.get("season", CONFIG.espn_season or 2026)),
                placeholder="2026",
            )
            swid = st.text_input(
                "SWID",
                value=str(saved.get("swid", CONFIG.espn_swid or "")),
                placeholder="{xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx}",
            )
            s2 = st.text_area(
                "ESPN_S2 cookie",
                value=str(saved.get("s2", CONFIG.espn_s2 or "")),
                placeholder="Paste the full espn_s2 cookie value",
                height=90,
            )
            submitted = st.form_submit_button("Use this league", use_container_width=True)

        if submitted:
            if not league_id.strip():
                st.error("League ID is required.")
                return
            try:
                int(season)
            except ValueError:
                st.error("Season must be a year like 2026.")
                return
            runtime_config = _clean_runtime_espn_config({
                "league_id": league_id,
                "season": season,
                "swid": swid,
                "s2": s2,
            })
            st.session_state["runtime_espn_config"] = runtime_config
            _clear_league_state()
            league = espn_client.load_league()
            if league.source == "espn":
                st.success(f"Connected: {len(league.teams)} teams loaded.")
            else:
                st.warning(f"Could not load ESPN league: {league.error or 'unknown ESPN error'}")

        if st.button("Clear pasted league", use_container_width=True):
            st.session_state.pop("runtime_espn_config", None)
            _clear_league_state()
            st.rerun()


def _team_default_index(teams) -> int:
    selected_id = st.session_state.get("selected_team_id")
    if selected_id not in (None, ""):
        try:
            selected_id = int(selected_id)
            for idx, team in enumerate(teams):
                if int(getattr(team, "team_id", -1)) == selected_id:
                    return idx
        except (TypeError, ValueError):
            pass
    for idx, team in enumerate(teams):
        if (team.owner or "").strip().lower() in {"you", "me", "my team"}:
            return idx
    return 0


def sidebar_team_selector():
    league = espn_client.load_league()
    if league.source != "espn" or not league.teams:
        st.caption("Connect an ESPN league to choose your team.")
        return

    previous_id = st.session_state.get("selected_team_id")
    selected = st.selectbox(
        "Your team",
        options=league.teams,
        index=_team_default_index(league.teams),
        format_func=lambda team: f"{team.name} ({team.owner})",
        key="sidebar_selected_team",
    )
    st.session_state["selected_team_id"] = int(selected.team_id)
    st.session_state["selected_team_name"] = selected.name
    st.session_state["selected_team_owner"] = selected.owner
    if previous_id not in (None, selected.team_id) and str(previous_id) != str(selected.team_id):
        _clear_league_state()
        st.rerun()


with st.sidebar:
    st.markdown("### Werbley's Squad")
    st.caption("IBM Experiential AI Lab | 4-agent fantasy baseball optimizer")
    provider_pill()
    sidebar_league_connector()
    sidebar_team_selector()
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
