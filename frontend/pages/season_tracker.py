"""Season tracker."""
from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

from backend.config import CONFIG
from backend.data import espn_client
from backend.draft import league_state
from backend.draft.scorer import (SEASON_END, SEASON_START, standings,
                                  player_weekly_trajectory)
from backend.season_tracker import build_espn_season_tracker
from frontend.components import agent_trace, ensure_pipeline, recommendation_card
from frontend.theme import PALETTE, page_header


def _breakdown_lines(scoring_profile):
    lines = []
    if getattr(scoring_profile, "uses_points", False):
        for stat, weight in sorted(scoring_profile.batter_weights.items()):
            lines.append(f"{stat}: {weight:g}")
        for stat, weight in sorted(scoring_profile.pitcher_weights.items()):
            lines.append(f"{stat}: {weight:g}")
        return lines
    return [
        "Using fallback category-style scoring.",
        "Batter weights: R=1, HR=4, RBI=1, SB=2, AVG bonus.",
        "Pitcher weights: W=5, SV=5, K=1, ERA/WHIP bonuses.",
    ]


def _stat_card(label: str, value: str):
    st.markdown(
        f"""
        <div style="background:{PALETTE['baseline_white']};border:1px solid {PALETTE['dirt_tan']};
             padding:16px 18px;border-radius:10px;box-shadow:0 1px 2px rgba(0,0,0,0.08);min-height:112px;">
          <div style="font-size:1rem;color:{PALETTE['away_navy']};opacity:0.9;">{label}</div>
          <div style="font-size:2.2rem;font-weight:700;color:{PALETTE['away_navy']};margin-top:8px;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_scoring_breakdown(scoring_profile):
    if "show_scoring_breakdown" not in st.session_state:
        st.session_state["show_scoring_breakdown"] = False
    button_label = "Hide scoring breakdown" if st.session_state["show_scoring_breakdown"] else "Show scoring breakdown"
    if st.button(button_label, key="toggle_scoring_breakdown"):
        st.session_state["show_scoring_breakdown"] = not st.session_state["show_scoring_breakdown"]
    if not st.session_state["show_scoring_breakdown"]:
        return

    with st.container(border=False):
        if getattr(scoring_profile, "uses_points", False):
            batter_rows = [
                {"group": "Batting", "stat": stat, "points": weight}
                for stat, weight in sorted(scoring_profile.batter_weights.items())
            ]
            pitcher_rows = [
                {"group": "Pitching", "stat": stat, "points": weight}
                for stat, weight in sorted(scoring_profile.pitcher_weights.items())
            ]
            st.dataframe(
                pd.DataFrame(batter_rows + pitcher_rows),
                hide_index=True,
                use_container_width=True,
            )
            st.caption("These are the scoring weights currently loaded from ESPN.")
        else:
            for line in _breakdown_lines(scoring_profile):
                st.write(line)
            st.caption("ESPN points settings were unavailable, so the app is using the fallback profile.")


def _my_team_name(state, league):
    if state is not None and getattr(state, "teams", None):
        return state.teams[state.human_index]
    for team in league.teams:
        owner = (team.owner or "").strip().lower()
        if owner in {"you", "me", "my team"}:
            return team.name
    return league.teams[0].name if league.teams else ""


def render():
    page_header("Season Tracker", f"Live league scoring for your {CONFIG.oot_season} ESPN season.")

    if "draft_state" not in st.session_state:
        imported_state, bundle, _ = league_state.load_existing_league_state()
        if imported_state is not None:
            st.session_state["draft_state"] = imported_state
            st.session_state["draft_bundle"] = bundle
            ensure_pipeline()

    state = st.session_state.get("draft_state")
    league = espn_client.load_league()
    scoring_profile = league.scoring_profile

    _, right = st.columns([4, 1])
    with right:
        today = date.today()
        default_as_of = min(max(today, SEASON_START), SEASON_END)
        as_of = st.date_input(
            "As-of date",
            value=default_as_of,
            min_value=SEASON_START,
            max_value=SEASON_END,
        )

    my_team = _my_team_name(state, league)
    using_exact_espn = league.source == "espn"

    if using_exact_espn:
        table, traj, outlook, _, _ = build_espn_season_tracker(as_of)
        if as_of < date.today():
            st.caption("Historical dates use completed ESPN matchup periods. Today's view reflects the live ESPN snapshot.")
    else:
        if state is None or not state.rosters.get(state.teams[state.human_index]):
            st.warning("Complete a draft on the Draft Room page first.")
            return
        table = standings(state.rosters, as_of, profile=scoring_profile)
        traj = player_weekly_trajectory(state.rosters[my_team], as_of, profile=scoring_profile)
        outlook = pd.DataFrame()

    st.session_state["standings_table"] = table
    st.subheader("Standings")

    my_rows = table[table["team"] == my_team]
    if my_rows.empty:
        st.warning("Could not match your team in the current league standings.")
        return
    my_row = my_rows.iloc[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        _stat_card("Your rank", f"#{int(my_row['rank'])} of {len(table)}")
    with c2:
        _stat_card("Points earned", f"{my_row['points_to_date']:.1f}")
        _render_scoring_breakdown(scoring_profile)
    with c3:
        if using_exact_espn and "current_matchup_projected" in table.columns:
            _stat_card("Current matchup projection", f"{my_row['current_matchup_projected']:.1f}")
        else:
            _stat_card("Projected full-season points", f"{my_row['projected_full_season']:.1f}")

    def _style(df):
        def highlight(row):
            if row["team"] == my_team:
                return [f"background-color: {PALETTE['dirt_tan']}"] * len(row)
            return [""] * len(row)

        return df.style.apply(highlight, axis=1)

    st.dataframe(_style(table), hide_index=True, use_container_width=True)

    st.subheader("Scoring trajectory")
    if not traj.empty:
        if using_exact_espn:
            fig = px.line(
                traj,
                x="date",
                y="cumulative_points",
                color="team",
                markers=True,
                title="League cumulative points by matchup period",
            )
        else:
            fig = px.line(
                traj,
                x="date",
                y="cumulative_pts",
                color="player",
                markers=True,
                title="Cumulative fantasy points",
            )
        fig.update_layout(
            legend=dict(orientation="h", y=-0.25),
            plot_bgcolor=PALETTE["baseline_white"],
        )
        st.plotly_chart(fig, use_container_width=True)

    if using_exact_espn and not outlook.empty:
        st.subheader("Season outlook")
        fig_outlook = px.line(
            outlook,
            x="date",
            y="cumulative_points",
            color="team",
            line_dash="series",
            line_dash_map={
                "Actual": "solid",
                "Current matchup": "solid",
                "Projected": "dash",
            },
            markers=True,
            title="Actual plus projected cumulative points by matchup period",
        )
        fig_outlook.for_each_trace(
            lambda trace: trace.update(
                line=dict(color=PALETTE["home_red"], width=4, dash="solid"),
                marker=dict(color=PALETTE["home_red"], size=9),
            )
            if "Current matchup" in str(trace.name)
            else None
        )
        fig_outlook.update_layout(
            legend=dict(orientation="h", y=-0.25),
            plot_bgcolor=PALETTE["baseline_white"],
        )
        st.plotly_chart(fig_outlook, use_container_width=True)
        st.caption("Red current-matchup segments bridge completed actual scoring to ESPN's live matchup projection; dashed projected segments extend each team using its average matchup score across the remaining schedule.")

    if st.button("Ask the assistant how I'm doing"):
        resp = ensure_pipeline().run(
            user_text=f"What are my current standings as of {as_of}?",
            skill_level=st.session_state.get("skill_level", "beginner"),
            standings_table=table,
        )
        recommendation_card(resp)
        agent_trace(resp)
