"""Season tracker & historical replay.

Pick any date in the 2025 season (Mar 27 – Sep 28). We approximate
accumulated fantasy points up to that date and rank every team.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

from backend.draft.scorer import (SEASON_END, SEASON_START, standings,
                                  player_weekly_trajectory)
from frontend.components import agent_trace, recommendation_card
from frontend.theme import PALETTE, page_header


def render():
    page_header("Season Tracker", "Historical replay of your 2025 draft.")

    state = st.session_state.get("draft_state")
    if state is None or not state.rosters.get(state.teams[state.human_index]):
        st.warning("Complete a draft on the **Draft Room** page first.")
        return

    # Date picker lives top-right per requirements.
    _, right = st.columns([4, 1])
    with right:
        as_of = st.date_input("As-of date", value=date(2025, 7, 15),
                              min_value=SEASON_START, max_value=SEASON_END)

    table = standings(state.rosters, as_of)
    st.session_state["standings_table"] = table

    st.subheader("Standings")
    my_team = state.teams[state.human_index]
    my_row = table[table["team"] == my_team].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Your rank", f"#{int(my_row['rank'])} of {len(table)}")
    c2.metric("Points earned", f"{my_row['points_to_date']:.1f}")
    c3.metric("Projected finish", f"{my_row['projected_full_season']:.1f}")

    def _style(df):
        def highlight(row):
            if row["team"] == my_team:
                return [f"background-color: {PALETTE['dirt_tan']}"] * len(row)
            return [""] * len(row)
        return df.style.apply(highlight, axis=1)
    st.dataframe(_style(table), hide_index=True, use_container_width=True)

    st.subheader("Your roster trajectory")
    traj = player_weekly_trajectory(state.rosters[my_team], as_of)
    if not traj.empty:
        fig = px.line(traj, x="date", y="cumulative_pts", color="player",
                      markers=True, title="Cumulative fantasy points")
        fig.update_layout(legend=dict(orientation="h", y=-0.25),
                          plot_bgcolor=PALETTE["baseline_white"])
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Ask the assistant how I'm doing"):
        pipeline = st.session_state.get("pipeline")
        if pipeline:
            resp = pipeline.run(
                user_text=f"What are my current standings as of {as_of}?",
                skill_level=st.session_state.get("skill_level", "beginner"),
                standings_table=table,
            )
            recommendation_card(resp)
            agent_trace(resp)
