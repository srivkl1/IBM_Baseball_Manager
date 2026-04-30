"""Trade analyzer page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from backend.data import espn_client
from backend.trade_analyzer import analyze_trades
from frontend.roster_layout import add_roster_layout
from frontend.theme import PALETTE, page_header


def _default_team_index(teams) -> int:
    for idx, team in enumerate(teams):
        if (team.owner or "").strip().lower() in {"you", "me", "my team"}:
            return idx
    return 0


def _roster_view(roster: pd.DataFrame) -> pd.DataFrame:
    roster = add_roster_layout(roster)
    columns = [
        "Roster area", "Slot", "photo", "Name", "position", "espn_avg_points", "league_percentile",
        "estimated_value_lost", "health", "trade_value", "advanced",
    ]
    labels = {
        "photo": "Photo",
        "Name": "Player",
        "position": "Pos",
        "espn_avg_points": "ESPN avg",
        "league_percentile": "League pct",
        "estimated_value_lost": "Value lost",
        "trade_value": "Trade value",
        "health": "Status",
        "advanced": "Advanced stats",
    }
    available = [column for column in columns if column in roster.columns]
    view = roster[available].copy().rename(columns=labels)
    if "Trade value" in view:
        view["Trade value"] = view["Trade value"].round(1)
    return view


def _table_height(rows: int) -> int:
    return min(760, max(260, (rows + 1) * 42 + 8))


def _column_config():
    return {
        "Photo": st.column_config.ImageColumn("Photo", width="small"),
        "Player": st.column_config.TextColumn("Player", width="medium"),
        "Advanced stats": st.column_config.TextColumn("Advanced stats", width="large"),
    }


def _legend():
    st.markdown("#### Trade Legend")
    st.markdown(
        """
        - **Trade value**: `Model proj + ESPN avg * 8 + WAR * 3`.
        - **League pct**: Player percentile versus compatible-position players in the league.
        - **Value lost**: Estimated fantasy value lost to IL/injury missed games.
        - **Your benefit / Target benefit**: Need-weighted score for how much the incoming player helps each roster.
        - **Value gap %**: Difference between the two players' trade values; lower is fairer.
        - **Fairness**: Higher means the trade value and benefit are more balanced.
        """
    )


def render():
    page_header(
        "Trade Analyzer",
        "Compare your roster against a target team and find balanced one-for-one trade ideas.",
    )

    league = espn_client.load_league()
    if len(league.teams) < 2:
        st.warning("Need at least two ESPN teams loaded to analyze trades.")
        return

    default_index = _default_team_index(league.teams)
    left_controls, right_controls, fairness_controls = st.columns([2, 2, 1])
    with left_controls:
        your_team = st.selectbox(
            "Your team",
            options=league.teams,
            index=default_index,
            format_func=lambda team: f"{team.name} ({team.owner})",
        )
    target_options = [team for team in league.teams if team.team_id != your_team.team_id]
    with right_controls:
        target_team = st.selectbox(
            "Target team",
            options=target_options,
            index=0,
            format_func=lambda team: f"{team.name} ({team.owner})",
        )
    with fairness_controls:
        max_gap = st.slider("Max value gap", 5, 40, 20, step=5, format="%d%%")

    with st.spinner("Searching for fair mutual-benefit trades..."):
        your_roster, target_roster, trades = analyze_trades(
            league,
            your_team,
            target_team,
            max_value_gap_pct=float(max_gap),
        )

    left, right = st.columns(2)
    with left:
        st.subheader(your_team.name)
        if your_roster.empty:
            st.info("No roster loaded for your team.")
        else:
            view = _roster_view(your_roster)
            st.dataframe(
                view,
                hide_index=True,
                use_container_width=True,
                height=_table_height(len(view)),
                column_config=_column_config(),
            )
    with right:
        st.subheader(target_team.name)
        if target_roster.empty:
            st.info("No roster loaded for the target team.")
        else:
            view = _roster_view(target_roster)
            st.dataframe(
                view,
                hide_index=True,
                use_container_width=True,
                height=_table_height(len(view)),
                column_config=_column_config(),
            )

    st.subheader("Recommended Balanced Trades")
    if trades.empty:
        st.info("No fair mutual-benefit one-for-one trades found with the current value-gap setting.")
    else:
        st.dataframe(trades, hide_index=True, use_container_width=True)
    _legend()
