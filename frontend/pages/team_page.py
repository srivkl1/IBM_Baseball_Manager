"""Individual team roster and free-agent advisor page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from backend.data import espn_client
from backend.team_advisor import build_team_advice
from frontend.roster_layout import add_roster_layout
from frontend.theme import PALETTE, page_header


def _team_default_index(teams) -> int:
    for idx, team in enumerate(teams):
        if (team.owner or "").strip().lower() in {"you", "me", "my team"}:
            return idx
    return 0


def _status_card(label: str, value: str, detail: str = ""):
    st.markdown(
        f"""
        <div style="background:{PALETTE['baseline_white']};border:1px solid {PALETTE['dirt_tan']};
             padding:14px 16px;border-radius:8px;min-height:96px;">
          <div style="font-size:.9rem;color:{PALETTE['away_navy']};opacity:.75;">{label}</div>
          <div style="font-size:1.8rem;font-weight:800;color:{PALETTE['field_green']};">{value}</div>
          <div style="font-size:.85rem;color:{PALETTE['away_navy']};opacity:.8;">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _display_roster(roster_df: pd.DataFrame) -> pd.DataFrame:
    roster_df = add_roster_layout(roster_df)
    cols = [
        "Roster area", "Slot", "photo", "Name", "position", "mlb_team", "espn_avg_points", "games_played", "espn_total_points",
        "espn_projected_points", "rostership", "league_percentile", "estimated_games_missed",
        "estimated_value_lost", "health", "proj_pts", "advanced",
    ]
    labels = {
        "photo": "Photo",
        "Name": "Player",
        "position": "Pos",
        "mlb_team": "MLB",
        "espn_avg_points": "ESPN avg",
        "games_played": "G",
        "espn_total_points": "ESPN pts",
        "espn_projected_points": "ESPN proj",
        "rostership": "Rostered %",
        "league_percentile": "League pct",
        "estimated_games_missed": "Est missed",
        "estimated_value_lost": "Value lost",
        "proj_pts": "Model proj",
        "advanced": "Advanced stats",
        "health": "Status",
    }
    available = [col for col in cols if col in roster_df.columns]
    return roster_df[available].rename(columns=labels)


def _display_free_agents(fa_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "photo", "Name", "position", "mlb_team", "espn_avg_points", "espn_projected_points",
        "rostership", "proj_pts", "add_score", "advanced",
    ]
    labels = {
        "photo": "Photo",
        "Name": "Player",
        "position": "Pos",
        "mlb_team": "MLB",
        "espn_avg_points": "ESPN avg",
        "espn_projected_points": "ESPN proj",
        "rostership": "Rostered %",
        "proj_pts": "Model proj",
        "advanced": "Advanced stats",
        "add_score": "Add score",
    }
    available = [col for col in cols if col in fa_df.columns]
    return fa_df[available].rename(columns=labels)


def _table_height(rows: int) -> int:
    return min(900, max(220, (rows + 1) * 42 + 8))


def _column_config():
    return {
        "Photo": st.column_config.ImageColumn("Photo", width="small"),
        "Player": st.column_config.TextColumn("Player", width="medium"),
        "Advanced stats": st.column_config.TextColumn("Advanced stats", width="large"),
    }


def _team_page_css():
    st.markdown(
        f"""
        <style>
          [data-testid="stDataFrame"] {{
            background: {PALETTE['baseline_white']};
            border: 1px solid {PALETTE['dirt_tan']};
            border-radius: 8px;
          }}
          [data-testid="stDataFrame"] div,
          [data-testid="stDataFrame"] span {{
            color: {PALETTE['away_navy']} !important;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _stat_legend():
    st.markdown("#### Stat Legend")
    st.markdown(
        """
        - **ESPN avg**: ESPN fantasy points per MLB game, calculated from ESPN total fantasy points divided by ESPN games played.
        - **G**: ESPN games played used for the average.
        - **ESPN pts**: Season fantasy points from your ESPN league.
        - **ESPN proj**: ESPN projected season fantasy points when ESPN provides it.
        - **Rostered %**: ESPN percentage of leagues where the player is rostered.
        - **League pct**: Player percentile versus compatible-position players across the league and free-agent pool.
        - **Est missed**: Estimated missed games for IL/injured players based on season games elapsed minus ESPN games played.
        - **Value lost**: Estimated fantasy value lost from missed games, using ESPN avg when available or model projection per game.
        - **Model proj**: App projection from recent fantasy scoring and advanced baseball stats.
        - **Advanced stats**: FanGraphs-style context such as wRC+, wOBA, ISO, FIP, xFIP, K%, BB%, and WAR.
        - **Status**: Stable is 50th percentile or better, Watch is 25th-50th percentile, Struggling is below 25th percentile.
        - **Add score**: Free-agent ranking score: `Model proj + ESPN avg * 8 + WAR * 3`.
        """
    )


def render():
    _team_page_css()
    page_header(
        "Team Advisor",
        "Review one roster, spot struggling players, and compare them with ESPN free agents.",
    )

    league = espn_client.load_league()
    if not league.teams:
        st.warning("No ESPN teams are loaded yet. Add your ESPN league settings in `.env` and restart the app.")
        return

    left, right = st.columns([3, 1])
    with left:
        selected_team = st.selectbox(
            "Team",
            options=league.teams,
            index=_team_default_index(league.teams),
            format_func=lambda team: f"{team.name} ({team.owner})",
        )
    with right:
        free_agent_size = st.number_input("Free agents to scan", 25, 250, 100, step=25)

    if not selected_team.roster:
        st.info(
            "This team does not have a roster in the current ESPN snapshot. "
            "If the app is in demo mode, configure `ESPN_LEAGUE_ID`, `ESPN_SWID`, and `ESPN_S2` in `.env`."
        )
        return

    with st.spinner("Building roster report from ESPN and advanced statistics..."):
        roster_df, fa_df, suggestions_df = build_team_advice(
            selected_team, league, free_agent_size=int(free_agent_size)
        )

    struggling_count = int((roster_df.get("health") == "Struggling").sum()) if "health" in roster_df else 0
    watch_count = int((roster_df.get("health") == "Watch").sum()) if "health" in roster_df else 0
    avg_points = float(roster_df["espn_avg_points"].mean()) if "espn_avg_points" in roster_df else 0.0
    il_df = roster_df[roster_df.get("is_il", False) == True].copy() if "is_il" in roster_df else pd.DataFrame()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _status_card("Roster size", str(len(roster_df)), selected_team.name)
    with c2:
        _status_card("Avg ESPN points", f"{avg_points:.2f}", "per scoring period when available")
    with c3:
        _status_card("Struggling", str(struggling_count), "bottom 25% at compatible positions")
    with c4:
        _status_card("Watch list", str(watch_count), "25th-50th percentile in the league")

    st.subheader("Your Team")
    roster_view = _display_roster(roster_df)
    st.dataframe(
        roster_view,
        hide_index=True,
        use_container_width=True,
        height=_table_height(len(roster_view)),
        column_config=_column_config(),
    )
    st.caption("ESPN avg is calculated from ESPN fantasy total points divided by ESPN games played. Status compares each player against compatible-position players across the league and free-agent pool.")

    st.subheader("Suggested Adds")
    if suggestions_df.empty:
        st.success("No clear free-agent upgrade beat the current roster filters.")
    else:
        st.dataframe(suggestions_df, hide_index=True, use_container_width=True)

    st.subheader("IL Impact")
    if il_df.empty:
        st.success("No IL or injury-tagged players found for this roster.")
    else:
        il_cols = [
            "Name", "position", "injury_status", "games_played",
            "estimated_games_missed", "estimated_value_lost", "proj_pts",
        ]
        il_labels = {
            "Name": "Player",
            "position": "Pos",
            "injury_status": "Injury",
            "games_played": "G",
            "estimated_games_missed": "Est missed",
            "estimated_value_lost": "Value lost",
            "proj_pts": "Model proj",
        }
        available = [col for col in il_cols if col in il_df.columns]
        il_view = il_df[available].rename(columns=il_labels)
        if "Injury" in il_view:
            il_view["Injury"] = il_view["Injury"].replace({"DL": "IL", "D10": "IL10", "D15": "IL15", "D60": "IL60"})
        st.dataframe(
            il_view,
            hide_index=True,
            use_container_width=True,
        )
        st.metric("Total estimated IL value lost", f"{il_df['estimated_value_lost'].sum():.1f}")

    st.subheader("Best Free Agents In League")
    if fa_df.empty:
        st.info("ESPN did not return free agents, and no fallback player pool was available.")
    else:
        fa_view = _display_free_agents(fa_df)
        st.dataframe(
            fa_view,
            hide_index=True,
            use_container_width=True,
            height=_table_height(min(len(fa_view), 12)),
            column_config=_column_config(),
        )

    _stat_legend()
