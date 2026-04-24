"""Interactive snake-draft simulation."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from backend.agents.data_retrieval import DataRetrieval
from backend.data import espn_client
from backend.draft import simulator as sim
from backend.workflow import AgentPipeline
from frontend.components import agent_trace, recommendation_card
from frontend.theme import PALETTE, page_header


FALLBACK_TEAM_NAMES = [
    "The Werbley Squad",
    "Bleacher Creatures",
    "Dingers & Things",
    "Bullpen Brigade",
    "Grand Slam Gang",
    "Curveball Kings",
    "Infield Flies",
    "Walk-Off Wonders",
    "Box Score Bandits",
    "Closer Crew",
    "Rally Cap Republic",
    "Fastball Fury",
]


@st.cache_data(show_spinner=False)
def _draft_setup_defaults():
    league = espn_client.load_league()
    team_names = [team.name for team in league.teams if team.name]
    if not team_names:
        team_names = FALLBACK_TEAM_NAMES[:4]

    human_index = 0
    for idx, team in enumerate(league.teams):
        owner = (team.owner or "").strip().lower()
        if owner in {"you", "me", "my team"}:
            human_index = idx
            break

    return {
        "source": league.source,
        "team_names": team_names[:12],
        "human_index": min(human_index, max(len(team_names[:12]) - 1, 0)),
    }


def _ensure_draft_state(team_names, human_index, rounds):
    if "draft_state" in st.session_state:
        return st.session_state["draft_state"]
    bundle = DataRetrieval().fetch({"needs_player_pool": True})
    pool = bundle["player_pool"]
    state = sim.new_draft(pool, team_names, human_index=human_index, rounds=rounds)
    st.session_state["draft_state"] = state
    st.session_state["draft_bundle"] = bundle
    st.session_state["pipeline"] = st.session_state.get("pipeline") or AgentPipeline()
    return state


def render():
    page_header("Draft Room", "Snake draft simulator powered by the 4-agent pipeline.")
    defaults = _draft_setup_defaults()

    with st.sidebar.expander("Draft setup", expanded="draft_state" not in st.session_state):
        default_names = defaults["team_names"] or FALLBACK_TEAM_NAMES[:4]
        n_teams = st.number_input("Teams", 2, 12, len(default_names), step=1)
        rounds = st.number_input("Rounds", 5, 25, 14, step=1)
        team_names = []
        if defaults["source"] == "espn":
            st.caption("Loaded team names from the ESPN league configured in `.env`.")
        else:
            st.caption("Using demo team names because ESPN league data is unavailable.")
        for i in range(int(n_teams)):
            suggested_name = (default_names[i] if i < len(default_names)
                              else FALLBACK_TEAM_NAMES[i])
            team_names.append(st.text_input(f"Team {i+1} name",
                                            value=suggested_name, key=f"team_{i}"))
        human_index = st.selectbox("You are…", range(len(team_names)),
                                   index=min(defaults["human_index"], len(team_names) - 1),
                                   format_func=lambda i: team_names[i])
        if st.button("Start / restart draft"):
            for k in ("draft_state", "draft_bundle", "standings_table"):
                st.session_state.pop(k, None)
            _ensure_draft_state(team_names, int(human_index), int(rounds))

    state = st.session_state.get("draft_state")
    if state is None:
        st.info("Set up the draft in the sidebar and click **Start / restart draft**.")
        return

    # Advance through CPUs until the human is on the clock.
    if not state.is_complete and not state.human_on_clock():
        picks = sim.fast_forward_to_human(state)
        if picks:
            st.toast(f"CPUs made {len(picks)} pick(s)")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Board")
        if state.is_complete:
            st.success("Draft complete — head to the **Season tracker** page.")
        else:
            round_num, slot = state.round_and_slot()
            on_clock = state.team_on_clock()
            st.markdown(
                f"**Round {round_num} · Pick {slot} · On the clock: "
                f"<span style='color:{PALETTE['home_red']};'>"
                f"{on_clock}</span>**", unsafe_allow_html=True,
            )
            if state.human_on_clock():
                if "pipeline" not in st.session_state:
                    st.session_state["pipeline"] = AgentPipeline()
                resp = st.session_state["pipeline"].run(
                    user_text=f"Recommend my round {round_num} pick",
                    skill_level=st.session_state.get("skill_level", "beginner"),
                    draft_state=state,
                )
                recommendation_card(resp)
                agent_trace(resp)

                avail = state.board[state.board["available"]]
                top_5_names = [c["name"] for c in resp.recommendation.candidates[:5]]
                choice = st.selectbox("Pick a player", options=top_5_names +
                                      sorted(set(avail["Name"]) - set(top_5_names)))
                if st.button("Draft this player"):
                    sim.apply_pick(state, choice)
                    st.rerun()
            else:
                st.info(f"Waiting for {on_clock}…")

        st.markdown("#### Top 15 still on the board")
        avail = state.board[state.board["available"]].head(15)
        st.dataframe(
            avail[["Name", "Team", "role", "proj_pts", "rank", "tier"]]
            .rename(columns={"Name": "Player", "role": "Pos", "proj_pts": "Proj pts",
                             "rank": "3-yr rank", "tier": "Tier"}),
            hide_index=True, use_container_width=True,
        )

    with right:
        st.subheader("Your roster")
        my_team = state.teams[state.human_index]
        if state.rosters[my_team]:
            st.dataframe(pd.DataFrame(state.rosters[my_team]),
                         hide_index=True, use_container_width=True)
        else:
            st.caption("No picks yet.")

        st.subheader("All rosters")
        for team, roster in state.rosters.items():
            with st.expander(f"{team} ({len(roster)})"):
                if roster:
                    st.dataframe(pd.DataFrame(roster), hide_index=True,
                                 use_container_width=True)
                else:
                    st.caption("empty")

        st.subheader("Pick log")
        if state.log:
            st.dataframe(pd.DataFrame(state.log).tail(20),
                         hide_index=True, use_container_width=True)
