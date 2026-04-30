"""Helpers for importing an existing ESPN league into the draft state."""
from __future__ import annotations

from typing import Optional, Tuple

from backend.agents.data_retrieval import DataRetrieval
from backend.data import espn_client
from backend.draft import simulator as sim


def _human_index(league: espn_client.LeagueSnapshot) -> int:
    for idx, team in enumerate(league.teams):
        owner = (team.owner or "").strip().lower()
        if owner in {"you", "me", "my team"}:
            return idx
    return 0


def has_existing_rosters(league: Optional[espn_client.LeagueSnapshot] = None) -> bool:
    league = league or espn_client.load_league()
    return league.source == "espn" and any(team.roster for team in league.teams)


def load_existing_league_state() -> Tuple[Optional[sim.DraftState], Optional[dict],
                                          espn_client.LeagueSnapshot]:
    league = espn_client.load_league()
    if not has_existing_rosters(league):
        return None, None, league

    bundle = DataRetrieval().fetch({"needs_player_pool": True})
    team_names = [team.name for team in league.teams if team.name]
    rosters = {
        team.name: [{
            "player": player.name,
            "mlb_team": player.mlb_team,
            "fantasy_position": player.fantasy_position,
            "lineup_slot": player.lineup_slot,
            "espn_total_points": player.total_points,
            "espn_projected_total_points": player.projected_total_points,
        } for player in team.roster]
        for team in league.teams if team.name
    }
    state = sim.from_existing_rosters(
        bundle["player_pool"],
        team_names,
        rosters,
        human_index=min(_human_index(league), max(len(team_names) - 1, 0)),
    )
    if getattr(league, "draft_picks", None):
        state.log = [{
            "pick_no": pick.pick_no,
            "round": pick.round,
            "slot": pick.slot,
            "team": pick.team,
            "player": pick.player,
        } for pick in league.draft_picks]
    return state, bundle, league
