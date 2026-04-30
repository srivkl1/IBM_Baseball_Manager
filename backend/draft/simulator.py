"""Snake-draft simulation engine.

Manages N teams (1 human + CPUs). The human's recommended pick each round is
supplied by the Analysis agent. The CPU teams draft using a simple BPA
strategy with a small positional-need nudge, so the board shifts realistically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


ROSTER_SLOTS = {"BAT": 8, "PIT": 6}  # 14 picks per team; tweak as desired.


@dataclass
class DraftState:
    teams: List[str]
    human_index: int
    rounds: int
    board: pd.DataFrame               # available players (mutated as picks occur)
    log: List[dict] = field(default_factory=list)   # every pick, in order
    rosters: dict = field(default_factory=dict)     # team_name -> list[dict]
    draft_locked: bool = False
    source: str = "simulation"

    @property
    def total_picks(self) -> int:
        return self.rounds * len(self.teams)

    @property
    def current_pick_number(self) -> int:
        return len(self.log) + 1

    @property
    def is_complete(self) -> bool:
        return self.draft_locked or len(self.log) >= self.total_picks

    def team_on_clock(self) -> str:
        if self.draft_locked:
            return "League already drafted"
        pick_idx = len(self.log)
        round_num = pick_idx // len(self.teams)
        slot = pick_idx % len(self.teams)
        order = list(range(len(self.teams)))
        if round_num % 2 == 1:           # snake
            order.reverse()
        return self.teams[order[slot]]

    def round_and_slot(self) -> tuple[int, int]:
        if self.draft_locked:
            return 0, 0
        pick_idx = len(self.log)
        return (pick_idx // len(self.teams)) + 1, (pick_idx % len(self.teams)) + 1

    def human_on_clock(self) -> bool:
        if self.draft_locked:
            return False
        return self.team_on_clock() == self.teams[self.human_index]


def new_draft(board: pd.DataFrame, teams: List[str], human_index: int = 0,
              rounds: int = 14) -> DraftState:
    board = board.copy().reset_index(drop=True)
    board["available"] = True
    rosters = {t: [] for t in teams}
    return DraftState(teams=teams, human_index=human_index, rounds=rounds,
                      board=board, rosters=rosters)


def from_existing_rosters(board: pd.DataFrame, teams: List[str], rosters: dict,
                          human_index: int = 0, source: str = "espn-import"
                          ) -> DraftState:
    board = board.copy().reset_index(drop=True)
    board["available"] = True
    normalized_rosters = {team: [] for team in teams}
    imported_log: List[dict] = []

    for team in teams:
        for player in rosters.get(team, []):
            name = player.get("player", "")
            role = player.get("role", "BAT")
            mlb_team = player.get("mlb_team", "")
            fantasy_position = player.get("fantasy_position", role)
            lineup_slot = player.get("lineup_slot", fantasy_position)
            proj_pts = float(player.get("proj_pts", 0.0))
            espn_total_points = float(player.get("espn_total_points", 0.0))
            espn_projected_total_points = float(player.get("espn_projected_total_points", 0.0))
            mask = (board["Name"] == name) & board["available"]
            if role:
                role_mask = mask & (board["role"] == role)
                if role_mask.any():
                    mask = role_mask
            if mask.any():
                row = board.loc[mask].iloc[0]
                role = row["role"]
                mlb_team = row.get("Team", mlb_team)
                fantasy_position = row.get("fantasy_position", fantasy_position)
                proj_pts = float(row.get("proj_pts", row.get("draft_score", proj_pts)))
                board.loc[[row.name], "available"] = False
            normalized_player = {
                "player": name,
                "role": role,
                "mlb_team": mlb_team,
                "fantasy_position": fantasy_position,
                "lineup_slot": lineup_slot,
                "proj_pts": proj_pts,
                "espn_total_points": espn_total_points,
                "espn_projected_total_points": espn_projected_total_points,
            }
            normalized_rosters[team].append(normalized_player)
            imported_log.append({
                "pick_no": len(imported_log) + 1,
                "round": 0,
                "slot": 0,
                "team": team,
                "player": name,
                "role": role,
                "mlb_team": mlb_team,
                "fantasy_position": fantasy_position,
                "lineup_slot": lineup_slot,
                "proj_pts": proj_pts,
                "espn_total_points": espn_total_points,
                "espn_projected_total_points": espn_projected_total_points,
            })

    rounds = max((len(roster) for roster in normalized_rosters.values()), default=1)
    return DraftState(
        teams=teams,
        human_index=human_index,
        rounds=rounds,
        board=board,
        log=imported_log,
        rosters=normalized_rosters,
        draft_locked=True,
        source=source,
    )


def _team_needs(state: DraftState, team: str) -> dict:
    roster = state.rosters[team]
    counts = {"BAT": sum(1 for r in roster if r["role"] == "BAT"),
              "PIT": sum(1 for r in roster if r["role"] == "PIT")}
    return {k: max(0, v - counts[k]) for k, v in ROSTER_SLOTS.items()}


def _score_candidate(state: DraftState, team: str, player_row: pd.Series) -> float:
    needs = _team_needs(state, team)
    role = player_row["role"]
    base = float(player_row.get("proj_pts", player_row.get("draft_score", 0)))
    roster = state.rosters.get(team, [])
    picks_made = len(roster)
    role_count = sum(1 for r in roster if r["role"] == role)
    same_team_count = sum(1 for r in roster if r.get("mlb_team") == player_row.get("Team"))

    need_bonus = 18.0 if needs.get(role, 0) > 0 else -40.0
    role_balance_bonus = max(0.0, needs.get(role, 0)) * 4.5
    early_pitching_bonus = 0.0
    if role == "PIT" and picks_made >= 2 and role_count == 0:
        early_pitching_bonus = 18.0
    stack_penalty = same_team_count * 4.0
    # Slight randomness so CPUs aren't perfectly deterministic.
    jitter = np.random.default_rng(
        hash((team, player_row["Name"])) % (2**32)
    ).normal(0, 3)
    return base + need_bonus + role_balance_bonus + early_pitching_bonus - stack_penalty + jitter


def recommend_pick(state: DraftState, team: Optional[str] = None,
                   top_n: int = 5) -> pd.DataFrame:
    team = team or state.team_on_clock()
    avail = state.board[state.board["available"]].copy()
    if avail.empty:
        return avail
    avail["suitability"] = avail.apply(
        lambda r: _score_candidate(state, team, r), axis=1
    )
    return avail.sort_values("suitability", ascending=False).head(top_n)


def apply_pick(state: DraftState, player_name: str) -> dict:
    mask = (state.board["Name"] == player_name) & state.board["available"]
    if not mask.any():
        raise ValueError(f"{player_name} is not available on the board")
    row = state.board[mask].iloc[0]
    team = state.team_on_clock()
    round_num, slot = state.round_and_slot()
    pick_record = {
        "pick_no": state.current_pick_number,
        "round": round_num,
        "slot": slot,
        "team": team,
        "player": player_name,
        "role": row["role"],
        "mlb_team": row.get("Team", ""),
        "fantasy_position": row.get("fantasy_position", row["role"]),
        "proj_pts": float(row.get("proj_pts", row.get("draft_score", 0))),
        "espn_total_points": float(row.get("espn_total_points", 0.0)),
        "espn_projected_total_points": float(row.get("espn_projected_total_points", 0.0)),
    }
    state.board.loc[mask, "available"] = False
    state.rosters[team].append({
        k: pick_record[k]
        for k in (
            "player",
            "role",
            "mlb_team",
            "fantasy_position",
            "proj_pts",
            "espn_total_points",
            "espn_projected_total_points",
        )
    })
    state.log.append(pick_record)
    return pick_record


def cpu_autopick(state: DraftState) -> dict:
    """Make the CPU on the clock pick its top suitability candidate."""
    team = state.team_on_clock()
    recs = recommend_pick(state, team=team, top_n=1)
    if recs.empty:
        raise RuntimeError("No players available")
    return apply_pick(state, recs.iloc[0]["Name"])


def fast_forward_to_human(state: DraftState) -> List[dict]:
    made = []
    while not state.is_complete and not state.human_on_clock():
        made.append(cpu_autopick(state))
    return made
