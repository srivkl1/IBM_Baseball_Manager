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

    @property
    def total_picks(self) -> int:
        return self.rounds * len(self.teams)

    @property
    def current_pick_number(self) -> int:
        return len(self.log) + 1

    @property
    def is_complete(self) -> bool:
        return len(self.log) >= self.total_picks

    def team_on_clock(self) -> str:
        pick_idx = len(self.log)
        round_num = pick_idx // len(self.teams)
        slot = pick_idx % len(self.teams)
        order = list(range(len(self.teams)))
        if round_num % 2 == 1:           # snake
            order.reverse()
        return self.teams[order[slot]]

    def round_and_slot(self) -> tuple[int, int]:
        pick_idx = len(self.log)
        return (pick_idx // len(self.teams)) + 1, (pick_idx % len(self.teams)) + 1

    def human_on_clock(self) -> bool:
        return self.team_on_clock() == self.teams[self.human_index]


def new_draft(board: pd.DataFrame, teams: List[str], human_index: int = 0,
              rounds: int = 14) -> DraftState:
    board = board.copy().reset_index(drop=True)
    board["available"] = True
    rosters = {t: [] for t in teams}
    return DraftState(teams=teams, human_index=human_index, rounds=rounds,
                      board=board, rosters=rosters)


def _team_needs(state: DraftState, team: str) -> dict:
    roster = state.rosters[team]
    counts = {"BAT": sum(1 for r in roster if r["role"] == "BAT"),
              "PIT": sum(1 for r in roster if r["role"] == "PIT")}
    return {k: max(0, v - counts[k]) for k, v in ROSTER_SLOTS.items()}


def _score_candidate(state: DraftState, team: str, player_row: pd.Series) -> float:
    needs = _team_needs(state, team)
    role = player_row["role"]
    base = float(player_row.get("proj_pts", player_row.get("draft_score", 0)))
    need_bonus = 15.0 if needs.get(role, 0) > 0 else -40.0
    # Slight randomness so CPUs aren't perfectly deterministic.
    jitter = np.random.default_rng(
        hash((team, player_row["Name"])) % (2**32)
    ).normal(0, 3)
    return base + need_bonus + jitter


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
        "proj_pts": float(row.get("proj_pts", row.get("draft_score", 0))),
    }
    state.board.loc[mask, "available"] = False
    state.rosters[team].append({k: pick_record[k] for k in ("player", "role", "proj_pts")})
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
