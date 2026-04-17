"""Season-long scoring & standings, with an 'as-of date' for historical replay.

We approximate in-season points by linearly accruing each player's 2025 final
fantasy points across the 2025 MLB regular season (Mar 27 – Sep 28). This lets
the UI slide a date and show evolving standings without needing per-game logs.
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List

import pandas as pd

from backend.config import CONFIG
from backend.data import pybaseball_client as pyb
from backend.draft.player_pool import _hitter_points, _pitcher_points


SEASON_START = date(2025, 3, 27)
SEASON_END = date(2025, 9, 28)


def _season_fraction(as_of: date) -> float:
    if as_of <= SEASON_START:
        return 0.0
    if as_of >= SEASON_END:
        return 1.0
    return (as_of - SEASON_START).days / (SEASON_END - SEASON_START).days


def _final_points_lookup() -> Dict[str, float]:
    season = CONFIG.oot_season
    bat = pyb.batting_stats(season).copy()
    pit = pyb.pitching_stats(season).copy()
    bat["pts"] = bat.apply(_hitter_points, axis=1)
    pit["pts"] = pit.apply(_pitcher_points, axis=1)
    combined = pd.concat([bat[["Name", "pts"]], pit[["Name", "pts"]]],
                         ignore_index=True)
    return dict(zip(combined["Name"], combined["pts"]))


def standings(rosters: Dict[str, List[dict]], as_of: date) -> pd.DataFrame:
    frac = _season_fraction(as_of)
    finals = _final_points_lookup()
    rows = []
    for team, roster in rosters.items():
        earned = 0.0
        projected = 0.0
        for pick in roster:
            final = finals.get(pick["player"], pick.get("proj_pts", 0.0))
            earned += final * frac
            projected += final
        rows.append({"team": team, "points_to_date": round(earned, 1),
                     "projected_full_season": round(projected, 1)})
    df = pd.DataFrame(rows).sort_values("points_to_date", ascending=False)
    df.insert(0, "rank", range(1, len(df) + 1))
    df["season_pct"] = f"{frac * 100:.0f}%"
    return df.reset_index(drop=True)


def player_weekly_trajectory(roster: List[dict], as_of: date) -> pd.DataFrame:
    """Per-player cumulative points at each week up to `as_of`."""
    finals = _final_points_lookup()
    weeks = pd.date_range(SEASON_START, min(as_of, SEASON_END), freq="W-SUN")
    out = []
    for pick in roster:
        final = finals.get(pick["player"], pick.get("proj_pts", 0.0))
        for w in weeks:
            frac = _season_fraction(w.date())
            out.append({"player": pick["player"], "date": w.date(),
                        "cumulative_pts": round(final * frac, 1)})
    return pd.DataFrame(out)
