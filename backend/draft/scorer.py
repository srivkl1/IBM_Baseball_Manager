"""Season-long scoring & standings from MLB game logs."""
from __future__ import annotations

from datetime import date
from typing import Dict, List

import pandas as pd

from backend.config import CONFIG
from backend.data import mlb_stats
from backend.scoring import (ScoringProfile, default_profile, hitter_game_points,
                             pitcher_game_points)


SEASON_START = date(CONFIG.oot_season, 3, 27)
SEASON_END = date(CONFIG.oot_season, 9, 28)


def _safe_float(value: object) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _season_fraction(as_of: date) -> float:
    if as_of <= SEASON_START:
        return 0.0
    if as_of >= SEASON_END:
        return 1.0
    return (as_of - SEASON_START).days / (SEASON_END - SEASON_START).days


def _player_game_log_with_points(player: dict, season: int,
                                 profile: ScoringProfile) -> pd.DataFrame:
    role = player.get("role", "")
    name = player["player"]
    logs = []

    if role == "BAT":
        hitting = mlb_stats.player_game_logs(name, season, "hitting").copy()
        if not hitting.empty:
            hitting["points"] = hitting.apply(
                lambda row: hitter_game_points(row.to_dict(), profile), axis=1
            )
            logs.append(hitting[["date", "points"]])
    elif role == "PIT":
        pitching = mlb_stats.player_game_logs(name, season, "pitching").copy()
        if not pitching.empty:
            pitching["points"] = pitching.apply(
                lambda row: pitcher_game_points(row.to_dict(), profile), axis=1
            )
            logs.append(pitching[["date", "points"]])

    if not logs:
        return pd.DataFrame(columns=["date", "points"])
    out = pd.concat(logs, ignore_index=True)
    out = out.groupby("date", as_index=False)["points"].sum().sort_values("date")
    return out.reset_index(drop=True)


def _player_points(player: dict, as_of: date, profile: ScoringProfile) -> tuple[float, float]:
    today = date.today()
    espn_total = _safe_float(player.get("espn_total_points"))
    espn_projected = _safe_float(player.get("espn_projected_total_points"))
    # ESPN already applies the exact league scoring rules. Use that directly
    # for current-day standings and projections whenever the league provides it.
    if as_of >= today and (espn_total or espn_projected):
        projected = espn_projected or max(espn_total, _safe_float(player.get("proj_pts")))
        return espn_total, projected

    logs = _player_game_log_with_points(player, CONFIG.oot_season, profile)
    if logs.empty:
        projected = espn_projected or float(player.get("proj_pts", 0.0))
        earned = projected * _season_fraction(as_of)
        return earned, projected

    earned = float(logs.loc[logs["date"] <= as_of, "points"].sum())
    season_total = float(logs["points"].sum())
    if season_total > 0:
        projected = season_total if as_of >= SEASON_END else max(
            season_total,
            earned / max(_season_fraction(as_of), 0.05),
        )
    else:
        projected = float(player.get("proj_pts", 0.0))
    return earned, projected


def standings(rosters: Dict[str, List[dict]], as_of: date,
              profile: ScoringProfile | None = None) -> pd.DataFrame:
    profile = profile or default_profile()
    rows = []
    for team, roster in rosters.items():
        earned = 0.0
        projected = 0.0
        for player in roster:
            player_earned, player_projected = _player_points(player, as_of, profile)
            earned += player_earned
            projected += player_projected
        rows.append({
            "team": team,
            "points_to_date": round(earned, 1),
            "projected_full_season": round(projected, 1),
        })
    df = pd.DataFrame(rows).sort_values("points_to_date", ascending=False)
    df.insert(0, "rank", range(1, len(df) + 1))
    df["season_pct"] = f"{_season_fraction(as_of) * 100:.0f}%"
    return df.reset_index(drop=True)


def player_weekly_trajectory(roster: List[dict], as_of: date,
                             profile: ScoringProfile | None = None) -> pd.DataFrame:
    """Per-player cumulative points at each week up to `as_of`."""
    profile = profile or default_profile()
    weeks = pd.date_range(SEASON_START, min(as_of, SEASON_END), freq="W-SUN")
    out = []
    for player in roster:
        logs = _player_game_log_with_points(player, CONFIG.oot_season, profile)
        if logs.empty:
            final = float(player.get("proj_pts", 0.0))
            for week in weeks:
                frac = _season_fraction(week.date())
                out.append({
                    "player": player["player"],
                    "date": week.date(),
                    "cumulative_pts": round(final * frac, 1),
                })
            continue

        for week in weeks:
            cumulative = float(logs.loc[logs["date"] <= week.date(), "points"].sum())
            out.append({
                "player": player["player"],
                "date": week.date(),
                "cumulative_pts": round(cumulative, 1),
            })
    return pd.DataFrame(out)
