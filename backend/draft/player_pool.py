"""Build the draftable player universe from the recent seasons ending at the target year.

For each player we compute:
  - avg_war: average WAR across the recent-history window
  - recent_form: most-recent season performance index
  - proj_points: fantasy-points projection for a standard rotisserie league
  - tier: 1 (elite) … 6 (deep bench)
"""
from __future__ import annotations

import pandas as pd

from backend.config import CONFIG
from backend.data import mlb_stats
from backend.data import pybaseball_client as pyb
from backend.scoring import ScoringProfile, default_profile, hitter_points, pitcher_points


def _hitter_points(row: pd.Series, profile: ScoringProfile | None = None) -> float:
    return hitter_points(row, profile or default_profile())


def _pitcher_points(row: pd.Series, profile: ScoringProfile | None = None) -> float:
    return pitcher_points(row, profile or default_profile())


def build_pool(profile: ScoringProfile | None = None) -> pd.DataFrame:
    profile = profile or default_profile()
    frames = []
    for season in CONFIG.recent_history_seasons:
        b = pyb.batting_stats(season).copy()
        b["role"] = "BAT"
        b["fantasy_pts"] = b.apply(lambda row: _hitter_points(row, profile), axis=1)
        p = pyb.pitching_stats(season).copy()
        p["role"] = "PIT"
        p["fantasy_pts"] = p.apply(lambda row: _pitcher_points(row, profile), axis=1)
        b["GS"] = 0
        frames.append(b[["Name", "Team", "Season", "role", "WAR", "fantasy_pts", "G", "GS"]])
        frames.append(p[["Name", "Team", "Season", "role", "WAR", "fantasy_pts", "G", "GS"]])
    long = pd.concat(frames, ignore_index=True)

    most_recent = long["Season"].max()
    recent = long[long["Season"] == most_recent][["Name", "role", "Team",
                                                  "fantasy_pts", "WAR", "G", "GS"]].rename(
        columns={"fantasy_pts": "recent_pts", "WAR": "recent_war",
                 "G": "recent_games", "GS": "recent_starts"}
    )

    agg = (long.groupby(["Name", "role"], as_index=False)
                .agg(avg_war=("WAR", "mean"),
                     avg_pts=("fantasy_pts", "mean")))
    pool = agg.merge(recent, on=["Name", "role"], how="left")
    pool["recent_pts"] = pool["recent_pts"].fillna(pool["avg_pts"])
    pool["recent_war"] = pool["recent_war"].fillna(pool["avg_war"])
    pool["recent_games"] = pool["recent_games"].fillna(0)
    pool["recent_starts"] = pool["recent_starts"].fillna(0)

    # Composite draft score: weight recent form slightly higher than the recent-window average.
    pool["draft_score"] = (0.55 * pool["recent_pts"]
                           + 0.30 * pool["avg_pts"]
                           + 1.5 * (pool["recent_war"] + pool["avg_war"]))
    pool["fantasy_position"] = pool.apply(
        lambda row: mlb_stats.position_for_player(
            row["Name"],
            role=row["role"],
            games_started=float(row.get("recent_starts", 0) or 0),
            games_played=float(row.get("recent_games", 0) or 0),
        ),
        axis=1,
    )
    pool = pool.sort_values("draft_score", ascending=False).reset_index(drop=True)
    pool["rank"] = pool.index + 1
    pool["tier"] = pd.cut(pool["rank"],
                          bins=[0, 12, 30, 60, 100, 150, 10_000],
                          labels=[1, 2, 3, 4, 5, 6]).astype(int)
    return pool
