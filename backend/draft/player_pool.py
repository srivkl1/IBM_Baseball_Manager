"""Build the draftable player universe from the 3-year stat window.

For each player we compute:
  - agg_war: sum of last 3 seasons' WAR (plus-stats scaled for pitchers)
  - recent_form: most-recent season performance index
  - proj_points: fantasy-points projection for a standard rotisserie league
  - tier: 1 (elite) … 6 (deep bench)
"""
from __future__ import annotations

import pandas as pd

from backend.config import CONFIG
from backend.data import pybaseball_client as pyb


# Fantasy-points weights for a standard roto-like league. Simple & readable.
BAT_POINTS = {"R": 1.0, "HR": 4.0, "RBI": 1.0, "SB": 2.0}
PIT_POINTS = {"W": 5.0, "SV": 5.0, "K": 1.0}


def _hitter_points(row: pd.Series) -> float:
    pts = sum(row.get(k, 0) * w for k, w in BAT_POINTS.items())
    # Reward batting average contribution (scaled to ~ a few points).
    pts += (row.get("AVG", 0) - 0.250) * 300
    return float(pts)


def _pitcher_points(row: pd.Series) -> float:
    pts = sum(row.get(k, 0) * w for k, w in PIT_POINTS.items())
    # Reward low ERA / WHIP.
    pts += max(0.0, (4.20 - row.get("ERA", 4.20))) * 8
    pts += max(0.0, (1.30 - row.get("WHIP", 1.30))) * 20
    return float(pts)


def build_pool() -> pd.DataFrame:
    frames = []
    for season in CONFIG.allowed_seasons:
        b = pyb.batting_stats(season).copy()
        b["role"] = "BAT"
        b["fantasy_pts"] = b.apply(_hitter_points, axis=1)
        p = pyb.pitching_stats(season).copy()
        p["role"] = "PIT"
        p["fantasy_pts"] = p.apply(_pitcher_points, axis=1)
        frames.append(b[["Name", "Team", "Season", "role", "WAR", "fantasy_pts"]])
        frames.append(p[["Name", "Team", "Season", "role", "WAR", "fantasy_pts"]])
    long = pd.concat(frames, ignore_index=True)

    most_recent = long["Season"].max()
    recent = long[long["Season"] == most_recent][["Name", "role", "Team",
                                                  "fantasy_pts", "WAR"]].rename(
        columns={"fantasy_pts": "recent_pts", "WAR": "recent_war"}
    )

    agg = (long.groupby(["Name", "role"], as_index=False)
                .agg(agg_war=("WAR", "sum"),
                     avg_pts=("fantasy_pts", "mean")))
    pool = agg.merge(recent, on=["Name", "role"], how="left")
    pool["recent_pts"] = pool["recent_pts"].fillna(pool["avg_pts"])
    pool["recent_war"] = pool["recent_war"].fillna(pool["agg_war"] / 3.0)

    # Composite draft score: weight recent form slightly higher than 3yr avg.
    pool["draft_score"] = (0.55 * pool["recent_pts"]
                           + 0.30 * pool["avg_pts"]
                           + 1.5 * pool["agg_war"])
    pool = pool.sort_values("draft_score", ascending=False).reset_index(drop=True)
    pool["rank"] = pool.index + 1
    pool["tier"] = pd.cut(pool["rank"],
                          bins=[0, 12, 30, 60, 100, 150, 10_000],
                          labels=[1, 2, 3, 4, 5, 6]).astype(int)
    return pool
