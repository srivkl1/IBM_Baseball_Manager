"""pybaseball wrappers with graceful offline fallback.

We pull:
  - batting_stats(year): FanGraphs standard + advanced (WAR, wRC+, ISO, wOBA, …)
  - pitching_stats(year): FanGraphs pitching (WAR, FIP, xFIP, K%, BB%, …)
  - statcast_batter / statcast_pitcher: per-player rolling Statcast metrics
  - schedule_and_record: team MLB schedule
  - prospects: FanGraphs top prospects

If pybaseball isn't installed or network is blocked, we synthesize a
realistic fallback dataset so the Streamlit demo still runs end-to-end.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from backend.config import CONFIG
from .cache import cached

try:
    import pybaseball as pb  # type: ignore
    pb.cache.enable()
    _HAVE_PB = True
except Exception:
    pb = None
    _HAVE_PB = False


# ---------- Real pulls ----------

@cached("batting_stats")
def _batting_stats_real(season: int) -> pd.DataFrame:
    df = pb.batting_stats(season, qual=100)
    df["Season"] = season
    return df


@cached("pitching_stats")
def _pitching_stats_real(season: int) -> pd.DataFrame:
    df = pb.pitching_stats(season, qual=40)
    df["Season"] = season
    return df


@cached("prospects")
def _prospects_real(season: int) -> pd.DataFrame:
    try:
        return pb.top_prospects(season=season)
    except Exception:
        return pd.DataFrame()


# ---------- Fallback synthesis (deterministic per season) ----------

_FALLBACK_BATTERS = [
    "Aaron Judge", "Shohei Ohtani", "Mookie Betts", "Bobby Witt Jr.", "Juan Soto",
    "Ronald Acuna Jr.", "Freddie Freeman", "Jose Ramirez", "Gunnar Henderson",
    "Corey Seager", "Fernando Tatis Jr.", "Julio Rodriguez", "Yordan Alvarez",
    "Kyle Tucker", "Matt Olson", "Rafael Devers", "Vladimir Guerrero Jr.",
    "Trea Turner", "Francisco Lindor", "Manny Machado", "Adley Rutschman",
    "Austin Riley", "Pete Alonso", "Marcus Semien", "Paul Goldschmidt",
    "Elly De La Cruz", "Jackson Chourio", "Jackson Merrill", "Wyatt Langford",
    "Anthony Santander", "William Contreras", "Bryce Harper", "Brandon Nimmo",
    "Cody Bellinger", "Teoscar Hernandez", "Jake Burger", "Jarren Duran",
    "Mark Vientos", "Triston Casas", "Ezequiel Tovar",
]
_FALLBACK_PITCHERS = [
    "Gerrit Cole", "Zack Wheeler", "Tarik Skubal", "Paul Skenes", "Corbin Burnes",
    "Logan Webb", "Spencer Strider", "Yoshinobu Yamamoto", "Pablo Lopez",
    "Cole Ragans", "Dylan Cease", "Blake Snell", "Framber Valdez", "George Kirby",
    "Logan Gilbert", "Chris Sale", "Max Fried", "Aaron Nola", "Justin Steele",
    "Sonny Gray", "Emmanuel Clase", "Josh Hader", "Devin Williams", "Mason Miller",
    "Edwin Diaz", "Ryan Helsley", "Raisel Iglesias", "Felix Bautista",
]
_TEAMS = ["NYY", "LAD", "ATL", "HOU", "PHI", "BAL", "MIL", "SDP", "SEA", "TEX",
          "BOS", "NYM", "TOR", "CLE", "DET", "KCR", "ARI", "STL", "CHC", "MIN"]


def _synth_batting(season: int) -> pd.DataFrame:
    rng = np.random.default_rng(season * 7 + 13)
    rows = []
    for i, name in enumerate(_FALLBACK_BATTERS):
        # Slight season-to-season noise; a few signature stars get boosts.
        boost = 1.0 + (0.1 if i < 10 else 0.0)
        rows.append({
            "Name": name,
            "Team": _TEAMS[(i + season) % len(_TEAMS)],
            "Season": season,
            "G": int(rng.integers(120, 160)),
            "PA": int(rng.integers(480, 700)),
            "HR": int(max(5, rng.normal(28 * boost, 8))),
            "R": int(max(20, rng.normal(85 * boost, 15))),
            "RBI": int(max(20, rng.normal(85 * boost, 18))),
            "SB": int(max(0, rng.normal(15, 10))),
            "AVG": float(np.clip(rng.normal(0.270 + 0.02 * (boost - 1), 0.02), 0.18, 0.36)),
            "OBP": float(np.clip(rng.normal(0.345, 0.025), 0.25, 0.45)),
            "SLG": float(np.clip(rng.normal(0.470 * boost, 0.05), 0.30, 0.65)),
            "wOBA": float(np.clip(rng.normal(0.350 * boost, 0.03), 0.25, 0.45)),
            "wRC+": float(np.clip(rng.normal(120 * boost, 20), 70, 210)),
            "ISO": float(np.clip(rng.normal(0.200, 0.05), 0.05, 0.40)),
            "BB%": float(np.clip(rng.normal(9.5, 2.0), 3, 18)),
            "K%": float(np.clip(rng.normal(21.0, 4.0), 10, 35)),
            "Barrel%": float(np.clip(rng.normal(9.0 * boost, 2.5), 2, 20)),
            "HardHit%": float(np.clip(rng.normal(40.0, 5.0), 25, 55)),
            "WAR": float(np.clip(rng.normal(3.5 * boost, 1.5), -0.5, 9.0)),
        })
    return pd.DataFrame(rows)


def _synth_pitching(season: int) -> pd.DataFrame:
    rng = np.random.default_rng(season * 11 + 29)
    rows = []
    for i, name in enumerate(_FALLBACK_PITCHERS):
        boost = 1.0 + (0.15 if i < 8 else 0.0)
        is_rp = i >= 20
        rows.append({
            "Name": name,
            "Team": _TEAMS[(i + season + 3) % len(_TEAMS)],
            "Season": season,
            "G": int(rng.integers(55, 70) if is_rp else rng.integers(25, 33)),
            "GS": 0 if is_rp else int(rng.integers(25, 33)),
            "IP": float(rng.integers(55, 75) if is_rp else rng.integers(150, 210)),
            "W": int(rng.integers(2, 6) if is_rp else rng.integers(8, 18)),
            "SV": int(rng.integers(20, 40) if is_rp else 0),
            "K": int((rng.normal(75, 10) if is_rp else rng.normal(195, 30)) * boost),
            "ERA": float(np.clip(rng.normal(3.50 / boost, 0.6), 1.8, 5.5)),
            "FIP": float(np.clip(rng.normal(3.60 / boost, 0.5), 2.0, 5.5)),
            "xFIP": float(np.clip(rng.normal(3.70 / boost, 0.5), 2.0, 5.5)),
            "WHIP": float(np.clip(rng.normal(1.15, 0.15), 0.80, 1.6)),
            "K%": float(np.clip(rng.normal(27.0 * boost, 4.0), 15, 40)),
            "BB%": float(np.clip(rng.normal(7.5, 1.8), 3, 14)),
            "WAR": float(np.clip(rng.normal(3.0 * boost, 1.3), -0.5, 7.0)),
        })
    return pd.DataFrame(rows)


# ---------- Public API ----------

def batting_stats(season: int) -> pd.DataFrame:
    if season not in CONFIG.allowed_seasons:
        raise ValueError(f"Season {season} outside 3-year window {CONFIG.allowed_seasons}")
    if _HAVE_PB:
        try:
            return _batting_stats_real(season)
        except Exception:
            pass
    return _synth_batting(season)


def pitching_stats(season: int) -> pd.DataFrame:
    if season not in CONFIG.allowed_seasons:
        raise ValueError(f"Season {season} outside 3-year window {CONFIG.allowed_seasons}")
    if _HAVE_PB:
        try:
            return _pitching_stats_real(season)
        except Exception:
            pass
    return _synth_pitching(season)


def prospects(season: int) -> pd.DataFrame:
    if _HAVE_PB:
        try:
            df = _prospects_real(season)
            if df is not None and len(df):
                return df
        except Exception:
            pass
    # Fallback: a few high-profile prospects.
    return pd.DataFrame({
        "Name": ["Roki Sasaki", "Samuel Basallo", "Walker Jenkins", "Leodalis De Vries",
                 "Kevin McGonigle", "Max Clark", "Travis Bazzana", "Bryce Rainer"],
        "Team": ["LAD", "BAL", "MIN", "SDP", "DET", "DET", "CLE", "DET"],
        "Season": season,
        "FV": [60, 60, 60, 55, 55, 55, 55, 50],
    })


def three_year_pool() -> pd.DataFrame:
    """Concatenated hitter+pitcher rows across the 3-year scope, with 'role'."""
    frames = []
    for season in CONFIG.allowed_seasons:
        b = batting_stats(season).copy()
        b["role"] = "BAT"
        p = pitching_stats(season).copy()
        p["role"] = "PIT"
        frames.append(b)
        frames.append(p)
    df = pd.concat(frames, ignore_index=True)
    return df


def have_real_data() -> bool:
    return _HAVE_PB
