"""League-aware fantasy scoring helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional

import pandas as pd

try:
    from espn_api.baseball.constant import STATS_MAP  # type: ignore
except Exception:
    STATS_MAP = {}


DEFAULT_BATTER_WEIGHTS = {"R": 1.0, "HR": 4.0, "RBI": 1.0, "SB": 2.0, "AVG_BONUS": 300.0}
DEFAULT_PITCHER_WEIGHTS = {"W": 5.0, "SV": 5.0, "K": 1.0, "ERA_BONUS": 8.0, "WHIP_BONUS": 20.0}


@dataclass
class ScoringProfile:
    scoring_type: str = "H2H_CATEGORY"
    source: str = "default"
    uses_points: bool = False
    batter_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_BATTER_WEIGHTS))
    pitcher_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PITCHER_WEIGHTS))


def _first_points_value(item: dict) -> float:
    overrides = item.get("pointsOverrides") or {}
    if overrides:
        try:
            return float(next(iter(overrides.values())))
        except (TypeError, ValueError, StopIteration):
            pass
    try:
        return float(item.get("points", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _safe_value(row: pd.Series, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        return float(default if pd.isna(value) else value)
    except (TypeError, ValueError):
        return default


def _safe_mapping_value(stats: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = stats.get(key, default)
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_ip_to_outs(value: object) -> float:
    if value in (None, ""):
        return 0.0
    text = str(value)
    if "." not in text:
        try:
            return float(text) * 3.0
        except ValueError:
            return 0.0
    whole, frac = text.split(".", 1)
    try:
        return float(int(whole) * 3 + int(frac))
    except ValueError:
        return 0.0


def _batting_1b(row: pd.Series) -> float:
    return max(0.0, _safe_value(row, "H") - _safe_value(row, "2B") - _safe_value(row, "3B") - _safe_value(row, "HR"))


def _batting_tb(row: pd.Series) -> float:
    return (_batting_1b(row)
            + 2 * _safe_value(row, "2B")
            + 3 * _safe_value(row, "3B")
            + 4 * _safe_value(row, "HR"))


def _pitching_outs(row: pd.Series) -> float:
    return _safe_value(row, "IP") * 3.0


def _game_1b(stats: Mapping[str, object]) -> float:
    return max(0.0, _safe_mapping_value(stats, "hits")
               - _safe_mapping_value(stats, "doubles")
               - _safe_mapping_value(stats, "triples")
               - _safe_mapping_value(stats, "homeRuns"))


def _game_tb(stats: Mapping[str, object]) -> float:
    total_bases = _safe_mapping_value(stats, "totalBases")
    if total_bases:
        return total_bases
    return (_game_1b(stats)
            + 2 * _safe_mapping_value(stats, "doubles")
            + 3 * _safe_mapping_value(stats, "triples")
            + 4 * _safe_mapping_value(stats, "homeRuns"))


BATTER_STAT_FNS: Dict[int, tuple[str, Callable[[pd.Series], float]]] = {
    0: ("AB", lambda row: _safe_value(row, "AB")),
    1: ("H", lambda row: _safe_value(row, "H")),
    2: ("AVG", lambda row: _safe_value(row, "AVG")),
    3: ("2B", lambda row: _safe_value(row, "2B")),
    4: ("3B", lambda row: _safe_value(row, "3B")),
    5: ("HR", lambda row: _safe_value(row, "HR")),
    7: ("1B", _batting_1b),
    8: ("TB", _batting_tb),
    10: ("BB", lambda row: _safe_value(row, "BB")),
    12: ("HBP", lambda row: _safe_value(row, "HBP")),
    13: ("SF", lambda row: _safe_value(row, "SF")),
    14: ("SH", lambda row: _safe_value(row, "SH")),
    15: ("SAC", lambda row: _safe_value(row, "SAC", _safe_value(row, "SF") + _safe_value(row, "SH"))),
    16: ("PA", lambda row: _safe_value(row, "PA")),
    17: ("OBP", lambda row: _safe_value(row, "OBP")),
    18: ("OPS", lambda row: _safe_value(row, "OPS", _safe_value(row, "OBP") + _safe_value(row, "SLG"))),
    20: ("R", lambda row: _safe_value(row, "R")),
    21: ("RBI", lambda row: _safe_value(row, "RBI")),
    23: ("SB", lambda row: _safe_value(row, "SB")),
    24: ("CS", lambda row: _safe_value(row, "CS")),
    25: ("SB-CS", lambda row: _safe_value(row, "SB") - _safe_value(row, "CS")),
    26: ("GDP", lambda row: _safe_value(row, "GDP")),
    27: ("B_SO", lambda row: _safe_value(row, "SO", _safe_value(row, "K"))),
    81: ("G", lambda row: _safe_value(row, "G")),
}

PITCHER_STAT_FNS: Dict[int, tuple[str, Callable[[pd.Series], float]]] = {
    32: ("G", lambda row: _safe_value(row, "G")),
    33: ("GS", lambda row: _safe_value(row, "GS")),
    34: ("OUTS", _pitching_outs),
    35: ("TBF", lambda row: _safe_value(row, "BF")),
    37: ("P_H", lambda row: _safe_value(row, "H")),
    39: ("P_BB", lambda row: _safe_value(row, "BB")),
    41: ("WHIP", lambda row: _safe_value(row, "WHIP")),
    44: ("P_R", lambda row: _safe_value(row, "R")),
    45: ("ER", lambda row: _safe_value(row, "ER")),
    46: ("P_HR", lambda row: _safe_value(row, "HR")),
    47: ("ERA", lambda row: _safe_value(row, "ERA")),
    48: ("K", lambda row: _safe_value(row, "K")),
    49: ("K/9", lambda row: _safe_value(row, "K/9", (_safe_value(row, "K") / max(_safe_value(row, "IP"), 1e-9)) * 9.0)),
    53: ("W", lambda row: _safe_value(row, "W")),
    54: ("L", lambda row: _safe_value(row, "L")),
    57: ("SV", lambda row: _safe_value(row, "SV")),
    60: ("HLD", lambda row: _safe_value(row, "HLD")),
    62: ("CG", lambda row: _safe_value(row, "CG")),
    63: ("QS", lambda row: _safe_value(row, "QS")),
    81: ("G", lambda row: _safe_value(row, "G")),
    83: ("SVHD", lambda row: _safe_value(row, "SVHD", _safe_value(row, "SV") + _safe_value(row, "HLD"))),
}

BATTER_GAME_STAT_FNS: Dict[str, Callable[[Mapping[str, object]], float]] = {
    "AB": lambda stats: _safe_mapping_value(stats, "atBats"),
    "H": lambda stats: _safe_mapping_value(stats, "hits"),
    "AVG": lambda stats: _safe_mapping_value(stats, "avg", _safe_mapping_value(stats, "battingAverage")),
    "2B": lambda stats: _safe_mapping_value(stats, "doubles"),
    "3B": lambda stats: _safe_mapping_value(stats, "triples"),
    "HR": lambda stats: _safe_mapping_value(stats, "homeRuns"),
    "1B": _game_1b,
    "TB": _game_tb,
    "BB": lambda stats: _safe_mapping_value(stats, "baseOnBalls"),
    "HBP": lambda stats: _safe_mapping_value(stats, "hitByPitch"),
    "SF": lambda stats: _safe_mapping_value(stats, "sacFlies"),
    "SH": lambda stats: _safe_mapping_value(stats, "sacBunts"),
    "SAC": lambda stats: _safe_mapping_value(stats, "sacFlies") + _safe_mapping_value(stats, "sacBunts"),
    "PA": lambda stats: _safe_mapping_value(stats, "plateAppearances"),
    "OBP": lambda stats: _safe_mapping_value(stats, "obp", _safe_mapping_value(stats, "onBasePercentage")),
    "OPS": lambda stats: _safe_mapping_value(stats, "ops"),
    "R": lambda stats: _safe_mapping_value(stats, "runs"),
    "RBI": lambda stats: _safe_mapping_value(stats, "rbi"),
    "SB": lambda stats: _safe_mapping_value(stats, "stolenBases"),
    "CS": lambda stats: _safe_mapping_value(stats, "caughtStealing"),
    "SB-CS": lambda stats: _safe_mapping_value(stats, "stolenBases") - _safe_mapping_value(stats, "caughtStealing"),
    "GDP": lambda stats: _safe_mapping_value(stats, "groundedIntoDoublePlay"),
    "B_SO": lambda stats: _safe_mapping_value(stats, "strikeOuts"),
    "G": lambda stats: 1.0,
}

PITCHER_GAME_STAT_FNS: Dict[str, Callable[[Mapping[str, object]], float]] = {
    "G": lambda stats: 1.0,
    "GS": lambda stats: _safe_mapping_value(stats, "gamesStarted"),
    "OUTS": lambda stats: _safe_mapping_value(stats, "outs", _parse_ip_to_outs(stats.get("inningsPitched"))),
    "TBF": lambda stats: _safe_mapping_value(stats, "battersFaced"),
    "P_H": lambda stats: _safe_mapping_value(stats, "hits"),
    "P_BB": lambda stats: _safe_mapping_value(stats, "baseOnBalls"),
    "WHIP": lambda stats: _safe_mapping_value(stats, "whip"),
    "P_R": lambda stats: _safe_mapping_value(stats, "runs"),
    "ER": lambda stats: _safe_mapping_value(stats, "earnedRuns"),
    "P_HR": lambda stats: _safe_mapping_value(stats, "homeRuns"),
    "ERA": lambda stats: _safe_mapping_value(stats, "era"),
    "K": lambda stats: _safe_mapping_value(stats, "strikeOuts"),
    "K/9": lambda stats: _safe_mapping_value(stats, "strikeoutsPer9Inn"),
    "W": lambda stats: _safe_mapping_value(stats, "wins"),
    "L": lambda stats: _safe_mapping_value(stats, "losses"),
    "SV": lambda stats: _safe_mapping_value(stats, "saves"),
    "HLD": lambda stats: _safe_mapping_value(stats, "holds"),
    "CG": lambda stats: _safe_mapping_value(stats, "completeGames"),
    "QS": lambda stats: _safe_mapping_value(stats, "qualityStarts"),
    "SVHD": lambda stats: _safe_mapping_value(stats, "saves") + _safe_mapping_value(stats, "holds"),
}


def default_profile() -> ScoringProfile:
    return ScoringProfile()


def profile_from_espn_settings(scoring_type: str, raw_scoring_settings: Optional[dict]) -> ScoringProfile:
    profile = ScoringProfile(scoring_type=scoring_type or "H2H_CATEGORY", source="espn")
    items = (raw_scoring_settings or {}).get("scoringItems", [])
    if not items:
        return profile

    batter_weights: Dict[str, float] = {}
    pitcher_weights: Dict[str, float] = {}
    for item in items:
        stat_id = item.get("statId")
        if stat_id in BATTER_STAT_FNS:
            batter_weights[BATTER_STAT_FNS[stat_id][0]] = _first_points_value(item)
        if stat_id in PITCHER_STAT_FNS:
            pitcher_weights[PITCHER_STAT_FNS[stat_id][0]] = _first_points_value(item)

    if batter_weights or pitcher_weights:
        profile.uses_points = True
        profile.batter_weights = batter_weights or dict(DEFAULT_BATTER_WEIGHTS)
        profile.pitcher_weights = pitcher_weights or dict(DEFAULT_PITCHER_WEIGHTS)
    return profile


def hitter_points(row: pd.Series, profile: Optional[ScoringProfile] = None) -> float:
    profile = profile or default_profile()
    if not profile.uses_points:
        pts = sum(_safe_value(row, k) * w for k, w in DEFAULT_BATTER_WEIGHTS.items() if k != "AVG_BONUS")
        pts += (_safe_value(row, "AVG") - 0.250) * DEFAULT_BATTER_WEIGHTS["AVG_BONUS"]
        return float(pts)

    total = 0.0
    for stat_id, (label, getter) in BATTER_STAT_FNS.items():
        weight = profile.batter_weights.get(label)
        if weight:
            total += getter(row) * weight
    return float(total)


def pitcher_points(row: pd.Series, profile: Optional[ScoringProfile] = None) -> float:
    profile = profile or default_profile()
    if not profile.uses_points:
        pts = sum(_safe_value(row, k) * w for k, w in DEFAULT_PITCHER_WEIGHTS.items()
                  if k not in {"ERA_BONUS", "WHIP_BONUS"})
        pts += max(0.0, (4.20 - _safe_value(row, "ERA", 4.20))) * DEFAULT_PITCHER_WEIGHTS["ERA_BONUS"]
        pts += max(0.0, (1.30 - _safe_value(row, "WHIP", 1.30))) * DEFAULT_PITCHER_WEIGHTS["WHIP_BONUS"]
        return float(pts)

    total = 0.0
    for stat_id, (label, getter) in PITCHER_STAT_FNS.items():
        weight = profile.pitcher_weights.get(label)
        if weight:
            total += getter(row) * weight
    return float(total)


def hitter_game_points(stats: Mapping[str, object], profile: Optional[ScoringProfile] = None) -> float:
    profile = profile or default_profile()
    if not profile.uses_points:
        return 0.0
    total = 0.0
    for label, getter in BATTER_GAME_STAT_FNS.items():
        weight = profile.batter_weights.get(label)
        if weight:
            total += getter(stats) * weight
    return float(total)


def pitcher_game_points(stats: Mapping[str, object], profile: Optional[ScoringProfile] = None) -> float:
    profile = profile or default_profile()
    if not profile.uses_points:
        return 0.0
    total = 0.0
    for label, getter in PITCHER_GAME_STAT_FNS.items():
        weight = profile.pitcher_weights.get(label)
        if weight:
            total += getter(stats) * weight
    return float(total)
