"""MLB Stats API helpers for player identity and per-game logs."""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional
from urllib.parse import quote

import pandas as pd
import requests

from backend.data.cache import cached
from backend.data.espn_client import normalize_fantasy_position

try:
    from pybaseball import playerid_lookup  # type: ignore
    _HAVE_PYBASEBALL_IDS = True
except Exception:
    playerid_lookup = None  # type: ignore
    _HAVE_PYBASEBALL_IDS = False


_BASE = "https://statsapi.mlb.com/api/v1"


def _request_json(url: str, params: Optional[dict] = None) -> dict:
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def _split_name(name: str) -> tuple[str, str]:
    parts = name.strip().split()
    if len(parts) == 1:
        return parts[0], ""
    return parts[-1], " ".join(parts[:-1])


@cached("mlb_player_identity")
def resolve_player_identity(name: str) -> dict:
    if _HAVE_PYBASEBALL_IDS:
        try:
            last, first = _split_name(name)
            matches = playerid_lookup(last, first)
            if matches is not None and len(matches):
                mlbam = int(matches.iloc[0]["key_mlbam"])
                return _request_json(f"{_BASE}/people/{mlbam}")["people"][0]
        except Exception:
            pass
    try:
        data = _request_json(f"{_BASE}/people/search", params={"names": name})
        people = data.get("people", [])
        if people:
            return people[0]
    except Exception:
        pass
    return {}


def normalize_primary_position(position: str, role: str = "", games_started: float = 0.0,
                               games_played: float = 0.0) -> str:
    position = normalize_fantasy_position(position)
    if position == "P":
        if games_started and games_started >= max(1.0, games_played * 0.4):
            return "SP"
        return "RP"
    return position or ("SP" if role == "PIT" else "")


def position_for_player(name: str, role: str = "", games_started: float = 0.0,
                        games_played: float = 0.0) -> str:
    identity = resolve_player_identity(name)
    primary = identity.get("primaryPosition", {}) if identity else {}
    return normalize_primary_position(
        str(primary.get("abbreviation", "")),
        role=role,
        games_started=games_started,
        games_played=games_played,
    )


@cached("mlb_player_game_logs")
def player_game_logs(name: str, season: int, group: str) -> pd.DataFrame:
    identity = resolve_player_identity(name)
    player_id = identity.get("id")
    if not player_id:
        return pd.DataFrame()
    try:
        data = _request_json(
            f"{_BASE}/people/{player_id}/stats",
            params={"stats": "gameLog", "group": group, "season": season},
        )
    except Exception:
        return pd.DataFrame()
    stats = data.get("stats", [])
    if not stats:
        return pd.DataFrame()
    splits = stats[0].get("splits", [])
    rows: List[dict] = []
    for split in splits:
        game_date = split.get("date")
        if not game_date:
            continue
        stat = split.get("stat", {})
        rows.append({"date": pd.to_datetime(game_date).date(), **stat})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
