"""ESPN fantasy-baseball league wrapper with demo fallback."""
from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass, field
from typing import List

import requests

from backend.config import CONFIG
from backend.scoring import ScoringProfile, default_profile, profile_from_espn_settings

try:
    from espn_api.baseball import League  # type: ignore
    _HAVE_ESPN = True
except Exception:
    League = None  # type: ignore
    _HAVE_ESPN = False


def normalize_fantasy_position(position: str) -> str:
    position = (position or "").upper().strip()
    if position in {"LF", "CF", "RF"}:
        return "OF"
    if position == "P":
        return "SP"
    return position


@dataclass
class FantasyPlayer:
    name: str
    player_id: int = 0
    mlb_team: str = ""
    fantasy_position: str = ""
    lineup_slot: str = ""
    eligible_positions: List[str] = field(default_factory=list)
    injury_status: str = ""
    status: str = ""
    total_points: float = 0.0
    avg_points: float = 0.0
    games_played: float = 0.0
    projected_total_points: float = 0.0
    rostership: float = 0.0


@dataclass
class FantasyTeam:
    team_id: int
    name: str
    owner: str
    roster: List[FantasyPlayer] = field(default_factory=list)
    wins: int = 0
    losses: int = 0
    ties: int = 0
    standing: int = 0


@dataclass
class FantasyDraftPick:
    pick_no: int
    round: int
    slot: int
    team: str
    player: str


@dataclass
class LeagueSnapshot:
    league_id: str
    season: int
    scoring_type: str
    teams: List[FantasyTeam]
    free_agents: List[str]
    source: str
    scoring_profile: ScoringProfile = field(default_factory=default_profile)
    draft_picks: List[FantasyDraftPick] = field(default_factory=list)
    current_matchup_period: int = 0
    current_scoring_period: int = 0
    error: str = ""


def _to_fantasy_player(player) -> FantasyPlayer:
    eligible = [normalize_fantasy_position(pos) for pos in getattr(player, "eligibleSlots", [])]
    lineup_slot = normalize_fantasy_position(getattr(player, "lineupSlot", ""))
    fantasy_position = normalize_fantasy_position(lineup_slot or getattr(player, "position", ""))
    if fantasy_position == "BE" and eligible:
        fantasy_position = eligible[0]
    if fantasy_position == "P":
        if "RP" in eligible and "SP" not in eligible:
            fantasy_position = "RP"
        else:
            fantasy_position = "SP"

    def _first_float(*attrs: str) -> float:
        for attr in attrs:
            try:
                value = getattr(player, attr, None)
                if value not in (None, ""):
                    return float(value)
            except (TypeError, ValueError):
                continue
        return 0.0

    season_stats = getattr(player, "stats", {}).get(0, {}) if getattr(player, "stats", None) else {}
    breakdown = season_stats.get("breakdown", {}) if isinstance(season_stats, dict) else {}
    games_played = _first_float("games_played", "gamesPlayed")
    if not games_played:
        try:
            games_played = float(
                breakdown.get("G", 0.0)
                or breakdown.get("GP", 0.0)
                or breakdown.get("GS", 0.0)
                or 0.0
            )
        except (TypeError, ValueError):
            games_played = 0.0
    total_points = float(getattr(player, "total_points", 0.0) or 0.0)
    avg_points = _first_float("avg_points", "average_points", "averagePoints", "appliedAverage")
    if not avg_points and games_played:
        avg_points = total_points / games_played

    return FantasyPlayer(
        name=getattr(player, "name", ""),
        player_id=int(getattr(player, "playerId", 0) or 0),
        mlb_team=getattr(player, "proTeam", ""),
        fantasy_position=fantasy_position,
        lineup_slot=lineup_slot,
        eligible_positions=eligible,
        injury_status=str(getattr(player, "injuryStatus", "") or ""),
        status=str(getattr(player, "status", "") or ""),
        total_points=total_points,
        avg_points=avg_points,
        games_played=games_played,
        projected_total_points=float(getattr(player, "projected_total_points", 0.0) or 0.0),
        rostership=_first_float("percent_owned", "percentOwned", "percent_owned_by", "ownership"),
    )


def _runtime_espn_config() -> dict:
    """Return ESPN settings scoped to the current Streamlit browser session."""
    try:
        import streamlit as st
        runtime = st.session_state.get("runtime_espn_config", {})
    except Exception:
        runtime = {}
    league_id = str(runtime.get("league_id", CONFIG.espn_league_id) or "").strip()
    try:
        season = int(str(runtime.get("season", CONFIG.espn_season) or CONFIG.espn_season).strip())
    except ValueError:
        season = CONFIG.espn_season
    return {
        "league_id": league_id,
        "season": season,
        "swid": str(runtime.get("swid", CONFIG.espn_swid) or "").strip(),
        "s2": str(runtime.get("s2", CONFIG.espn_s2) or "").strip(),
    }


def runtime_league_cache_key() -> tuple[str, int, bool]:
    settings = _runtime_espn_config()
    return (settings["league_id"], settings["season"], bool(settings["swid"] and settings["s2"]))


def _demo_snapshot(error: str = "", season: int | None = None) -> LeagueSnapshot:
    teams = [
        FantasyTeam(i + 1, n, o)
        for i, (n, o) in enumerate([
            ("The Werbley Squad", "You"),
            ("Bleacher Creatures", "CPU-1"),
            ("Dingers & Things", "CPU-2"),
            ("Bullpen Brigade", "CPU-3"),
        ])
    ]
    return LeagueSnapshot(
        league_id="demo-league",
        season=season or CONFIG.espn_season,
        scoring_type="H2H Categories (R/HR/RBI/SB/AVG + W/SV/K/ERA/WHIP)",
        teams=teams,
        free_agents=[],
        source="demo",
        scoring_profile=default_profile(),
        error=error,
    )


@lru_cache(maxsize=8)
def _load_native_league_cached(league_id: str, season: int, swid: str, s2: str):
    if not (_HAVE_ESPN and league_id):
        return None
    kwargs = {"league_id": int(league_id), "year": season}
    if swid and s2:
        kwargs.update({"espn_s2": s2, "swid": swid})
    return League(**kwargs)


def load_native_league():
    settings = _runtime_espn_config()
    return _load_native_league_cached(
        settings["league_id"],
        settings["season"],
        settings["swid"],
        settings["s2"],
    )


def clear_runtime_caches():
    _load_native_league_cached.cache_clear()


def _draft_picks_for_league(lg) -> List[FantasyDraftPick]:
    picks: List[FantasyDraftPick] = []
    for idx, pick in enumerate(getattr(lg, "draft", []), start=1):
        team = getattr(getattr(pick, "team", None), "team_name", "")
        picks.append(
            FantasyDraftPick(
                pick_no=idx,
                round=int(getattr(pick, "round_num", 0) or 0),
                slot=int(getattr(pick, "round_pick", 0) or 0),
                team=team,
                player=str(getattr(pick, "playerName", "") or ""),
            )
        )
    return picks


def load_league() -> LeagueSnapshot:
    settings = _runtime_espn_config()
    if not (_HAVE_ESPN and settings["league_id"]):
        reason = "ESPN package unavailable." if not _HAVE_ESPN else "ESPN_LEAGUE_ID is missing."
        return _demo_snapshot(reason, season=settings["season"])
    try:
        lg = load_native_league()
        if lg is None:
            return _demo_snapshot("ESPN client returned no league.", season=settings["season"])
        teams = [
            FantasyTeam(
                team_id=t.team_id,
                name=t.team_name,
                owner=(t.owners[0].get("displayName") if t.owners else "?"),
                roster=[_to_fantasy_player(p) for p in getattr(t, "roster", [])],
                wins=int(getattr(t, "wins", 0) or 0),
                losses=int(getattr(t, "losses", 0) or 0),
                ties=int(getattr(t, "ties", 0) or 0),
                standing=int(getattr(t, "standing", 0) or 0),
            )
            for t in lg.teams
        ]
        fas: List[str] = []
        try:
            fas = [p.name for p in lg.free_agents(size=100)]
        except Exception:
            fas = []
        return LeagueSnapshot(
            league_id=str(settings["league_id"]),
            season=settings["season"],
            scoring_type=str(getattr(lg.settings, "scoring_type", "H2H Categories")),
            teams=teams,
            free_agents=fas,
            source="espn",
            scoring_profile=profile_from_espn_settings(
                str(getattr(lg.settings, "scoring_type", "H2H Categories")),
                getattr(lg.settings, "_raw_scoring_settings", {}),
            ),
            draft_picks=_draft_picks_for_league(lg),
            current_matchup_period=int(getattr(lg, "currentMatchupPeriod", 0) or 0),
            current_scoring_period=int(getattr(lg, "current_week", 0) or 0),
        )
    except Exception as exc:
        return _demo_snapshot(str(exc), season=settings["season"])


def load_free_agent_players(size: int = 100) -> List[FantasyPlayer]:
    """Return detailed ESPN free-agent rows when the native ESPN client is available."""
    settings = _runtime_espn_config()
    if not (_HAVE_ESPN and settings["league_id"]):
        return []
    try:
        lg = load_native_league()
        if lg is None:
            return []
        return [_to_fantasy_player(player) for player in lg.free_agents(size=size)]
    except Exception:
        return []


@lru_cache(maxsize=1)
def public_mlb_injury_map() -> dict[str, dict]:
    """Return current public ESPN MLB injury statuses keyed by player name."""
    try:
        response = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/injuries",
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return {}

    injuries: dict[str, dict] = {}
    for team_block in data.get("injuries", []):
        team_name = team_block.get("displayName", "")
        for item in team_block.get("injuries", []):
            athlete = item.get("athlete", {}) or {}
            name = athlete.get("displayName") or item.get("displayName") or item.get("name")
            if not name:
                continue
            status = item.get("status") or (item.get("status", {}) or {}).get("name", "")
            if isinstance(status, dict):
                status = status.get("name") or status.get("type") or status.get("abbreviation") or ""
            details = item.get("details", {}) or {}
            fantasy_status = details.get("fantasyStatus", {}) if isinstance(details, dict) else {}
            injuries[name.strip().casefold()] = {
                "status": str(status or ""),
                "fantasy_status": str(fantasy_status.get("description") or fantasy_status.get("abbreviation") or ""),
                "type": str((item.get("type", {}) or {}).get("description") or ""),
                "detail": str(details.get("detail") or details.get("type") or ""),
                "return_date": str(details.get("returnDate") or ""),
                "team": team_name,
                "summary": item.get("shortComment") or item.get("longComment") or "",
            }
    return injuries


def player_news(name: str, limit: int = 3) -> List[dict]:
    settings = _runtime_espn_config()
    if not (_HAVE_ESPN and settings["league_id"]):
        return []
    try:
        lg = load_native_league()
        if lg is None:
            return []
        player_id = getattr(lg, "player_map", {}).get(name)
        if not player_id:
            lowered = name.strip().casefold()
            for key, value in getattr(lg, "player_map", {}).items():
                if isinstance(key, str) and key.strip().casefold() == lowered:
                    player_id = value
                    break
        if not player_id:
            return []
        raw = lg.espn_request.get_player_news(int(player_id))
    except Exception:
        return []

    items = raw if isinstance(raw, list) else raw.get("feed", raw.get("articles", [])) if isinstance(raw, dict) else []
    rows = []
    for item in items[:limit]:
        headline = item.get("headline") or item.get("title") or item.get("description", "")
        summary = item.get("description") or item.get("story") or item.get("summary", "")
        link = ""
        links = item.get("links") or {}
        if isinstance(links, dict):
            link = (links.get("web") or {}).get("href", "") if isinstance(links.get("web"), dict) else ""
        rows.append({
            "headline": headline,
            "summary": summary,
            "published": item.get("published") or item.get("lastModified") or "",
            "link": link,
        })
    return rows
