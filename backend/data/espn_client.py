"""ESPN fantasy-baseball league wrapper with demo fallback."""
from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass, field
from typing import List

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


def _demo_snapshot() -> LeagueSnapshot:
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
        season=CONFIG.espn_season,
        scoring_type="H2H Categories (R/HR/RBI/SB/AVG + W/SV/K/ERA/WHIP)",
        teams=teams,
        free_agents=[],
        source="demo",
        scoring_profile=default_profile(),
    )


@lru_cache(maxsize=1)
def load_native_league():
    if not (_HAVE_ESPN and CONFIG.espn_league_id):
        return None
    kwargs = {"league_id": int(CONFIG.espn_league_id), "year": CONFIG.espn_season}
    if CONFIG.espn_swid and CONFIG.espn_s2:
        kwargs.update({"espn_s2": CONFIG.espn_s2, "swid": CONFIG.espn_swid})
    return League(**kwargs)


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
    if not (_HAVE_ESPN and CONFIG.espn_league_id):
        return _demo_snapshot()
    try:
        lg = load_native_league()
        if lg is None:
            return _demo_snapshot()
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
            league_id=str(CONFIG.espn_league_id),
            season=CONFIG.espn_season,
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
    except Exception:
        return _demo_snapshot()


def load_free_agent_players(size: int = 100) -> List[FantasyPlayer]:
    """Return detailed ESPN free-agent rows when the native ESPN client is available."""
    if not (_HAVE_ESPN and CONFIG.espn_league_id):
        return []
    try:
        lg = load_native_league()
        if lg is None:
            return []
        return [_to_fantasy_player(player) for player in lg.free_agents(size=size)]
    except Exception:
        return []
