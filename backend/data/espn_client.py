"""ESPN fantasy-baseball league wrapper (espn-api) with demo fallback."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from backend.config import CONFIG

try:
    from espn_api.baseball import League  # type: ignore
    _HAVE_ESPN = True
except Exception:
    League = None  # type: ignore
    _HAVE_ESPN = False


@dataclass
class FantasyTeam:
    team_id: int
    name: str
    owner: str
    roster: List[str] = field(default_factory=list)


@dataclass
class LeagueSnapshot:
    league_id: str
    season: int
    scoring_type: str
    teams: List[FantasyTeam]
    free_agents: List[str]
    source: str  # "espn" | "demo"


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
    )


def load_league() -> LeagueSnapshot:
    if not (_HAVE_ESPN and CONFIG.espn_league_id):
        return _demo_snapshot()
    try:
        kwargs = {"league_id": int(CONFIG.espn_league_id), "year": CONFIG.espn_season}
        if CONFIG.espn_swid and CONFIG.espn_s2:
            kwargs.update({"espn_s2": CONFIG.espn_s2, "swid": CONFIG.espn_swid})
        lg = League(**kwargs)
        teams = [
            FantasyTeam(
                team_id=t.team_id,
                name=t.team_name,
                owner=(t.owners[0].get("displayName") if t.owners else "?"),
                roster=[p.name for p in getattr(t, "roster", [])],
            )
            for t in lg.teams
        ]
        fas = []
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
        )
    except Exception:
        return _demo_snapshot()
