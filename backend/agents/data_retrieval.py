"""Agent 2 — Data Retrieval.

Given a structured request from the Orchestrator, gather:
  - historical player data from 2000 through the target season
  - a draft pool built from the most recent seasons ending at the target year
  - per-player Statcast / FanGraphs rows
  - top prospects
  - ESPN league snapshot (rosters, free agents, scoring)
  - ML-projected fantasy points for the target season

Returns a dict passed downstream to the Analysis agent.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from backend.config import CONFIG
from backend.data import pybaseball_client as pyb
from backend.data import espn_client
from backend.draft.player_pool import build_pool
from backend.models import draft_optimizer


def _norm_name(name: str) -> str:
    return str(name or "").strip().casefold()


def _espn_role(position: str, eligible: list[str]) -> str:
    positions = {str(position or "").upper(), *(str(pos or "").upper() for pos in eligible)}
    return "PIT" if positions & {"SP", "RP", "P"} else "BAT"


def _espn_player_pool(league: espn_client.LeagueSnapshot, free_agent_size: int = 500) -> pd.DataFrame:
    """Build a real fantasy ranking pool from ESPN when advanced stats are unavailable."""
    players = []
    seen: set[str] = set()
    for team in getattr(league, "teams", []):
        for player in getattr(team, "roster", []):
            key = _norm_name(player.name)
            if key and key not in seen:
                seen.add(key)
                players.append((player, team.name, True))
    for player in espn_client.load_free_agent_players(size=free_agent_size):
        key = _norm_name(player.name)
        if key and key not in seen:
            seen.add(key)
            players.append((player, "", False))

    rows = []
    for player, fantasy_team, rostered in players:
        avg_points = float(player.avg_points or 0.0)
        total_points = float(player.total_points or 0.0)
        games = float(player.games_played or 0.0)
        projected = float(player.projected_total_points or 0.0)
        if not avg_points and games:
            avg_points = total_points / games
        if not projected:
            projected = max(total_points, avg_points * max(games, 1.0))
        position = espn_client.normalize_fantasy_position(player.fantasy_position)
        eligible = [espn_client.normalize_fantasy_position(pos) for pos in player.eligible_positions]
        rows.append({
            "Name": player.name,
            "Team": player.mlb_team,
            "role": _espn_role(position, eligible),
            "fantasy_position": position,
            "positions_key": "|".join(sorted(set([position, *eligible]) - {""})),
            "proj_pts": projected,
            "draft_score": projected,
            "recent_pts": total_points,
            "avg_pts": avg_points,
            "espn_total_points": total_points,
            "espn_avg_points": avg_points,
            "games_played": games,
            "espn_projected_points": projected,
            "rostership": float(player.rostership or 0.0),
            "espn_injury_status": player.injury_status,
            "espn_status": player.status,
            "espn_lineup_slot": player.lineup_slot,
            "fantasy_team": fantasy_team,
            "is_rostered": rostered,
        })
    if not rows:
        return pd.DataFrame()

    pool = pd.DataFrame(rows)
    penalties = pool.apply(
        lambda row: _health_penalty(
            row.get("espn_injury_status", ""),
            row.get("espn_status", ""),
            row.get("espn_lineup_slot", ""),
        ),
        axis=1,
    )
    pool["health_status"] = [label for label, _ in penalties]
    pool["health_multiplier"] = [multiplier for _, multiplier in penalties]
    pool["health_adjusted_proj_pts"] = pool["proj_pts"].fillna(0).astype(float) * pool["health_multiplier"]
    pool = pool.sort_values("health_adjusted_proj_pts", ascending=False).reset_index(drop=True)
    pool["rank"] = pool.index + 1
    pool["proj_rank"] = pool["rank"]
    pool["tier"] = pd.cut(
        pool["rank"],
        bins=[0, 12, 30, 60, 100, 150, 10_000],
        labels=[1, 2, 3, 4, 5, 6],
    ).astype(int)
    return pool


class DataRetrieval:
    def fetch(self, data_request: Dict[str, Any]) -> Dict[str, Any]:
        league = espn_client.load_league()
        data_source = "pybaseball" if pyb.have_real_data() else (
            "synthetic-demo" if CONFIG.allow_synthetic_data else "real-data-unavailable"
        )
        bundle: Dict[str, Any] = {
            "seasons_scope": list(CONFIG.allowed_seasons),
            "data_source": data_source,
            "league": league,
            "scoring_profile": league.scoring_profile,
            "user_text": data_request.get("user_text", ""),
        }

        if data_request.get("needs_player_pool", True):
            pool = build_pool(league.scoring_profile)
            if pool.empty:
                espn_pool = _espn_player_pool(league)
                bundle["player_pool"] = espn_pool
                bundle["prospects"] = pyb.prospects(CONFIG.oot_season)
                if espn_pool.empty:
                    bundle["data_error"] = (
                        "Real advanced-stat data is unavailable, and ESPN did not return enough "
                        "roster/free-agent data to build a ranking pool."
                    )
                else:
                    bundle["data_source"] = "espn-fantasy"
                    bundle["data_warning"] = (
                        "Advanced-stat rankings were unavailable, so this answer uses real ESPN "
                        "fantasy totals, averages, projections, roster status, and injury status."
                    )
            else:
                # Fold in ML projections for the OOT season (used for draft).
                if league.scoring_profile.uses_points:
                    pool["proj_pts"] = 0.65 * pool["recent_pts"] + 0.35 * pool["avg_pts"]
                    pool = pool.sort_values("proj_pts", ascending=False).reset_index(drop=True)
                    pool["proj_rank"] = pool.index + 1
                else:
                    try:
                        proj = draft_optimizer.score_players_for_season(CONFIG.oot_season)
                        pool = pool.merge(
                            proj[["Name", "role", "proj_pts", "proj_rank"]],
                            on=["Name", "role"], how="left",
                        )
                    except Exception as exc:
                        pool["proj_pts"] = pool["recent_pts"]
                        pool["proj_rank"] = pool["rank"]
                        bundle["projection_error"] = str(exc)
                pool = _add_espn_health_context(pool, league)
                bundle["player_pool"] = pool
                bundle["prospects"] = pyb.prospects(CONFIG.oot_season)

        if data_request.get("needs_recent_form"):
            most_recent = CONFIG.oot_season
            bundle["recent_batting"] = pyb.batting_stats(most_recent)
            bundle["recent_pitching"] = pyb.pitching_stats(most_recent)

        return bundle


def _health_penalty(injury_status: str, status: str, lineup_slot: str = "") -> tuple[str, float]:
    raw = " ".join(str(value or "") for value in (injury_status, status, lineup_slot)).upper()
    if any(token in raw for token in ("IL60", "D60", "60-DAY", "60 DAY", "SIXTY_DAY_DL")):
        return "IL60", 0.35
    if any(token in raw for token in ("IL15", "D15", "15-DAY", "15 DAY", "FIFTEEN_DAY_DL")):
        return "IL15", 0.45
    if any(token in raw for token in ("IL10", "D10", "10-DAY", "10 DAY", "TEN_DAY_DL")):
        return "IL10", 0.50
    if any(token in raw for token in (" IL", "IL ", "INJURED", "OUT", "DL", "D7")):
        return "Injured/IL", 0.50
    if any(token in raw for token in ("DTD", "DAY-TO-DAY", "DAY TO DAY", "QUESTIONABLE")):
        return "Day-to-day", 0.80
    if any(token in raw for token in ("SUSP", "BEREAVEMENT", "PATERNITY", "NA")):
        return "Unavailable", 0.65
    return "Active", 1.00


def _add_espn_health_context(pool: pd.DataFrame, league) -> pd.DataFrame:
    out = pool.copy()
    out["espn_injury_status"] = ""
    out["espn_status"] = ""
    out["espn_lineup_slot"] = ""

    health_by_name: dict[str, dict[str, str]] = {}
    for team in getattr(league, "teams", []):
        for player in getattr(team, "roster", []):
            if not getattr(player, "name", ""):
                continue
            health_by_name[player.name.strip().casefold()] = {
                "injury": player.injury_status,
                "status": player.status,
                "slot": player.lineup_slot,
            }
    try:
        for player in espn_client.load_free_agent_players(size=500):
            if not getattr(player, "name", ""):
                continue
            health_by_name.setdefault(
                player.name.strip().casefold(),
                {
                    "injury": player.injury_status,
                    "status": player.status,
                    "slot": player.lineup_slot,
                },
            )
    except Exception:
        pass

    if health_by_name:
        keys = out["Name"].astype(str).str.strip().str.casefold()
        out["espn_injury_status"] = keys.map(lambda key: health_by_name.get(key, {}).get("injury", ""))
        out["espn_status"] = keys.map(lambda key: health_by_name.get(key, {}).get("status", ""))
        out["espn_lineup_slot"] = keys.map(lambda key: health_by_name.get(key, {}).get("slot", ""))

    public_injuries = espn_client.public_mlb_injury_map()
    if public_injuries:
        keys = out["Name"].astype(str).str.strip().str.casefold()
        public_status = keys.map(lambda key: public_injuries.get(key, {}).get("fantasy_status", ""))
        public_detail = keys.map(lambda key: public_injuries.get(key, {}).get("detail", ""))
        public_summary = keys.map(lambda key: public_injuries.get(key, {}).get("summary", ""))
        out["public_injury_status"] = public_status.fillna("")
        out["public_injury_detail"] = public_detail.fillna("")
        out["public_injury_summary"] = public_summary.fillna("")
        out["espn_injury_status"] = out["espn_injury_status"].where(
            out["espn_injury_status"].fillna("").astype(str).str.len().gt(0),
            out["public_injury_status"],
        )
    else:
        out["public_injury_status"] = ""
        out["public_injury_detail"] = ""
        out["public_injury_summary"] = ""

    penalties = out.apply(
        lambda row: _health_penalty(
            row.get("espn_injury_status", ""),
            row.get("espn_status", ""),
            row.get("espn_lineup_slot", ""),
        ),
        axis=1,
    )
    out["health_status"] = [label for label, _ in penalties]
    out["health_multiplier"] = [multiplier for _, multiplier in penalties]
    out["health_penalty"] = (1.0 - out["health_multiplier"]).round(2)
    for source_col, adjusted_col in (
        ("proj_pts", "health_adjusted_proj_pts"),
        ("draft_score", "health_adjusted_draft_score"),
    ):
        if source_col in out:
            out[adjusted_col] = out[source_col].fillna(0).astype(float) * out["health_multiplier"]
    if "proj_pts" in out:
        out = out.sort_values("health_adjusted_proj_pts", ascending=False).reset_index(drop=True)
        out["proj_rank"] = out.index + 1
    return out
