"""Exact ESPN-backed season tracker helpers."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Tuple

import pandas as pd

from backend.config import CONFIG
from backend.data import espn_client


SEASON_START = date(CONFIG.oot_season, 3, 27)
SEASON_END = date(CONFIG.oot_season, 9, 28)


def _display_date_for_period(period: int) -> date:
    return min(SEASON_END, SEASON_START + timedelta(days=max(period - 1, 0) * 7))


def _period_for_date(as_of: date, current_period: int) -> int:
    if as_of <= SEASON_START:
        return 0
    approx = ((as_of - SEASON_START).days // 7) + 1
    return max(0, min(current_period, approx))


def _team_lookup(league_snapshot: espn_client.LeagueSnapshot) -> Dict[int, espn_client.FantasyTeam]:
    return {team.team_id: team for team in league_snapshot.teams}


def _team_name(team_obj) -> str:
    return getattr(team_obj, "team_name", getattr(team_obj, "name", str(team_obj)))


def _period_box_scores(native_league, period: int):
    return native_league.box_scores(matchup_period=period, scoring_period=period)


def _total_matchup_periods(native_league, current_period: int) -> int:
    schedule_lengths = [len(getattr(team, "schedule", [])) for team in getattr(native_league, "teams", [])]
    return max([current_period, *schedule_lengths]) if schedule_lengths else current_period


def build_espn_season_tracker(as_of: date) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float], str]:
    league_snapshot = espn_client.load_league()
    native_league = espn_client.load_native_league()
    if league_snapshot.source != "espn" or native_league is None:
        raise ValueError("ESPN league data is unavailable")

    team_lookup = _team_lookup(league_snapshot)
    today = date.today()
    current_period = max(1, league_snapshot.current_matchup_period or getattr(native_league, "currentMatchupPeriod", 1))
    target_period = _period_for_date(as_of, current_period)
    include_live_period = as_of >= today and target_period == current_period
    total_periods = _total_matchup_periods(native_league, current_period)

    period_rows: List[dict] = []
    cumulative: Dict[str, float] = {team.name: 0.0 for team in league_snapshot.teams}
    period_points_by_team: Dict[str, List[float]] = {team.name: [] for team in league_snapshot.teams}
    current_period_scores: Dict[str, float] = {team.name: 0.0 for team in league_snapshot.teams}
    current_period_projected: Dict[str, float] = {team.name: 0.0 for team in league_snapshot.teams}

    for period in range(1, target_period + 1):
        box_scores = _period_box_scores(native_league, period)
        is_live_period = include_live_period and period == current_period
        for box in box_scores:
            teams = [
                (_team_name(box.home_team), float(getattr(box, "home_score", 0.0) or 0.0),
                 float(getattr(box, "home_projected", -1.0) or -1.0)),
                (_team_name(box.away_team), float(getattr(box, "away_score", 0.0) or 0.0),
                 float(getattr(box, "away_projected", -1.0) or -1.0)),
            ]
            for team_name, score, projected in teams:
                if not team_name or team_name == "0":
                    continue
                cumulative[team_name] = cumulative.get(team_name, 0.0) + score
                period_points_by_team.setdefault(team_name, []).append(score)
                period_rows.append({
                    "team": team_name,
                    "period": period,
                    "date": _display_date_for_period(period),
                    "period_points": round(score, 1),
                    "cumulative_points": round(cumulative[team_name], 1),
                    "series": "Actual",
                })
                if is_live_period:
                    current_period_scores[team_name] = score
                    current_period_projected[team_name] = projected if projected >= 0 else score

    standings_order = {
        _team_name(team): idx + 1
        for idx, team in enumerate(native_league.standings())
    }
    rows = []
    for team in league_snapshot.teams:
        team_name = team.name
        rows.append({
            "rank": standings_order.get(team_name, team.standing or len(rows) + 1),
            "team": team_name,
            "record": f"{team.wins}-{team.losses}" + (f"-{team.ties}" if team.ties else ""),
            "points_to_date": round(cumulative.get(team_name, 0.0), 1),
            "current_matchup_points": round(current_period_scores.get(team_name, 0.0), 1),
            "current_matchup_projected": round(current_period_projected.get(team_name, 0.0), 1),
        })

    table = pd.DataFrame(rows).sort_values(["rank", "points_to_date"], ascending=[True, False]).reset_index(drop=True)
    trajectory = pd.DataFrame(period_rows)
    outlook_rows: List[dict] = []
    for team in league_snapshot.teams:
        team_name = team.name
        actual_rows = trajectory[trajectory["team"] == team_name].sort_values("period")
        actual_scores = period_points_by_team.get(team_name, [])
        if actual_rows.empty:
            continue
        for _, row in actual_rows.iterrows():
            outlook_rows.append(dict(row))

        cumulative_projection = float(actual_rows.iloc[-1]["cumulative_points"])
        if include_live_period:
            current_score = current_period_scores.get(team_name, 0.0)
            current_projected = current_period_projected.get(team_name, current_score)
            cumulative_before_current = cumulative_projection - current_score
            current_projected_cumulative = cumulative_before_current + current_projected

            anchor_period = max(target_period - 1, 0)
            outlook_rows.append({
                "team": team_name,
                "period": anchor_period,
                "date": _display_date_for_period(anchor_period),
                "period_points": 0.0,
                "cumulative_points": round(cumulative_before_current, 1),
                "series": "Current matchup",
            })
            outlook_rows.append({
                "team": team_name,
                "period": target_period,
                "date": _display_date_for_period(target_period),
                "period_points": round(current_projected, 1),
                "cumulative_points": round(current_projected_cumulative, 1),
                "series": "Current matchup",
            })
            outlook_rows.append({
                "team": team_name,
                "period": target_period,
                "date": _display_date_for_period(target_period),
                "period_points": round(current_projected, 1),
                "cumulative_points": round(current_projected_cumulative, 1),
                "series": "Projected",
            })
            cumulative_projection = current_projected_cumulative

        average_points = sum(actual_scores) / len(actual_scores) if actual_scores else 0.0
        for period in range(target_period + 1, total_periods + 1):
            cumulative_projection += average_points
            outlook_rows.append({
                "team": team_name,
                "period": period,
                "date": _display_date_for_period(period),
                "period_points": round(average_points, 1),
                "cumulative_points": round(cumulative_projection, 1),
                "series": "Projected",
            })
    outlook = pd.DataFrame(outlook_rows)
    leaderboard = table.iloc[0].to_dict() if not table.empty else {}
    meta = {
        "as_of_period": float(target_period),
        "current_period": float(current_period),
        "include_live_period": 1.0 if include_live_period else 0.0,
        "total_periods": float(total_periods),
    }
    return table, trajectory, outlook, meta, leaderboard.get("team", "")
