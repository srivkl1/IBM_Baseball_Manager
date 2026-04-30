"""Mutual-benefit trade recommendation helpers."""
from __future__ import annotations

from typing import Dict

import pandas as pd

from backend.data import espn_client
from backend.team_advisor import (
    _advanced_pool,
    _compatibility,
    _espn_rows,
    _health_from_percentile,
    _league_percentile,
    _merge_player_context,
)


def _roster_frame(team: espn_client.FantasyTeam, league: espn_client.LeagueSnapshot,
                  pool: pd.DataFrame) -> pd.DataFrame:
    periods = max(int(league.current_scoring_period or league.current_matchup_period or 1), 1)
    roster = _merge_player_context(_espn_rows(team.roster, periods), pool)
    if roster.empty:
        return roster

    population = _merge_player_context(
        _espn_rows(
            [player for league_team in league.teams for player in league_team.roster],
            periods,
        ),
        pool,
    )
    roster["league_percentile"] = roster.apply(
        lambda row: _league_percentile(row, population),
        axis=1,
    )
    roster["health"] = roster["league_percentile"].apply(_health_from_percentile)
    roster["trade_value"] = (
        roster.get("proj_pts", 0.0).fillna(0.0).astype(float)
        + roster.get("espn_avg_points", 0.0).fillna(0.0).astype(float) * 8.0
        + roster.get("WAR", 0.0).fillna(0.0).astype(float) * 3.0
    )
    return roster.sort_values("trade_value", ascending=False).reset_index(drop=True)


def _positions(positions_key: object) -> set[str]:
    return set(str(positions_key or "").split("|")) - {""}


def _position_strength(roster: pd.DataFrame) -> Dict[str, float]:
    buckets: Dict[str, list[float]] = {}
    for _, row in roster.iterrows():
        percentile = float(row.get("league_percentile", 50.0) or 50.0)
        for position in _positions(row.get("positions_key", "")):
            buckets.setdefault(position, []).append(percentile)
    return {
        position: sum(values) / len(values)
        for position, values in buckets.items()
        if values
    }


def _need_for_player(roster_strength: Dict[str, float], player: pd.Series) -> float:
    positions = _positions(player.get("positions_key", ""))
    if not positions:
        return 50.0
    strengths = [roster_strength.get(position, 50.0) for position in positions]
    return max(0.0, 100.0 - min(strengths))


def _player_label(row: pd.Series) -> str:
    pos = row.get("position", "")
    team = row.get("mlb_team", "")
    return f"{row['Name']} ({pos}, {team})"


def _benefit(receiver_strength: Dict[str, float], incoming: pd.Series, outgoing: pd.Series) -> float:
    incoming_need = _need_for_player(receiver_strength, incoming)
    outgoing_need = _need_for_player(receiver_strength, outgoing)
    incoming_quality = float(incoming.get("league_percentile", 50.0) or 50.0) / 100.0
    outgoing_quality = float(outgoing.get("league_percentile", 50.0) or 50.0) / 100.0
    return round((incoming_need * incoming_quality) - (outgoing_need * outgoing_quality * 0.55), 1)


def _trade_rows(your_roster: pd.DataFrame, target_roster: pd.DataFrame,
                max_value_gap_pct: float) -> pd.DataFrame:
    your_strength = _position_strength(your_roster)
    target_strength = _position_strength(target_roster)
    rows = []
    for _, you_give in your_roster.iterrows():
        for _, you_get in target_roster.iterrows():
            your_value = float(you_give.get("trade_value", 0.0) or 0.0)
            target_value = float(you_get.get("trade_value", 0.0) or 0.0)
            max_value = max(abs(your_value), abs(target_value), 1.0)
            value_gap_pct = abs(your_value - target_value) / max_value * 100.0
            if value_gap_pct > max_value_gap_pct:
                continue

            your_benefit = _benefit(your_strength, you_get, you_give)
            target_benefit = _benefit(target_strength, you_give, you_get)
            if your_benefit <= 0 or target_benefit <= 0:
                continue

            balance_gap = abs(your_benefit - target_benefit)
            mutual_score = your_benefit + target_benefit - balance_gap - value_gap_pct * 0.25
            rows.append({
                "You receive": _player_label(you_get),
                "You send": _player_label(you_give),
                "Your benefit": your_benefit,
                "Target benefit": target_benefit,
                "Value gap %": round(value_gap_pct, 1),
                "Fairness": round(100.0 - value_gap_pct - balance_gap, 1),
                "Why it works": (
                    f"You address {you_get.get('eligible', you_get.get('position', 'a need'))}; "
                    f"they address {you_give.get('eligible', you_give.get('position', 'a need'))}."
                ),
                "mutual_score": round(mutual_score, 1),
            })
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["mutual_score", "Fairness"], ascending=[False, False])
        .head(12)
        .drop(columns=["mutual_score"])
        .reset_index(drop=True)
    )


def analyze_trades(league: espn_client.LeagueSnapshot, your_team: espn_client.FantasyTeam,
                   target_team: espn_client.FantasyTeam,
                   max_value_gap_pct: float = 20.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool = _advanced_pool(league.scoring_profile)
    your_roster = _roster_frame(your_team, league, pool)
    target_roster = _roster_frame(target_team, league, pool)
    if your_roster.empty or target_roster.empty:
        return your_roster, target_roster, pd.DataFrame()
    trades = _trade_rows(your_roster, target_roster, max_value_gap_pct)
    return your_roster, target_roster, trades
