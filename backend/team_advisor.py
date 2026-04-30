"""Team-level roster health and free-agent replacement helpers."""
from __future__ import annotations

from typing import Iterable, List

import pandas as pd

from backend.config import CONFIG
from backend.data import espn_client
from backend.data import mlb_stats
from backend.data import pybaseball_client as pyb
from backend.draft.player_pool import build_pool


CORE_POSITIONS = {"C", "1B", "2B", "3B", "SS", "OF", "SP", "RP", "UTIL", "DH"}
ESTIMATED_GAMES_ELAPSED = 30


def _il_label(value: str) -> str:
    text = str(value or "").strip()
    if text.upper().startswith("DL"):
        return "IL" + text[2:]
    return text


def _norm_name(name: str) -> str:
    return str(name or "").strip().casefold()


def _position_set(player: espn_client.FantasyPlayer) -> set[str]:
    positions = {player.fantasy_position, *player.eligible_positions}
    normalized = {espn_client.normalize_fantasy_position(pos) for pos in positions if pos}
    return {pos for pos in normalized if pos in CORE_POSITIONS}


def _espn_rows(players: Iterable[espn_client.FantasyPlayer], periods: int) -> pd.DataFrame:
    rows = []
    for player in players:
        avg_points = player.avg_points
        if not avg_points and player.games_played:
            avg_points = player.total_points / player.games_played
        player_id = _mlb_player_id(player.name)
        positions = sorted(_position_set(player))
        display_position = player.fantasy_position
        if player.lineup_slot in {"BE", "IL"} and positions:
            display_position = positions[0]
        rows.append({
            "photo": _headshot_url(player_id),
            "Name": player.name,
            "mlb_player_id": player_id,
            "mlb_team": player.mlb_team,
            "position": display_position or "/".join(positions),
            "lineup_slot": player.lineup_slot,
            "eligible": ", ".join(positions),
            "injury_status": _il_label(player.injury_status),
            "status": _il_label(player.status),
            "espn_total_points": round(float(player.total_points or 0.0), 1),
            "espn_avg_points": round(float(avg_points or 0.0), 2),
            "games_played": round(float(player.games_played or 0.0), 0),
            "espn_projected_points": round(float(player.projected_total_points or 0.0), 1),
            "rostership": round(float(player.rostership or 0.0), 1),
            "positions_key": "|".join(sorted(_position_set(player))),
        })
    return pd.DataFrame(rows)


def add_il_impact(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "injury_status" in out:
        out["injury_status"] = out["injury_status"].apply(_il_label)
    if "status" in out:
        out["status"] = out["status"].apply(_il_label)
    lineup = out.get("lineup_slot", pd.Series("", index=out.index)).fillna("").astype(str).str.upper()
    injury = out.get("injury_status", pd.Series("", index=out.index)).fillna("").astype(str)
    status = out.get("status", pd.Series("", index=out.index)).fillna("").astype(str)
    out["is_il"] = (lineup == "IL") | injury.str.len().gt(0) | status.str.upper().isin({"IL", "IL10", "IL15", "IL60", "D60"})

    games_played = out.get("games_played", pd.Series(0.0, index=out.index)).fillna(0).astype(float)
    avg_points = out.get("espn_avg_points", pd.Series(0.0, index=out.index)).fillna(0).astype(float)
    projected = out.get("proj_pts", pd.Series(0.0, index=out.index)).fillna(0).astype(float)

    out["estimated_games_missed"] = 0.0
    out.loc[out["is_il"], "estimated_games_missed"] = (
        ESTIMATED_GAMES_ELAPSED - games_played
    ).clip(lower=0)
    replacement_rate = projected / max(ESTIMATED_GAMES_ELAPSED, 1)
    value_rate = avg_points.where(avg_points > 0, replacement_rate)
    out["estimated_value_lost"] = (out["estimated_games_missed"] * value_rate).round(1)
    return out


def _mlb_player_id(name: str) -> int:
    try:
        return int(mlb_stats.resolve_player_identity(name).get("id", 0) or 0)
    except Exception:
        return 0


def _headshot_url(player_id: int) -> str:
    if not player_id:
        return ""
    return f"https://img.mlbstatic.com/mlb-photos/image/upload/w_96,q_auto:best/v1/people/{player_id}/headshot/67/current"


def _advanced_pool(scoring_profile) -> pd.DataFrame:
    pool = build_pool(scoring_profile)
    current = CONFIG.oot_season
    batting = pyb.batting_stats(current).copy()
    pitching = pyb.pitching_stats(current).copy()

    batting_cols = [col for col in ["Name", "wRC+", "wOBA", "ISO", "Barrel%", "HardHit%", "WAR"] if col in batting]
    pitching_cols = [col for col in ["Name", "FIP", "xFIP", "K%", "BB%", "WAR"] if col in pitching]
    batting = batting[batting_cols].copy() if batting_cols else pd.DataFrame(columns=["Name"])
    pitching = pitching[pitching_cols].copy() if pitching_cols else pd.DataFrame(columns=["Name"])
    batting["advanced_type"] = "Bat"
    pitching["advanced_type"] = "Pitch"

    advanced = pd.concat([batting, pitching], ignore_index=True, sort=False)
    merged = pool.merge(advanced, on="Name", how="left")
    if "proj_pts" not in merged:
        merged["proj_pts"] = merged.get("recent_pts", merged.get("draft_score", 0.0))
    return merged


def _advanced_label(row: pd.Series) -> str:
    role = str(row.get("role", ""))
    if role == "PIT":
        bits = []
        for col in ("FIP", "xFIP", "K%", "BB%", "WAR"):
            if col in row and pd.notna(row[col]):
                value = float(row[col])
                bits.append(f"{col} {value:.1f}")
        return " | ".join(bits[:3]) or "No advanced line"

    bits = []
    for col in ("wRC+", "wOBA", "ISO", "Barrel%", "WAR"):
        if col in row and pd.notna(row[col]):
            value = float(row[col])
            if col in {"wOBA", "ISO"}:
                bits.append(f"{col} {value:.3f}")
            else:
                bits.append(f"{col} {value:.1f}")
    return " | ".join(bits[:3]) or "No advanced line"


def _merge_player_context(espn_df: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    if espn_df.empty:
        return espn_df
    merged = espn_df.merge(pool, on="Name", how="left", suffixes=("", "_pool"))
    for col in ("Team", "fantasy_position"):
        if col not in merged:
            merged[col] = ""
    merged["proj_pts"] = merged["proj_pts"].fillna(merged["espn_projected_points"])
    merged["draft_score"] = merged["draft_score"].fillna(merged["proj_pts"])
    merged["advanced"] = merged.apply(_advanced_label, axis=1)
    return merged


def _compatibility(player_positions: str, candidate_positions: str) -> bool:
    player = set(str(player_positions or "").split("|")) - {""}
    candidate = set(str(candidate_positions or "").split("|")) - {""}
    if not player or not candidate:
        return True
    if player & candidate:
        return True
    if "UTIL" in player and candidate - {"SP", "RP"}:
        return True
    return False


def _league_percentile(row: pd.Series, population: pd.DataFrame) -> float:
    if population.empty:
        return 50.0
    compatible = population[
        population["positions_key"].apply(lambda pos: _compatibility(row.get("positions_key", ""), pos))
    ].copy()
    if compatible.empty:
        compatible = population.copy()
    avg_values = compatible.get("espn_avg_points", pd.Series(dtype=float)).fillna(0.0).astype(float)
    if int((avg_values > 0).sum()) >= 5 and float(row.get("espn_avg_points", 0.0) or 0.0) > 0:
        values = avg_values
        score = float(row.get("espn_avg_points", 0.0) or 0.0)
    else:
        values = compatible.get("proj_pts", pd.Series(dtype=float)).fillna(
            compatible.get("draft_score", pd.Series(dtype=float))
        ).fillna(0.0).astype(float)
        score = float(row.get("proj_pts", row.get("draft_score", 0.0)) or 0.0)
    if values.empty:
        return 50.0
    return round(float((values <= score).mean() * 100.0), 1)


def _health_from_percentile(percentile: float) -> str:
    if percentile < 25:
        return "Struggling"
    if percentile < 50:
        return "Watch"
    return "Stable"


def build_team_advice(team: espn_client.FantasyTeam, league: espn_client.LeagueSnapshot,
                      free_agent_size: int = 100) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    periods = max(int(league.current_scoring_period or league.current_matchup_period or 1), 1)
    pool = _advanced_pool(league.scoring_profile)
    roster_df = add_il_impact(_merge_player_context(_espn_rows(team.roster, periods), pool))
    league_rosters_df = _merge_player_context(
        _espn_rows(
            [
                player
                for league_team in league.teams
                for player in league_team.roster
            ],
            periods,
        ),
        pool,
    )

    rostered_names = {
        _norm_name(player.name)
        for league_team in league.teams
        for player in league_team.roster
        if player.name
    }

    detailed_fas = espn_client.load_free_agent_players(size=free_agent_size)
    if detailed_fas:
        fa_df = add_il_impact(_merge_player_context(_espn_rows(detailed_fas, periods), pool))
    else:
        fa_df = pool[~pool["Name"].map(_norm_name).isin(rostered_names)].copy()
        fa_df["mlb_team"] = fa_df.get("Team", "")
        fa_df["position"] = fa_df.get("fantasy_position", fa_df.get("role", ""))
        fa_df["eligible"] = fa_df["position"]
        fa_df["player_id"] = fa_df["Name"].apply(_mlb_player_id)
        fa_df["photo"] = fa_df["player_id"].apply(_headshot_url)
        fa_df["espn_total_points"] = 0.0
        fa_df["espn_avg_points"] = 0.0
        fa_df["games_played"] = 0.0
        fa_df["espn_projected_points"] = fa_df.get("proj_pts", fa_df.get("draft_score", 0.0))
        fa_df["rostership"] = 0.0
        fa_df["positions_key"] = fa_df["position"]
        fa_df["advanced"] = fa_df.apply(_advanced_label, axis=1)

    if roster_df.empty:
        return roster_df, fa_df.head(25), pd.DataFrame()

    fa_df["add_score"] = (
        fa_df.get("proj_pts", 0.0).fillna(0.0).astype(float)
        + fa_df.get("espn_avg_points", 0.0).fillna(0.0).astype(float) * 8.0
        + fa_df.get("WAR", 0.0).fillna(0.0).astype(float) * 3.0
    )
    fa_df = fa_df.sort_values("add_score", ascending=False).reset_index(drop=True)
    league_population = pd.concat([league_rosters_df, fa_df], ignore_index=True, sort=False)
    roster_df["league_percentile"] = roster_df.apply(
        lambda row: _league_percentile(row, league_population),
        axis=1,
    )
    roster_df["health"] = roster_df["league_percentile"].apply(_health_from_percentile)

    struggling = roster_df[roster_df["health"].isin(["Struggling", "Watch"])].sort_values(
        ["league_percentile", "espn_avg_points", "proj_pts"], ascending=[True, True, True]
    )
    suggestions: List[dict] = []
    for _, player in struggling.head(8).iterrows():
        compatible = fa_df[
            fa_df["positions_key"].apply(lambda pos: _compatibility(player.get("positions_key", ""), pos))
        ].head(5)
        for _, candidate in compatible.iterrows():
            projected_gain = float(candidate.get("proj_pts", 0.0) or 0.0) - float(player.get("proj_pts", 0.0) or 0.0)
            avg_gain = float(candidate.get("espn_avg_points", 0.0) or 0.0) - float(player.get("espn_avg_points", 0.0) or 0.0)
            suggestions.append({
                "Drop candidate": player["Name"],
                "Add candidate": candidate["Name"],
                "Pos": candidate.get("position", ""),
                "Projected gain": round(projected_gain, 1),
                "Avg pts gain": round(avg_gain, 2),
                "Rostership": round(float(candidate.get("rostership", 0.0) or 0.0), 1),
                "Why": candidate.get("advanced", "No advanced line"),
            })

    suggestions_df = pd.DataFrame(suggestions)
    if not suggestions_df.empty:
        suggestions_df = suggestions_df.sort_values(
            ["Projected gain", "Avg pts gain"], ascending=[False, False]
        ).head(12)
    return roster_df, fa_df.head(25), suggestions_df
