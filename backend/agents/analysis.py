"""Agent 3 - Analysis.

Produces a structured recommendation from the data bundle. For draft picks it
consults the draft simulator to propose the top candidates for the team on
the clock, weighing ML projected points plus roster needs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.draft import simulator as sim
from backend.trade_analyzer import analyze_trades


@dataclass
class Recommendation:
    intent: str
    headline: str
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    rationale_bullets: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class Analysis:
    def analyze(self, intent: str, bundle: Dict[str, Any],
                draft_state: Optional[sim.DraftState] = None) -> Recommendation:
        if intent == "draft_pick":
            return self._analyze_draft(bundle, draft_state)
        if intent == "roster_lookup":
            return self._analyze_roster_lookup(bundle, draft_state)
        if intent == "roster_move":
            return self._analyze_roster_move(bundle)
        if intent == "waiver_scan":
            return self._analyze_waiver_scan(bundle)
        if intent == "team_diagnosis":
            return self._analyze_team_diagnosis(bundle)
        if intent == "trade_analysis":
            return self._analyze_trade_analysis(bundle)
        if intent == "lineup_optimization":
            return self._analyze_lineup_optimization(bundle)
        if intent == "risk_check":
            return self._analyze_risk_check(bundle)
        if intent == "player_trend":
            return self._analyze_trend(bundle)
        if intent == "standings_check":
            return self._analyze_standings(bundle)
        return Recommendation(intent=intent, headline="No analysis path for this intent.")

    @staticmethod
    def _available_pool(pool: pd.DataFrame, bundle: Dict[str, Any],
                        draft_state: Optional[sim.DraftState] = None) -> pd.DataFrame:
        if draft_state is not None:
            return draft_state.board[draft_state.board["available"]].copy()

        filtered = pool.copy()
        league = bundle.get("league")
        if league is None:
            return filtered

        rostered = {
            player.name.strip().casefold()
            for team in league.teams
            for player in getattr(team, "roster", [])
            if getattr(player, "name", "")
        }
        if rostered:
            filtered = filtered[~filtered["Name"].str.strip().str.casefold().isin(rostered)]

        free_agents = {
            player.strip().casefold()
            for player in getattr(league, "free_agents", [])
            if player
        }
        if free_agents:
            fa_only = filtered[filtered["Name"].str.strip().str.casefold().isin(free_agents)]
            if len(fa_only) >= 5:
                return fa_only
        return filtered

    def _analyze_draft(self, bundle: Dict[str, Any],
                       draft_state: Optional[sim.DraftState]) -> Recommendation:
        pool = self._available_pool(bundle["player_pool"], bundle, draft_state)
        if draft_state is None:
            top = pool.head(10)
            cands = [{
                "name": r["Name"], "role": r["role"], "position": r.get("fantasy_position", r["role"]), "team": r.get("Team", ""),
                "proj_pts": round(float(r.get("proj_pts", r.get("draft_score", 0))), 1),
                "rank": int(r["rank"]),
                "tier": int(r["tier"]),
            } for _, r in top.iterrows()]
            bullets = [
                f"{c['name']} ({c['position']}) - projected {c['proj_pts']} pts, tier {c['tier']}"
                for c in cands[:3]
            ]
            return Recommendation(
                intent="draft_pick",
                headline="Top available players by recent-form composite plus historical ML projection",
                candidates=cands,
                rationale_bullets=bullets,
                metrics={"pool_size": int(len(pool))},
            )

        rec_team = (draft_state.teams[draft_state.human_index]
                    if draft_state.draft_locked else None)
        recs = sim.recommend_pick(draft_state, team=rec_team, top_n=5)
        cands = [{
            "name": r["Name"], "role": r["role"], "position": r.get("fantasy_position", r["role"]), "team": r.get("Team", ""),
            "proj_pts": round(float(r.get("proj_pts", r.get("draft_score", 0))), 1),
            "suitability": round(float(r["suitability"]), 1),
            "tier": int(r["tier"]),
        } for _, r in recs.iterrows()]

        round_num, slot = draft_state.round_and_slot()
        if draft_state.draft_locked:
            headline = (f"Best available fit for {draft_state.teams[draft_state.human_index]}: "
                        f"{cands[0]['name']} ({cands[0]['position']})" if cands
                        else "No players left on the board")
        else:
            headline = (f"Round {round_num}, pick {slot} - top option: "
                        f"{cands[0]['name']} ({cands[0]['position']})" if cands
                        else "No players left on the board")

        bullets = []
        for c in cands[:3]:
            bullets.append(
                f"{c['name']} ({c['position']}, {c['team']}) - projected "
                f"{c['proj_pts']} pts, tier {c['tier']}, fit {c['suitability']}"
            )
        return Recommendation(
            intent="draft_pick",
            headline=headline,
            candidates=cands,
            rationale_bullets=bullets,
            metrics={"pick_number": draft_state.current_pick_number,
                     "team_on_clock": rec_team or draft_state.team_on_clock()},
        )

    def _analyze_roster_move(self, bundle: Dict[str, Any]) -> Recommendation:
        pool: pd.DataFrame = bundle.get("player_pool")
        if pool is None:
            return Recommendation(intent="roster_move",
                                  headline="Need a player pool to evaluate roster moves.")
        adds = self._available_pool(pool, bundle).head(10)
        cands = [{"name": r["Name"], "role": r["role"], "position": r.get("fantasy_position", r["role"]),
                  "proj_pts": round(float(r.get("proj_pts", 0)), 1),
                  "tier": int(r["tier"])}
                 for _, r in adds.iterrows()]
        return Recommendation(
            intent="roster_move",
            headline="Top available adds based on unrostered players",
            candidates=cands,
            rationale_bullets=[f"Consider adding {c['name']} ({c['position']}) - projected {c['proj_pts']} pts"
                               for c in cands[:3]],
        )

    @staticmethod
    def _my_team(league):
        if league is None or not getattr(league, "teams", None):
            return None
        for team in league.teams:
            if (team.owner or "").strip().lower() in {"you", "me", "my team"}:
                return team
        return league.teams[0]

    @staticmethod
    def _player_rows(team, pool: pd.DataFrame) -> pd.DataFrame:
        if team is None:
            return pd.DataFrame()
        rows = []
        for player in getattr(team, "roster", []):
            rows.append({
                "Name": player.name,
                "position": player.fantasy_position or "/".join(player.eligible_positions),
                "lineup_slot": player.lineup_slot,
                "eligible": ", ".join(player.eligible_positions),
                "team": player.mlb_team,
                "espn_avg": player.avg_points,
                "espn_pts": player.total_points,
                "games": player.games_played,
                "injury": player.injury_status,
                "status": player.status,
            })
        roster = pd.DataFrame(rows)
        if roster.empty:
            return roster
        merged = roster.merge(pool, on="Name", how="left")
        if "proj_pts" not in merged:
            merged["proj_pts"] = merged.get("recent_pts", merged["espn_pts"])
        merged["proj_pts"] = merged["proj_pts"].fillna(merged.get("recent_pts", merged["espn_pts"]))
        war = merged["WAR"] if "WAR" in merged else pd.Series(0.0, index=merged.index)
        merged["value"] = (
            merged["proj_pts"].fillna(0).astype(float)
            + merged["espn_avg"].fillna(0).astype(float) * 8.0
            + war.fillna(0).astype(float) * 3.0
        )
        return merged.sort_values("value", ascending=False).reset_index(drop=True)

    def _analyze_waiver_scan(self, bundle: Dict[str, Any]) -> Recommendation:
        pool: pd.DataFrame = bundle.get("player_pool")
        if pool is None:
            return Recommendation("waiver_scan", "Need a player pool before scanning waivers.")
        adds = self._available_pool(pool, bundle).copy().head(12)
        cands = []
        for _, r in adds.iterrows():
            confidence = "high" if float(r.get("proj_pts", 0) or 0) >= float(adds["proj_pts"].median() if "proj_pts" in adds else 0) else "medium"
            cands.append({
                "name": r["Name"],
                "position": r.get("fantasy_position", r.get("role", "")),
                "team": r.get("Team", ""),
                "proj_pts": round(float(r.get("proj_pts", r.get("draft_score", 0)) or 0), 1),
                "tier": int(r.get("tier", 6)),
                "confidence": confidence,
            })
        return Recommendation(
            intent="waiver_scan",
            headline="Best available waiver targets by projection and roster availability",
            candidates=cands,
            rationale_bullets=[
                f"{c['name']} ({c['position']}) projects for {c['proj_pts']} pts; confidence {c['confidence']}"
                for c in cands[:5]
            ],
            metrics={"available_pool": int(len(adds)), "confidence_basis": "projection tier plus ESPN availability"},
        )

    def _analyze_team_diagnosis(self, bundle: Dict[str, Any]) -> Recommendation:
        league = bundle.get("league")
        pool: pd.DataFrame = bundle.get("player_pool")
        team = self._my_team(league)
        roster = self._player_rows(team, pool) if pool is not None else pd.DataFrame()
        if roster.empty:
            return Recommendation("team_diagnosis", "No roster is loaded for diagnosis.")

        grouped = (
            roster.groupby("position", dropna=False)
            .agg(players=("Name", "count"), avg_value=("value", "mean"), top_player=("Name", "first"))
            .reset_index()
            .sort_values("avg_value", ascending=False)
        )
        strengths = grouped.head(3)
        weaknesses = grouped.tail(3).sort_values("avg_value")
        cands = grouped.to_dict(orient="records")
        bullets = [
            f"Strength: {row['position']} led by {row['top_player']} with average value {row['avg_value']:.1f}"
            for _, row in strengths.iterrows()
        ] + [
            f"Need: {row['position']} has lower roster value; consider waiver or trade upgrades"
            for _, row in weaknesses.iterrows()
        ]
        return Recommendation(
            "team_diagnosis",
            f"{team.name} is strongest at {strengths.iloc[0]['position']} and thinnest at {weaknesses.iloc[0]['position']}",
            candidates=cands,
            rationale_bullets=bullets,
            metrics={"team": team.name, "roster_size": int(len(roster)), "confidence": "medium"},
        )

    def _analyze_trade_analysis(self, bundle: Dict[str, Any]) -> Recommendation:
        league = bundle.get("league")
        my_team = self._my_team(league)
        if league is None or my_team is None or len(getattr(league, "teams", [])) < 2:
            return Recommendation("trade_analysis", "Need at least two league teams to analyze trades.")

        rows = []
        for target in league.teams:
            if target.team_id == my_team.team_id:
                continue
            _, _, trades = analyze_trades(league, my_team, target, max_value_gap_pct=25.0)
            if trades.empty:
                continue
            best = trades.iloc[0].to_dict()
            best["Target team"] = target.name
            rows.append(best)
        if not rows:
            return Recommendation(
                "trade_analysis",
                "No balanced one-for-one trade ideas cleared the fairness filter.",
                rationale_bullets=["Try loosening value gap, considering two-for-one offers, or waiting for team needs to shift."],
                metrics={"confidence": "low"},
            )
        trade_df = pd.DataFrame(rows).sort_values(["Fairness", "Your benefit"], ascending=[False, False]).head(5)
        cands = trade_df.to_dict(orient="records")
        return Recommendation(
            "trade_analysis",
            f"Best trade partner fit: {cands[0]['Target team']}",
            candidates=cands,
            rationale_bullets=[
                f"Offer {c['You send']} for {c['You receive']} to {c['Target team']} - fairness {c['Fairness']}"
                for c in cands[:3]
            ],
            metrics={"confidence": "medium", "trade_type": "one-for-one balanced swaps"},
        )

    def _analyze_lineup_optimization(self, bundle: Dict[str, Any]) -> Recommendation:
        league = bundle.get("league")
        pool: pd.DataFrame = bundle.get("player_pool")
        team = self._my_team(league)
        roster = self._player_rows(team, pool) if pool is not None else pd.DataFrame()
        if roster.empty:
            return Recommendation("lineup_optimization", "No roster is loaded for lineup optimization.")
        starters = roster.head(10)
        bench_watch = roster.tail(min(5, len(roster))).sort_values("value")
        cands = [
            {
                "name": row["Name"],
                "position": row.get("position", ""),
                "team": row.get("team", ""),
                "value": round(float(row.get("value", 0)), 1),
                "recommendation": "Start/core" if idx in starters.index else "Bench/watch",
            }
            for idx, row in pd.concat([starters, bench_watch]).drop_duplicates("Name").iterrows()
        ]
        return Recommendation(
            "lineup_optimization",
            f"Start your highest-value core; watch {bench_watch.iloc[0]['Name']} as the lowest-value roster spot",
            candidates=cands,
            rationale_bullets=[
                f"Core play: {row['Name']} ({row.get('position', '')}) value {row.get('value', 0):.1f}"
                for _, row in starters.head(3).iterrows()
            ] + [
                f"Bench/watch: {row['Name']} value {row.get('value', 0):.1f}"
                for _, row in bench_watch.head(3).iterrows()
            ],
            metrics={"confidence": "medium", "method": "projection plus ESPN average plus WAR"},
        )

    def _analyze_risk_check(self, bundle: Dict[str, Any]) -> Recommendation:
        league = bundle.get("league")
        pool: pd.DataFrame = bundle.get("player_pool")
        team = self._my_team(league)
        roster = self._player_rows(team, pool) if pool is not None else pd.DataFrame()
        if roster.empty:
            return Recommendation("risk_check", "No roster is loaded for risk review.")

        def risk(row):
            score = 0
            reasons = []
            if str(row.get("injury", "")).strip():
                score += 35
                reasons.append(f"injury status {row.get('injury')}")
            if float(row.get("games", 0) or 0) < 5:
                score += 25
                reasons.append("limited games played")
            if float(row.get("espn_avg", 0) or 0) <= 0 and float(row.get("espn_pts", 0) or 0) <= 0:
                score += 20
                reasons.append("no ESPN production yet")
            median_projection = roster["proj_pts"].median()
            median_projection = 0.0 if pd.isna(median_projection) else float(median_projection)
            if float(row.get("proj_pts", 0) or 0) < median_projection:
                score += 15
                reasons.append("below team median projection")
            return pd.Series({"risk_score": min(score, 100), "risk_reasons": "; ".join(reasons) or "normal profile"})

        risk_cols = roster.apply(risk, axis=1)
        risk_df = pd.concat([roster, risk_cols], axis=1).sort_values("risk_score", ascending=False)
        cands = [
            {
                "name": row["Name"],
                "position": row.get("position", ""),
                "risk_score": int(row["risk_score"]),
                "reason": row["risk_reasons"],
                "recommendation": "Shop/replace" if row["risk_score"] >= 50 else "Monitor",
            }
            for _, row in risk_df.head(8).iterrows()
        ]
        return Recommendation(
            "risk_check",
            f"Highest risk roster spot: {cands[0]['name']} ({cands[0]['risk_score']}/100)",
            candidates=cands,
            rationale_bullets=[
                f"{c['name']}: risk {c['risk_score']}/100 - {c['reason']}"
                for c in cands[:5]
            ],
            metrics={"confidence": "medium", "risk_inputs": "injury, games played, ESPN production, projection"},
        )

    def _analyze_roster_lookup(self, bundle: Dict[str, Any],
                               draft_state: Optional[sim.DraftState]) -> Recommendation:
        if draft_state is None:
            return Recommendation(
                intent="roster_lookup",
                headline="I do not have a league roster loaded yet.",
                rationale_bullets=["Connect an ESPN league or load a draft state first."],
            )

        team_name = draft_state.teams[draft_state.human_index]
        roster = draft_state.rosters.get(team_name, [])
        if not roster:
            return Recommendation(
                intent="roster_lookup",
                headline=f"{team_name} does not have any players loaded yet.",
                rationale_bullets=["Once the roster is available, I can list everyone on your team."],
            )

        candidates = [{
            "player": player["player"],
            "role": player.get("role", ""),
            "position": player.get("fantasy_position", player.get("role", "")),
            "team": player.get("mlb_team", ""),
            "proj_pts": round(float(player.get("proj_pts", 0)), 1),
        } for player in roster]
        bullets = [
            f"{player['player']} ({player.get('fantasy_position', player.get('role', ''))})"
            for player in roster[:8]
        ]
        return Recommendation(
            intent="roster_lookup",
            headline=f"{team_name} roster: {len(roster)} players loaded",
            candidates=candidates,
            rationale_bullets=bullets,
            metrics={"team": team_name, "roster_size": len(roster)},
        )

    def _analyze_trend(self, bundle: Dict[str, Any]) -> Recommendation:
        recent_bat = bundle.get("recent_batting")
        if recent_bat is None or len(recent_bat) == 0:
            return Recommendation(intent="player_trend",
                                  headline="No recent data available.")
        hot = recent_bat.sort_values("wRC+", ascending=False).head(5)
        cands = [{"name": r["Name"], "wRC+": round(float(r["wRC+"]), 0),
                  "HR": int(r["HR"]), "AVG": round(float(r["AVG"]), 3)}
                 for _, r in hot.iterrows()]
        return Recommendation(
            intent="player_trend",
            headline="Hottest bats by wRC+ this season",
            candidates=cands,
            rationale_bullets=[f"{c['name']}: wRC+ {c['wRC+']} with {c['HR']} HR "
                               f"and a .{int(c['AVG']*1000):03d} AVG"
                               for c in cands[:3]],
        )

    def _analyze_standings(self, bundle: Dict[str, Any]) -> Recommendation:
        table = bundle.get("standings_table")
        if table is None:
            return Recommendation(intent="standings_check",
                                  headline="No draft has been locked in yet.")
        leader = table.iloc[0]
        record = leader.get("record")
        points = leader.get("points_to_date", leader.get("current_matchup_points", 0))
        headline = f"{leader['team']} leads"
        if record:
            headline += f" at {record}"
        headline += f" with {points} pts"
        bullets = []
        if "season_pct" in leader:
            bullets.append(f"Season {leader['season_pct']} complete")
        if "projected_full_season" in leader:
            bullets.append(f"Projected full-season: {leader['projected_full_season']} pts")
        if "current_matchup_projected" in leader:
            bullets.append(f"Current matchup projection: {leader['current_matchup_projected']} pts")
        return Recommendation(
            intent="standings_check",
            headline=headline,
            candidates=table.to_dict(orient="records"),
            rationale_bullets=bullets,
        )
