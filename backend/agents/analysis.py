"""Agent 3 - Analysis.

Produces a structured recommendation from the data bundle. For draft picks it
consults the draft simulator to propose the top candidates for the team on
the clock, weighing ML projected points plus roster needs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re

import pandas as pd

from backend.config import CONFIG
from backend.data import espn_client, mlb_stats
from backend.draft import simulator as sim
from backend.team_advisor import add_il_impact
from backend.trade_analyzer import analyze_trades


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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
        if intent == "player_bio":
            return self._analyze_player_bio(bundle)
        if intent == "player_trend":
            return self._analyze_trend(bundle)
        if intent == "standings_check":
            return self._analyze_standings(bundle)
        if intent == "general_qa":
            return self._analyze_general_qa(bundle)
        return Recommendation(intent=intent, headline="No analysis path for this intent.")

    @staticmethod
    def _missing_real_data_rec(intent: str, bundle: Dict[str, Any]) -> Recommendation:
        return Recommendation(
            intent=intent,
            headline="Real player ranking data is unavailable right now.",
            rationale_bullets=[
                bundle.get("data_error")
                or "The app could not load the real advanced-stat player pool.",
                "Synthetic fallback rankings are disabled so the app does not show false teams, projections, or draft advice.",
                "Player-specific questions can still use MLB API data when the player name is provided.",
            ],
            metrics={"data_source": bundle.get("data_source"), "real_data_required": True},
        )

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
        user_text = str(bundle.get("user_text", ""))
        if self._is_player_list_query(user_text):
            return self._analyze_player_list(bundle)

        pool = bundle.get("player_pool")
        if pool is None or pool.empty:
            return self._missing_real_data_rec("draft_pick", bundle)
        pool = self._available_pool(pool, bundle, draft_state)
        if draft_state is None:
            top = pool.head(10)
            cands = [{
                "name": r["Name"], "role": r["role"], "position": r.get("fantasy_position", r["role"]), "team": r.get("Team", ""),
                "proj_pts": round(float(r.get("proj_pts", r.get("draft_score", 0))), 1),
                "health_adjusted_proj_pts": round(float(r.get("health_adjusted_proj_pts", r.get("proj_pts", r.get("draft_score", 0)))), 1),
                "health": r.get("health_status", "Active"),
                "injury_note": self._injury_note(r),
                "rank": int(r["rank"]),
                "tier": int(r["tier"]),
            } for _, r in top.iterrows()]
            bullets = [
                f"{c['name']} ({c['position']}) - health-adjusted {c['health_adjusted_proj_pts']} pts, tier {c['tier']}, {c['health']}"
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
            "health_adjusted_proj_pts": round(float(r.get("health_adjusted_proj_pts", r.get("proj_pts", r.get("draft_score", 0)))), 1),
            "health": r.get("health_status", "Active"),
            "injury_note": self._injury_note(r),
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
                f"{c['proj_pts']} pts, health-adjusted {c['health_adjusted_proj_pts']}, "
                f"{c['health']}, tier {c['tier']}, fit {c['suitability']}"
            )
        return Recommendation(
            intent="draft_pick",
            headline=headline,
            candidates=cands,
            rationale_bullets=bullets,
            metrics={"pick_number": draft_state.current_pick_number,
                     "team_on_clock": rec_team or draft_state.team_on_clock()},
        )

    @staticmethod
    def _is_player_list_query(text: str) -> bool:
        t = text.lower()
        list_words = ("top", "best", "rank", "ranking", "list", "good", "elite")
        subject_words = (
            "players", "draft picks", "picks", "catchers", "pitchers", "hitters",
            "outfielders", "infielders", "infielder", "infield", "basemen", "shortstops", "starters", "relievers",
            "1b", "2b", "3b", "ss", "of", "sp", "rp", "c",
        )
        return any(word in t for word in list_words) and any(word in t for word in subject_words)

    @staticmethod
    def _position_filter_from_text(text: str) -> Optional[set[str]]:
        t = text.lower()
        position_aliases = [
            (("catcher", "catchers", " c "), {"C"}),
            (("first base", "first basemen", "1b"), {"1B"}),
            (("second base", "second basemen", "2b"), {"2B"}),
            (("third base", "third basemen", "3b"), {"3B"}),
            (("shortstop", "shortstops", " ss "), {"SS"}),
            (("infield", "infielder", "infielders"), {"C", "1B", "2B", "3B", "SS"}),
            (("outfield", "outfielder", "outfielders", " of "), {"OF"}),
            (("starting pitcher", "starting pitchers", "starter", "starters", "sp"), {"SP"}),
            (("relief pitcher", "relief pitchers", "reliever", "relievers", "closer", "closers", "rp"), {"RP"}),
            (("pitcher", "pitchers"), {"SP", "RP", "PIT", "P"}),
            (("hitter", "hitters", "batter", "batters"), {"C", "1B", "2B", "3B", "SS", "OF", "BAT"}),
        ]
        padded = f" {t} "
        for aliases, positions in position_aliases:
            if any(alias in padded for alias in aliases):
                return positions
        return None

    def _analyze_player_list(self, bundle: Dict[str, Any]) -> Recommendation:
        pool: pd.DataFrame = bundle.get("player_pool")
        if pool is None or pool.empty:
            return self._missing_real_data_rec("draft_pick", bundle)

        user_text = str(bundle.get("user_text", ""))
        positions = self._position_filter_from_text(user_text)
        ranked = pool.copy()
        scope = "overall"
        if positions:
            pos_series = ranked.get("fantasy_position", ranked.get("role", "")).astype(str).str.upper()
            role_series = ranked.get("role", "").astype(str).str.upper()
            ranked = ranked[pos_series.isin(positions) | role_series.isin(positions)]
            scope = "/".join(sorted(positions))
        if ranked.empty:
            return Recommendation(
                "draft_pick",
                f"No players matched that position filter ({scope}).",
                rationale_bullets=["Try asking for top hitters, pitchers, OF, SP, RP, SS, 1B, 2B, 3B, or C."],
                metrics={"response_style": "player_list", "position_scope": scope},
            )

        sort_col = (
            "health_adjusted_proj_pts" if "health_adjusted_proj_pts" in ranked.columns
            else "health_adjusted_draft_score" if "health_adjusted_draft_score" in ranked.columns
            else "draft_score" if "draft_score" in ranked.columns
            else "proj_pts"
        )
        if sort_col in ranked:
            ranked = ranked.sort_values(sort_col, ascending=False)
        elif "rank" in ranked:
            ranked = ranked.sort_values("rank", ascending=True)

        # Always verify list/ranking answers against MLB API before showing
        # teams/stat lines. This prevents synthetic or stale team abbreviations
        # from being presented as real context in hosted deployments.
        ranked = self._rerank_with_mlb_api(ranked.head(30))
        top = ranked.head(10)
        cands = []
        for idx, (_, r) in enumerate(top.iterrows(), start=1):
            real = r.get("real_context", {}) if isinstance(r.get("real_context", {}), dict) else {}
            cands.append({
                "rank": idx,
                "name": r["Name"],
                "position": r.get("fantasy_position", r.get("role", "")),
                "team": real.get("team") or r.get("Team", ""),
                "proj_pts": round(float(r.get("proj_pts", r.get("draft_score", 0)) or 0), 1),
                "health_adjusted_proj_pts": round(float(r.get("health_adjusted_proj_pts", r.get("proj_pts", r.get("draft_score", 0))) or 0), 1),
                "health": r.get("health_status", "Active"),
                "injury_note": self._injury_note(r),
                "real_stat_line": real.get("stat_line", ""),
                "real_score": round(float(r.get("real_data_score", 0.0) or 0.0), 1),
                "tier": int(r.get("tier", 0) or 0),
                "three_year_rank": int(r.get("rank", idx) or idx),
            })
        title_scope = "overall" if scope == "overall" else f"at {scope}"
        bullets = [
            f"{c['rank']}. {c['name']} ({c['position']}, {c['team']}) - "
            f"{c['real_stat_line'] or str(c['health_adjusted_proj_pts']) + ' health-adjusted pts'}, "
            f"{c['health']}, tier {c['tier']}"
            for c in cands
        ]
        return Recommendation(
            intent="draft_pick",
            headline=f"Top {len(cands)} draft targets {title_scope}",
            candidates=cands,
            rationale_bullets=bullets,
            metrics={
                "response_style": "player_list",
                "position_scope": scope,
                "ranked_pool_size": int(len(ranked)),
                "basis": (
                    "MLB API season stats/current team, health status, and health-adjusted projection"
                ),
                "data_source_warning": "",
            },
        )

    def _rerank_with_mlb_api(self, ranked: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in ranked.iterrows():
            row = row.copy()
            real = self._real_player_context(str(row.get("Name", "")))
            row["real_context"] = real
            health_multiplier = _safe_float(row.get("health_multiplier"), 1.0)
            row["real_data_score"] = real.get("score", 0.0) * health_multiplier
            if real.get("team"):
                row["Team"] = real["team"]
            rows.append(row)
        out = pd.DataFrame(rows)
        if "real_data_score" in out:
            out = out.sort_values(
                ["real_data_score", "health_adjusted_proj_pts" if "health_adjusted_proj_pts" in out else "proj_pts"],
                ascending=[False, False],
            )
        return out.reset_index(drop=True)

    @staticmethod
    def _real_player_context(name: str) -> dict:
        bio = mlb_stats.player_bio(name)
        summary = mlb_stats.player_season_summary(name, CONFIG.oot_season)
        team = (
            (summary.get("hitting", {}) or {}).get("team")
            or (summary.get("pitching", {}) or {}).get("team")
            or bio.get("current_team")
            or ""
        )
        hitting = summary.get("hitting") or {}
        pitching = summary.get("pitching") or {}
        if hitting:
            games = _safe_float(hitting.get("games"))
            ops = _safe_float(hitting.get("ops"))
            avg = str(hitting.get("avg") or "").lstrip("0")
            hr = _safe_float(hitting.get("home_runs"))
            rbi = _safe_float(hitting.get("rbi"))
            score = games * 0.4 + ops * 220 + hr * 4 + rbi * 1.2
            if int(summary.get("season") or CONFIG.oot_season) < CONFIG.oot_season:
                score *= 0.65
            return {
                "team": team,
                "score": score,
                "stat_line": (
                    f"MLB {summary.get('season')}: {int(games)} G, {avg or 'N/A'} AVG, "
                    f"{hitting.get('ops') or 'N/A'} OPS, {int(hr)} HR, {int(rbi)} RBI"
                ),
            }
        if pitching:
            games = _safe_float(pitching.get("games"))
            innings = _safe_float(pitching.get("innings"))
            era = _safe_float(pitching.get("era"), 9.99)
            whip = _safe_float(pitching.get("whip"), 2.00)
            strikeouts = _safe_float(pitching.get("strikeouts"))
            score = innings * 1.2 + strikeouts * 1.5 + max(0, 5.00 - era) * 35 + max(0, 1.50 - whip) * 60
            if int(summary.get("season") or CONFIG.oot_season) < CONFIG.oot_season:
                score *= 0.65
            return {
                "team": team,
                "score": score,
                "stat_line": (
                    f"MLB {summary.get('season')}: {int(games)} G, {pitching.get('innings') or '0'} IP, "
                    f"{pitching.get('era') or 'N/A'} ERA, {pitching.get('whip') or 'N/A'} WHIP, "
                    f"{int(strikeouts)} K"
                ),
            }
        return {
            "team": team,
            "score": 0.0,
            "stat_line": "MLB API: current team verified; no season stat line found",
        }

    @staticmethod
    def _injury_note(row: pd.Series) -> str:
        if str(row.get("health_status", "Active")) == "Active":
            return ""
        return str(row.get("public_injury_detail", "") or row.get("espn_injury_status", "") or "")

    def _analyze_roster_move(self, bundle: Dict[str, Any]) -> Recommendation:
        pool: pd.DataFrame = bundle.get("player_pool")
        if pool is None or pool.empty:
            return self._missing_real_data_rec("roster_move", bundle)
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
    def _my_team(league, bundle: Optional[Dict[str, Any]] = None):
        if league is None or not getattr(league, "teams", None):
            return None
        context = (bundle or {}).get("user_context", {}) or {}
        selected_id = context.get("selected_team_id")
        if selected_id not in (None, ""):
            try:
                selected_id = int(selected_id)
                for team in league.teams:
                    if int(getattr(team, "team_id", -1)) == selected_id:
                        return team
            except (TypeError, ValueError):
                pass
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
        merged = add_il_impact(merged.rename(columns={"injury": "injury_status"}))
        merged["value"] = (
            merged["proj_pts"].fillna(0).astype(float)
            + merged["espn_avg"].fillna(0).astype(float) * 8.0
            + war.fillna(0).astype(float) * 3.0
            - merged.get("estimated_value_lost", 0.0).fillna(0).astype(float) * 0.35
        )
        return merged.sort_values("value", ascending=False).reset_index(drop=True)

    def _analyze_waiver_scan(self, bundle: Dict[str, Any]) -> Recommendation:
        pool: pd.DataFrame = bundle.get("player_pool")
        if pool is None or pool.empty:
            return self._missing_real_data_rec("waiver_scan", bundle)
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
        if pool is None or pool.empty:
            return self._missing_real_data_rec("team_diagnosis", bundle)
        team = self._my_team(league, bundle)
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
        il_lost = float(roster.get("estimated_value_lost", pd.Series(dtype=float)).fillna(0).sum())
        if il_lost:
            bullets.append(f"IL impact: estimated {il_lost:.1f} fantasy points of value lost to missed games")
        return Recommendation(
            "team_diagnosis",
            f"{team.name} is strongest at {strengths.iloc[0]['position']} and thinnest at {weaknesses.iloc[0]['position']}",
            candidates=cands,
            rationale_bullets=bullets,
            metrics={"team": team.name, "roster_size": int(len(roster)), "il_value_lost": round(il_lost, 1), "confidence": "medium"},
        )

    def _analyze_trade_analysis(self, bundle: Dict[str, Any]) -> Recommendation:
        league = bundle.get("league")
        my_team = self._my_team(league, bundle)
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
        if pool is None or pool.empty:
            return self._missing_real_data_rec("lineup_optimization", bundle)
        team = self._my_team(league, bundle)
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
        if pool is None or pool.empty:
            return self._missing_real_data_rec("risk_check", bundle)
        team = self._my_team(league, bundle)
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
            if float(row.get("estimated_value_lost", 0) or 0) > 0:
                score += min(30, int(float(row.get("estimated_value_lost", 0)) / 5))
                reasons.append(f"estimated {row.get('estimated_value_lost', 0):.1f} value lost")
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
        user_text = str(bundle.get("user_text", ""))
        fantasy_team_query = self._extract_fantasy_team_query(user_text)
        mlb_team_query = self._extract_mlb_team_query(user_text)
        if fantasy_team_query:
            league = bundle.get("league")
            rec = self._analyze_fantasy_team_roster(fantasy_team_query, league)
            if rec is not None:
                return rec
        if mlb_team_query:
            return self._analyze_mlb_team_roster(mlb_team_query)

        league = bundle.get("league")
        selected_team = self._my_team(league, bundle)
        if selected_team is not None and any(
            phrase in user_text.lower()
            for phrase in ("my team", "my roster", "my lineup", "who do i have")
        ):
            rec = self._analyze_fantasy_team_roster(selected_team.name, league)
            if rec is not None:
                return rec

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

    @staticmethod
    def _extract_fantasy_team_query(text: str) -> str:
        cleaned = re.sub(r"[?!.]", "", text or "").strip()
        patterns = [
            r"who(?: is|'s)? on\s+(.+?)\s+fantasy team$",
            r"show(?: me)?\s+(.+?)\s+fantasy team$",
            r"who(?: is|'s)? on\s+(.+?)'s team$",
            r"who(?: is|'s)? on\s+(.+?) roster$",
        ]
        for pattern in patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    @staticmethod
    def _extract_mlb_team_query(text: str) -> str:
        cleaned = re.sub(r"[?!.]", "", text or "").strip()
        patterns = [
            r"who(?: is|'s)? on\s+the\s+(.+)$",
            r"who(?: is|'s)? on\s+(.+)$",
            r"who plays for\s+the\s+(.+)$",
            r"who plays for\s+(.+)$",
            r"show(?: me)?\s+the\s+(.+?)\s+roster$",
            r"show(?: me)?\s+(.+?)\s+roster$",
        ]
        for pattern in patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                query = match.group(1).strip()
                if "fantasy" not in query.lower() and not query.lower().startswith(("my team", "my roster")):
                    return query
        return ""

    @staticmethod
    def _matches_team_query(team, query: str) -> bool:
        q = query.strip().casefold()
        values = [
            getattr(team, "name", ""),
            getattr(team, "owner", ""),
            f"{getattr(team, 'owner', '')} {getattr(team, 'name', '')}",
        ]
        return any(q and q in str(value).casefold() for value in values)

    def _analyze_fantasy_team_roster(self, query: str, league) -> Optional[Recommendation]:
        if league is None or not getattr(league, "teams", None):
            return None
        matches = [team for team in league.teams if self._matches_team_query(team, query)]
        if not matches:
            return Recommendation(
                "roster_lookup",
                f"I could not find a fantasy team matching '{query}'.",
                rationale_bullets=[
                    "Use the exact fantasy team name or owner name from your ESPN league.",
                ],
                metrics={"query": query, "league_source": getattr(league, "source", "")},
            )
        team = matches[0]
        cands = [{
            "player": player.name,
            "position": player.fantasy_position,
            "mlb_team": player.mlb_team,
            "slot": player.lineup_slot,
            "espn_points": round(float(player.total_points), 1),
            "espn_avg": round(float(player.avg_points), 2),
        } for player in getattr(team, "roster", [])]
        return Recommendation(
            "roster_lookup",
            f"{team.name} roster: {len(cands)} players loaded",
            candidates=cands,
            rationale_bullets=[
                f"{c['player']} ({c.get('position', '')}, {c.get('mlb_team', '')})"
                for c in cands[:12]
            ],
            metrics={"team": team.name, "owner": team.owner, "roster_size": len(cands), "source": "ESPN fantasy"},
        )

    def _analyze_mlb_team_roster(self, query: str) -> Recommendation:
        roster = mlb_stats.team_roster(query, CONFIG.oot_season)
        if not roster:
            return Recommendation(
                "roster_lookup",
                f"I could not find an MLB team matching '{query}'.",
                rationale_bullets=["Try the full team name, city, nickname, or abbreviation."],
                metrics={"query": query, "source": "MLB Stats API"},
            )
        cands = roster.get("players", [])
        return Recommendation(
            "roster_lookup",
            f"{roster.get('team')} active roster: {len(cands)} players",
            candidates=cands,
            rationale_bullets=[
                f"{player.get('name')} ({player.get('position')})"
                for player in cands[:15]
            ],
            metrics={
                "team": roster.get("team"),
                "roster_size": len(cands),
                "season": roster.get("season"),
                "source": "MLB Stats API",
            },
        )

    def _analyze_trend(self, bundle: Dict[str, Any]) -> Recommendation:
        query_name = self._extract_player_query(bundle.get("user_text", ""))
        recent_bat = bundle.get("recent_batting")
        recent_pitch = bundle.get("recent_pitching")
        if query_name:
            player_rec = self._analyze_named_player(query_name, recent_bat, recent_pitch, bundle)
            if player_rec is not None:
                return player_rec
            return self._analyze_missing_named_player(query_name, bundle)
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

    def _analyze_missing_named_player(self, query_name: str, bundle: Dict[str, Any]) -> Recommendation:
        prospects = bundle.get("prospects")
        prospect_rows = self._name_match(prospects, query_name) if prospects is not None else pd.DataFrame()
        bio = mlb_stats.player_bio(query_name)
        mlb_summary = mlb_stats.player_season_summary(query_name, CONFIG.oot_season)
        candidate: Dict[str, Any] = {"name": query_name}
        bullets = [
            f"I found the player name '{query_name}', but not a current-season batting or pitching row in the loaded stats table.",
            "That usually means the player has a small MLB sample or is not qualified for the pulled advanced-stat leaderboards.",
        ]
        if bio:
            candidate.update(bio)
            if bio.get("primary_position") == "TWP":
                candidate["primary_position"] = "two-way player"
            bullets.append(
                f"Bio context: {bio.get('name', query_name)} is listed as {candidate.get('primary_position', 'a player')} for {bio.get('current_team') or 'an MLB organization'}."
            )
        if mlb_summary:
            candidate["mlb_api_summary"] = mlb_summary
            hitting = mlb_summary.get("hitting")
            pitching = mlb_summary.get("pitching")
            if hitting:
                bullets.append(
                    f"MLB API {mlb_summary.get('season')} hitting line: "
                    f"{hitting.get('games')} G, AVG {hitting.get('avg')}, OPS {hitting.get('ops')}, "
                    f"{hitting.get('home_runs')} HR, {hitting.get('rbi')} RBI."
                )
            if pitching:
                candidate["performance_read"] = (
                    "struggling" if _safe_float(pitching.get("era"), 0.0) >= 5.00
                    or _safe_float(pitching.get("whip"), 0.0) >= 1.40 else "limited sample"
                )
                bullets.append(
                    f"MLB API {mlb_summary.get('season')} pitching line: "
                    f"{pitching.get('games')} G/{pitching.get('games_started')} GS, "
                    f"{pitching.get('innings')} IP, ERA {pitching.get('era')}, WHIP {pitching.get('whip')}, "
                    f"{pitching.get('strikeouts')} K, {pitching.get('walks')} BB."
                )
        fantasy_line = self._fantasy_context_for_player(query_name, bundle.get("league"))
        if fantasy_line:
            candidate["espn_fantasy"] = fantasy_line
            bullets.append(
                f"ESPN fantasy context: {fantasy_line['team_status']} for {fantasy_line.get('fantasy_team', 'free agent')}, "
                f"{fantasy_line.get('espn_points', 0)} pts, {fantasy_line.get('espn_avg', 0)} avg."
            )
        if not prospect_rows.empty:
            row = prospect_rows.iloc[0].to_dict()
            candidate.update({"prospect": row})
            fv = row.get("FV", row.get("future_value", ""))
            team = row.get("Team", "")
            bullets.append(f"Prospect context: {row.get('Name', query_name)} appears in the prospect data for {team} with FV {fv}.")
        headline = f"{query_name} has limited current-stat data loaded"
        pitching = mlb_summary.get("pitching") if mlb_summary else None
        hitting = mlb_summary.get("hitting") if mlb_summary else None
        if pitching:
            headline = (
                f"{query_name}: MLB API shows a rough pitching line "
                f"({pitching.get('era')} ERA, {pitching.get('whip')} WHIP)"
            )
        elif hitting:
            headline = (
                f"{query_name}: MLB API hitting line "
                f"({hitting.get('avg')} AVG, {hitting.get('ops')} OPS)"
            )

        return Recommendation(
            intent="player_trend",
            headline=headline,
            candidates=[candidate],
            rationale_bullets=bullets,
            metrics={"matched_query": query_name, "data_gap": "no current batting/pitching row"},
        )

    @staticmethod
    def _fantasy_context_for_player(query_name: str, league) -> Optional[dict]:
        if league is None:
            return None
        query = query_name.strip().casefold()
        for team in getattr(league, "teams", []):
            for player in getattr(team, "roster", []):
                if player.name.strip().casefold() == query:
                    return {
                        "team_status": "rostered",
                        "fantasy_team": team.name,
                        "position": player.fantasy_position,
                        "espn_points": round(float(player.total_points), 1),
                        "espn_avg": round(float(player.avg_points), 2),
                        "rostership": round(float(player.rostership), 1),
                        "games": round(float(player.games_played), 0),
                        "projected_points": round(float(player.projected_total_points), 1),
                    }
        return None

    def _analyze_player_bio(self, bundle: Dict[str, Any]) -> Recommendation:
        query_name = self._extract_player_query(bundle.get("user_text", ""))
        if not query_name:
            return Recommendation(
                intent="player_bio",
                headline="Which player should I look up?",
                rationale_bullets=["Ask something like: Who is Shohei Ohtani?"],
            )

        bio = mlb_stats.player_bio(query_name)
        if not bio:
            return Recommendation(
                intent="player_bio",
                headline=f"I could not find a bio for {query_name}.",
                rationale_bullets=["Try the player's full MLB name."],
            )

        recent_bat = bundle.get("recent_batting")
        recent_pitch = bundle.get("recent_pitching")
        trend_rec = self._analyze_named_player(bio.get("name", query_name), recent_bat, recent_pitch, bundle)
        stats = trend_rec.candidates if trend_rec else []
        teams = mlb_stats.teams_played(bio.get("name", query_name), CONFIG.oot_season)
        news = espn_client.player_news(bio.get("name", query_name), limit=3)

        birthplace = ", ".join(
            part for part in (bio.get("birth_city"), bio.get("birth_state"), bio.get("birth_country")) if part
        )
        if bio.get("primary_position") == "TWP":
            bio["primary_position"] = "two-way player"
        candidate = {
            **bio,
            "birthplace": birthplace,
            "teams_played_recent": teams,
            "recent_stats": stats,
            "news": news,
        }
        bullets = [
            f"{bio.get('name')} is a {bio.get('primary_position', 'baseball player')} for {bio.get('current_team') or 'an MLB organization'}.",
        ]
        age = bio.get("age")
        if age or birthplace:
            bullets.append(f"Bio: age {age or 'unknown'}, from {birthplace or 'unknown birthplace'}.")
        if bio.get("bats") or bio.get("throws"):
            bullets.append(f"Bats {bio.get('bats') or 'unknown'} and throws {bio.get('throws') or 'unknown'}.")
        if bio.get("mlb_debut"):
            bullets.append(f"MLB debut: {bio.get('mlb_debut')}.")
        if teams:
            bullets.append(f"Recent team data includes: {', '.join(teams[:5])}.")
        if stats:
            for row in stats[:2]:
                if row.get("role") == "Batter":
                    bullets.append(f"Current batting context: wRC+ {row.get('wRC+')}, {row.get('HR')} HR, WAR {row.get('WAR')}.")
                elif row.get("role") == "Pitcher":
                    bullets.append(f"Current pitching context: FIP {row.get('FIP')}, xFIP {row.get('xFIP')}, WAR {row.get('WAR')}.")
        for item in news[:3]:
            if item.get("headline"):
                bullets.append(f"Recent news: {item['headline']}")

        return Recommendation(
            intent="player_bio",
            headline=f"{bio.get('name')} bio and fantasy context",
            candidates=[candidate],
            rationale_bullets=bullets,
            metrics={"matched_query": query_name, "news_items": len(news), "source": "MLB Stats API + ESPN news"},
        )

    @staticmethod
    def _extract_player_query(text: str) -> str:
        cleaned = re.sub(r"[?!.]", "", text or "").strip()
        patterns = [
            r"who is\s+(.+)$",
            r"tell me about\s+(.+)$",
            r"what team does\s+(.+?)\s+play for$",
            r"which team does\s+(.+?)\s+play for$",
            r"where does\s+(.+?)\s+play$",
            r"how old is\s+(.+)$",
            r"where is\s+(.+?)\s+from$",
            r"where was\s+(.+?)\s+born$",
            r"how is\s+(.+)$",
            r"why is\s+(.+?)(?:\s+so\s+\w+)?$",
            r"why has\s+(.+?)(?:\s+been\s+\w+)?$",
            r"what happened to\s+(.+)$",
            r"what about\s+(.+)$",
            r"(.+?)\s+(?:trend|hot|cold|slump|streak)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                return Analysis._clean_player_query(match.group(1))
        tokens = cleaned.split()
        if 2 <= len(tokens) <= 4 and not cleaned.lower().startswith(("scan ", "show ", "find ")):
            return Analysis._clean_player_query(cleaned)
        return ""

    @staticmethod
    def _clean_player_query(name: str) -> str:
        name = re.sub(r"^(the player|player)\s+", "", name.strip(), flags=re.IGNORECASE)
        trailing_context = [
            r"\s+performing\s+(?:poorly|badly|well|great|terribly)$",
            r"\s+playing\s+(?:poorly|badly|well|great|terribly)$",
            r"\s+(?:doing|looking)\s+(?:poorly|badly|well|great|terrible|rough)$",
            r"\s+(?:doing|performing|playing|looking)$",
            r"\s+(?:struggling|slumping|cold|hot|washed|rough)$",
            r"\s+(?:bad|good|great|terrible|poor|elite|mid)$",
        ]
        for pattern in trailing_context:
            name = re.sub(pattern, "", name, flags=re.IGNORECASE).strip()
        return name

    @staticmethod
    def _name_match(df: Optional[pd.DataFrame], query_name: str) -> pd.DataFrame:
        if df is None or df.empty or "Name" not in df:
            return pd.DataFrame()
        query = query_name.strip().casefold()
        names = df["Name"].astype(str).str.strip().str.casefold()
        exact = df[names == query]
        if not exact.empty:
            return exact
        parts = [part for part in query.split() if len(part) > 1]
        if not parts:
            return pd.DataFrame()
        mask = names.apply(lambda name: all(part in name for part in parts))
        return df[mask]

    def _analyze_named_player(self, query_name: str, recent_bat: Optional[pd.DataFrame],
                              recent_pitch: Optional[pd.DataFrame],
                              bundle: Dict[str, Any]) -> Optional[Recommendation]:
        bat_match = self._name_match(recent_bat, query_name)
        pitch_match = self._name_match(recent_pitch, query_name)
        rows = []
        if not bat_match.empty:
            r = bat_match.iloc[0]
            wrc = float(r.get("wRC+", 0) or 0)
            hr = int(r.get("HR", 0) or 0)
            avg = float(r.get("AVG", 0) or 0)
            war = float(r.get("WAR", 0) or 0)
            peer_wrc = float(recent_bat["wRC+"].median()) if recent_bat is not None and "wRC+" in recent_bat else 100.0
            peer_war = float(recent_bat["WAR"].median()) if recent_bat is not None and "WAR" in recent_bat else 0.0
            rows.append({
                "name": r["Name"],
                "role": "Batter",
                "team": r.get("Team", ""),
                "wRC+": round(wrc, 0),
                "HR": hr,
                "AVG": round(avg, 3),
                "WAR": round(war, 1),
                "peer_wRC+_median": round(peer_wrc, 0),
                "peer_WAR_median": round(peer_war, 1),
                "performance_read": "below average" if wrc < 95 else "above average" if wrc > 110 else "around average",
            })
        if not pitch_match.empty:
            r = pitch_match.iloc[0]
            fip = float(r.get("FIP", 0) or 0)
            xfip = float(r.get("xFIP", 0) or 0)
            k_pct = float(r.get("K%", 0) or 0)
            war = float(r.get("WAR", 0) or 0)
            peer_fip = float(recent_pitch["FIP"].median()) if recent_pitch is not None and "FIP" in recent_pitch else 4.20
            peer_k = float(recent_pitch["K%"].median()) if recent_pitch is not None and "K%" in recent_pitch else 22.0
            rows.append({
                "name": r["Name"],
                "role": "Pitcher",
                "team": r.get("Team", ""),
                "FIP": round(fip, 2),
                "xFIP": round(xfip, 2),
                "K%": round(k_pct, 1),
                "WAR": round(war, 1),
                "peer_FIP_median": round(peer_fip, 2),
                "peer_K%_median": round(peer_k, 1),
                "performance_read": "struggling" if fip > peer_fip + 0.5 else "strong" if fip < peer_fip - 0.5 else "around average",
            })
        if not rows:
            return None

        resolved_name = rows[0]["name"]
        fantasy_line = self._fantasy_context_for_player(resolved_name, bundle.get("league"))
        mlb_summary = mlb_stats.player_season_summary(resolved_name, CONFIG.oot_season)
        has_mlb_stats = bool(mlb_summary.get("hitting") or mlb_summary.get("pitching"))
        candidates = rows.copy()
        if fantasy_line:
            candidates[0]["espn_fantasy"] = fantasy_line
            candidates[0].update({
                "fantasy_team": fantasy_line.get("fantasy_team"),
                "position": fantasy_line.get("position"),
                "espn_points": fantasy_line.get("espn_points"),
                "espn_avg": fantasy_line.get("espn_avg"),
                "rostership": fantasy_line.get("rostership"),
                "games": fantasy_line.get("games"),
                "projected_points": fantasy_line.get("projected_points"),
            })
        if has_mlb_stats:
            candidates[0]["mlb_api_summary"] = mlb_summary

        bullets = []
        if has_mlb_stats:
            hitting = mlb_summary.get("hitting")
            pitching = mlb_summary.get("pitching")
            if hitting:
                bullets.append(
                    f"MLB API {mlb_summary.get('season')} hitting line: "
                    f"{hitting.get('games')} G, AVG {hitting.get('avg')}, OPS {hitting.get('ops')}, "
                    f"{hitting.get('home_runs')} HR, {hitting.get('rbi')} RBI."
                )
            if pitching:
                bullets.append(
                    f"MLB API {mlb_summary.get('season')} pitching line: "
                    f"{pitching.get('games')} G/{pitching.get('games_started')} GS, "
                    f"{pitching.get('innings')} IP, ERA {pitching.get('era')}, WHIP {pitching.get('whip')}, "
                    f"{pitching.get('strikeouts')} K, {pitching.get('walks')} BB."
                )
        for row in rows:
            if row["role"] == "Batter":
                bullets.append(
                    f"Advanced stats: {row['name']} is a batter with wRC+ {row['wRC+']} vs peer median {row['peer_wRC+_median']}, {row['HR']} HR, AVG {row['AVG']:.3f}, WAR {row['WAR']}; read: {row['performance_read']}"
                )
            else:
                bullets.append(
                    f"Advanced stats: {row['name']} is a pitcher with FIP {row['FIP']} vs peer median {row['peer_FIP_median']}, xFIP {row['xFIP']}, K% {row['K%']} vs peer median {row['peer_K%_median']}, WAR {row['WAR']}; read: {row['performance_read']}"
                )
        if fantasy_line:
            bullets.append(
                f"ESPN fantasy context: {fantasy_line['team_status']} for {fantasy_line.get('fantasy_team', 'free agent')}, "
                f"{fantasy_line.get('espn_points', 0)} pts, {fantasy_line.get('espn_avg', 0)} avg."
            )

        return Recommendation(
            intent="player_trend",
            headline=f"{resolved_name} player profile",
            candidates=candidates,
            rationale_bullets=bullets,
            metrics={
                "matched_query": query_name,
                "roles_found": [row["role"] for row in rows],
                "sources": ["advanced_stats"] + (["mlb_api"] if has_mlb_stats else []) + (["espn_fantasy"] if fantasy_line else []),
            },
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

    def _analyze_general_qa(self, bundle: Dict[str, Any]) -> Recommendation:
        question = str(bundle.get("user_text", "")).strip()
        league = bundle.get("league")
        league_context = {}
        if league is not None:
            league_context = {
                "league_source": league.source,
                "season": league.season,
                "teams": len(league.teams),
            }
        return Recommendation(
            intent="general_qa",
            headline=question or "General baseball question",
            candidates=[],
            rationale_bullets=[],
            metrics={"question": question, **league_context},
        )
