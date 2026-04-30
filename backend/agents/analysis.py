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
        query_name = self._extract_player_query(bundle.get("user_text", ""))
        recent_bat = bundle.get("recent_batting")
        recent_pitch = bundle.get("recent_pitching")
        if query_name:
            player_rec = self._analyze_named_player(query_name, recent_bat, recent_pitch, bundle)
            if player_rec is not None:
                return player_rec
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
            r"how is\s+(.+)$",
            r"what about\s+(.+)$",
            r"(.+?)\s+(?:trend|hot|cold|slump|streak)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                return re.sub(r"^(the player|player)\s+", "", name, flags=re.IGNORECASE)
        tokens = cleaned.split()
        if 2 <= len(tokens) <= 4 and not cleaned.lower().startswith(("scan ", "show ", "find ")):
            return cleaned
        return ""

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
            rows.append({
                "name": r["Name"],
                "role": "Batter",
                "team": r.get("Team", ""),
                "wRC+": round(float(r.get("wRC+", 0) or 0), 0),
                "HR": int(r.get("HR", 0) or 0),
                "AVG": round(float(r.get("AVG", 0) or 0), 3),
                "WAR": round(float(r.get("WAR", 0) or 0), 1),
            })
        if not pitch_match.empty:
            r = pitch_match.iloc[0]
            rows.append({
                "name": r["Name"],
                "role": "Pitcher",
                "team": r.get("Team", ""),
                "FIP": round(float(r.get("FIP", 0) or 0), 2),
                "xFIP": round(float(r.get("xFIP", 0) or 0), 2),
                "K%": round(float(r.get("K%", 0) or 0), 1),
                "WAR": round(float(r.get("WAR", 0) or 0), 1),
            })
        if not rows:
            return None

        league = bundle.get("league")
        fantasy_line = None
        if league is not None:
            query = rows[0]["name"].strip().casefold()
            for team in league.teams:
                for player in team.roster:
                    if player.name.strip().casefold() == query:
                        fantasy_line = {
                            "fantasy_team": team.name,
                            "position": player.fantasy_position,
                            "espn_points": round(float(player.total_points), 1),
                            "espn_avg": round(float(player.avg_points), 2),
                            "rostership": round(float(player.rostership), 1),
                        }
                        break
                if fantasy_line:
                    break
        candidates = rows.copy()
        if fantasy_line:
            candidates[0].update(fantasy_line)

        bullets = []
        for row in rows:
            if row["role"] == "Batter":
                bullets.append(
                    f"{row['name']} is a batter: wRC+ {row['wRC+']}, {row['HR']} HR, AVG {row['AVG']:.3f}, WAR {row['WAR']}"
                )
            else:
                bullets.append(
                    f"{row['name']} is a pitcher: FIP {row['FIP']}, xFIP {row['xFIP']}, K% {row['K%']}, WAR {row['WAR']}"
                )
        if fantasy_line:
            bullets.append(
                f"Fantasy context: {fantasy_line['espn_points']} ESPN pts, {fantasy_line['espn_avg']} avg, rostered by {fantasy_line['fantasy_team']}"
            )

        return Recommendation(
            intent="player_trend",
            headline=f"{rows[0]['name']} player profile",
            candidates=candidates,
            rationale_bullets=bullets,
            metrics={"matched_query": query_name, "roles_found": [row["role"] for row in rows]},
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
