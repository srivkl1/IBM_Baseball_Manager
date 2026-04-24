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
