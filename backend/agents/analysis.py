"""Agent 3 — Analysis.

Produces a structured recommendation from the data bundle. For draft picks it
consults the draft simulator to propose the top candidates for the team on
the clock, weighing ML projected points + roster needs + bye-risk (proxied by
recent-form volatility).
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
        if intent == "roster_move":
            return self._analyze_roster_move(bundle)
        if intent == "player_trend":
            return self._analyze_trend(bundle)
        if intent == "standings_check":
            return self._analyze_standings(bundle)
        return Recommendation(intent=intent, headline="No analysis path for this intent.")

    # ---- draft pick ----
    def _analyze_draft(self, bundle: Dict[str, Any],
                       draft_state: Optional[sim.DraftState]) -> Recommendation:
        pool: pd.DataFrame = bundle["player_pool"]
        if draft_state is None:
            # No live draft — just surface global top-10.
            top = pool.head(10)
            cands = [{
                "name": r["Name"], "role": r["role"], "team": r.get("Team", ""),
                "proj_pts": round(float(r.get("proj_pts", r.get("draft_score", 0))), 1),
                "rank": int(r["rank"]),
                "tier": int(r["tier"]),
            } for _, r in top.iterrows()]
            bullets = [
                f"{c['name']} ({c['role']}) — projected {c['proj_pts']} pts, tier {c['tier']}"
                for c in cands[:3]
            ]
            return Recommendation(
                intent="draft_pick",
                headline="Top overall prospects by 3-year composite + ML projection",
                candidates=cands, rationale_bullets=bullets,
                metrics={"pool_size": int(len(pool))},
            )

        # Live-draft recommendation for the team on the clock.
        recs = sim.recommend_pick(draft_state, top_n=5)
        cands = [{
            "name": r["Name"], "role": r["role"], "team": r.get("Team", ""),
            "proj_pts": round(float(r.get("proj_pts", r.get("draft_score", 0))), 1),
            "suitability": round(float(r["suitability"]), 1),
            "tier": int(r["tier"]),
        } for _, r in recs.iterrows()]
        round_num, slot = draft_state.round_and_slot()
        headline = (f"Round {round_num}, pick {slot} — top option: "
                    f"{cands[0]['name']} ({cands[0]['role']})" if cands
                    else "No players left on the board")
        bullets = []
        for c in cands[:3]:
            bullets.append(
                f"{c['name']} ({c['role']}, {c['team']}) — projected "
                f"{c['proj_pts']} pts, tier {c['tier']}, fit {c['suitability']}"
            )
        return Recommendation(
            intent="draft_pick",
            headline=headline,
            candidates=cands,
            rationale_bullets=bullets,
            metrics={"pick_number": draft_state.current_pick_number,
                     "team_on_clock": draft_state.team_on_clock()},
        )

    # ---- roster move ----
    def _analyze_roster_move(self, bundle: Dict[str, Any]) -> Recommendation:
        pool: pd.DataFrame = bundle.get("player_pool")
        if pool is None:
            return Recommendation(intent="roster_move",
                                  headline="Need a player pool to evaluate roster moves.")
        adds = pool.head(15).tail(10)
        cands = [{"name": r["Name"], "role": r["role"],
                  "proj_pts": round(float(r.get("proj_pts", 0)), 1),
                  "tier": int(r["tier"])}
                 for _, r in adds.iterrows()]
        return Recommendation(
            intent="roster_move",
            headline="Top available adds (league-agnostic ranking)",
            candidates=cands,
            rationale_bullets=[f"Consider adding {c['name']} — projected {c['proj_pts']} pts"
                               for c in cands[:3]],
        )

    # ---- player trend ----
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

    # ---- standings ----
    def _analyze_standings(self, bundle: Dict[str, Any]) -> Recommendation:
        table = bundle.get("standings_table")
        if table is None:
            return Recommendation(intent="standings_check",
                                  headline="No draft has been locked in yet.")
        leader = table.iloc[0]
        return Recommendation(
            intent="standings_check",
            headline=f"{leader['team']} leads with {leader['points_to_date']} pts",
            candidates=table.to_dict(orient="records"),
            rationale_bullets=[
                f"Season {leader['season_pct']} complete",
                f"Projected full-season: {leader['projected_full_season']} pts",
            ],
        )
