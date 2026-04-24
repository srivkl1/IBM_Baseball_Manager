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


class DataRetrieval:
    def fetch(self, data_request: Dict[str, Any]) -> Dict[str, Any]:
        league = espn_client.load_league()
        bundle: Dict[str, Any] = {
            "seasons_scope": list(CONFIG.allowed_seasons),
            "data_source": "pybaseball" if pyb.have_real_data() else "synthetic-fallback",
            "league": league,
            "scoring_profile": league.scoring_profile,
        }

        if data_request.get("needs_player_pool", True):
            pool = build_pool(league.scoring_profile)
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
            bundle["player_pool"] = pool
            bundle["prospects"] = pyb.prospects(CONFIG.oot_season)

        if data_request.get("needs_recent_form"):
            most_recent = CONFIG.oot_season
            bundle["recent_batting"] = pyb.batting_stats(most_recent)
            bundle["recent_pitching"] = pyb.pitching_stats(most_recent)

        return bundle
