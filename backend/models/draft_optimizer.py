"""Gradient-boosted regressor that predicts next-season fantasy points.

Train on year-to-year transitions across the configured historical range.
Validate on the final transition into the target season.
Inference: apply (season-1) features to score draft candidates for `season`.
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from backend.config import CONFIG
from backend.data import pybaseball_client as pyb
from backend.draft.player_pool import _hitter_points, _pitcher_points

_MODEL_PATH = CONFIG.data_cache_dir / "draft_optimizer_2000_2026.joblib"

BAT_FEATURES = ["PA", "HR", "R", "RBI", "SB", "AVG", "OBP", "SLG", "wOBA",
                "wRC+", "ISO", "BB%", "K%", "Barrel%", "HardHit%", "WAR"]
PIT_FEATURES = ["IP", "K", "ERA", "FIP", "xFIP", "WHIP", "K%", "BB%", "W", "SV", "WAR"]


@dataclass
class OptimizerMetrics:
    train_range: tuple
    train_pairs: int
    oot_pair: tuple
    mae_bat_oot: float
    mae_pit_oot: float
    r2_bat_oot: float
    r2_pit_oot: float


def _join_year_pair(season_a: int, season_b: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    bat_a = pyb.batting_stats(season_a).copy()
    bat_b = pyb.batting_stats(season_b).copy()
    bat_b["target_pts"] = bat_b.apply(_hitter_points, axis=1)
    bat = bat_a.merge(bat_b[["Name", "target_pts"]], on="Name", how="inner")

    pit_a = pyb.pitching_stats(season_a).copy()
    pit_b = pyb.pitching_stats(season_b).copy()
    pit_b["target_pts"] = pit_b.apply(_pitcher_points, axis=1)
    pit = pit_a.merge(pit_b[["Name", "target_pts"]], on="Name", how="inner")
    return bat, pit


def _training_pairs() -> list[tuple[int, int]]:
    seasons = list(CONFIG.allowed_seasons)
    return list(zip(seasons[:-2], seasons[1:-1]))


def train_and_evaluate(force: bool = False) -> OptimizerMetrics:
    if _MODEL_PATH.exists() and not force:
        bundle = joblib.load(_MODEL_PATH)
        return bundle["metrics"]

    training_pairs = _training_pairs()
    if not training_pairs:
        raise ValueError("Need at least three seasons to train and evaluate the optimizer.")

    bat_train_frames = []
    pit_train_frames = []
    for season_a, season_b in training_pairs:
        bat_pair, pit_pair = _join_year_pair(season_a, season_b)
        bat_train_frames.append(bat_pair)
        pit_train_frames.append(pit_pair)

    bat_train = pd.concat(bat_train_frames, ignore_index=True)
    pit_train = pd.concat(pit_train_frames, ignore_index=True)
    oot_pair = (CONFIG.oot_season - 1, CONFIG.oot_season)
    bat_oot, pit_oot = _join_year_pair(*oot_pair)

    bat_model = GradientBoostingRegressor(random_state=42).fit(
        bat_train[BAT_FEATURES].fillna(0), bat_train["target_pts"]
    )
    pit_model = GradientBoostingRegressor(random_state=42).fit(
        pit_train[PIT_FEATURES].fillna(0), pit_train["target_pts"]
    )

    bat_pred = bat_model.predict(bat_oot[BAT_FEATURES].fillna(0))
    pit_pred = pit_model.predict(pit_oot[PIT_FEATURES].fillna(0))

    metrics = OptimizerMetrics(
        train_range=(training_pairs[0][0], training_pairs[-1][1]),
        train_pairs=len(training_pairs),
        oot_pair=oot_pair,
        mae_bat_oot=float(mean_absolute_error(bat_oot["target_pts"], bat_pred)),
        mae_pit_oot=float(mean_absolute_error(pit_oot["target_pts"], pit_pred)),
        r2_bat_oot=float(r2_score(bat_oot["target_pts"], bat_pred)),
        r2_pit_oot=float(r2_score(pit_oot["target_pts"], pit_pred)),
    )
    joblib.dump({"bat": bat_model, "pit": pit_model, "metrics": metrics}, _MODEL_PATH)
    return metrics


def _load_models():
    if not _MODEL_PATH.exists():
        train_and_evaluate()
    return joblib.load(_MODEL_PATH)


def score_players_for_season(season: int) -> pd.DataFrame:
    """Return projected fantasy points for every player using (season-1) features."""
    feature_season = season - 1
    if feature_season not in CONFIG.allowed_seasons:
        feature_season = max(CONFIG.allowed_seasons)

    bundle = _load_models()
    bat = pyb.batting_stats(feature_season).copy()
    pit = pyb.pitching_stats(feature_season).copy()

    bat["proj_pts"] = bundle["bat"].predict(bat[BAT_FEATURES].fillna(0))
    pit["proj_pts"] = bundle["pit"].predict(pit[PIT_FEATURES].fillna(0))

    bat["role"] = "BAT"
    pit["role"] = "PIT"
    cols = ["Name", "Team", "role", "proj_pts"]
    scored = pd.concat([bat[cols], pit[cols]], ignore_index=True)
    scored = scored.sort_values("proj_pts", ascending=False).reset_index(drop=True)
    scored["proj_rank"] = scored.index + 1
    return scored
