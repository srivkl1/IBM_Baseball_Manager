"""Environment + runtime configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock").lower()

    watsonx_apikey: str = os.getenv("WATSONX_APIKEY", "")
    watsonx_url: str = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    watsonx_project_id: str = os.getenv("WATSONX_PROJECT_ID", "")
    watsonx_model_id: str = os.getenv("WATSONX_MODEL_ID", "ibm/granite-3-8b-instruct")

    custom_base_url: str = os.getenv("CUSTOM_LLM_BASE_URL", "http://localhost:8080/v1")
    custom_api_key: str = os.getenv("CUSTOM_LLM_API_KEY", "not-needed")
    custom_model: str = os.getenv("CUSTOM_LLM_MODEL", "llama-3.1-8b-instruct")

    espn_league_id: str = os.getenv("ESPN_LEAGUE_ID", "")
    espn_season: int = int(os.getenv("ESPN_SEASON", "2025"))
    espn_swid: str = os.getenv("ESPN_SWID", "")
    espn_s2: str = os.getenv("ESPN_S2", "")

    data_cache_dir: Path = Path(os.getenv("DATA_CACHE_DIR", "./data_cache"))

    # Scope: only consider data from these seasons (3-year window).
    allowed_seasons: tuple = (2023, 2024, 2025)
    train_seasons: tuple = (2023, 2024)
    oot_season: int = 2025

    def __post_init__(self):
        self.data_cache_dir.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
