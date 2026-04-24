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
    espn_season: int = int(os.getenv("ESPN_SEASON", "2026"))
    espn_swid: str = os.getenv("ESPN_SWID", "")
    espn_s2: str = os.getenv("ESPN_S2", "")

    data_cache_dir: Path = Path(os.getenv("DATA_CACHE_DIR", "./data_cache"))

    # Historical scope for the project.
    data_start_season: int = int(os.getenv("DATA_START_SEASON", "2000"))
    oot_season: int = int(os.getenv("TARGET_SEASON", "2026"))
    recent_history_window: int = int(os.getenv("RECENT_HISTORY_WINDOW", "3"))

    allowed_seasons: tuple = ()
    recent_history_seasons: tuple = ()

    def __post_init__(self):
        self.allowed_seasons = tuple(range(self.data_start_season, self.oot_season + 1))
        recent_start = max(self.data_start_season, self.oot_season - self.recent_history_window + 1)
        self.recent_history_seasons = tuple(range(recent_start, self.oot_season + 1))
        self.data_cache_dir.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
