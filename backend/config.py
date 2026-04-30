"""Environment + runtime configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _load_streamlit_secrets_into_env():
    """Mirror Streamlit Cloud secrets into os.environ when available.

    Local runs use .env. Hosted Streamlit runs often rely on st.secrets, and
    making this explicit keeps backend modules independent from Streamlit UI
    code while still supporting both deployment styles.
    """
    try:
        import streamlit as st
    except Exception:
        return
    try:
        secrets = st.secrets
    except Exception:
        return

    def set_env(key: str, value):
        if key in os.environ or value is None:
            return
        os.environ[key] = str(value)

    for key in (
        "LLM_PROVIDER",
        "WATSONX_APIKEY",
        "WATSONX_URL",
        "WATSONX_PROJECT_ID",
        "WATSONX_MODEL_ID",
        "CUSTOM_LLM_BASE_URL",
        "CUSTOM_LLM_API_KEY",
        "CUSTOM_LLM_MODEL",
        "ESPN_LEAGUE_ID",
        "ESPN_SEASON",
        "ESPN_SWID",
        "ESPN_S2",
        "DATA_CACHE_DIR",
        "DATA_START_SEASON",
        "TARGET_SEASON",
        "RECENT_HISTORY_WINDOW",
    ):
        try:
            set_env(key, secrets.get(key))
        except Exception:
            pass

    # Also support grouped secrets such as [watsonx] and [espn].
    section_map = {
        "watsonx": {
            "apikey": "WATSONX_APIKEY",
            "url": "WATSONX_URL",
            "project_id": "WATSONX_PROJECT_ID",
            "model_id": "WATSONX_MODEL_ID",
        },
        "espn": {
            "league_id": "ESPN_LEAGUE_ID",
            "season": "ESPN_SEASON",
            "swid": "ESPN_SWID",
            "s2": "ESPN_S2",
        },
    }
    for section, keys in section_map.items():
        try:
            values = secrets.get(section, {})
        except Exception:
            values = {}
        for secret_key, env_key in keys.items():
            try:
                set_env(env_key, values.get(secret_key))
            except Exception:
                pass


_load_streamlit_secrets_into_env()


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
