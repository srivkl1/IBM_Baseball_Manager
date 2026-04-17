"""IBM watsonx.ai foundation model provider."""
from __future__ import annotations

from typing import Optional

from backend.config import CONFIG
from .base import LLM


class WatsonxLLM(LLM):
    name = "watsonx"

    def __init__(self):
        if not (CONFIG.watsonx_apikey and CONFIG.watsonx_project_id):
            raise RuntimeError(
                "watsonx provider requires WATSONX_APIKEY and WATSONX_PROJECT_ID in .env"
            )
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference

        self._model = ModelInference(
            model_id=CONFIG.watsonx_model_id,
            credentials=Credentials(api_key=CONFIG.watsonx_apikey, url=CONFIG.watsonx_url),
            project_id=CONFIG.watsonx_project_id,
        )

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 512,
                 temperature: float = 0.2) -> str:
        full = f"<<SYS>>\n{system}\n<</SYS>>\n\n{prompt}" if system else prompt
        params = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "decoding_method": "greedy" if temperature == 0 else "sample",
        }
        try:
            resp = self._model.generate_text(prompt=full, params=params)
        except Exception as exc:  # network / auth / quota issues
            return f"[watsonx error: {exc}]"
        if isinstance(resp, dict):
            return resp.get("results", [{}])[0].get("generated_text", "")
        return str(resp)
