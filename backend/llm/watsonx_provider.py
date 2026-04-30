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
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            resp = self._model.chat(messages=messages, params=params)
        except Exception as exc:  # network / auth / quota issues
            return f"[watsonx error: {exc}]"
        if isinstance(resp, dict):
            choices = resp.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                return str(message.get("content", ""))
            return resp.get("results", [{}])[0].get("generated_text", "")
        return str(resp)
