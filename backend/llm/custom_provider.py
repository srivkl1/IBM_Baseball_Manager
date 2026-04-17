"""OpenAI-compatible endpoint (vLLM, Ollama, LM Studio, OpenAI itself)."""
from __future__ import annotations

from typing import Optional

from backend.config import CONFIG
from .base import LLM


class CustomLLM(LLM):
    name = "custom"

    def __init__(self):
        from openai import OpenAI
        self._client = OpenAI(
            base_url=CONFIG.custom_base_url,
            api_key=CONFIG.custom_api_key or "not-needed",
        )
        self._model = CONFIG.custom_model

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 512,
                 temperature: float = 0.2) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            return f"[custom LLM error: {exc}]"
